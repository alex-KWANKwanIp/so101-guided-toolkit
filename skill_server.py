#!/usr/bin/env python3
"""
Skill server: wraps trained LeRobot policies as *named skills* that an AI agent or a BCI can call over HTTP.

Usage (in your LeRobot environment; VLA skills such as SmolVLA need one with `transformers`):
    python skill_server.py                                   # local only (127.0.0.1:8000)
    python skill_server.py --host 0.0.0.0 --token SECRET     # reachable from the network, requires the token

API (all JSON; with --token every request needs the header `Authorization: Bearer SECRET`):
    GET  /skills                list skills
    GET  /status                is the arm idle or running
    POST /jobs                  run a skill, body: {"skill": "name", "request_id": "unique id"}
    GET  /jobs/<job_id>         job state
    POST /jobs/<job_id>/cancel  cancel a job (the arm slowly returns to where it started)
    POST /stop                  stop whatever is running

Job states:
    running    running
    finished   the policy ran for the given time without errors. This does NOT mean the task succeeded:
               verify with a camera!
    failed     error (see log_tail or skill_logs/)
    cancelled  cancelled
    timeout    exceeded the time limit and was stopped

Every job runs, in the background:
    lerobot-rollout --strategy.type=base --policy.path=<checkpoint> ... --duration=<seconds>

The robot (type, port, id, cameras) comes from config.json. Skills live in skills.json (register_skill.py
writes them). A skill may add:
    "policy_overrides": {"n_action_steps": 20}
        → --policy.n_action_steps=20 (ACT: re-plan every 20 steps, reacts faster when the object moves)
    "policy_overrides": {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01}
        → ACT temporal ensembling (re-plan every step and average)
    "rollout_args": ["--inference.type=rtc", "--rename_map={...}"]
        → extra lerobot-rollout arguments (filled in automatically for VLA policies by register_skill.py)
"""

import argparse
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

HERE = Path(__file__).resolve().parent
# output of every run is kept here for debugging
LOG_DIR = HERE / "skill_logs"

# loading the policy and connecting arm and cameras takes time, so the limit is longer than the skill itself
STARTUP_ALLOWANCE_S = 60


def tail(path: str, n: int = 15) -> list[str]:
    """Last n lines of a log file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()[-n:]]
    except FileNotFoundError:
        return []


def check_policy_overrides(name: str, overrides) -> None:
    """Validate policy_overrides from skills.json; refuse to start on errors."""
    if not isinstance(overrides, dict):
        raise SystemExit(f"Skill '{name}': policy_overrides must be {{name: value}}, e.g. {{\"n_action_steps\": 20}}")
    for key, value in overrides.items():
        if not re.fullmatch(r"[a-z_][a-z0-9_.]*", key) or key in ("path", "type"):
            raise SystemExit(f"Skill '{name}': invalid policy_overrides key '{key}'")
        if not isinstance(value, (int, float, str, bool)):
            raise SystemExit(f"Skill '{name}': policy_overrides['{key}'] must be a number, string or true/false")
    # ACT rule: temporal ensembling requires n_action_steps == 1
    if overrides.get("temporal_ensemble_coeff") is not None and overrides.get("n_action_steps", 100) != 1:
        raise SystemExit(f"Skill '{name}': temporal_ensemble_coeff requires n_action_steps = 1")


def cli_value(value) -> str:
    """JSON value → command-line value (lower-case true/false)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class SkillRunner:
    """Starts, tracks and stops skills. Only one skill runs at a time."""

    def __init__(self, skills_cfg: dict):
        cfg = config.load()
        self.robot = {**cfg["robot"], "cameras": config.cameras_arg(), **skills_cfg.get("robot", {})}
        self.skills = skills_cfg["skills"]
        self.policy_device = cfg.get("policy_device")
        self.jobs: dict[str, dict] = {}
        self.request_ids: dict[str, str] = {}  # request_id -> job_id, so a retried request does not move the arm twice
        self.current_job_id: str | None = None
        self.lock = threading.Lock()

        self.rollout_bin = shutil.which("lerobot-rollout")
        if self.rollout_bin is None:
            raise SystemExit("lerobot-rollout not found: activate your LeRobot environment first")
        vla = [n for n, sk in self.skills.items() if sk.get("rollout_args")]
        if vla:
            import importlib.util

            if importlib.util.find_spec("transformers") is None:
                raise SystemExit(f"Skills {vla} use a VLA policy (SmolVLA / π0.5) and need `transformers`: "
                                 "start the server in an environment that has it")

        for name, skill in self.skills.items():
            for key in ("policy_path", "task", "max_duration_s"):
                if key not in skill:
                    raise SystemExit(f"Skill '{name}' in skills.json is missing '{key}'")
            check_policy_overrides(name, skill.get("policy_overrides", {}))

        LOG_DIR.mkdir(exist_ok=True)

    # ---------- public ----------

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": name,
                "description": skill.get("description", ""),
                "max_duration_s": skill["max_duration_s"],
                "trained_info": skill.get("trained_info"),
                "policy_overrides": skill.get("policy_overrides", {}),
            }
            for name, skill in self.skills.items()
        ]

    def status(self) -> dict:
        with self.lock:
            job = self.jobs.get(self.current_job_id) if self.current_job_id else None
            return {
                "state": "running" if job else "idle",
                "current_job": self._public(job) if job else None,
            }

    def start(self, skill_name: str, request_id: str | None) -> tuple[int, dict]:
        with self.lock:
            # same request_id again (e.g. a network retry): return the existing job, the arm does not move again
            if request_id and request_id in self.request_ids:
                return HTTPStatus.OK, self._public(self.jobs[self.request_ids[request_id]])

            if skill_name not in self.skills:
                return HTTPStatus.NOT_FOUND, {
                    "error": f"No such skill: '{skill_name}'",
                    "skills": list(self.skills),
                }

            if self.current_job_id is not None:
                return HTTPStatus.CONFLICT, {
                    "error": "The arm is busy with another job",
                    "current_job": self.current_job_id,
                }

            skill = self.skills[skill_name]
            policy_path = Path(os.path.expanduser(skill["policy_path"]))
            if not policy_path.exists():
                return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Checkpoint not found: {policy_path}"}

            job_id = uuid.uuid4().hex[:8]
            log_path = LOG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{job_id}_{skill_name}.log"
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                self._build_command(skill, policy_path),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # so it can be stopped on its own, like pressing Ctrl+C
            )

            job = {
                "job_id": job_id,
                "skill": skill_name,
                "request_id": request_id,
                "state": "running",
                "started_at": time.time(),
                "ended_at": None,
                "return_code": None,
                "log_path": str(log_path),
                "_proc": proc,
                "_log_file": log_file,
                "_cancel_requested": False,
                "_timed_out": False,
            }
            self.jobs[job_id] = job
            if request_id:
                self.request_ids[request_id] = job_id
            self.current_job_id = job_id

        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        print(f"[skill-server] started skill '{skill_name}' (job {job_id})")
        return HTTPStatus.ACCEPTED, self._public(job)

    def get_job(self, job_id: str) -> tuple[int, dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return HTTPStatus.NOT_FOUND, {"error": f"No such job: '{job_id}'"}
            return HTTPStatus.OK, self._public(job)

    def cancel(self, job_id: str) -> tuple[int, dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return HTTPStatus.NOT_FOUND, {"error": f"No such job: '{job_id}'"}
            if job["state"] != "running":
                return HTTPStatus.OK, self._public(job)
            job["_cancel_requested"] = True

        threading.Thread(target=self._stop_process, args=(job,), daemon=True).start()
        print(f"[skill-server] cancelling job {job_id}")
        return HTTPStatus.ACCEPTED, self._public(job)

    def stop_current(self) -> tuple[int, dict]:
        with self.lock:
            job_id = self.current_job_id
        if job_id is None:
            return HTTPStatus.OK, {"state": "idle"}
        return self.cancel(job_id)

    def shutdown(self) -> None:
        """Stop a running skill safely before the server exits."""
        with self.lock:
            job = self.jobs.get(self.current_job_id) if self.current_job_id else None
            if job:
                job["_cancel_requested"] = True
        if job:
            print("[skill-server] stopping the running skill...")
            self._stop_process(job)
            job["_proc"].wait()

    # ---------- internal ----------

    def _build_command(self, skill: dict, policy_path: Path) -> list[str]:
        return [
            self.rollout_bin,
            "--strategy.type=base",
            f"--policy.path={policy_path}",
            f"--robot.type={self.robot['type']}",
            f"--robot.port={self.robot['port']}",
            f"--robot.id={self.robot['id']}",
            f"--robot.cameras={self.robot['cameras']}",
            f"--task={skill['task']}",
            f"--duration={skill['max_duration_s']}",
        ] + [f"--policy.{key}={cli_value(value)}" for key, value in skill.get("policy_overrides", {}).items()] + (
            [f"--policy.device={self.policy_device}"]
            if self.policy_device and "device" not in skill.get("policy_overrides", {}) else []) + \
            list(skill.get("rollout_args", []))  # VLA: --inference.type=rtc, --rename_map (filled by register_skill.py)

    def _watch(self, job: dict) -> None:
        """Wait in the background for the skill to end and update its state."""
        proc = job["_proc"]
        limit = self.skills[job["skill"]]["max_duration_s"] + STARTUP_ALLOWANCE_S
        try:
            proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            job["_timed_out"] = True
            self._stop_process(job)
            proc.wait()

        with self.lock:
            job["return_code"] = proc.returncode
            job["ended_at"] = time.time()
            job["_log_file"].close()
            if job["_timed_out"]:
                job["state"] = "timeout"
            elif job["_cancel_requested"]:
                job["state"] = "cancelled"
            elif proc.returncode == 0:
                job["state"] = "finished"
            else:
                job["state"] = "failed"
            if self.current_job_id == job["job_id"]:
                self.current_job_id = None
        print(f"[skill-server] job {job['job_id']} ended: {job['state']}")

    def _stop_process(self, job: dict) -> None:
        """Ask it to stop like Ctrl+C first (the arm returns to its start pose); escalate only if it does not respond."""
        proc = job["_proc"]
        for sig, wait_s in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            if proc.poll() is not None:
                return
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue

    def _public(self, job: dict) -> dict:
        info = {k: v for k, v in job.items() if not k.startswith("_")}
        info["log_tail"] = tail(job["log_path"])
        return info


def make_handler(runner: SkillRunner, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not token:
                return True
            got = self.headers.get("Authorization", "")
            if hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
                return True
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "missing or wrong token (Authorization: Bearer <token>)"})
            return False

        def _parts(self) -> list[str]:
            return [p for p in self.path.split("?")[0].split("/") if p]

        def do_GET(self) -> None:
            if not self._authorized():
                return
            parts = self._parts()
            if parts == ["skills"]:
                self._send(HTTPStatus.OK, {"skills": runner.list_skills()})
            elif parts == ["status"]:
                self._send(HTTPStatus.OK, runner.status())
            elif len(parts) == 2 and parts[0] == "jobs":
                self._send(*runner.get_job(parts[1]))
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "no such endpoint"})

        def do_POST(self) -> None:
            if not self._authorized():
                return
            parts = self._parts()
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
            except (ValueError, json.JSONDecodeError):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "body must be JSON"})
                return

            if parts == ["jobs"]:
                self._send(*runner.start(body.get("skill", ""), body.get("request_id")))
            elif len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                self._send(*runner.cancel(parts[1]))
            elif parts == ["stop"]:
                self._send(*runner.stop_current())
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "no such endpoint"})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[http] {self.address_string()} {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose LeRobot policies as callable skills over HTTP")
    parser.add_argument("--config", default=str(HERE / "skills.json"), help="skills file")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="default: this computer only. 0.0.0.0 lets other computers on the network call it (use --token!)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=os.environ.get("SKILL_SERVER_TOKEN"),
                        help="require 'Authorization: Bearer <token>' on every request (or set SKILL_SERVER_TOKEN)")
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise SystemExit(f"{args.config} not found: register a checkpoint first with register_skill.py "
                         "(or copy skills.example.json to skills.json)")
    skills_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runner = SkillRunner(skills_cfg)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runner, args.token))

    print(f"[skill-server] listening on http://{args.host}:{args.port}")
    print(f"[skill-server] skills: {', '.join(runner.skills) or '(none yet: use register_skill.py)'}")
    if args.host not in ("127.0.0.1", "localhost") and not args.token:
        print("[skill-server] ⚠ reachable from the network WITHOUT a token: anyone on the network can move the arm. "
              "Restart with --token.")
    print("[skill-server] Ctrl+C to quit")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runner.shutdown()
        print("[skill-server] stopped")


if __name__ == "__main__":
    main()
