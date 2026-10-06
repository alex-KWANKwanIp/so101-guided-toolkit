#!/usr/bin/env python3
"""
Register a trained checkpoint as a skill (run once after each training run you have tested).

Examples:
    python register_skill.py --name pick_cube_to_box \\
        --checkpoint ~/outputs/train/my_act/checkpoints/030000 --intent cube --eval 8/10

    # ACT: re-plan more often (reacts in time when the object is moved)
    python register_skill.py --name pick_cube_to_box --checkpoint ... --n-action-steps 20
    python register_skill.py --name pick_cube_to_box --checkpoint ... --temporal-ensemble

    # SmolVLA: RTC and camera renaming are added automatically; extra rollout options with --rollout-arg
    python register_skill.py --name pick_cube_vla --checkpoint ~/outputs/train/my_smolvla/checkpoints/040000 \\
        --intent cube --rollout-arg=--inference.rtc.execution_horizon=20

What it does:
    1. finds the checkpoint and checks it is complete (checkpoints/last is resolved to the real step)
    2. checks that the cameras the policy needs match the robot cameras (config.json)
    3. reads the task sentence from the training dataset and derives a max run time from the average episode length
    4. updates skills.json (the previous one is kept as skills.json.bak)
    5. with --intent, updates intents.json so a BCI / agent intent maps to this skill
    6. appends the registration to registry_history.csv (handy for reports)

Restart skill_server.py afterwards for the change to take effect.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import config
import policy_args

HERE = Path(__file__).resolve().parent
LEROBOT_HOME = Path(
    os.environ.get("HF_LEROBOT_HOME")
    or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "lerobot"
)


def ok(msg: str) -> None:
    print(f"✓ {msg}")


def warn(msg: str) -> None:
    print(f"! {msg}")


def fail(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def short_path(p: Path) -> str:
    """Write /home/<user>/... as ~/... for readability."""
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def resolve_checkpoint(path_str: str) -> Path:
    """Accept a training output folder, checkpoints/<step>, checkpoints/last or pretrained_model; return the model folder."""
    p = Path(os.path.expanduser(path_str))
    if not p.exists():
        fail(f"Path not found: {p}")
    if (p / "checkpoints").is_dir():  # whole training folder: use the latest checkpoint
        p = p / "checkpoints" / "last"
    # `last` is only a link. Resolve it to the real step so continued training never silently swaps the skill's model
    p = p.resolve()
    if p.name != "pretrained_model" and (p / "pretrained_model").is_dir():
        p = p / "pretrained_model"
    for name in ("config.json", "model.safetensors"):
        if not (p / name).exists():
            fail(f"{name} is missing in {p}; this is not a complete model folder")
    return p


def read_step(model_dir: Path) -> int | None:
    step_file = model_dir.parent / "training_state" / "training_step.json"
    if step_file.exists():
        return json.loads(step_file.read_text())["step"]
    return int(model_dir.parent.name) if model_dir.parent.name.isdigit() else None


def policy_cameras(model_dir: Path) -> tuple[str, dict]:
    """Policy type and the cameras it needs (name → (height, width)) from the checkpoint's config.json."""
    cfg = json.loads((model_dir / "config.json").read_text())
    cams = {}
    for key, feat in cfg.get("input_features", {}).items():
        if feat.get("type") == "VISUAL" and key.startswith("observation.images."):
            shape = feat.get("shape", [])
            cams[key.removeprefix("observation.images.")] = (
                (shape[1], shape[2]) if len(shape) == 3 else None
            )
    return cfg.get("type", "?"), cams


def robot_cameras(cameras_str: str) -> dict | None:
    """--robot.cameras string → {name: (height, width)}."""
    try:
        import yaml
    except ImportError:
        warn("PyYAML not installed, skipping the camera check")
        return None
    cams = yaml.safe_load(cameras_str) or {}
    result = {}
    for name, c in cams.items():
        size = (c.get("height"), c.get("width"))
        if c.get("use_rgb", True):
            result[name] = size
        if c.get("use_depth", False):  # depth cameras add <name>_depth
            result[f"{name}_depth"] = size
    return result


def dataset_info(model_dir: Path) -> dict:
    """Find the training dataset from train_config.json; read its task sentences and average episode length."""
    info = {"repo_id": None, "episodes": None, "avg_s": None, "tasks": None}
    tc_path = model_dir / "train_config.json"
    if not tc_path.exists():
        warn("train_config.json not found, cannot read dataset information")
        return info
    ds = json.loads(tc_path.read_text()).get("dataset", {})
    info["repo_id"] = ds.get("repo_id")
    root = Path(ds["root"]) if ds.get("root") else LEROBOT_HOME / (ds.get("repo_id") or "")

    meta = root / "meta" / "info.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        info["episodes"] = m.get("total_episodes")
        if m.get("total_episodes") and m.get("fps"):
            info["avg_s"] = m["total_frames"] / m["total_episodes"] / m["fps"]
    else:
        warn(f"Dataset not found on this computer: {root}")

    tasks_file = root / "meta" / "tasks.parquet"
    if tasks_file.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(tasks_file)
            tasks = df["task"] if "task" in df.columns else df.index
            info["tasks"] = [str(t) for t in tasks]
        except Exception as err:  # e.g. no pandas / pyarrow
            warn(f"Could not read the dataset's task sentences ({err})")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a trained checkpoint as a skill")
    parser.add_argument("--name", required=True, help="skill name (letters, digits, underscores), e.g. fetch_phone")
    parser.add_argument("--checkpoint", required=True, help="checkpoint, e.g. ~/outputs/train/xxx/checkpoints/060000")
    parser.add_argument("--intent", help="intent name, e.g. phone (what the BCI / agent calls)")
    parser.add_argument("--description", help="skill description")
    parser.add_argument("--task", help="task sentence (default: read from the dataset)")
    parser.add_argument("--max-duration", type=int, help="max run time in seconds (default: estimated from the dataset)")
    parser.add_argument("--eval", help="your measured success rate, e.g. 8/10 (recorded for reports)")
    parser.add_argument("--n-action-steps", type=int,
                        help="ACT: re-plan every N steps (default 100; 20 recommended). Default: keep the skill's setting")
    parser.add_argument("--temporal-ensemble", nargs="?", type=float, const=0.01, metavar="COEFF",
                        help="ACT: re-plan every step and average (temporal ensembling), coefficient 0.01 by default")
    parser.add_argument("--default-actions", action="store_true",
                        help="clear --n-action-steps / --temporal-ensemble and use the model defaults")
    parser.add_argument("--rollout-arg", action="append", default=[], metavar="ARG",
                        help="extra lerobot-rollout argument (repeatable), e.g. --rollout-arg=--inference.rtc.execution_horizon=20")
    parser.add_argument("--config", default=str(HERE / "skills.json"))
    parser.add_argument("--intents", default=str(HERE / "intents.json"))
    args = parser.parse_args()
    if sum(x is not None for x in (args.n_action_steps, args.temporal_ensemble)) + args.default_actions > 1:
        fail("Use only one of --n-action-steps, --temporal-ensemble, --default-actions")

    if not args.name.replace("_", "").isalnum() or not args.name.isascii():
        fail("Skill names may only contain letters, digits and underscores, e.g. fetch_phone")

    # 1. checkpoint
    model_dir = resolve_checkpoint(args.checkpoint)
    step = read_step(model_dir)
    policy_type, p_cams = policy_cameras(model_dir)
    ok(f"Checkpoint: {short_path(model_dir)} ({policy_type}, step {step})")

    # 2. cameras
    skills_path = Path(args.config)
    skills_cfg = json.loads(skills_path.read_text(encoding="utf-8")) if skills_path.exists() else {"skills": {}}
    skills_cfg.setdefault("skills", {})
    r_cams = robot_cameras(skills_cfg.get("robot", {}).get("cameras") or config.cameras_arg())
    is_vla = policy_type in policy_args.VLA_TYPES
    if is_vla:
        # VLA: training renamed front / wrist to camera1 / camera2 (rename_map); an unused camera3 is fine,
        # and images are resized, so resolution is not compared
        inv = {v.removeprefix("observation.images."): k.removeprefix("observation.images.")
               for k, v in policy_args.rename_map(model_dir).items()}
        p_cams = {inv[n]: None for n in p_cams if n in inv} if inv else {n: None for n in p_cams}
    if r_cams is not None:
        missing = sorted(set(p_cams) - set(r_cams))
        if missing:
            fail(f"The policy needs cameras {missing} but the robot has {sorted(r_cams)}. "
                 "Camera names must match the ones used for recording")
        for name, size in p_cams.items():
            if size and r_cams[name] != size:
                fail(f"Camera {name} resolution differs: policy {size[1]}x{size[0]}, robot {r_cams[name][1]}x{r_cams[name][0]}")
        extra = sorted(set(r_cams) - set(p_cams))
        if extra:
            warn(f"The robot has cameras this policy does not use: {extra} (works, just opens extra cameras)")
        ok(f"Cameras match: {sorted(p_cams)}")

    # 3. task sentence and run time
    ds = dataset_info(model_dir)
    task = args.task
    if ds["tasks"]:
        if task is None:
            if len(ds["tasks"]) > 1:
                fail(f"The dataset has several task sentences; pick one with --task: {ds['tasks']}")
            task = ds["tasks"][0]
        elif task not in ds["tasks"]:
            warn(f"--task differs from the dataset's sentences: {ds['tasks']}")
    if not task:
        fail("Could not read the task sentence; pass --task with the single_task used for recording")
    ok(f"Task sentence: {task}")

    max_duration = args.max_duration
    if max_duration is None:
        if ds["avg_s"]:
            max_duration = max(30, math.ceil(ds["avg_s"] * 2.5 / 10) * 10)
            ok(f"Average demonstration {ds['avg_s']:.1f} s → max run time {max_duration} s")
        else:
            max_duration = 60
            warn("Could not estimate the run time; using 60 s")

    # 4. runtime policy settings (policy_overrides) and extra rollout arguments
    old = skills_cfg["skills"].get(args.name)
    if args.default_actions:
        overrides = {}
    elif args.temporal_ensemble is not None:
        overrides = {"n_action_steps": 1, "temporal_ensemble_coeff": args.temporal_ensemble}
    elif args.n_action_steps is not None:
        if args.n_action_steps < 1:
            fail("--n-action-steps must be an integer >= 1")
        overrides = {"n_action_steps": args.n_action_steps}
    else:
        overrides = (old or {}).get("policy_overrides", {})
    if overrides:
        if is_vla and any(k in overrides for k in ("temporal_ensemble_coeff", "n_action_steps")):
            warn("--temporal-ensemble / --n-action-steps are for ACT; VLA policies use RTC, so they are ignored")
            overrides = {k: v for k, v in overrides.items() if k not in ("temporal_ensemble_coeff", "n_action_steps")}
        if overrides:
            ok(f"Runtime policy settings: {overrides}")
    rollout = policy_args.rollout_args(model_dir)
    rollout += [a for a in args.rollout_arg if a.split("=")[0] not in {r.split("=")[0] for r in rollout}]
    if rollout:
        ok(f"Extra lerobot-rollout arguments: {' '.join(rollout)}")

    # 5. update skills.json
    registered_at = time.strftime("%Y-%m-%d %H:%M")
    skills_cfg["skills"][args.name] = {
        "description": args.description or (old or {}).get("description") or task,
        "policy_path": short_path(model_dir),
        "task": task,
        "max_duration_s": max_duration,
        **({"policy_overrides": overrides} if overrides else {}),
        **({"rollout_args": rollout} if rollout else {}),
        "trained_info": {
            "policy_type": policy_type,
            "step": step,
            "dataset": ds["repo_id"],
            "eval": args.eval,
            "registered_at": registered_at,
        },
    }
    if skills_path.exists():
        shutil.copy2(skills_path, skills_path.with_suffix(".json.bak"))
    skills_path.write_text(json.dumps(skills_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if old:
        ok(f"Updated skill {args.name}: {old['policy_path']} → {short_path(model_dir)}")
    else:
        ok(f"Added skill {args.name}")

    # 6. update intents.json
    if args.intent:
        intents_path = Path(args.intents)
        intents = json.loads(intents_path.read_text(encoding="utf-8")) if intents_path.exists() else {}
        intents[args.intent] = args.name
        intents_path.write_text(json.dumps(intents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ok(f"Intent '{args.intent}' → skill {args.name}")

    # 7. history
    history = HERE / "registry_history.csv"
    new_file = not history.exists()
    with open(history, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["registered_at", "skill", "intent", "policy_type", "step", "eval",
                             "max_duration_s", "task", "dataset", "dataset_episodes", "policy_path",
                             "policy_overrides"])
        writer.writerow([registered_at, args.name, args.intent or "", policy_type, step, args.eval or "",
                         max_duration, task, ds["repo_id"] or "", ds["episodes"] or "", short_path(model_dir),
                         json.dumps(overrides) if overrides else ""])

    print("\nNext:")
    note = " (in an environment with `transformers`, needed for VLA policies)" if rollout else ""
    print(f"  1. Restart the skill server: Ctrl+C in its terminal, then python skill_server.py{note}")
    if args.intent:
        print(f"  2. Test: python intent_client.py {args.intent}")
    else:
        print(f"  2. Test: curl -X POST http://127.0.0.1:8000/jobs -H 'Content-Type: application/json' "
              f"-d '{{\"skill\": \"{args.name}\"}}'")


if __name__ == "__main__":
    main()
