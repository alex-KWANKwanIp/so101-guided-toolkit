#!/usr/bin/env python3
"""
Guided evaluation: run a policy episode by episode with lerobot-rollout (episodic strategy), place the object
according to guided_record.py's position modes, mark every episode success / fail with one key, and get the
success rate at the end.

The rollout itself is LeRobot's own lerobot-rollout (every episode is saved as a rollout_ dataset, so you can
watch the videos later). This script only opens a window on the front camera before the first episode, after
each episode and during resets: the green circle shows where to put the object.

Usage (align the scene first with scene_check.py --layout and close it with q):
    python guided_eval.py --policy ~/outputs/train/my_act/checkpoints/050000/pretrained_model --mode standard --rounds 2 --tag act50k \\
        -- --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1
    python guided_eval.py --policy <checkpoint> --mode far --rounds 1 --points 8 --tag act50k_far
    python guided_eval.py --policy <checkpoint> --mode select --points 8 --tag smolvla_select
    python guided_eval.py --policy <checkpoint> --mode standard --rounds 1 --show-plan      # only show the test positions

    Everything after `--` is passed to lerobot-rollout unchanged (smoothing options, extra RTC settings...).
    VLA checkpoints (SmolVLA, π0.5) automatically get --inference.type=rtc and the training --rename_map
    (see policy_args.py); SmolVLA / π0.5 need an environment with `transformers` installed.

Modes (same as guided_record.py): standard 20 points; dense 40; far 16 around the edge; wide 36 over the whole
reachable region; multi / select with several coloured objects. move is for recording only.
--holdout: only the 5 positions that guided_record.py's wide / move modes never recorded, to see how the
policy does on unseen positions (the region placement/wide_region.json and --seed must match the recording).
--rounds: each round tests every point once (shuffled); --points picks fewer, evenly spread points. No jitter.

Each episode:
    1. The window shows the target (green circle). Put the object there and press → (the policy moves on its own).
    2. When the policy is done or clearly failed, press → to end the episode (at most --episode-time seconds).
    3. Mark the result: s = success, f = fail (Esc any time if something is dangerous). → does nothing until
       you mark. To discard and redo this episode press ← (not counted).
    4. The arm returns to its start pose by itself; place the next position and press →.
    5. At the end (or Esc) the success rate is printed and saved in eval_results/<tag>_<mode>_<date time>/:
       results.csv (every episode), summary.txt (overall, per point / condition / instruction, per round),
       plan.json (test positions).

Task instruction box (bottom of the window): VLA policies follow this sentence. Edit it during a reset and
press Enter (or Apply); it is used from the next episode on, and recorded per episode in results.csv. While
typing in the box, s / f / b are just text and arrow keys / Esc are ignored. ACT ignores the instruction.

Keys: s / f work whether the terminal or the window has focus; b (grab background: objects out, arm in the
start pose; needed for object detection) needs the window focused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import config
import guided_record as gr
import policy_args
import scene_check as sc

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "eval_results"


class EvalWindow(gr.GuideWindow):
    def __init__(self, *args, task: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.mark = None
        self.default_task, self.task, self.typing, self.on_task = task, task, False, None
        self._task_box()
        self.root.title("guided_eval - green circle = this position   → start/end   s success   f fail   "
                        "← discard   Esc stop   b background")
        # s / f use a global key listener like LeRobot's arrow keys: they work with the terminal or the window focused
        try:
            from pynput import keyboard

            self._listener = keyboard.Listener(on_press=lambda key: self.set_mark(getattr(key, "char", None)))
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:  # no pynput or no permission (macOS: Accessibility / Input Monitoring)
            print(f"(global key listener failed: {e}; click the window before pressing s / f)", flush=True)

    def _task_box(self) -> None:
        """Instruction box under the image: VLA policies follow this sentence. Enter (or Apply) uses it from the next episode."""
        import tkinter as tk

        box = tk.Frame(self.root)
        box.pack(fill="x", padx=4, pady=4)
        tk.Label(box, text="Task instruction (VLA follows it):").pack(side="left")
        self.task_var = tk.StringVar(value=self.task)
        self.entry = tk.Entry(box, textvariable=self.task_var, width=70, font=("", 12))
        self.entry.pack(side="left", fill="x", expand=True)
        tk.Button(box, text="Apply", command=self.apply_task).pack(side="left", padx=2)
        tk.Button(box, text="Reset to default", command=lambda: (self.task_var.set(self.default_task), self.apply_task())).pack(side="left")
        self.entry.bind("<Return>", lambda _e: self.apply_task())
        self.entry.bind("<FocusIn>", lambda _e: setattr(self, "typing", True))
        self.entry.bind("<FocusOut>", lambda _e: setattr(self, "typing", False))

    def apply_task(self) -> None:
        text = " ".join(self.task_var.get().split())
        if not text:
            self.task_var.set(self.task)
        elif text != self.task:
            self.task = text
            print(f"Task instruction changed to: {text} (used from the next episode)", flush=True)
            if self.on_task:
                self.on_task(text)
        self.root.focus_set()  # leave the box so keys control the evaluation again

    def set_mark(self, char) -> None:
        if char not in ("s", "f") or self.typing:  # s / f typed into the instruction box are not marks
            return
        mark = "success" if char == "s" else "fail"
        if mark != self.mark:  # window and global listener may both fire: print once
            self.mark = mark
            print(f"Marked: {mark}", flush=True)

    def on_key(self, event) -> None:
        if self.typing:  # typing in the box: b, s, f are just text
            return
        super().on_key(event)
        self.set_mark(event.char)


def write_summary(out: Path, rows: list[dict], header: str) -> str:
    done = [r for r in rows if r["result"] in ("success", "fail")]
    lines = [header, ""]
    if not done:
        lines.append("No results were marked.")
    else:
        ok = sum(r["result"] == "success" for r in done)
        lines.append(f"Overall success: {ok}/{len(done)} = {ok / len(done) * 100:.0f}%")
        times = [float(r["duration_s"]) for r in done if r["result"] == "success"]
        if times:
            lines.append(f"Average time of successes: {sum(times) / len(times):.1f} s")
        keys = (("target", "Per point"), ("round", "Per round"))
        if any(r.get("near") for r in done):  # select: distractor next to a target vs far away
            keys = (("near", "Distractor position"), ("round", "Per round"))
        if len({r.get("task") for r in done}) > 1:  # several instructions were tested: split by instruction
            keys = (("task", "Per instruction"),) + keys
        for key, title in keys:
            groups = defaultdict(list)
            for r in done:
                groups[r[key]].append(r["result"] == "success")
            lines.append(f"\n{title}:")
            for k in sorted(groups, key=lambda x: (len(str(x)), str(x))):
                v = groups[k]
                lines.append(f"  {k}: {sum(v)}/{len(v)}")
    skipped = [r for r in rows if r["result"] not in ("success", "fail")]
    if skipped:
        lines.append(f"\nDiscarded or unmarked: {len(skipped)} (not counted)")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    return text


def install_hooks(plan: list[dict], plan_path: Path, out: Path, meta: dict):
    from lerobot.rollout.strategies.episodic import EpisodicStrategy

    orig_policy_loop = EpisodicStrategy._policy_loop
    orig_process = EpisodicStrategy._process_observation_and_notify
    state = {"gui": None, "started": False, "dataset": None, "rows": []}
    fieldnames = ["time", "episode", "round", "target", "target_x", "target_y", "placed_x", "placed_y",
                  "dist_px", "result", "duration_s", "policy", "mode", "near", "task"]
    csv_path = out / "results.csv"

    def gui() -> EvalWindow:
        if state["gui"] is None:
            state["gui"] = EvalWindow("eval", plan, plan_path, task=meta["task"])
            state["gui"].on_task = set_engine_task
        return state["gui"]

    def set_engine_task(task: str) -> None:
        """Swap the task sentence inside the inference engine (both sync and RTC engines read `_task`)."""
        eng = state.get("engine")
        if eng is not None and hasattr(eng, "_task"):
            eng._task = task

    def round_of(ep: int) -> int:
        return ep // meta["points"] + 1

    def watch(ctx, robot, events, ep, title, sub, until, teleop=None):
        """Loop without the policy: refresh the window every frame; with a leader arm, keep teleoperating.
        Ends when until() returns True or on Ctrl+C."""
        g = gui()
        processors = ctx.processors
        snap = None
        while True:
            if g.typing:  # LeRobot also sees arrow keys / Esc typed in the instruction box: ignore them while typing
                keys = ("exit_early", "rerecord_episode", "stop_recording")
                if snap is None:
                    snap = {k: events[k] for k in keys}
                for k in keys:
                    events[k] = snap[k]
            else:
                snap = None
            if until():
                break
            if ctx.runtime.shutdown_event.is_set():  # Ctrl+C
                events["stop_recording"] = True
                return
            t0 = time.perf_counter()
            obs = robot.get_observation()
            if teleop is not None:
                act = processors.teleop_action_processor((teleop.get_action(), obs))
                robot.send_action(processors.robot_action_processor((act, obs)))
            if "front" in obs:
                g.show(obs["front"], ep, title(ep), sub)
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))

    def policy_loop(self, ctx, robot, events, features, fps, control_time_s, dataset, single_task):
        state["dataset"] = dataset
        ep = dataset.num_episodes
        g = gui()
        if not state["started"]:  # first episode: place the object before starting
            state["started"] = True
            print(f"\nPut the object in the green circle and press → to start episode {ep} (Esc to quit)", flush=True)

            def started():
                if events["stop_recording"]:
                    return True
                if events["exit_early"]:
                    events["exit_early"] = False
                    return True
                return False

            watch(ctx, robot, events, ep, lambda e: f"Prepare episode {e} (round {round_of(e)}, target {g.target(e)['id']}): "
                                                    "put the object in the green circle",
                  "When the position is OK press → to start; Esc to quit", started)
            if events["stop_recording"]:
                raise KeyboardInterrupt("Esc pressed before the evaluation started")

        target, placed = g.target(ep), g.last_det
        task = g.task
        state["engine"] = getattr(self, "_engine", None)
        set_engine_task(task)
        if task != meta["task"]:
            print(f"Episode {ep} instruction: {task}", flush=True)
        g.show_recording(ep)
        t0 = time.time()
        orig_policy_loop(self, ctx, robot, events, features, fps, control_time_s, dataset, task)
        duration = time.time() - t0

        # no frame recorded (→ pressed too early, or the first VLA inference was not ready): LeRobot would fail
        # saving an empty episode, so discard it and redo
        if not dataset.has_pending_frames():
            events["rerecord_episode"] = True
            print(f"Episode {ep} recorded no frames (ended after {duration:.1f} s); discarded, redo it. "
                  "After → wait until the arm starts moving; the first VLA inference can take a few seconds.", flush=True)
            return

        # mark the result: s / f; ← discard; Esc stop. → does nothing before marking
        g.mark = None
        warned = {"v": False}

        def marked():
            if g.mark or events["rerecord_episode"] or events["stop_recording"]:
                return True
            if events["exit_early"]:
                events["exit_early"] = False
                if not warned["v"]:
                    print("Not marked yet: press s (success) or f (fail) first", flush=True)
                    warned["v"] = True
            return False

        watch(ctx, robot, events, ep, lambda e: f"Episode {e} ended ({duration:.0f} s): s = success, f = fail",
              "← to discard and redo; Esc if dangerous", marked)
        events["exit_early"] = False  # leave time in the reset to place the next position
        result = "redo" if events["rerecord_episode"] else (g.mark or "unmarked")
        row = {"time": datetime.now().isoformat(timespec="seconds"), "episode": ep, "round": round_of(ep),
               "target": target["id"], "target_x": round(target["x"], 1), "target_y": round(target["y"], 1),
               "placed_x": round(placed["x"], 1) if placed else "", "placed_y": round(placed["y"], 1) if placed else "",
               "dist_px": round(float(placed["d"]), 1) if placed else "", "result": result,
               "duration_s": round(duration, 1), "policy": meta["policy"], "mode": meta["mode"], "task": task,
               "near": ("distractor near target" if target.get("near") else "distractor far") if "objects" in target and
                       any(o["role"] == "distractor" for o in target["objects"]) else ""}
        state["rows"].append(row)
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"Episode {ep} (target {target['id']}): {result}", flush=True)

    def reset_loop(self, ctx, robot, teleop, events, fps, control_time_s, display_data, display_mode,
                   display_compressed):
        ds = state["dataset"]

        def next_ep():
            return ds.num_episodes if events["rerecord_episode"] else ds.num_episodes + 1

        def done():
            if events["stop_recording"]:
                return True
            if events["exit_early"]:
                events["exit_early"] = False
                return True
            return False

        g = gui()
        ep = next_ep()
        redo = " (redo)" if events["rerecord_episode"] else ""
        watch(ctx, robot, events, ep,
              lambda e: f"Reset: put the object in the green circle for episode {e}{redo} "
                        f"(round {round_of(e)}, target {g.target(e)['id']})",
              "When the position is OK press → to start; Esc to stop", done, teleop)
        set_engine_task(g.task)  # switch before the next episode so RTC's background inference starts with it

    def process(self, processors, obs):
        if state["gui"] is not None:
            state["gui"].keep_alive()
        return orig_process(self, processors, obs)

    EpisodicStrategy._policy_loop = policy_loop
    EpisodicStrategy._reset_loop = reset_loop
    EpisodicStrategy._process_observation_and_notify = process
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="checkpoint folder (.../checkpoints/<step>/pretrained_model)")
    ap.add_argument("--mode", choices=[k for k, v in gr.MODES.items() if not v.get("moves")], default="standard")
    ap.add_argument("--holdout", action="store_true", help="only the held-out positions never recorded (wide mode's red ×)")
    ap.add_argument("--rounds", type=int, default=1, help="rounds: each round tests every point once")
    ap.add_argument("--points", type=int, default=None, help="test only this many points (default: all of the mode's)")
    ap.add_argument("--expand", type=float, default=None, help="override the mode's area growth (px)")
    ap.add_argument("--tag", default="", help="name of this evaluation, e.g. act50k_standard")
    ap.add_argument("--task", default=None,
                    help="initial instruction (default: config default_task; generated for multi / select). "
                         "Can be changed in the window's instruction box during the evaluation")
    ap.add_argument("--objects", default=gr.DEFAULT_OBJECTS, help="multi / select: object colours (same as recording)")
    ap.add_argument("--targets", default=gr.DEFAULT_TARGETS, help="select: colours to pick (same as recording)")
    ap.add_argument("--episode-time", type=float, default=60, help="maximum seconds per episode")
    ap.add_argument("--teleop", action="store_true", help="teleoperate during resets (default: the follower returns by itself)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-plan", action="store_true", help="only create and show the test positions")
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]

    policy = Path(os.path.expanduser(args.policy))
    if not (policy / "config.json").exists():
        sc.fail(f"Checkpoint not found: {policy} (point to the pretrained_model folder)")
    if args.holdout:
        args.mode = "holdout"
        m = {**gr.MODES["wide"], "points": len(gr.holdout_points(args.seed))}
    else:
        m = {**gr.MODES[args.mode], **({"points": args.points} if args.points else {}),
             **({"expand": args.expand} if args.expand is not None else {})}
    n = args.rounds * m["points"]
    if m.get("objects"):  # multi / select: new combinations every episode; seed + 500 avoids the recorded ones
        colors = gr.parse_colors(args.objects)
        targets = colors if m.get("all_targets") else gr.parse_colors(args.targets)
        plan = gr.make_object_plan(colors, targets, n, m["points"], 0, args.seed + 500,
                                   prefix="M" if m.get("all_targets") else "S", holdout_seed=args.seed)
        args.task = args.task or gr.object_task(colors, targets)
    else:
        plan = gr.make_plan(m["points"], n, m["expand"], 0, args.seed, m["outside"],  # no jitter when testing
                            region=m.get("region", False), holdout_only=args.holdout)
    args.task = args.task or gr.DEFAULT_TASK

    tag = args.tag or policy.parent.parent.parent.name + "_" + policy.parent.name
    out = RESULT_DIR / f"{tag}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    meta = {"policy": str(policy), "mode": args.mode, "rounds": args.rounds, "points": m["points"],
            "expand": m["expand"], "task": args.task, "start_episode": 0, "created": datetime.now().isoformat(timespec="seconds")}
    plan_path.write_text(json.dumps({**meta, "targets": plan}, indent=1, ensure_ascii=False))
    img = gr.plan_image(f"eval_{args.mode}_{m['points']}", plan)
    print(f"Test positions: {m['points']} points × {args.rounds} round(s) = {n} episodes  plan: {plan_path}  image: {img}")
    if args.show_plan:
        return

    safe_tag = "".join(c if c.isalnum() or c == "_" else "_" for c in tag)
    argv = [
        "lerobot-rollout", "--strategy.type=episodic", f"--policy.path={policy}", *config.robot_args(),
        f"--dataset.repo_id={config.repo_id('rollout_eval_' + safe_tag)}", f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={n}", f"--dataset.episode_time_s={args.episode_time}", "--dataset.reset_time_s=3600",
        "--dataset.push_to_hub=false", "--dataset.streaming_encoding=true", "--dataset.encoder_threads=2",
        "--dataset.rgb_encoder.vcodec=h264", "--display_data=false",
    ]
    if args.teleop:
        argv += config.teleop_args()
    if policy_args.policy_type(policy) not in policy_args.VLA_TYPES:
        print("Note: this policy is not a VLA (e.g. ACT) and ignores the instruction; changing it has no effect.")
    for a in policy_args.rollout_args(policy):  # VLA: add RTC and camera renaming unless given explicitly
        if not any(x.split("=")[0] == a.split("=")[0] for x in extra):
            argv.append(a)
    pdev = config.load().get("policy_device")
    if pdev and not any(a.startswith("--policy.device") for a in extra):  # e.g. mps on a Mac
        argv.append(f"--policy.device={pdev}")
    argv += extra
    print("\nRunning: " + " ".join(a if " " not in a else repr(a) for a in argv) + "\n", flush=True)

    state = install_hooks(plan, plan_path, out, meta)
    import lerobot.scripts.lerobot_rollout as lr

    sys.argv = argv
    try:
        lr.main()
    except KeyboardInterrupt as e:
        print(f"\nStopped: {e}")
    finally:
        header = (f"Policy: {policy}\nMode: {args.mode}, {m['points']} points × {args.rounds} round(s), task \"{args.task}\"\n"
                  f"rollout arguments: {' '.join(extra) or '(none)'}\nTime: {datetime.now():%Y-%m-%d %H:%M}")
        print("\n" + "=" * 60 + "\n" + write_summary(out, state["rows"], header) + "\n" + "=" * 60)
        print(f"Results: {out} (results.csv, summary.txt); videos in "
              f"~/.cache/huggingface/lerobot/{config.hf_user()}/rollout_eval_{safe_tag}_<date time>")


if __name__ == "__main__":
    main()
