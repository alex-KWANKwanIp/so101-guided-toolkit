#!/usr/bin/env python3
"""
Placement guide: while recording (or testing a policy), mark on the live front-camera image *where the object
should go this time*, detect where it is now, and tell you which way to move it.

It first analyses a dataset: from the first frame of every episode it finds where the object was, works out
which spots have already been recorded (A, B, C...), then plans the next batch of positions.

Usage (like scene_check.py, a camera can only be opened by one program at a time):
    python placement_guide.py --dataset PATH --analyze                 # analyse only: list recorded spots, save a coverage image
    python placement_guide.py --dataset PATH --plan fill --count 20    # plan 20 *new* spots (filling the gaps), live guidance
    python placement_guide.py --dataset PATH --plan anchors --repeat 2 # go back to recorded spots A, B, C... twice each (e.g. for testing)
    python placement_guide.py --dataset PATH --plan-file placement/xxx_plan_....json   # continue a saved plan
    python placement_guide.py --dataset PATH --plan fill --count 20 --record          # guided recording: place, press space, record one episode

The --analyze step also writes placement/<dataset>_positions.json, which guided_record.py uses for its
standard / dense / far modes (set `positions_file` in config.json or pass --positions there).

Window keys:
    b      grab background (remove the object, arm in start pose, then press b; detection becomes more reliable)
    space  position OK: without --record just log it and move on; with --record start recording this episode
    n / p  next / previous target      s save a screenshot      q or Esc quit

--record flow (each episode):
    1. The window shows the target; move the object into the green circle (green = OK), arm in start pose.
    2. Press space: the window hides, releases the camera and runs lerobot-record for one episode (--resume).
    3. Record as usual: move the leader arm after "Recording episode N"; when done and back in the start pose
       press →; ← to redo; Esc to stop.
    4. You are back in the guide with the next target. Target vs actual position is logged in
       placement/<dataset>_placement_log.csv. (Each episode reconnects arm and cameras: about 10 s extra.)

On screen: grey dots = positions in the dataset; letters = recorded spots (A, B...); green circle = this target;
yellow box = detected object; arrow = where to move it. Distances in cm are estimates (layout.px_per_cm).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
import scene_check as sc

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "placement"
CFG = config.load()
DEFAULT_DATASET = CFG["reference_dataset"]
FONTS = [("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),  # Linux
         ("/System/Library/Fonts/Helvetica.ttc", 0),              # macOS
         ("C:/Windows/Fonts/arial.ttf", 0)]                       # Windows
PX_PER_CM = float(CFG["layout"]["px_per_cm"])
DIFF_THRESHOLD = 40  # difference from the background (0-255) that counts as "something is there"
CLUSTER_PX = 25  # positions closer than this count as the same spot (A, B...)


# ---------- object detection ----------

def detect_object(rgb: np.ndarray, bg: np.ndarray, roi: tuple[int, int, int, int], area: float) -> dict | None:
    """Compare with the background (frame without the object); return the blob inside the work area whose size
    is closest to the object's. Returns centre and bounding box."""
    diff = cv2.absdiff(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)).max(axis=2)
    mask = cv2.morphologyEx((diff > DIFF_THRESHOLD).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    x0, y0, x1, y1 = roi
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    n, _, stats, cent = cv2.connectedComponentsWithStats(clipped)
    best = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 0.35 * area <= a <= 3.0 * area and (best is None or abs(a - area) < abs(best["area"] - area)):
            best = {"x": float(cent[i][0]), "y": float(cent[i][1]), "area": int(a), "bbox": stats[i, :4].tolist()}
    return best


# ---------- dataset analysis ----------

def analyze_dataset(dataset: Path, refresh: bool = False) -> dict:
    """Object position in the first frame of every episode; cached in placement/<dataset>_positions.json."""
    OUT_DIR.mkdir(exist_ok=True)
    cache = OUT_DIR / f"{dataset.name}_positions.json"
    n = sc.num_episodes(dataset)
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        if data.get("num_episodes") == n:
            return data
    print(f"Analysing object positions in {dataset.name} ({n} episodes)...")
    frames = [sc.load_reference(dataset, e, "front") for e in range(n)]
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)  # the object moves, so the median is the empty background
    cv2.imwrite(str(OUT_DIR / f"{dataset.name}_background.png"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
    h, w = bg.shape[:2]
    episodes = []
    for e, f in enumerate(frames):
        d = detect_object(f, bg, (0, 0, w, h), 1700)
        if d is None:
            print(f"  episode {e}: object not found (skipped)")
            continue
        episodes.append({"episode": e, **d})
    if not episodes:
        sc.fail("No object positions found in the dataset.")
    # cluster: consecutive episodes placed close together are one spot (people record one spot several times in a row);
    # then merge clusters with nearby centres (e.g. coming back to A later). First spot is A, next B...
    groups: list[list[dict]] = []
    for ep in episodes:
        if groups:
            cx, cy = np.mean([p["x"] for p in groups[-1]]), np.mean([p["y"] for p in groups[-1]])
            if np.hypot(ep["x"] - cx, ep["y"] - cy) < CLUSTER_PX:
                groups[-1].append(ep)
                continue
        groups.append([ep])
    anchors: list[dict] = []
    for g in groups:
        gx, gy = float(np.mean([p["x"] for p in g])), float(np.mean([p["y"] for p in g]))
        near = next((a for a in anchors if np.hypot(a["x"] - gx, a["y"] - gy) < CLUSTER_PX / 2), None)
        if near is None:
            anchors.append({"name": chr(ord("A") + len(anchors)), "x": gx, "y": gy, "episodes": [p["episode"] for p in g]})
        else:
            near["episodes"] += [p["episode"] for p in g]
            pts = [(p["x"], p["y"]) for p in episodes if p["episode"] in near["episodes"]]
            near["x"], near["y"] = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
    data = {
        "dataset": str(dataset), "num_episodes": n, "size": [w, h],
        "object_area": float(np.median([e["area"] for e in episodes])),
        "episodes": episodes, "anchors": anchors,
    }
    cache.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return data


def load_background(dataset: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(OUT_DIR / f"{dataset.name}_background.png")), cv2.COLOR_BGR2RGB)


def work_roi(pos: dict, margin: int = 110) -> tuple[int, int, int, int]:
    w, h = pos["size"]
    xs = [e["x"] for e in pos["episodes"]]
    ys = [e["y"] for e in pos["episodes"]]
    return (max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin),
            min(w, int(max(xs)) + margin), min(h, int(max(ys)) + margin))


# ---------- planning ----------

def make_plan(pos: dict, mode: str, count: int, repeat: int, expand: float) -> list[dict]:
    if mode == "anchors":
        return [{"id": f"{a['name']}{k + 1}", "x": a["x"], "y": a["y"], "kind": f"recorded spot {a['name']}"}
                for k in range(repeat) for a in pos["anchors"]]
    # fill: inside the area enclosed by recorded positions (grown by `expand` px), repeatedly pick the point
    # farthest from everything recorded or planned so far
    pts = np.array([[e["x"], e["y"]] for e in pos["episodes"]], np.float32)
    hull = cv2.convexHull(pts)
    xs = np.arange(pts[:, 0].min() - expand, pts[:, 0].max() + expand + 1, 3)
    ys = np.arange(pts[:, 1].min() - expand, pts[:, 1].max() + expand + 1, 3)
    cand = np.array([(x, y) for y in ys for x in xs
                     if cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -expand], np.float32)
    taken = [tuple(p) for p in pts]
    plan = []
    for k in range(count):
        d = np.min(np.linalg.norm(cand[:, None, :] - np.array(taken, np.float32)[None], axis=2), axis=1)
        i = int(np.argmax(d))
        x, y = float(cand[i, 0]), float(cand[i, 1])
        near = min(pos["anchors"], key=lambda a: np.hypot(a["x"] - x, a["y"] - y))
        plan.append({"id": f"N{k + 1}", "x": x, "y": y, "kind": f"new spot (nearest recorded: {near['name']}, about "
                     f"{np.hypot(near['x'] - x, near['y'] - y) / PX_PER_CM:.1f} cm away)"})
        taken.append((x, y))
    return plan


def save_plan(dataset: Path, plan: list[dict], mode: str) -> Path:
    path = OUT_DIR / f"{dataset.name}_plan_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"dataset": str(dataset), "mode": mode, "done": [], "targets": plan},
                               indent=1, ensure_ascii=False))
    return path


# ---------- drawing ----------

def put_text(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], size: int = 17) -> np.ndarray:
    """Append a text panel under the image (PIL for nicer fonts and Unicode such as ✓ → ×). img is RGB."""
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for path, index in FONTS:
        try:
            font = ImageFont.truetype(path, size, index=index)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    panel = Image.new("RGB", (img.shape[1], 10 + (size + 8) * len(lines)), (25, 25, 25))
    draw = ImageDraw.Draw(panel)
    for i, (text, color) in enumerate(lines):
        draw.text((8, 5 + i * (size + 8)), text, font=font, fill=color)
    return np.concatenate([img, np.array(panel)], axis=0)


def draw_overlay(rgb, pos, plan, idx, det, tol, done) -> np.ndarray:
    img = rgb.copy()
    for e in pos["episodes"]:
        cv2.circle(img, (int(e["x"]), int(e["y"])), 3, (170, 170, 170), -1)
    for a in pos["anchors"]:
        cv2.putText(img, a["name"], (int(a["x"]) + 6, int(a["y"]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    for j, t in enumerate(plan):
        if j != idx:
            color = (80, 160, 255) if j in done else (90, 90, 90)
            cv2.circle(img, (int(t["x"]), int(t["y"])), 4, color, 1)
    t = plan[idx]
    tx, ty = int(t["x"]), int(t["y"])
    ok = det is not None and np.hypot(det["x"] - t["x"], det["y"] - t["y"]) <= tol
    color = (0, 230, 0) if ok else (0, 200, 255)
    cv2.circle(img, (tx, ty), int(tol), color, 2)
    cv2.drawMarker(img, (tx, ty), color, cv2.MARKER_CROSS, 14, 1)
    if det is not None:
        x, y, w, h = det["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 220, 0), 2)
        if not ok:
            cv2.arrowedLine(img, (int(det["x"]), int(det["y"])), (tx, ty), (255, 220, 0), 2, tipLength=0.2)
    return img


def advice(det, target, tol, have_bg) -> tuple[str, tuple[int, int, int], bool]:
    if det is None:
        hint = "" if have_bg else " (remove the object and press b to grab the background for better detection)"
        return f"Object not seen: put it in the green circle{hint}", (255, 200, 0), False
    dx, dy = target["x"] - det["x"], target["y"] - det["y"]
    dist = np.hypot(dx, dy)
    if dist <= tol:
        return f"✓ Position OK (off by {dist:.0f} px, about {dist / PX_PER_CM:.1f} cm)", (0, 230, 0), True
    parts = []
    if abs(dx) > tol / 2:
        parts.append(f"{abs(dx) / PX_PER_CM:.1f} cm to the {'right' if dx > 0 else 'left'}")
    if abs(dy) > tol / 2:
        parts.append(f"{abs(dy) / PX_PER_CM:.1f} cm {'down' if dy > 0 else 'up'} the image")
    return "Follow the yellow arrow: " + ", ".join(parts) + f" ({dist:.0f} px to go)", (255, 200, 0), False


def coverage_image(dataset: Path, pos: dict, plan: list[dict] | None) -> Path:
    img = load_background(dataset)
    if plan:
        img = draw_overlay(img, pos, plan, 0, None, 10, set())
        for t in plan:
            cv2.putText(img, t["id"], (int(t["x"]) + 5, int(t["y"]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255),
                        1, cv2.LINE_AA)
    else:
        img = draw_overlay(img, pos, [{"x": -50, "y": -50}], 0, None, 1, set())
    x0, y0, x1, y1 = work_roi(pos, 60)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    path = OUT_DIR / f"{dataset.name}_coverage{'_plan' if plan else ''}.png"
    cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return path


# ---------- guided recording ----------

def dataset_task(dataset: Path) -> str:
    import pandas as pd

    df = pd.read_parquet(dataset / "meta" / "tasks.parquet")
    return str(df["task"].iloc[0] if "task" in df.columns else df.index[0])


def record_one(dataset: Path, args) -> bool:
    """Record one more episode with lerobot-record (--resume). Returns whether the episode count went up."""
    exe = shutil.which("lerobot-record")
    if exe is None:
        sc.fail("lerobot-record not found: activate your LeRobot environment first")
    before = sc.num_episodes(dataset)
    r = CFG["robot"]
    cmd = [exe, f"--robot.type={r['type']}", f"--robot.port={r['port']}", f"--robot.id={r['id']}",
           f"--robot.cameras={config.cameras_arg(args.front, args.wrist)}", *config.teleop_args(),
           f"--display_data={'true' if args.display else 'false'}",
           f"--dataset.repo_id={config.repo_id(dataset.name)}", f"--dataset.root={dataset}",
           f"--dataset.single_task={dataset_task(dataset)}",
           "--dataset.num_episodes=1", "--dataset.episode_time_s=600", "--dataset.reset_time_s=0",
           "--dataset.push_to_hub=false", "--dataset.streaming_encoding=true", "--dataset.encoder_threads=2",
           "--resume=true"]
    print("\n" + "=" * 70)
    print(f"Recording episode {before} (dataset has {before} episodes). Move the leader arm after 'Recording episode';")
    print("when done and back in the start pose press →; ← to redo; Esc to stop.")
    print("=" * 70, flush=True)
    subprocess.run(cmd)
    after = sc.num_episodes(dataset)
    print(f"Dataset now has {after} episodes" + (" (saved)" if after > before else " (nothing saved this time)"))
    return after > before


def append_log(dataset: Path, row: dict) -> None:
    path = OUT_DIR / f"{dataset.name}_placement_log.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------- live window ----------

class Guide:
    def __init__(self, args, dataset: Path, pos: dict, plan_path: Path):
        self.args, self.dataset, self.pos, self.plan_path = args, dataset, pos, plan_path
        data = json.loads(plan_path.read_text())
        self.plan, self.done = data["targets"], set(data.get("done", []))
        self.idx = next((i for i in range(len(self.plan)) if i not in self.done), 0)
        self.bg, self.have_bg = load_background(dataset), False
        self.roi, self.area = work_roi(pos), pos["object_area"]
        self.size = tuple(pos["size"])
        self.dev = sc.resolve_device("front", args.front)
        self.cap = sc.open_camera(self.dev)
        self.live, self.det, self.last_print = None, None, ""
        # targets are in the dataset's pixel coordinates: if the scene (camera, arm, box) is not aligned,
        # targets are wrong, so check periodically. Compare with the saved layout if any, else the dataset.
        if (sc.REF_DIR / "layout_current.txt").exists():
            _, self.ref = sc.load_layout(None, "front")
        else:
            self.ref = sc.load_reference(dataset, 0, "front")
        self.patches = sc.pick_patches(sc.grad_mag(self.ref))
        self.scene, self.scene_t = None, 0.0

    def run(self) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self.Image, self.ImageTk = Image, ImageTk
        self.root = tk.Tk()
        self.root.title("placement_guide - space: OK   b: background   n/p: target   q: quit")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.after(0, self.tick)
        try:
            self.root.mainloop()
        finally:
            if self.cap is not None:
                self.cap.release()

    def quit(self) -> None:
        self.root.quit()
        self.root.destroy()

    def save_progress(self) -> None:
        data = json.loads(self.plan_path.read_text())
        data["done"] = sorted(self.done)
        self.plan_path.write_text(json.dumps(data, indent=1, ensure_ascii=False))

    def tick(self) -> None:
        self.live = sc.read_rgb(self.cap, self.size)
        if time.time() - self.scene_t > 3:
            self.scene, self.scene_t = sc.evaluate(self.ref, self.live, self.patches), time.time()
        self.det = detect_object(self.live, self.bg, self.roi, self.area)
        t = self.plan[self.idx]
        msg, color, ok = advice(self.det, t, self.args.tol, self.have_bg)
        img = draw_overlay(self.live, self.pos, self.plan, self.idx, self.det, self.args.tol, self.done)
        action = "press space to record this episode" if self.args.record else "press space to log it and go to the next"
        lines = [
            (f"Target {self.idx + 1}/{len(self.plan)}: {t['id']}  {t['kind']}  done: {len(self.done)}", (255, 255, 255)),
            (msg, color),
            ((action + " (you can also press it when not perfectly aligned)") if ok
             else "b background  n/p target  s screenshot  q quit", (190, 190, 190)),
        ]
        if not self.scene["ok"] and not self.scene["median"] < 12:
            lines.insert(0, (f"⚠ Scene differs from the recording (median shift {self.scene['median']:.0f} px): targets will be off. "
                             "Realign camera, arm and box with scene_check.py --layout first", (255, 90, 90)))
        if msg != self.last_print:
            print(f"[{t['id']}] {msg}")
            self.last_print = msg
        view = put_text(img, lines)
        photo = self.ImageTk.PhotoImage(self.Image.fromarray(view))
        self.label.configure(image=photo)
        self.label.image = photo
        self.root.after(30, self.tick)

    def confirm(self) -> None:
        t = self.plan[self.idx]
        row = {"time": datetime.now().isoformat(timespec="seconds"), "episode": sc.num_episodes(self.dataset),
               "target": t["id"], "target_x": round(t["x"], 1), "target_y": round(t["y"], 1),
               "placed_x": round(self.det["x"], 1) if self.det else "", "placed_y": round(self.det["y"], 1) if self.det else "",
               "dist_px": round(float(np.hypot(self.det["x"] - t["x"], self.det["y"] - t["y"])), 1) if self.det else "",
               "plan": self.plan_path.name, "recorded": ""}
        if self.args.record:
            self.cap.release()
            self.cap = None
            self.root.withdraw()
            self.root.update()
            saved = record_one(self.dataset, self.args)
            row["recorded"] = "yes" if saved else "no"
            for _ in range(20):  # lerobot-record just released the camera; it may take a moment
                try:
                    self.cap = sc.open_camera(self.dev)
                    break
                except SystemExit:
                    time.sleep(0.5)
            self.root.deiconify()
            if not saved:
                append_log(self.dataset, row)
                print("Nothing was saved; the target stays the same.")
                return
        append_log(self.dataset, row)
        self.done.add(self.idx)
        self.save_progress()
        remaining = [i for i in range(len(self.plan)) if i not in self.done]
        if not remaining:
            print(f"All {len(self.plan)} positions in the plan are done!")
        else:
            self.idx = next((i for i in remaining if i > self.idx), remaining[0])

    def on_key(self, event) -> None:
        ch = event.char
        if ch == "q" or event.keysym == "Escape":
            self.quit()
        elif ch == " " or event.keysym == "Return":
            self.confirm()
        elif ch == "b":
            self.bg, self.have_bg = self.live.copy(), True
            print("Background grabbed (no object and no hands in view)")
        elif ch in ("n", "p"):
            self.idx = (self.idx + (1 if ch == "n" else -1)) % len(self.plan)
        elif ch == "s":
            OUT_DIR.mkdir(exist_ok=True)
            path = OUT_DIR / f"guide_{datetime.now():%Y%m%d_%H%M%S}.png"
            img = draw_overlay(self.live, self.pos, self.plan, self.idx, self.det, self.args.tol, self.done)
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Screenshot: {path}")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset folder (where recorded spots are read from; --record appends to it). "
                         "Default: config reference_dataset")
    ap.add_argument("--analyze", action="store_true", help="only analyse placements in the dataset, no camera")
    ap.add_argument("--refresh", action="store_true", help="re-analyse (done automatically when the dataset changes)")
    ap.add_argument("--plan", choices=["fill", "anchors"], help="fill: new spots (fill gaps); anchors: back to recorded spots")
    ap.add_argument("--count", type=int, default=20, help="how many spots for fill")
    ap.add_argument("--repeat", type=int, default=1, help="how many times each spot for anchors")
    ap.add_argument("--expand", type=float, default=12, help="how far fill may go beyond the recorded area, in px")
    ap.add_argument("--plan-file", help="continue a saved plan (resumes unfinished targets)")
    ap.add_argument("--tol", type=float, default=10, help="within how many px of the target counts as OK")
    ap.add_argument("--record", action="store_true", help="guided recording: record one episode into --dataset per placement")
    ap.add_argument("--display", action="store_true", help="open Rerun while recording (off by default, faster)")
    ap.add_argument("--front", default=str(CFG["cameras"]["front"]))
    ap.add_argument("--wrist", default=str(CFG["cameras"]["wrist"]))
    args = ap.parse_args()

    if not args.dataset:
        sc.fail("Pass --dataset PATH (or set reference_dataset in config.json)")
    dataset = Path(os.path.expanduser(args.dataset))
    if not (dataset / "meta" / "info.json").exists():
        sc.fail(f"Dataset not found: {dataset}")
    pos = analyze_dataset(dataset, args.refresh)
    print(f"\n{dataset.name}: object found in {len(pos['episodes'])} episodes, grouped into {len(pos['anchors'])} spots")
    for a in pos["anchors"]:
        print(f"  {a['name']}: image ({a['x']:.0f}, {a['y']:.0f}), episodes {a['episodes']}")

    if args.analyze:
        print(f"Coverage image: {coverage_image(dataset, pos, None)}")
        print(f"Positions file (for guided_record.py standard/dense/far): {OUT_DIR / (dataset.name + '_positions.json')}")
        return
    if args.plan_file:
        plan_path = Path(args.plan_file)
    elif args.plan:
        plan = make_plan(pos, args.plan, args.count, args.repeat, args.expand)
        plan_path = save_plan(dataset, plan, args.plan)
        print(f"\nPlanned {len(plan)} positions: {plan_path}")
        for t in plan:
            print(f"  {t['id']}: ({t['x']:.0f}, {t['y']:.0f})  {t['kind']}")
        print(f"Plan image: {coverage_image(dataset, pos, plan)}")
    else:
        sc.fail("Add --analyze, --plan fill|anchors or --plan-file")
    Guide(args, dataset, pos, plan_path).run()


if __name__ == "__main__":
    main()
