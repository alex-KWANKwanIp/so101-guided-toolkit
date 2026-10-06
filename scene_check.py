#!/usr/bin/env python3
"""
Scene alignment check: overlay what a camera sees *now* on what it saw while recording the training data,
so you can put the camera, arm and box back exactly where they were before testing or recording more data.

Why: an imitation policy (ACT, SmolVLA...) decides where to move from *where things are in the image*.
If the camera or the arm base moves a few centimetres, the policy keeps reaching to the wrong place,
silently, with no error message.

Usage (close lerobot-rollout / record / skill_server first: a camera can only be opened by one program):
    python scene_check.py --layout            # compare with the saved standard layout (recommended)
    python scene_check.py --save-layout NAME  # save the current scene as a standard layout (front + wrist)
    python scene_check.py --dataset PATH      # compare with a dataset (auto-picks the closest episode's first frame)
    python scene_check.py --camera wrist      # wrist camera instead (also checks the arm's start pose)
    python scene_check.py --episode 0         # use a specific reference episode
    python scene_check.py --snapshot          # no window: grab one frame, save a comparison image, print the result
    python scene_check.py --image some.png    # compare a saved image instead of a live camera
    python scene_check.py --reset-camera      # switch white balance / exposure back to auto and exit

Before aligning: put the arm in the start pose used for recording and remove the object
(or place it where it is in the reference).

Window keys:
    1 blend (50/50)  2 reference edges (green)  3 side by side  4 difference
    c switch front / wrist        n / p next / previous reference episode   r re-pick the closest reference
    s save a comparison image     q or Esc quit
    [ / ]  colour temperature -200 K / +200 K (turns off auto white balance): B/R too high (too blue) → press ]
    - / =  exposure down / up (turns off auto exposure)
    a      back to auto white balance and auto exposure

The numbers:
    arrows = where each reference feature is now (shorter is better).
    median shift < 8 px and 80 % of features < 20 px → aligned. Median > 20 px must be fixed.
    (Within one recording session the shift is typically 0-3 px; a bumped camera is often 20-30 px.)
    B/R = blue / red ratio (colour tone). A difference > 0.03 from the reference means the lighting or
    white balance changed. Manual white balance / exposure persist until the camera is unplugged or the
    computer reboots; on exit the script prints commands to restore them.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config

CFG = config.load()
DEFAULT_DATASET = CFG["reference_dataset"]
CAM = CFG["cameras"]
# config.json first, then the fixed names created by make_udev_rules.py
DEVICES = {"front": [str(CAM["front"]), "/dev/cam_front"], "wrist": [str(CAM["wrist"]), "/dev/cam_wrist"]}
FRAME_SIZE = (int(CAM["width"]), int(CAM["height"]))
HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "scene_refs"
OUT_DIR = HERE / "scene_checks"
PATCH, SEARCH, N_PATCHES, MIN_SCORE = 64, 160, 20, 0.6
MODES = {1: "blend", 2: "edges", 3: "side-by-side", 4: "difference"}


def fail(msg: str) -> None:
    print(f"\n✗ {msg}")
    sys.exit(1)


# ---------- reference: first frame of an episode in the training dataset ----------

def load_reference(dataset: Path, episode: int, camera: str) -> np.ndarray:
    """Return an RGB image (H, W, 3). The first read caches a PNG in scene_refs/."""
    cache = REF_DIR / f"{dataset.name}_ep{episode:03d}_{camera}.png"
    if cache.exists():
        return cv2.cvtColor(cv2.imread(str(cache)), cv2.COLOR_BGR2RGB)

    import av
    import pandas as pd

    info = json.loads((dataset / "meta" / "info.json").read_text())
    files = sorted(glob.glob(str(dataset / "meta" / "episodes" / "*" / "*.parquet")))
    if not files:
        fail(f"No episode metadata found in dataset: {dataset}")
    eps = pd.concat([pd.read_parquet(f) for f in files])
    row = eps[eps["episode_index"] == episode]
    if row.empty:
        fail(f"The dataset has {len(eps)} episodes (0-{len(eps) - 1}); there is no episode {episode}")
    row = row.iloc[0]
    key = f"observation.images.{camera}"
    video = dataset / info["video_path"].format(
        video_key=key,
        chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]),
    )
    ts = float(row[f"videos/{key}/from_timestamp"])
    half_frame = 0.5 / info["fps"]
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(ts - 1.0, 0) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is not None and float(frame.pts * stream.time_base) >= ts - half_frame:
                rgb = frame.to_ndarray(format="rgb24")
                break
        else:
            fail(f"Could not read a frame from {video} at {ts:.2f} s")
    REF_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(cache), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return rgb


def num_episodes(dataset: Path) -> int:
    return int(json.loads((dataset / "meta" / "info.json").read_text())["total_episodes"])


def best_reference(dataset: Path, camera: str, live: np.ndarray) -> int:
    """Try 5 evenly spaced episodes and return the one closest to the live view
    (the scene can differ slightly between recording sessions)."""
    n = num_episodes(dataset)
    scores = []
    for ep in sorted({int(round(x)) for x in np.linspace(0, n - 1, min(5, n))}):
        ref = load_reference(dataset, ep, camera)
        r = evaluate(ref, live, pick_patches(grad_mag(ref)))
        scores.append((r["level"], r["median"] if r["median"] == r["median"] else 1e9, ep))
        print(f"  reference episode {ep:3d}: median shift {r['median']:.1f} px, matched {r['matched']}/{N_PATCHES}")
    ep = min(scores)[2]
    print(f"→ using episode {ep} as reference")
    return ep


# ---------- comparison: feature shift, colour tone, brightness ----------

def grad_mag(rgb: np.ndarray) -> np.ndarray:
    """Compare edge strength rather than raw colour: more robust to lighting changes."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2.0)
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))


def pick_patches(ref_grad: np.ndarray) -> list[tuple[int, int]]:
    """Pick the most textured, non-overlapping small patches of the reference (arm, box edges...)."""
    h, w = ref_grad.shape
    candidates = sorted(
        ((float(ref_grad[y : y + PATCH, x : x + PATCH].std()), x, y)
         for y in range(0, h - PATCH + 1, PATCH // 2)
         for x in range(0, w - PATCH + 1, PATCH // 2)),
        reverse=True,
    )
    chosen: list[tuple[int, int]] = []
    for _, x, y in candidates:
        if all(abs(x - cx) >= PATCH or abs(y - cy) >= PATCH for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) == N_PATCHES:
            break
    return chosen


def patch_shifts(ref_grad, live_grad, patches):
    """How far each patch moved in the live image: [(centre x, centre y, dx, dy, similarity)]."""
    h, w = ref_grad.shape
    out = []
    for x, y in patches:
        x0, y0 = max(0, x - SEARCH), max(0, y - SEARCH)
        x1, y1 = min(w, x + PATCH + SEARCH), min(h, y + PATCH + SEARCH)
        res = cv2.matchTemplate(live_grad[y0:y1, x0:x1], ref_grad[y : y + PATCH, x : x + PATCH], cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        out.append((x + PATCH // 2, y + PATCH // 2, x0 + mx - x, y0 + my - y, float(score)))
    return out


def color_stats(rgb: np.ndarray) -> tuple[float, float]:
    mean = rgb.reshape(-1, 3).astype(np.float32).mean(0) / 255
    return float(mean[2] / max(mean[0], 1e-6)), float(mean.mean())  # (B/R, brightness)


def evaluate(ref: np.ndarray, live: np.ndarray, patches) -> dict:
    shifts = patch_shifts(grad_mag(ref), grad_mag(live), patches)
    good = [s for s in shifts if s[4] >= MIN_SCORE]
    dist = np.array([np.hypot(s[2], s[3]) for s in good]) if good else np.array([])
    ref_br, ref_light = color_stats(ref)
    live_br, live_light = color_stats(live)
    result = {
        "shifts": shifts,
        "matched": len(good),
        "median": float(np.median(dist)) if len(dist) else float("nan"),
        "p80": float(np.percentile(dist, 80)) if len(dist) else float("nan"),
        "median_dx": float(np.median([s[2] for s in good])) if good else float("nan"),
        "median_dy": float(np.median([s[3] for s in good])) if good else float("nan"),
        "ref_br": ref_br, "live_br": live_br, "ref_light": ref_light, "live_light": live_light,
    }
    if len(good) < N_PATCHES // 3 or result["median"] >= 60:
        result["verdict"], result["ok"], result["level"] = "SCENE VERY DIFFERENT", False, 3
    elif result["median"] < 8 and result["p80"] < 20:
        result["verdict"], result["ok"], result["level"] = "ALIGNED", True, 0
    elif result["median"] < 20:
        result["verdict"], result["ok"], result["level"] = "SMALL OFFSET - adjust", False, 1
    else:
        result["verdict"], result["ok"], result["level"] = "MOVED - realign camera/arm/box", False, 2
    return result


def explain(r: dict, camera: str) -> str:
    lines = [f"[{camera}] matched features: {r['matched']}/{N_PATCHES}"]
    if r["level"] == 3 and r["matched"] < N_PATCHES // 3:
        lines.append("  ✗ Almost nothing matches: the camera direction or the whole scene differs from the recording.")
        lines.append("    Look at side by side (key 3): the arm and box must sit where they are in the reference on the left.")
    elif r["level"] == 3:
        lines.append(f"  ✗ Far off (median shift about {r['median']:.0f} px): arrows are unreliable at this distance, coarse-align first.")
        lines.append("    Use reference edges (key 2): move camera, arm base and box onto the green lines; arrows become accurate once close.")
    else:
        lines.append(f"  median shift {r['median']:.1f} px, 80 % of features within {r['p80']:.1f} px "
                     f"(overall: horizontal {r['median_dx']:+.0f} px [+ right / - left], "
                     f"vertical {r['median_dy']:+.0f} px [+ down / - up])")
        if r["ok"]:
            lines.append("  ✓ Position aligned")
        else:
            lines.append("  ✗ Offset: green arrows point to where reference features are now. "
                         "Long arrows only near the arm → the arm base moved; all arrows shifted together → the camera moved.")
    diff = r["live_br"] - r["ref_br"]
    tone = "bluer / cooler" if diff > 0 else "yellower / warmer"
    lines.append(f"  tone B/R: now {r['live_br']:.3f}, reference {r['ref_br']:.3f}"
                 + (f" → {tone} (diff {diff:+.3f})" if abs(diff) > 0.03 else " → close ✓"))
    lines.append(f"  brightness: now {r['live_light']:.2f}, reference {r['ref_light']:.2f}"
                 + (" → very different, check lights or exposure" if abs(r["live_light"] - r["ref_light"]) > 0.06 else " → close ✓"))
    return "\n".join(lines)


# ---------- drawing ----------

def compose(ref: np.ndarray, live: np.ndarray, r: dict, mode: int, header: str, footer: str) -> np.ndarray:
    """Return a BGR image for display."""
    ref_bgr, live_bgr = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR), cv2.cvtColor(live, cv2.COLOR_RGB2BGR)
    if mode == 1:
        view = cv2.addWeighted(live_bgr, 0.5, ref_bgr, 0.5, 0)
    elif mode == 2:
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), (5, 5), 0), 40, 120)
        view = live_bgr.copy()
        view[cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0] = (0, 255, 0)
    elif mode == 3:
        view = np.concatenate([ref_bgr, live_bgr], axis=1)
    else:
        d = cv2.absdiff(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), cv2.cvtColor(live, cv2.COLOR_RGB2GRAY))
        view = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=3), cv2.COLORMAP_JET)
    offset = ref.shape[1] if mode == 3 else 0
    for cx, cy, dx, dy, score in (r["shifts"] if r["level"] < 3 else []):  # arrows are meaningless when far off
        p0 = (int(cx) + offset, int(cy))
        if score >= MIN_SCORE:
            cv2.arrowedLine(view, p0, (int(cx + dx) + offset, int(cy + dy)), (0, 255, 0), 2, tipLength=0.25)
        else:
            cv2.circle(view, p0, 3, (0, 0, 255), -1)
    color = (0, 200, 0) if r["ok"] else (0, 0, 255)
    status = (f"{r['verdict']}  median {r['median']:.1f}px  p80 {r['p80']:.1f}px  matched {r['matched']}/{N_PATCHES}"
              f"  |  B/R live {r['live_br']:.3f} ref {r['ref_br']:.3f}")
    bar = np.zeros((66, view.shape[1], 3), np.uint8)
    for i, (text, c) in enumerate([(header, (255, 255, 255)), (status, color), (footer, (200, 200, 200))]):
        cv2.putText(bar, text, (6, 18 + 21 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return np.concatenate([bar, view], axis=0)


def save_report(ref, live, r, camera, episode) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    header = f"reference ep{episode} {camera}  vs  now {datetime.now():%Y-%m-%d %H:%M:%S}"
    img = np.concatenate([compose(ref, live, r, 3, header, "left: reference   right: now"),
                          np.concatenate([compose(ref, live, r, 2, "edges of reference (green) on current view", ""),
                                          compose(ref, live, r, 1, "50/50 blend", "")], axis=1)], axis=0)
    path = OUT_DIR / f"scene_check_{camera}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(path), img)
    return path


# ---------- camera ----------

def resolve_device(camera: str, override: str | None) -> str:
    if override and override.isdigit():  # macOS / Windows: camera index 0, 1, 2...
        return override
    if not override and DEVICES[camera][0].isdigit():  # config.json uses an index
        return DEVICES[camera][0]
    for dev in ([override] if override else DEVICES[camera]):
        if os.path.exists(dev):
            return os.path.realpath(dev)
    fail(f"{camera} camera not found (tried {', '.join(DEVICES[camera])}). "
         "Pass --device, set it in config.json, or create fixed names with make_udev_rules.py.")


def open_camera(dev: str) -> cv2.VideoCapture:
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY  # no V4L2 on macOS
    cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev, backend)
    if CAM.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, CAM["fps"])
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        fail(f"Cannot open {dev}. Usually another program holds it: close lerobot-rollout / record / teleoperate / skill_server.")
    return cap


def read_rgb(cap: cv2.VideoCapture, size: tuple[int, int]) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        fail("No frame from the camera (unplugged?)")
    if frame.shape[1::-1] != size:
        frame = cv2.resize(frame, size)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def v4l2(dev: str, *args: str) -> str | None:
    if not shutil.which("v4l2-ctl"):
        print("  (v4l2-ctl not installed, cannot change white balance / exposure: sudo apt install v4l-utils)")
        return None
    r = subprocess.run(["v4l2-ctl", "-d", dev, *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  v4l2-ctl failed: {r.stderr.strip()}")
        return None
    return r.stdout


def v4l2_get(dev: str, name: str) -> int | None:
    out = v4l2(dev, "-C", name)
    try:
        return int(out.split(":")[1]) if out else None
    except (IndexError, ValueError):
        return None


def camera_settings(dev: str) -> dict:
    if not sys.platform.startswith("linux") or str(dev).isdigit():
        return {}
    names = ["white_balance_automatic", "white_balance_temperature", "auto_exposure", "exposure_time_absolute"]
    return {n: v4l2_get(dev, n) for n in names}


def print_settings(dev: str, camera: str) -> None:
    s = camera_settings(dev)
    if not s:
        return
    print(f"\n{camera} camera settings now: {s}")
    if s.get("white_balance_automatic") == 0 or s.get("auto_exposure") == 1:
        print("To restore these settings next time (after replugging or rebooting), run:")
        if s.get("white_balance_automatic") == 0:
            print(f"  v4l2-ctl -d {dev} -c white_balance_automatic=0 && "
                  f"v4l2-ctl -d {dev} -c white_balance_temperature={s['white_balance_temperature']}")
        if s.get("auto_exposure") == 1:
            print(f"  v4l2-ctl -d {dev} -c auto_exposure=1 && "
                  f"v4l2-ctl -d {dev} -c exposure_time_absolute={s['exposure_time_absolute']}")
        print(f"Back to auto: python {Path(__file__).name} --camera {camera} --reset-camera")


def adjust(dev: str, key: int) -> None:
    if key in (ord("["), ord("]")):
        v4l2(dev, "-c", "white_balance_automatic=0")
        cur = v4l2_get(dev, "white_balance_temperature") or 4600
        new = int(np.clip(cur + (200 if key == ord("]") else -200), 2800, 6500))
        v4l2(dev, "-c", f"white_balance_temperature={new}")
        print(f"  colour temperature {cur} → {new} K (auto white balance off)")
    elif key in (ord("-"), ord("=")):
        v4l2(dev, "-c", "auto_exposure=1")
        cur = v4l2_get(dev, "exposure_time_absolute") or 157
        new = max(1, int(round(cur * (1.15 if key == ord("=") else 1 / 1.15))))
        v4l2(dev, "-c", f"exposure_time_absolute={new}")
        print(f"  exposure {cur} → {new} (auto exposure off)")
    elif key == ord("a"):
        v4l2(dev, "-c", "white_balance_automatic=1")
        v4l2(dev, "-c", "auto_exposure=3")
        print("  auto white balance and auto exposure restored")


# ---------- standard layout: save the scene at one moment and always compare with it ----------

def save_layout(name: str | None) -> None:
    name = name or f"{datetime.now():%Y%m%d_%H%M%S}"
    REF_DIR.mkdir(exist_ok=True)
    info = {"name": name, "saved": datetime.now().isoformat(timespec="seconds"), "cameras": {}}
    for camera in ("front", "wrist"):
        dev = resolve_device(camera, None)
        cap = open_camera(dev)
        for _ in range(45):  # let auto exposure / white balance settle
            rgb = read_rgb(cap, FRAME_SIZE)
        cap.release()
        path = REF_DIR / f"layout_{name}_{camera}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        info["cameras"][camera] = {"image": path.name, "settings": camera_settings(dev)}
        print(f"  {camera}: {path}")
    (REF_DIR / f"layout_{name}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    (REF_DIR / "layout_current.txt").write_text(name)
    print(f"Saved standard layout '{name}'. Compare with: python {Path(__file__).name} --layout")


def load_layout(name: str | None, camera: str) -> tuple[str, np.ndarray]:
    if not name:
        cur = REF_DIR / "layout_current.txt"
        if not cur.exists():
            fail(f"No standard layout saved yet: run python {Path(__file__).name} --save-layout NAME first")
        name = cur.read_text().strip()
    path = REF_DIR / f"layout_{name}_{camera}.png"
    if not path.exists():
        fail(f"Standard layout image not found: {path}")
    return name, cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="training dataset folder (default: config reference_dataset)")
    ap.add_argument("--episode", type=int, help="use this episode's first frame as reference (default: auto-pick the closest)")
    ap.add_argument("--camera", choices=["front", "wrist"], default="front")
    ap.add_argument("--device", help="camera device (path or index); default from config.json, then /dev/cam_front, /dev/cam_wrist")
    ap.add_argument("--snapshot", action="store_true", help="no window: grab one frame, save a comparison image, print the result")
    ap.add_argument("--image", help="compare this image instead of the live camera")
    ap.add_argument("--reset-camera", action="store_true", help="switch white balance / exposure back to auto, then exit")
    ap.add_argument("--save-layout", nargs="?", const="", metavar="NAME", help="save the current scene as a standard layout")
    ap.add_argument("--layout", nargs="?", const="", metavar="NAME", help="compare with a saved standard layout (default: latest)")
    args = ap.parse_args()

    if args.save_layout is not None:
        save_layout(args.save_layout or None)
        return

    camera, episode = args.camera, args.episode

    if args.reset_camera:
        dev = resolve_device(camera, args.device)
        adjust(dev, ord("a"))
        print_settings(dev, camera)
        return

    layout, dataset = None, None
    if args.layout is not None:
        layout, ref = load_layout(args.layout or None, camera)
        episode = -1
        print(f"Reference: standard layout '{layout}'")
    else:
        if not args.dataset:
            fail("Nothing to compare with: use --layout (after --save-layout), --dataset PATH, "
                 "or set reference_dataset in config.json")
        dataset = Path(os.path.expanduser(args.dataset))
        ref = load_reference(dataset, episode if episode is not None else 0, camera)
    size = (ref.shape[1], ref.shape[0])

    if args.image:
        live = cv2.imread(args.image)
        if live is None:
            fail(f"Cannot read image: {args.image}")
        live = cv2.cvtColor(cv2.resize(live, size), cv2.COLOR_BGR2RGB)
        if layout is None:
            if episode is None:
                episode = best_reference(dataset, camera, live)
            ref = load_reference(dataset, episode, camera)
        r = evaluate(ref, live, pick_patches(grad_mag(ref)))
        print(explain(r, camera))
        print(f"Comparison image: {save_report(ref, live, r, camera, episode)}")
        return

    dev = resolve_device(camera, args.device)
    cap = open_camera(dev)
    print(f"{camera} camera: {dev}  reference: " + (f"standard layout {layout}" if layout is not None else f"dataset {dataset.name}"))
    for _ in range(45):  # drop ~1.5 s of frames so auto exposure / white balance settle
        read_rgb(cap, size)
    live = read_rgb(cap, size)
    if layout is None:
        if episode is None:
            episode = best_reference(dataset, camera, live)
        ref = load_reference(dataset, episode, camera)
    patches = pick_patches(grad_mag(ref))

    if args.snapshot:
        cap.release()
        r = evaluate(ref, live, patches)
        print(explain(r, camera))
        print(f"Comparison image: {save_report(ref, live, r, camera, episode)}")
        print_settings(dev, camera)
        return

    view = LiveView(dataset, camera, dev, cap, episode, ref, patches, size)
    view.layout = layout
    view.run()


class LiveView:
    """Live window (tkinter: the OpenCV in a LeRobot env is often headless, so cv2.imshow is unavailable)."""

    def __init__(self, dataset, camera, dev, cap, episode, ref, patches, size):
        self.dataset, self.camera, self.dev, self.cap = dataset, camera, dev, cap
        self.episode, self.ref, self.patches, self.size = episode, ref, patches, size
        self.mode, self.frame_i, self.r, self.live, self.last_print = 2, 0, None, None, 0.0
        self.layout = None

    def run(self) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self.Image, self.ImageTk = Image, ImageTk
        self.root = tk.Tk()
        self.root.title("scene_check - q: quit")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.after(0, self.tick)
        try:
            self.root.mainloop()
        finally:
            self.cap.release()
        print_settings(self.dev, self.camera)

    def quit(self) -> None:
        self.root.quit()
        self.root.destroy()

    def set_reference(self, episode: int) -> None:
        self.episode = episode
        self.ref = load_reference(self.dataset, episode, self.camera)
        self.patches, self.r = pick_patches(grad_mag(self.ref)), None
        print(f"Reference: episode {episode}, {self.camera} camera ({self.dev})")

    def tick(self) -> None:
        self.live = read_rgb(self.cap, self.size)
        if self.r is None or self.frame_i % 10 == 0:
            self.r = evaluate(self.ref, self.live, self.patches)
            if time.time() - self.last_print > 3:
                print("\n" + explain(self.r, self.camera))
                self.last_print = time.time()
        self.frame_i += 1
        ref_name = f"layout {self.layout}" if self.layout is not None else f"ep{self.episode}"
        header = f"ref: {ref_name} {self.camera} | mode {self.mode}: {MODES[self.mode]} | 1-4 mode  c cam  n/p/r ref  s save  q quit"
        footer = "[ ] WB temp   - = exposure   a auto   (arrows: where reference features are now)"
        bgr = compose(self.ref, self.live, self.r, self.mode, header, footer)
        photo = self.ImageTk.PhotoImage(self.Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        self.label.configure(image=photo)
        self.label.image = photo  # keep a reference or the image is garbage-collected
        self.root.after(10, self.tick)

    def on_key(self, event) -> None:
        ch = event.char
        if ch == "q" or event.keysym == "Escape":
            self.quit()
        elif ch and ch in "1234":
            self.mode = int(ch)
        elif ch == "s":
            print(f"Comparison image: {save_report(self.ref, self.live, self.r, self.camera, self.episode)}")
        elif ch == "c":
            print_settings(self.dev, self.camera)
            self.cap.release()
            self.camera = "wrist" if self.camera == "front" else "front"
            self.dev = resolve_device(self.camera, None)
            self.cap = open_camera(self.dev)
            if self.layout is not None:
                _, self.ref = load_layout(self.layout, self.camera)
                self.patches, self.r = pick_patches(grad_mag(self.ref)), None
            else:
                self.set_reference(best_reference(self.dataset, self.camera, read_rgb(self.cap, self.size)))
        elif self.layout is not None and ch in ("r", "n", "p"):
            print("n/p/r are not available when comparing with a standard layout (not a dataset)")
        elif ch == "r":
            self.set_reference(best_reference(self.dataset, self.camera, self.live))
        elif ch in ("n", "p"):
            n = num_episodes(self.dataset)
            self.set_reference(int(np.clip(self.episode + (1 if ch == "n" else -1), 0, n - 1)))
        elif ch in ("[", "]", "-", "=", "a"):
            adjust(self.dev, ord(ch))


if __name__ == "__main__":
    main()
