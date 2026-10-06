#!/usr/bin/env python3
"""
Guided recording: record with lerobot-record and, during every reset, open a window that shows with a green
circle where to put the object for the next episode.

Recording, saving and keys (→ ← Esc) are all LeRobot's own lerobot-record; this script only draws guidance on
the front camera image before the first episode and during each reset. It never changes the recorded data.

Usage (align the scene first with scene_check.py --layout and close it with q):
    python guided_record.py --name my_pick_place --episodes 80                        # new dataset, 80 episodes
    python guided_record.py --resume ~/.cache/huggingface/lerobot/<user>/<dataset> --episodes 20   # add to a dataset
    python guided_record.py --name ... --show-plan                                    # only show the position plan
    python guided_record.py --set-region                                              # click the reachable area (wide, move, multi, select)
    python guided_record.py --name my_rgb_select --mode select --episodes 96 --plan-size 192   # red/blue/green: pick red & blue only
    python guided_record.py --name my_rgb_multi --mode multi --episodes 48                     # put all three cubes in the box

Seven position modes (--mode):
    standard  20 evenly spread points in the work area (area of a previously recorded dataset grown by ~2 cm),
              each placed several times. Needs a positions file (placement_guide.py --analyze; config positions_file).
    dense     same area, 40 points with less jitter: more, closer positions to fill "between the points".
    far       16 points only in a ring 2-5 cm *outside* the work area: farther, less usual places. Never in the box
              or where the arm rests. The arm may not reach them: try with the leader arm first.
    wide      36 points spread over the whole reachable region (clicked with --set-region, default estimate
              otherwise); rotate the object randomly. 5 positions are never recorded (red × on the plan) so
              guided_eval.py --holdout can test truly unseen positions.
    move      "object suddenly moved" demos, needs two people. Green circle = where the object starts; orange
              circles 1 (and 2) = where the helper moves it *during* recording, when the gripper is 3-5 cm away,
              then the hand leaves the view at once. About 1/3 of episodes move twice. The window shows the live
              image with the orange targets while recording. Same region and held-out positions as wide.
    multi     several objects of different colours at once (default red, blue, green); one episode puts all of
              them in the box. Each object gets a circle in its own colour; numbers 1, 2, 3 are the pick order.
    select    "pick the right ones": 3 objects, pick only the --targets colours (default red, blue); the others
              (default green) are distractors to be left alone, drawn as ×. About 1/3 of episodes place the
              distractor right next to a target (5-7 cm) so the policy must look at colour, not position.
              multi / select use the wide region (and its held-out positions), keep objects ~5 cm apart, and
              should go into a new dataset (--name). Colours via --objects (red, blue, green, yellow); the task
              sentence is generated automatically, or give your own with --task.
    Each mode has its own plan file placement/<name>_<mode>_plan.json; --points, --expand, --jitter override defaults.

How positions are chosen: pseudo-random with a fixed --seed (same seed → same plan, so split sessions and
different policies line up). Round-robin: every round places each point once in shuffled order with a small
jitter, so counts never differ by more than one. In multi / select every colour, distractor included, cycles
through the points on its own, and the planner avoids repeating the same pair of positions. Use an episode
count that is a multiple of the point count (wide 36 → 180 / 216; select 24 → 192). When recording one plan
over several sessions, pass --plan-size <total> the first time; later --resume sessions continue it.

Each episode:
    1. The window shows the *next* target (green) and the detected object (yellow box). Move the object into the
       circle following the arrow; bring the arm back to the start pose with the leader arm.
    2. Position OK → press →: recording starts (as usual: when done and back in the start pose, press →).
    3. Bad episode: press ← *while recording*. The reset then shows the *same* target; put the object back and
       press → to redo. (Pressing ← during the reset starts the redo immediately, with no time to place.)
    4. Esc: stop and save (everything recorded so far is kept).

Window key (click the window first): b = grab background. Do it once before starting: object(s) out, arm in
the start pose, hands out of view, then b. Grab it again if the lighting changes or the camera is bumped.
Actual placements are logged in placement/<name>_placement_log.csv (multi / select: <name>_objects_log.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
import placement_guide as pg
import scene_check as sc

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "placement"
CFG = config.load()
DEFAULT_TASK = CFG["default_task"]
TOL_PX = 10  # within this many px of the target counts as OK (~0.8 cm)
BOX_RIGHT_X = int(CFG["layout"]["box_right_x"])  # targets are never placed left of this x (the box side)
# modes: points = distinct points, expand = grow the work area by px, outside = at least this far outside it
# (0 = no limit), jitter = random offset per episode
MODES = {
    "standard": {"points": 20, "expand": 24, "outside": 0, "jitter": 4},
    "dense": {"points": 40, "expand": 24, "outside": 0, "jitter": 2},
    "far": {"points": 16, "expand": 60, "outside": 24, "jitter": 4},
    # region = use the whole reachable region (REGION_FILE), moves = move the object during recording
    "wide": {"points": 36, "expand": 0, "outside": 0, "jitter": 6, "region": True},
    "move": {"points": 24, "expand": 0, "outside": 0, "jitter": 6, "region": True, "moves": True},
    # objects = several objects at once (told apart by colour); all_targets = pick all (multi), else only --targets (select)
    "multi": {"points": 24, "expand": 0, "outside": 0, "jitter": 6, "region": True, "objects": True, "all_targets": True},
    "select": {"points": 24, "expand": 0, "outside": 0, "jitter": 6, "region": True, "objects": True},
}
# object colours: OpenCV HSV hue ranges (0-179) and the RGB used to draw them
COLORS = {
    "red": {"hue": [(0, 8), (165, 179)], "rgb": (255, 60, 60)},
    "yellow": {"hue": [(18, 34)], "rgb": (255, 220, 0)},
    "green": {"hue": [(35, 85)], "rgb": (60, 230, 60)},
    "blue": {"hue": [(90, 130)], "rgb": (70, 150, 255)},
}
DEFAULT_OBJECTS, DEFAULT_TARGETS = "red,blue,green", "red,blue"
OBJ_MIN_SEP = 60  # minimum px between objects (~5 cm, so the gripper does not hit the neighbour)
NEAR_MAX = 90  # select: max px between the distractor and a target when it is placed "next to" it (~7 cm)
NEAR_FRAC = 1 / 3  # select: fraction of episodes with the distractor next to a target
REGION_FILE = OUT_DIR / "wide_region.json"
# default estimate of the reachable region (front image px) until you click your own with --set-region
DEFAULT_REGION = CFG["layout"]["default_region"]
HOLDOUT = 5  # positions never recorded (to test unseen positions)
HOLDOUT_GAP = 18  # recorded points stay at least this many px from held-out positions (plus jitter, ~2 cm)
MOVE_MIN, MOVE_MAX = 96, 260  # how far one "sudden move" goes (px, ~8-22 cm)
DOUBLE_MOVE = 1 / 3  # fraction of move episodes with two moves
ORANGE = (255, 150, 0)


def positions_file(override: str | None = None) -> Path | None:
    p = override or CFG.get("positions_file")
    return Path(os.path.expanduser(p)) if p else None


# ---------- position plans ----------

def load_region() -> list[list[float]]:
    if REGION_FILE.exists():
        return json.loads(REGION_FILE.read_text())["polygon"]
    return DEFAULT_REGION


def region_candidates(region: list, margin: float = 6) -> np.ndarray:
    """One candidate every 2 px inside the reachable region (at least `margin` px from the edge, not on the box side)."""
    poly = np.array(region, np.float32)
    (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
    return np.array([(x, y) for y in np.arange(y0, y1 + 1, 2) for x in np.arange(x0, x1 + 1, 2)
                     if x >= BOX_RIGHT_X and cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= margin],
                    np.float32)


def holdout_points(seed: int = 0) -> np.ndarray:
    """HOLDOUT random positions in the reachable region (>= 6 cm apart, >= 2 cm from the edge), never recorded.
    Depends only on the region and the seed, so wide, move and guided_eval --holdout get the same set."""
    cand = region_candidates(load_region(), margin=24)
    rng = np.random.default_rng(seed + 1000)
    chosen: list[np.ndarray] = []
    for i in rng.permutation(len(cand)):
        if all(np.hypot(*(cand[i] - c)) >= 72 for c in chosen):
            chosen.append(cand[i])
            if len(chosen) == HOLDOUT:
                break
    return np.array(chosen, np.float32).reshape(-1, 2)


def plan_moves(rng: np.random.Generator, start: np.ndarray, cand: np.ndarray) -> list[dict]:
    """"Sudden move": from `start` to a random spot MOVE_MIN-MOVE_MAX px away; about 1/3 of the time move again."""
    moves, cur = [], np.asarray(start, np.float32)
    for _ in range(2 if rng.random() < DOUBLE_MOVE else 1):
        d = np.linalg.norm(cand - cur, axis=1)
        ok = np.flatnonzero((d >= MOVE_MIN) & (d <= MOVE_MAX))
        if len(ok) == 0:
            break
        cur = cand[rng.choice(ok)]
        moves.append({"x": float(cur[0]), "y": float(cur[1])})
    return moves


def parse_colors(text: str) -> list[str]:
    colors = [c.strip().lower() for c in text.split(",") if c.strip()]
    bad = [c for c in colors if c not in COLORS]
    if bad or len(set(colors)) != len(colors):
        sc.fail(f"Colours must be among {', '.join(COLORS)} without repeats (got: {text})")
    return colors


def object_task(colors: list[str], targets: list[str]) -> str:
    """Task sentence for multi / select (must be identical when recording, testing and running)."""
    def names(cs):
        return cs[0] if len(cs) == 1 else ", ".join(cs[:-1]) + " and " + cs[-1]
    others = [c for c in colors if c not in targets]
    task = f"Put the {names(targets)} cube{'s' if len(targets) > 1 else ''} in the box"
    if others:
        task += f" and leave the {names(others)} cube{'s' if len(others) > 1 else ''} on the table"
    return task


def spread_points(cand: np.ndarray, points: int) -> np.ndarray:
    """Pick `points` evenly spread candidates (farthest-point sampling + k-means, same as wide)."""
    if len(cand) < points:
        sc.fail(f"Only {len(cand)} candidates in the region, not enough for {points} points; enlarge it with --set-region "
                "or lower --points")
    center = cand.mean(0)
    chosen = [cand[np.argmin(np.linalg.norm(cand - center, axis=1))]]
    while len(chosen) < points:
        d = np.min(np.linalg.norm(cand[:, None] - np.array(chosen)[None], axis=2), axis=1)
        chosen.append(cand[int(np.argmax(d))])
    c = np.array(chosen)
    for _ in range(15):
        near = np.argmin(np.linalg.norm(cand[:, None] - c[None], axis=2), axis=1)
        c = np.array([cand[near == i].mean(0) if np.any(near == i) else c[i] for i in range(len(c))])
    return np.array([cand[np.argmin(np.linalg.norm(cand - p, axis=1))] for p in c], np.float32)


def make_object_plan(colors: list[str], targets: list[str], episodes: int, points: int, jitter: float, seed: int,
                     near_frac: float = NEAR_FRAC, prefix: str = "S", holdout_seed: int | None = None) -> list[dict]:
    """multi / select: a position for every object in every episode.
    Each colour (distractors included) cycles through `points` evenly spread points on its own, so each colour
    appears about equally often in each area. The same pair of positions (e.g. red at A, green at B) is not
    repeated more than needed: pairs already used max_pair times are skipped first.
    select puts the distractor next to a target (OBJ_MIN_SEP-NEAR_MAX px) in about `near_frac` of episodes.
    Objects are always at least OBJ_MIN_SEP px apart."""
    cand = region_candidates(load_region())
    hold = holdout_points(seed if holdout_seed is None else holdout_seed)
    if len(hold):
        cand = cand[np.min(np.linalg.norm(cand[:, None] - hold[None], axis=2), axis=1) >= HOLDOUT_GAP + max(jitter, 4) * 1.5]
    anchors = spread_points(cand, points)
    rng = np.random.default_rng(seed)
    others = [c for c in colors if c not in targets]
    queues: dict[str, list[int]] = {c: [] for c in colors}
    pair_count: dict[tuple, int] = {}  # (colour1, point1, colour2, point2) → times used together
    max_pair = max(1, int(np.ceil(episodes / points / points * 2)))  # about twice the average
    plan = []
    for ep in range(episodes):
        placed: list[tuple[str, int, np.ndarray]] = []  # (colour, point index (-1 = not a round-robin point), position)
        objs = []
        near = bool(others) and rng.random() < near_frac

        def pick(c: str, want_near: bool) -> tuple[int, np.ndarray]:
            q = queues[c]
            if len(q) < len(anchors):
                q.extend(rng.permutation(len(anchors)).tolist())
            n_t = sum(1 for col, _, _ in placed if col in targets)
            for strict in (True, False):  # first pass skips over-used pairs; relax if nothing fits
                for j, k in enumerate(q[:len(anchors)]):
                    trial = anchors[k] + rng.uniform(-jitter, jitter, 2)
                    if any(np.hypot(*(trial - pos)) < OBJ_MIN_SEP for _, _, pos in placed):
                        continue
                    if want_near and min(np.hypot(*(trial - pos)) for _, _, pos in placed[:n_t]) > NEAR_MAX:
                        continue
                    if strict and any(pair_count.get((col, kk, c, k), 0) >= max_pair for col, kk, _ in placed):
                        continue
                    q.pop(j)
                    return k, trial
            # none of the queued points fit: search the whole region
            d = np.min(np.linalg.norm(cand[:, None] - np.array([pos for _, _, pos in placed])[None], axis=2), axis=1)
            ok = np.flatnonzero((d >= OBJ_MIN_SEP) & (d <= NEAR_MAX)) if want_near else np.array([], int)
            if len(ok) == 0:
                ok = np.flatnonzero(d >= OBJ_MIN_SEP)
            return -1, cand[rng.choice(ok)]

        for c in targets + others:
            is_target = c in targets
            k, p = pick(c, want_near=(not is_target) and near and c == others[0])
            for col, kk, _ in placed:
                if k >= 0 and kk >= 0:
                    pair_count[(col, kk, c, k)] = pair_count.get((col, kk, c, k), 0) + 1
            placed.append((c, k, np.asarray(p, np.float32)))
            objs.append({"color": c, "role": "target" if is_target else "distractor",
                         "order": len(objs) + 1 if is_target else 0, "x": float(p[0]), "y": float(p[1])})
        plan.append({"id": f"{prefix}{ep + 1:03d}", "x": objs[0]["x"], "y": objs[0]["y"], "near": near, "objects": objs})
    return plan


def make_plan(points: int, episodes: int, expand: float, jitter: float, seed: int, outside: float = 0,
              region: bool = False, moves: bool = False, holdout_only: bool = False,
              positions: Path | None = None) -> list[dict]:
    """Spread `points` points, then order them into `episodes`: each round places every point once, shuffled,
    with a small random offset. Area: region=False is the recorded work area from the positions file (grown by
    `expand` px, only a little towards the arm); region=True is the whole reachable region (avoiding
    holdout_points). moves=True also plans "where to move it"; holdout_only=True uses only the held-out positions."""
    center = None
    if region:
        cand = region_candidates(load_region())
        hold = holdout_points(seed)
        if holdout_only:
            if len(hold) == 0:
                sc.fail("Region too small for held-out positions; click a larger one with --set-region")
            chosen, points = list(hold), len(hold)
        else:
            if len(hold):
                gap = HOLDOUT_GAP + jitter * 1.5  # still far enough after the random offset
                cand = cand[np.min(np.linalg.norm(cand[:, None] - hold[None], axis=2), axis=1) >= gap]
            center = cand.mean(0) if len(cand) else None
    else:
        positions = positions or positions_file()
        if positions is None or not positions.exists():
            sc.fail("standard / dense / far need a positions file: run `placement_guide.py --dataset PATH --analyze` "
                    "and set positions_file in config.json (or pass --positions), or use --mode wide")
        pos = json.loads(positions.read_text())
        pts = np.array([[e["x"], e["y"]] for e in pos["episodes"]], np.float32)
        hull = cv2.convexHull(pts)
        y_max = pts[:, 1].max() + 8  # below this is where the arm rests
        xs = np.arange(pts[:, 0].min() - expand, pts[:, 0].max() + expand + 1, 2)
        ys = np.arange(pts[:, 1].min() - expand, y_max + 1, 2)

        def ok(x, y):
            d = cv2.pointPolygonTest(hull, (float(x), float(y)), True)  # inside positive, outside negative distance
            return -expand <= d <= -outside if outside > 0 else d >= -expand

        cand = np.array([(x, y) for y in ys for x in xs if x >= BOX_RIGHT_X and ok(x, y)], np.float32)
        center = pts.mean(0)
    if not holdout_only:
        if len(cand) < points:
            sc.fail(f"Only {len(cand)} candidates in the area, not enough for {points} points; enlarge it "
                    "(--expand / --set-region) or lower --points")
        if outside > 0:  # ring mode: start from the point farthest from the centre
            chosen = [cand[np.argmax(np.linalg.norm(cand - center, axis=1))]]
        else:
            chosen = [cand[np.argmin(np.linalg.norm(cand - center, axis=1))]]
        while len(chosen) < points:  # farthest-point sampling spreads the points evenly
            d = np.min(np.linalg.norm(cand[:, None] - np.array(chosen)[None], axis=2), axis=1)
            chosen.append(cand[int(np.argmax(d))])
        if region:  # farthest-point pushes points to the edges; a few k-means steps even them out
            c = np.array(chosen)
            for _ in range(15):
                near = np.argmin(np.linalg.norm(cand[:, None] - c[None], axis=2), axis=1)
                c = np.array([cand[near == i].mean(0) if np.any(near == i) else c[i] for i in range(len(c))])
            chosen = [cand[np.argmin(np.linalg.norm(cand - p, axis=1))] for p in c]  # concave region: snap back inside
    rng = np.random.default_rng(seed)
    prefix = "H" if holdout_only else "M" if moves else "W" if region else "P"
    plan = []
    while len(plan) < episodes:
        for k in rng.permutation(points):
            if len(plan) == episodes:
                break
            x, y = chosen[k] + rng.uniform(-jitter, jitter, 2)
            t = {"id": f"{prefix}{k + 1:02d}", "x": float(x), "y": float(y)}
            if moves:
                t["moves"] = plan_moves(rng, np.array([x, y]), cand)
            plan.append(t)
    return plan


def plan_file(name: str, mode: str) -> Path:
    legacy = OUT_DIR / f"{name}_plan.json"  # plans created before modes existed are standard plans
    if mode == "standard" and legacy.exists():
        return legacy
    return OUT_DIR / f"{name}_{mode}_plan.json"


def load_or_make_plan(name: str, args) -> tuple[Path, list[dict]]:
    OUT_DIR.mkdir(exist_ok=True)
    path = plan_file(name, args.mode)
    if path.exists():
        meta = json.loads(path.read_text())
        if meta.get("region") and meta.get("polygon") != load_region():
            print(f"⚠ {path.name} was planned with an older region (changed later with --set-region). "
                  "Delete this plan file to re-plan with the new region.")
        return path, meta["targets"]
    m = {**MODES[args.mode], **{k: v for k, v in
                                 {"points": args.points, "expand": args.expand, "jitter": args.jitter}.items() if v is not None}}
    if m.get("objects"):
        colors = parse_colors(args.objects)
        targets = colors if m.get("all_targets") else parse_colors(args.targets)
        if not set(targets) <= set(colors) or (not m.get("all_targets") and len(targets) == len(colors)):
            sc.fail(f"--targets ({args.targets}) must be a subset of --objects ({args.objects}) and leave at least one out")
        plan = make_object_plan(colors, targets, args.plan_size or args.episodes, m["points"], m["jitter"], args.seed,
                                prefix="M" if m.get("all_targets") else "S")
        extra = {"polygon": load_region(), "holdout": holdout_points(args.seed).tolist(), "colors": colors,
                 "pick": targets, "task": args.task or object_task(colors, targets)}
        path.write_text(json.dumps({"name": name, "mode": args.mode, "created": datetime.now().isoformat(timespec="seconds"),
                                    **m, **extra, "start_episode": None, "targets": plan}, indent=1, ensure_ascii=False))
        return path, plan
    plan = make_plan(m["points"], args.plan_size or args.episodes, m["expand"], m["jitter"], args.seed, m["outside"],
                     region=m.get("region", False), moves=m.get("moves", False), positions=positions_file(args.positions))
    extra = {"polygon": load_region(), "holdout": holdout_points(args.seed).tolist()} if m.get("region") else {}
    path.write_text(json.dumps({"name": name, "mode": args.mode, "created": datetime.now().isoformat(timespec="seconds"),
                                **m, **extra, "start_episode": None, "targets": plan}, indent=1, ensure_ascii=False))
    return path, plan


def plan_background() -> np.ndarray:
    """Background for plan images: the saved layout, else the analysed dataset's background, else grey."""
    if (sc.REF_DIR / "layout_current.txt").exists():
        return sc.load_layout(None, "front")[1]
    p = positions_file()
    if p and p.exists():
        bg_file = OUT_DIR / f"{Path(json.loads(p.read_text())['dataset']).name}_background.png"
        if bg_file.exists():
            return cv2.cvtColor(cv2.imread(str(bg_file)), cv2.COLOR_BGR2RGB)
    w, h = config.load()["cameras"]["width"], config.load()["cameras"]["height"]
    return np.full((h, w, 3), 60, np.uint8)


def plan_image(name: str, plan: list[dict], region: list | None = None, holdout: list | None = None) -> Path:
    """Plan image: green dots = object positions, orange lines = where it is moved (move),
    red × = held-out test positions, green outline = reachable region."""
    img = plan_background().copy()
    if region:
        cv2.polylines(img, [np.array(region, np.int32)], True, (0, 160, 0), 1)
    for t in plan:
        prev = (int(t["x"]), int(t["y"]))
        for mv in t.get("moves", []):
            cur = (int(mv["x"]), int(mv["y"]))
            cv2.line(img, prev, cur, ORANGE, 1)
            cv2.circle(img, cur, 2, ORANGE, -1)
            prev = cur
    for t in plan:
        if "objects" in t:
            for o in t["objects"]:
                col = COLORS[o["color"]]["rgb"]
                if o["role"] == "target":
                    cv2.circle(img, (int(o["x"]), int(o["y"])), 2, col, -1)
                else:
                    cv2.drawMarker(img, (int(o["x"]), int(o["y"])), col, cv2.MARKER_TILTED_CROSS, 5, 1)
        else:
            cv2.circle(img, (int(t["x"]), int(t["y"])), 2, (0, 230, 0), -1)
    for x, y in holdout or []:
        cv2.drawMarker(img, (int(x), int(y)), (255, 40, 40), cv2.MARKER_TILTED_CROSS, 10, 2)
    seen = {}
    for t in plan:
        if "objects" not in t:  # multi-object plans have a new id every episode: no labels
            seen.setdefault(t["id"], t)
    for pid, t in seen.items():
        cv2.putText(img, pid[1:], (int(t["x"]) + 4, int(t["y"]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    xs, ys = plan_xy(plan)
    xs, ys = xs + [p[0] for p in holdout or []], ys + [p[1] for p in holdout or []]
    x0, y0 = max(0, int(min(xs)) - 60), max(0, int(min(ys)) - 60)
    crop = img[y0:int(max(ys)) + 90, x0:int(max(xs)) + 60]
    path = OUT_DIR / f"{name}_plan.png"
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
                                        cv2.COLOR_RGB2BGR))
    return path


def plan_xy(plan: list[dict]) -> tuple[list[float], list[float]]:
    """Every position used in a plan (including move destinations and all objects)."""
    pts = [(t["x"], t["y"]) for t in plan] + [(m["x"], m["y"]) for t in plan for m in t.get("moves", [])] + \
        [(o["x"], o["y"]) for t in plan for o in t.get("objects", [])]
    return [p[0] for p in pts], [p[1] for p in pts]


def draw_moves(img: np.ndarray, t: dict) -> None:
    prev = (int(t["x"]), int(t["y"]))
    for i, mv in enumerate(t.get("moves", []), 1):
        cur = (int(mv["x"]), int(mv["y"]))
        cv2.arrowedLine(img, prev, cur, ORANGE, 1, tipLength=0.06)
        cv2.circle(img, cur, TOL_PX, ORANGE, 2)
        cv2.putText(img, str(i), (cur[0] + TOL_PX + 3, cur[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE, 2)
        prev = cur


def move_hint(t: dict) -> str:
    order = "orange circle 1, then to 2 when the gripper comes close again" if len(t.get("moves", [])) > 1 else "orange circle 1"
    return f"Helper: when the gripper is 3-5 cm from the object, move it to {order}, then take your hand away"


# ---------- object detection ----------

def detect(rgb: np.ndarray, bg: np.ndarray, roi, target) -> dict | None:
    """Compare with the background; return the object-sized blob in the work area closest to the target
    (big blobs such as the passing arm are ignored)."""
    diff = cv2.absdiff(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)).max(axis=2)
    mask = cv2.morphologyEx((diff > pg.DIFF_THRESHOLD).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    x0, y0, x1, y1 = roi
    clipped = np.zeros_like(mask)
    clipped[max(0, y0):y1, max(0, x0):x1] = mask[max(0, y0):y1, max(0, x0):x1]
    n, _, stats, cent = cv2.connectedComponentsWithStats(clipped)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not 250 <= area <= 5000:
            continue
        d = np.hypot(cent[i][0] - target["x"], cent[i][1] - target["y"])
        if best is None or d < best["d"]:
            best = {"x": float(cent[i][0]), "y": float(cent[i][1]), "area": int(area), "bbox": stats[i, :4].tolist(), "d": d}
    return best


def hue_class(hues: np.ndarray, colors: list[str]) -> str | None:
    """Hues of the saturated pixels of a blob → colour name (only among `colors`; more than half the pixels must
    agree). Red wraps around 0 (0-8 and 165-179), so count fractions instead of taking a median."""
    if len(hues) < 20:
        return None
    for c in colors:
        frac = np.mean(np.any([(hues >= lo) & (hues <= hi) for lo, hi in COLORS[c]["hue"]], axis=0))
        if frac > 0.5:
            return c
    return None


def detect_colors(rgb: np.ndarray, bg: np.ndarray | None, roi, colors: list[str]) -> list[dict]:
    """Find every coloured object in the work area: by difference from the background if there is one, else by
    saturation; then classify by hue. Anything on the box side (left of BOX_RIGHT_X) is ignored."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2HSV)
    sat = (hsv[..., 1] > 80) & (hsv[..., 2] > 50)
    if bg is not None:
        diff = cv2.absdiff(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.GaussianBlur(bg, (5, 5), 0)).max(axis=2)
        mask = (diff > pg.DIFF_THRESHOLD) & sat
    else:
        mask = sat
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    x0, y0, x1, y1 = roi
    clipped = np.zeros_like(mask)
    xs = max(0, x0, BOX_RIGHT_X - 5)
    clipped[max(0, y0):y1, xs:x1] = mask[max(0, y0):y1, xs:x1]
    n, lab, stats, cent = cv2.connectedComponentsWithStats(clipped)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not 150 <= area <= 5000:
            continue
        c = hue_class(hsv[..., 0][(lab == i) & sat], colors)
        if c:
            out.append({"color": c, "x": float(cent[i][0]), "y": float(cent[i][1]), "area": int(area),
                        "bbox": stats[i, :4].tolist()})
    return out


def match_objects(t: dict, dets: list[dict]) -> dict[str, dict | None]:
    """Match every planned object with the closest detection of the same colour (d = px from its target)."""
    res = {}
    for o in t["objects"]:
        same = [dict(d, d=float(np.hypot(d["x"] - o["x"], d["y"] - o["y"]))) for d in dets if d["color"] == o["color"]]
        res[o["color"]] = min(same, key=lambda d: d["d"]) if same else None
    return res


def objects_hint(t: dict) -> str:
    order = " → ".join(f"{o['order']} {o['color']}" for o in t["objects"] if o["role"] == "target")
    skip = ", ".join(o["color"] for o in t["objects"] if o["role"] == "distractor")
    return f"Pick order: {order}" + (f"; do not touch {skip} (leave it on the table)" if skip else "")


# ---------- guide window ----------

class GuideWindow:
    def __init__(self, name: str, plan: list[dict], plan_path: Path):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.Image, self.ImageTk = Image, ImageTk
        self.name, self.plan, self.plan_path = name, plan, plan_path
        self.bg = None  # grabbed live with b (objects out, arm in start pose): a saved layout may contain objects
        xs, ys = plan_xy(plan)
        self.roi = (int(min(xs)) - 60, int(min(ys)) - 60, int(max(xs)) + 60, int(max(ys)) + 40)
        self.root = tk.Tk()
        self.root.title("guided_record - green circle = next position   → record   ← redo   Esc stop   b background")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # stop with Esc, not by closing the window
        self.frame, self.last_det, self.last_dets, self.frame_i = None, None, {}, 0
        self.start = json.loads(plan_path.read_text()).get("start_episode") or 0  # dataset episode of the plan's first entry

    def on_key(self, event) -> None:
        if event.char == "b" and self.frame is not None:
            self.bg = self.frame.copy()
            print("Background grabbed (no objects and no hands in view)", flush=True)

    def target(self, episode: int) -> dict:
        return self.plan[(episode - self.start) % len(self.plan)]

    def _render(self, img: np.ndarray, lines: list) -> None:
        view = pg.put_text(img, lines, size=18)
        photo = self.ImageTk.PhotoImage(self.Image.fromarray(view))
        self.label.configure(image=photo)
        self.label.image = photo
        self.root.update()

    def show(self, rgb: np.ndarray, episode: int, title: str, sub: str) -> None:
        self.frame = rgb
        t = self.target(episode)
        if "objects" in t:
            return self.show_objects(rgb, t, title, sub)
        det = detect(rgb, self.bg, self.roi, t) if self.bg is not None else None
        self.last_det = det
        img = rgb.copy()
        for other in {p["id"]: p for p in self.plan}.values():
            cv2.circle(img, (int(other["x"]), int(other["y"])), 2, (120, 120, 120), -1)
        ok = det is not None and det["d"] <= TOL_PX
        green = (0, 255, 0)
        tx, ty = int(t["x"]), int(t["y"])
        cv2.circle(img, (tx, ty), TOL_PX, green, 2)
        cv2.circle(img, (tx, ty), TOL_PX + 14, green, 1)
        cv2.line(img, (tx - 30, ty), (tx - TOL_PX - 3, ty), green, 1)
        cv2.line(img, (tx + TOL_PX + 3, ty), (tx + 30, ty), green, 1)
        cv2.line(img, (tx, ty - 30), (tx, ty - TOL_PX - 3), green, 1)
        cv2.line(img, (tx, ty + TOL_PX + 3), (tx, ty + 30), green, 1)
        draw_moves(img, t)
        if det is not None:
            x, y, w, h = det["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 220, 0), 2)
            if not ok:
                cv2.arrowedLine(img, (int(det["x"]), int(det["y"])), (tx, ty), (255, 220, 0), 2, tipLength=0.2)
        if self.bg is None:
            msg, color = ("Remove the object, put the arm in the start pose, click this window and press b to grab "
                          "the background (needed to detect the object)"), (255, 90, 90)
        else:
            msg, color, _ = pg.advice(det, t, TOL_PX, True)
        lines = [(title, (255, 255, 255)), (msg, color), (sub, (190, 190, 190))]
        if t.get("moves"):
            lines.insert(2, ("While recording: " + move_hint(t), ORANGE))
        self._render(img, lines)

    def show_objects(self, rgb: np.ndarray, t: dict, title: str, sub: str) -> None:
        """multi / select: a circle per object in its own colour (targets numbered 1, 2..., distractors ×);
        detected objects get a box, misplaced ones an arrow."""
        colors = [o["color"] for o in t["objects"]]
        matched = match_objects(t, detect_colors(rgb, self.bg, self.roi, colors))
        self.last_dets = matched
        self.last_det = matched.get(t["objects"][0]["color"])  # used by guided_eval (first object)
        img = rgb.copy()
        status = []
        for o in t["objects"]:
            col, name = COLORS[o["color"]]["rgb"], o["color"]
            tx, ty = int(o["x"]), int(o["y"])
            cv2.circle(img, (tx, ty), TOL_PX, col, 2)
            cv2.circle(img, (tx, ty), TOL_PX + 12, col, 1)
            if o["role"] == "target":
                cv2.putText(img, str(o["order"]), (tx + TOL_PX + 6, ty - TOL_PX), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            else:
                cv2.drawMarker(img, (tx, ty), col, cv2.MARKER_TILTED_CROSS, 2 * TOL_PX + 10, 2)
            d = matched[o["color"]]
            if d is None:
                status.append(f"{name}: not seen" + ("" if self.bg is not None else " (press b first)"))
                continue
            x, y, w, h = d["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
            if d["d"] <= TOL_PX:
                status.append(f"{name}: ✓")
            else:
                cv2.arrowedLine(img, (int(d["x"]), int(d["y"])), (tx, ty), col, 2, tipLength=0.2)
                status.append(f"{name}: move {d['d'] / pg.PX_PER_CM:.1f} cm along the arrow")
        all_ok = all(matched[o["color"]] is not None and matched[o["color"]]["d"] <= TOL_PX for o in t["objects"])
        summary = ("✓ All positions OK" if all_ok else "   ".join(status), (0, 230, 0) if all_ok else (255, 200, 0))
        if self.bg is None:
            summary = ("Remove all objects, arm in the start pose, click this window and press b to grab the background",
                       (255, 90, 90))
        near = " (this time the distractor is deliberately next to a target)" if t.get("near") else ""
        self._render(img, [(title, (255, 255, 255)), summary, (objects_hint(t) + near, (255, 255, 255)),
                           (sub, (190, 190, 190))])

    def show_recording(self, episode: int) -> None:
        if self.frame is None:
            return
        t = self.target(episode)
        img = (self.frame * 0.4).astype(np.uint8)
        lines = [(f"● Recording episode {episode} (target {t['id']})", (255, 80, 80))]
        if "objects" in t:
            lines.append((objects_hint(t), (255, 255, 255)))
        lines.append(("When done and back in the start pose press → ; bad episode: press ← now", (200, 200, 200)))
        self._render(img, lines)

    def show_live(self, rgb: np.ndarray, episode: int) -> None:
        """move mode while recording: live image + orange circles so the helper knows where to move the object."""
        t = self.target(episode)
        img = rgb.copy()
        cv2.circle(img, (int(t["x"]), int(t["y"])), 4, (0, 255, 0), -1)
        draw_moves(img, t)
        self._render(img, [(f"● Recording episode {episode} ({t['id']}, {len(t['moves'])} move(s))", (255, 80, 80)),
                           (move_hint(t), ORANGE),
                           ("When done and back in the start pose press → ; bad episode: press ← now", (200, 200, 200))])

    def keep_alive(self) -> None:
        self.frame_i += 1
        if self.frame_i % 15 == 0:
            self.root.update()


def log_objects(name: str, repo_id: str, episode: int, target: dict, dets: dict) -> None:
    """multi / select: one row per object (colour, role, target position, actual position)."""
    path = OUT_DIR / f"{name}_objects_log.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "dataset", "episode", "plan_id", "near", "color", "role", "order",
                        "target_x", "target_y", "placed_x", "placed_y", "dist_px"])
        for o in target["objects"]:
            d = dets.get(o["color"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), repo_id, episode, target["id"], int(target.get("near", False)),
                        o["color"], o["role"], o["order"], round(o["x"], 1), round(o["y"], 1),
                        round(d["x"], 1) if d else "", round(d["y"], 1) if d else "", round(float(d["d"]), 1) if d else ""])


def log_placement(name: str, repo_id: str, episode: int, target: dict, det: dict | None) -> None:
    path = OUT_DIR / f"{name}_placement_log.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "dataset", "episode", "target", "target_x", "target_y", "placed_x", "placed_y", "dist_px"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), repo_id, episode, target["id"],
                    round(target["x"], 1), round(target["y"], 1),
                    round(det["x"], 1) if det else "", round(det["y"], 1) if det else "",
                    round(float(det["d"]), 1) if det else ""])


# ---------- hooks into lerobot-record ----------

def install_hooks(name: str, plan: list[dict], plan_path: Path):
    import lerobot.scripts.lerobot_record as lr

    original = lr.record_loop
    state = {"started": False, "gui": None}

    def gui() -> GuideWindow:
        if state["gui"] is None:
            state["gui"] = GuideWindow(name, plan, plan_path)
        return state["gui"]

    def guide_loop(kwargs, episode_fn, title_fn, sub):
        """Run a record_loop that saves nothing (the leader still drives the follower), updating the window every frame."""
        g = gui()
        obs_proc = kwargs["robot_observation_processor"]

        def observe(obs):
            if "front" in obs:
                ep = episode_fn()
                g.show(obs["front"], ep, title_fn(ep), sub)
            return obs_proc(obs)

        return original(**{**kwargs, "robot_observation_processor": observe})

    def patched_record_loop(*args, **kwargs):
        if args:
            raise TypeError("guided_record only supports record_loop called with keyword arguments")
        dataset, events = kwargs.get("dataset"), kwargs["events"]
        if dataset is not None:
            state["dataset"] = dataset
            if not state["started"]:  # before the first episode: let the user place the first position
                state["started"] = True
                ep = dataset.num_episodes
                meta = json.loads(plan_path.read_text())
                if meta.get("start_episode") is None and "mode" in meta:  # new plan: map it from the current episode
                    meta["start_episode"] = ep
                    plan_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
                gui().start = meta.get("start_episode") or 0
                print(f"\nPlace episode {ep}'s position, then press → to start recording (Esc to quit)", flush=True)
                guide_loop({**kwargs, "dataset": None, "control_time_s": 24 * 3600},
                           lambda: ep, lambda e: f"Prepare episode {e}: put the object in the green circle (target {gui().target(e)['id']})",
                           "Bring the arm to the start pose with the leader; when the position is OK press → to record; Esc to quit")
                if events["stop_recording"]:
                    raise KeyboardInterrupt("Esc pressed before recording started")
            ep = dataset.num_episodes
            g = gui()
            if "objects" in g.target(ep):
                log_objects(name, dataset.repo_id, ep, g.target(ep), g.last_dets)
            else:
                log_placement(name, dataset.repo_id, ep, g.target(ep), g.last_det)
            g.show_recording(ep)
            obs_proc = kwargs["robot_observation_processor"]
            moving = bool(g.target(ep).get("moves"))

            def observe_rec(obs):
                if moving and "front" in obs:  # move mode: refresh the live view every 4 frames (~7 Hz, keeps recording fast)
                    g.frame_i += 1
                    if g.frame_i % 4 == 0:
                        g.show_live(obs["front"], ep)
                else:
                    g.keep_alive()
                return obs_proc(obs)

            return original(**{**kwargs, "robot_observation_processor": observe_rec})

        # reset phase: episode N was just recorded (not saved yet). After ← it is a redo of N, otherwise next is N+1
        ds = state["dataset"]

        def next_episode():
            return ds.num_episodes if events["rerecord_episode"] else ds.num_episodes + 1

        def title(e):
            redo = " (redo)" if events["rerecord_episode"] else ""
            return f"Reset: put the object in the green circle for episode {e}{redo} (target {gui().target(e)['id']})"

        return guide_loop(kwargs, next_episode, title,
                          "Bring the arm to the start pose with the leader; when OK press → to record; Esc to stop "
                          "(to redo, press ← while recording)")

    lr.record_loop = patched_record_loop
    return lr


def set_region() -> None:
    """Click the whole reachable region on the standard front layout image; saved to REGION_FILE (wide, move, multi, select)."""
    import tkinter as tk
    from PIL import Image, ImageTk

    layout, img0 = sc.load_layout(None, "front")
    pts = [list(p) for p in load_region()]
    p = positions_file()
    old = [(e["x"], e["y"]) for e in json.loads(p.read_text())["episodes"]] if p and p.exists() else []
    root = tk.Tk()
    root.title("guided_record --set-region: click the whole reachable region")
    label = tk.Label(root, bd=0, highlightthickness=0)
    label.pack()

    def redraw() -> None:
        img = img0.copy()
        cv2.line(img, (BOX_RIGHT_X, 0), (BOX_RIGHT_X, img.shape[0]), (255, 0, 255), 1)
        for x, y in old:
            cv2.circle(img, (int(x), int(y)), 2, (0, 200, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) >= 3, (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 4, (255, 220, 0), -1)
        view = pg.put_text(img, [
            ("Left click: add a corner. Go around the edge of where the arm reaches AND the front camera sees; "
             "avoid where the arm rests", (255, 255, 255)),
            ("Right click: remove last corner   c: clear   d: default   Enter: save   Esc: quit without saving", (190, 190, 190)),
            (f"{len(pts)} corners. Left of the magenta line is the box (never a target); blue dots = recorded positions",
             (190, 190, 190))], size=16)
        photo = ImageTk.PhotoImage(Image.fromarray(view))
        label.configure(image=photo)
        label.image = photo

    def click(e) -> None:
        if e.y < img0.shape[0]:
            pts.append([e.x, e.y])
            redraw()

    def undo(_e) -> None:
        if pts:
            pts.pop()
            redraw()

    def key(e) -> None:
        if e.char == "c":
            pts.clear()
        elif e.char == "d":
            pts[:] = [list(p) for p in DEFAULT_REGION]
        elif e.keysym == "Return":
            if len(pts) < 3:
                print("At least 3 corners are needed")
                return
            OUT_DIR.mkdir(exist_ok=True)
            REGION_FILE.write_text(json.dumps({"polygon": pts, "layout": layout,
                                               "created": datetime.now().isoformat(timespec="seconds")}, indent=1))
            print(f"Saved: {REGION_FILE} ({len(pts)} corners)")
            stale = [f for pat in ("wide", "move", "multi", "select") for f in sorted(OUT_DIR.glob(f"*_{pat}_plan.json"))]
            if stale:
                print("⚠ These plans use the old region; delete them to re-plan with the new one: "
                      + ", ".join(p.name for p in stale))
            root.destroy()
            return
        elif e.keysym == "Escape":
            root.destroy()
            return
        redraw()

    root.bind("<Button-1>", click)
    root.bind("<Button-2>", undo)  # right click on macOS
    root.bind("<Button-3>", undo)
    root.bind("<Key>", key)
    redraw()
    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="name of a new dataset (LeRobot appends date and time)")
    ap.add_argument("--resume", help="add episodes to an existing dataset: its folder")
    ap.add_argument("--episodes", type=int, default=80, help="episodes to record in this session")
    ap.add_argument("--task", default=None,
                    help="task sentence (identical for recording, testing and running; default: config default_task, "
                         "generated automatically for multi / select)")
    ap.add_argument("--mode", choices=list(MODES), default="standard",
                    help="standard: 20 points; dense: 40 closer points; far: 16 points around the edge; "
                         "wide: 36 points in the whole reachable region; move: object moved during recording (two people); "
                         "multi: several objects, pick all; select: pick only the --targets colours")
    ap.add_argument("--objects", default=DEFAULT_OBJECTS, help=f"multi / select: object colours (default {DEFAULT_OBJECTS})")
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help=f"select: colours to pick (default {DEFAULT_TARGETS}; the rest are distractors)")
    ap.add_argument("--points", type=int, default=None, help="number of distinct points (default: the mode's)")
    ap.add_argument("--plan-size", type=int, default=None, help="total episodes in the plan (default: --episodes)")
    ap.add_argument("--expand", type=float, default=None, help="grow the work area by this many px (default: the mode's)")
    ap.add_argument("--jitter", type=float, default=None, help="random offset around each point in px (default: the mode's)")
    ap.add_argument("--positions", help="positions file for standard / dense / far (default: config positions_file)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--display", action="store_true", help="also open Rerun")
    ap.add_argument("--show-plan", action="store_true", help="only create and show the plan, do not record")
    ap.add_argument("--set-region", action="store_true", help="click the whole reachable region on the standard front layout")
    args, extra = ap.parse_known_args()

    if args.set_region:
        set_region()
        return

    if bool(args.name) == bool(args.resume):
        sc.fail("Use exactly one of --name (new dataset) or --resume (add to a dataset)")
    if args.resume:
        root = Path(os.path.expanduser(args.resume))
        if not (root / "meta" / "info.json").exists():
            sc.fail(f"Dataset not found: {root}")
        base = root.name
        bases = set()  # names that already have plans (without _plan and _<mode>); dataset names carry a timestamp
        for f in OUT_DIR.glob("*_plan.json"):
            stem = f.stem.removesuffix("_plan")
            for m in MODES:
                stem = stem.removesuffix(f"_{m}")
            bases.add(stem)
        plan_name = next((b for b in sorted(bases, key=len, reverse=True) if base.startswith(b)), base)
    else:
        plan_name = args.name
    plan_path, plan = load_or_make_plan(plan_name, args)
    if "objects" in plan[0]:
        print(f"Position plan: {plan_path} ({len(plan)} episodes, {len(plan[0]['objects'])} objects each)")
    else:
        print(f"Position plan: {plan_path} ({len({t['id'] for t in plan})} points, {len(plan)} episodes)")
    meta = json.loads(plan_path.read_text())
    img = plan_image(plan_name + ('' if plan_path.stem == plan_name + '_plan' else '_' + args.mode), plan,
                     meta.get("polygon"), meta.get("holdout"))
    print(f"Mode: {args.mode}  plan image: {img}")
    if meta.get("region"):
        src = REGION_FILE.name if REGION_FILE.exists() else "default estimate (not clicked with --set-region yet)"
        print(f"Reachable region: {src}; red × = {len(meta.get('holdout', []))} held-out test positions (guided_eval.py --holdout)")
    if meta.get("mode") == "move":
        print("move mode needs two people: one drives the leader arm, one watches the orange circles and moves the object.")
    if meta.get("objects"):
        if args.task and args.task != meta["task"]:
            print(f"⚠ --task differs from the plan file's sentence; using the plan's: {meta['task']}")
        args.task = meta["task"]
        skip = [c for c in meta["colors"] if c not in meta["pick"]]
        print(f"Objects: {', '.join(meta['colors'])}; pick: {', '.join(meta['pick'])}"
              + (f"; leave: {', '.join(skip)}" if skip else ""))
        print(f"Task sentence: {args.task}")
    args.task = args.task or DEFAULT_TASK
    if args.show_plan:
        return

    argv = [
        "lerobot-record", *config.robot_args(), *config.teleop_args(),
        f"--display_data={'true' if args.display else 'false'}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.episodes}",
        "--dataset.episode_time_s=600", "--dataset.reset_time_s=600",
        "--dataset.push_to_hub=false", "--dataset.streaming_encoding=true", "--dataset.encoder_threads=2",
    ]
    if args.resume:
        argv += [f"--dataset.repo_id={config.repo_id(root.name)}", f"--dataset.root={root}", "--resume=true"]
    else:
        argv += [f"--dataset.repo_id={config.repo_id(args.name)}"]
    argv += extra
    print("\nRunning: " + " ".join(a if " " not in a else repr(a) for a in argv) + "\n", flush=True)

    lr = install_hooks(plan_name, plan, plan_path)
    sys.argv = argv
    try:
        lr.main()
    except KeyboardInterrupt as e:
        print(f"\nStopped: {e}")
    print(f"\nActual placements: {OUT_DIR / (plan_name + '_placement_log.csv')}")
    print(f"The dataset is in ~/.cache/huggingface/lerobot/{config.hf_user()}/ (name ends with date and time).")


if __name__ == "__main__":
    main()
