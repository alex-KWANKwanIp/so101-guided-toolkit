"""
Shared settings for every tool in this folder: where the arms and cameras are, your Hugging Face
user name, and a few numbers that describe your table layout.

Defaults live in DEFAULTS below. Put your own values in `config.json` next to this file (copy
`config.example.json`); you only need the keys you want to change, nested sections are merged.

    hf_user           Hugging Face user/org used for dataset repo ids. null = ask `huggingface_hub`
                      who is logged in, falling back to "local".
    robot / teleop    LeRobot type, serial port and calibration id of the follower / leader arm.
                      Linux: prefer /dev/serial/by-id/... (stable across reboots); macOS: /dev/tty.usbmodem...
    cameras           front / wrist: a /dev/... path (Linux) or an index 0, 1, 2 (macOS / Windows),
                      plus width, height, fps and fourcc ("MJPG" on Linux, null on macOS).
    policy_device     null = use the device stored in the checkpoint (usually cuda); "mps" on Apple Silicon.
    default_task      Task sentence used when --task is not given (must match the dataset exactly).
    layout            Front-camera pixel coordinates (640x480) describing your table:
                        box_right_x     targets are never placed left of this x (where the box is); 0 = off
                        default_region  polygon of the reachable area, used until you run
                                        `guided_record.py --set-region`
                        px_per_cm       rough scale, only used to print distances in cm
    reference_dataset Dataset folder whose first frames scene_check.py / placement_guide.py compare against
                      when no saved layout is used.
    positions_file    placement/<dataset>_positions.json written by `placement_guide.py --analyze`; defines the
                      work area for guided_record.py's standard / dense / far modes (wide / move / multi / select
                      use the clicked region instead).
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config.json"

DEFAULTS: dict = {
    "hf_user": None,
    "robot": {"type": "so101_follower", "port": "/dev/ttyACM0", "id": "follower01"},
    "teleop": {"type": "so101_leader", "port": "/dev/ttyACM1", "id": "leader01"},
    "cameras": {"front": "/dev/video0", "wrist": "/dev/video2", "width": 640, "height": 480, "fps": 30,
                "fourcc": "MJPG"},
    "policy_device": None,
    "default_task": "Pick up the cube and place it in the box",
    "layout": {
        "box_right_x": 345,
        "default_region": [[350, 120], [625, 120], [625, 330], [590, 330], [590, 245], [405, 245], [405, 330],
                           [350, 330]],
        "px_per_cm": 12.0,
    },
    "reference_dataset": None,
    "positions_file": None,
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    if CONFIG_FILE.exists():
        user = json.loads(CONFIG_FILE.read_text())
        return _merge(DEFAULTS, {k: v for k, v in user.items() if not k.startswith("_")})
    return copy.deepcopy(DEFAULTS)


@lru_cache(maxsize=1)
def hf_user() -> str:
    """Hugging Face user for dataset repo ids (config.json `hf_user`, else whoever is logged in, else "local")."""
    if load()["hf_user"]:
        return load()["hf_user"]
    try:
        from huggingface_hub import whoami

        return whoami()["name"]
    except Exception:
        return "local"


def repo_id(name: str) -> str:
    return f"{hf_user()}/{name}"


def camera_entry(name: str, device) -> str:
    cam = load()["cameras"]
    extra = f", fourcc: {cam['fourcc']}" if cam.get("fourcc") else ""
    return (f"{name}: {{type: opencv, index_or_path: {device}, width: {cam['width']}, height: {cam['height']}, "
            f"fps: {cam['fps']}{extra}}}")


def cameras_arg(front=None, wrist=None) -> str:
    """Value for --robot.cameras (front / wrist default to config)."""
    cam = load()["cameras"]
    return "{" + camera_entry("front", front if front is not None else cam["front"]) + ", " + \
        camera_entry("wrist", wrist if wrist is not None else cam["wrist"]) + "}"


def robot_args(with_cameras: bool = True) -> list[str]:
    r = load()["robot"]
    args = [f"--robot.type={r['type']}", f"--robot.port={r['port']}", f"--robot.id={r['id']}"]
    return args + ([f"--robot.cameras={cameras_arg()}"] if with_cameras else [])


def teleop_args() -> list[str]:
    t = load()["teleop"]
    return [f"--teleop.type={t['type']}", f"--teleop.port={t['port']}", f"--teleop.id={t['id']}"]
