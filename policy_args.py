"""
Extra `lerobot-rollout` arguments each policy type needs. Shared by register_skill.py and guided_eval.py.

- ACT: nothing extra.
- SmolVLA, π0 / π0.5 and other flow-matching VLAs (they plan a whole action chunk):
    --inference.type=rtc        background inference + Real-Time Chunking, smoother hand-over between chunks
    --rename_map=...            if training renamed front / wrist to the model's camera1 / camera2, rollout must
                                rename the same way (read from the checkpoint's train_config.json; π0.5 trained
                                without renaming gets none)
"""

from __future__ import annotations

import json
from pathlib import Path

VLA_TYPES = {"smolvla", "pi0", "pi05", "pi0_fast"}  # policies that use RTC


def policy_type(model_dir: Path) -> str:
    return json.loads((Path(model_dir) / "config.json").read_text()).get("type", "?")


def rename_map(model_dir: Path) -> dict:
    """Camera renaming used in training (robot name → model name); {} if none."""
    tc = Path(model_dir) / "train_config.json"
    if not tc.exists():
        return {}
    return json.loads(tc.read_text()).get("rename_map") or {}


def rollout_args(model_dir: Path) -> list[str]:
    """Arguments this checkpoint needs for lerobot-rollout."""
    if policy_type(model_dir) not in VLA_TYPES:
        return []
    args = ["--inference.type=rtc"]
    rm = rename_map(model_dir)
    if rm:
        args.append("--rename_map=" + json.dumps(rm))
    return args
