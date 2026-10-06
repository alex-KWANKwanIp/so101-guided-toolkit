#!/usr/bin/env python3
"""
Check that all 6 motors of an arm answer (read-only: the arm does not move and calibration is untouched).

Use it when you see `Failed to write ... on id_=N ... There is no status packet!`:
    python motor_check.py              # follower arm
    python motor_check.py --leader     # leader arm

Reading the result:
    all ✓ and voltage normal       → it was a one-off glitch; restart the program.
    one ✗ and all after it ✗       → loose cable near that motor (motors are daisy-chained, everything after a break is lost).
    only one ✗, the rest fine      → that motor tripped overload protection or is broken: unplug arm power for 10 s.
    voltage < 6 V or very uneven   → power supply not plugged in properly or wrong supply.
"""

from __future__ import annotations

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

import config

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STATUS_BITS = {0: "voltage error", 1: "angle sensor error", 2: "overheating", 3: "over-current", 5: "overload"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leader", action="store_true", help="check the leader arm (default: follower)")
    ap.add_argument("--port", help="serial port (default: from config.json)")
    args = ap.parse_args()

    cfg = config.load()
    port = args.port or cfg["teleop" if args.leader else "robot"]["port"]
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES, 1)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)
    print(f"{'Leader' if args.leader else 'Follower'}: {port}\n")
    try:
        for i, name in enumerate(NAMES, 1):
            if bus.ping(i, num_retry=2) is None:
                print(f"  ✗ {i} {name:<14} no response")
                continue
            info = []
            try:
                volt = bus.read("Present_Voltage", name, normalize=False, num_retry=2) / 10
                temp = bus.read("Present_Temperature", name, normalize=False, num_retry=2)
                status = bus.read("Status", name, normalize=False, num_retry=2)
                info.append(f"voltage {volt:.1f} V  temperature {temp} °C")
                errors = [txt for bit, txt in STATUS_BITS.items() if status >> bit & 1]
                if errors:
                    info.append("⚠ " + ", ".join(errors))
            except ConnectionError as e:
                info.append(f"read failed: {e}")
            print(f"  ✓ {i} {name:<14} " + "  ".join(info))
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
