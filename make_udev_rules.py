#!/usr/bin/env python3
"""
Give the arms and cameras fixed device names (Linux only), so it no longer matters which USB port they use.

Names created (default):
    /dev/so101_follower   follower arm
    /dev/so101_leader     leader arm
    /dev/cam_front        front camera
    /dev/cam_wrist        wrist camera

Usage (no sudo needed for this script; it prints the sudo commands to run at the end):
    python make_udev_rules.py
The script asks you to unplug → replug each device in turn and works out which is which.

When you add devices later (a second follower, a head camera...), regenerate everything at once:
    python make_udev_rules.py --arms so101_follower so101_follower2 so101_leader \
                              --cameras cam_front cam_wrist cam_head

Why: Linux numbers devices in plug-in order (ttyACM0, video2...), so the numbers change.
This script reads each device's own serial number and writes udev rules: "see this serial → create this name".
If two devices share a serial (or have none), it falls back to the physical USB port (it will warn you).

Depth cameras (RealSense) do not need this: LeRobot finds them by serial_number_or_name.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RULES_FILE = "99-so101.rules"

LABELS = {
    "so101_follower": "follower arm",
    "so101_leader": "leader arm",
    "cam_front": "front camera",
    "cam_wrist": "wrist camera",
}


def fail(msg: str) -> None:
    print(f"\n✗ {msg}")
    sys.exit(1)


# ---------- device info ----------

def udev_props(dev: str) -> dict:
    """Device properties from udevadm (vendor id, model id, serial, USB path...)."""
    out = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={dev}"],
        capture_output=True, text=True, check=True,
    ).stdout
    props = {}
    for line in out.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value
    return props


def list_serial_ports() -> set:
    return set(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))


def is_capture_node(dev: str) -> bool:
    """A USB camera usually has two /dev/video nodes: index 0 is the image, the other is metadata."""
    name = os.path.basename(os.path.realpath(dev))
    try:
        return Path(f"/sys/class/video4linux/{name}/index").read_text().strip() == "0"
    except OSError:
        return False


def list_cameras() -> set:
    return {d for d in glob.glob("/dev/video*") if is_capture_node(d)}


# ---------- unplug / replug identification ----------

def wait_for_change(list_fn, before: set, want: str, timeout: float) -> set:
    """Wait for a device to disappear ("removed") or appear ("added"); return the changed devices."""
    deadline = time.time() + timeout
    while True:
        now = list_fn()
        diff = (before - now) if want == "removed" else (now - before)
        if diff or time.time() > deadline:
            break
        time.sleep(0.3)
    if diff:
        time.sleep(1.0)  # let the system finish setting the device up
        now = list_fn()
        diff = (before - now) if want == "removed" else (now - before)
    return diff


def identify(name: str, kind: str) -> str:
    label = LABELS.get(name, name)
    list_fn = list_serial_ports if kind == "arm" else list_cameras
    before = list_fn()
    if not before:
        fail("No " + ("arm" if kind == "arm" else "camera") + " found. Plug everything in first.")

    input(f"\nUnplug the USB cable of [{label}], then press Enter...")
    removed = wait_for_change(list_fn, before, "removed", timeout=10)
    if len(removed) != 1:
        fail(f"{len(removed)} devices disappeared {sorted(removed)}; expected exactly 1. Run again and unplug one cable at a time.")
    after_unplug = list_fn()

    input(f"Found it (it was {removed.pop()}). Plug it back in (any port), then press Enter...")
    added = wait_for_change(list_fn, after_unplug, "added", timeout=15)
    if len(added) != 1:
        fail(f"{len(added)} new devices {sorted(added)}; expected exactly 1. Check the cable and run again.")
    dev = added.pop()
    print(f"✓ [{label}] is now {dev}")
    return dev


# ---------- rules ----------

def usb_key(props: dict, field: str) -> str:
    """Newer systemd also provides ID_USB_* (not overwritten by other rules); prefer it when present."""
    return f"ID_USB_{field}" if f"ID_USB_{field}" in props else f"ID_{field}"


def make_rule(name: str, dev: str, kind: str, all_props: dict) -> tuple:
    """Return (rule text, how it matches, warning)."""
    p = udev_props(dev)
    k_vid, k_pid, k_serial = usb_key(p, "VENDOR_ID"), usb_key(p, "MODEL_ID"), usb_key(p, "SERIAL_SHORT")
    vid, pid, serial, path = p.get(k_vid), p.get(k_pid), p.get(k_serial), p.get("ID_PATH")
    if not vid or not pid:
        fail(f"{dev} does not look like a USB device; cannot create a rule.")

    same_model = [q for q in all_props.values() if q.get(k_vid) == vid and q.get(k_pid) == pid]
    serial_unique = bool(serial) and sum(q.get(k_serial) == serial for q in same_model) == 1

    if kind == "arm":
        head = 'SUBSYSTEM=="tty"'
        tail = f', SYMLINK+="{name}", MODE="0666"'  # MODE also fixes serial-port permissions
    else:
        head = 'SUBSYSTEM=="video4linux", ATTR{index}=="0"'
        tail = f', SYMLINK+="{name}"'
    match = f'{head}, ENV{{{k_vid}}}=="{vid}", ENV{{{k_pid}}}=="{pid}"'

    warning = None
    if serial_unique:
        rule = f'{match}, ENV{{{k_serial}}}=="{serial}"{tail}'
        how = f"matched by serial {serial}: any USB port works"
    else:
        if not path:
            fail(f"{dev} has no unique serial and no USB path; cannot create a rule.")
        rule = f'{match}, ENV{{ID_PATH}}=="{path}"{tail}'
        how = "matched by USB port position"
        warning = (
            f"[{LABELS.get(name, name)}] shares its serial with another device of the same model (or has none), "
            "so it is matched by USB port: always plug it into this same port (label the port)."
        )
    return rule, how, warning


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed device names for arms and cameras (udev rules)")
    parser.add_argument("--arms", nargs="*", default=["so101_follower", "so101_leader"])
    parser.add_argument("--cameras", nargs="*", default=["cam_front", "cam_wrist"])
    parser.add_argument("--output", default=RULES_FILE)
    args = parser.parse_args()

    if shutil.which("udevadm") is None:
        fail("udevadm not found (this script runs on Linux only).")

    print("Plug in all arms and cameras first. You will be asked to unplug and replug them one by one.")
    print("Close lerobot-teleoperate, lerobot-record, skill_server.py and anything else using these devices first.")

    found = {}
    for name in args.arms:
        found[name] = ("arm", identify(name, "arm"))
    for name in args.cameras:
        found[name] = ("camera", identify(name, "camera"))

    # check serial uniqueness against everything currently connected
    arm_props = {d: udev_props(d) for d in list_serial_ports()}
    cam_props = {d: udev_props(d) for d in list_cameras()}

    lines = [
        "# Generated by make_udev_rules.py: fixed names for SO-101 arms and cameras",
        f"# Created: {time.strftime('%Y-%m-%d %H:%M')}",
    ]
    warnings = []
    print("\nRules:")
    for name, (kind, dev) in found.items():
        rule, how, warning = make_rule(name, dev, kind, arm_props if kind == "arm" else cam_props)
        lines += [f"# {LABELS.get(name, name)} ({how})", rule]
        print(f"  /dev/{name:<16} ← {how}")
        if warning:
            warnings.append(warning)

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ Rules written to {Path(args.output).resolve()}")
    for w in warnings:
        print(f"! {w}")

    print("\nNext: install the rules (asks for your password)")
    print(f"  sudo cp {args.output} /etc/udev/rules.d/{RULES_FILE}")
    print("  sudo udevadm control --reload-rules")
    print("  sudo udevadm trigger")
    print("  ls -l " + " ".join(f"/dev/{n}" for n in found))

    print("\nThen use the fixed names in config.json, e.g.:")
    for name, (kind, _) in found.items():
        if name == "so101_follower":
            print('  "robot": {"port": "/dev/so101_follower"}')
        elif name == "so101_leader":
            print('  "teleop": {"port": "/dev/so101_leader"}')
        elif kind == "camera":
            print(f'  "cameras": {{"{name.removeprefix("cam_")}": "/dev/{name}"}}')


if __name__ == "__main__":
    main()
