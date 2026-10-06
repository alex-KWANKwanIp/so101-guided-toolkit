#!/usr/bin/env python3
"""
Intent → skill example (reference for the BCI / AI-agent side).

A BCI or voice front end only needs to output an "intent" and a confidence, e.g.:
    {"intent": "cube", "confidence": 0.82}
This script:
    1. asks the user to confirm when confidence is low ("clarify unclear intents first")
    2. maps the intent to a skill name (intents.json)
    3. calls the skill server and waits for the result

Simulate a BCI output to test:
    python intent_client.py cube --confidence 0.9
    python intent_client.py cube --confidence 0.5     # low confidence: asks you to confirm first
"""

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# intent → skill table, e.g. {"cube": "pick_cube_to_box", "phone": "fetch_phone"}
# updated automatically by `register_skill.py --intent`
INTENTS_FILE = Path(__file__).resolve().parent / "intents.json"

# below this confidence, ask the user first
CONFIRM_THRESHOLD = 0.8


def load_intents() -> dict:
    if not INTENTS_FILE.exists():
        return {}
    return json.loads(INTENTS_FILE.read_text(encoding="utf-8"))


def call(server: str, method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    """Send an HTTP request, return (status code, JSON body)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    req = urllib.request.Request(server + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def run_intent(server: str, intent: str, confidence: float, token: str | None) -> None:
    # 1. the intent must be in the table
    intents = load_intents()
    skill = intents.get(intent)
    if skill is None:
        print(f"No skill is mapped to '{intent}'. Known intents: {list(intents)}")
        return

    # 2. low confidence: confirm first. A BCI can be wrong, so gate it before the arm moves
    if confidence < CONFIRM_THRESHOLD:
        answer = input(f"Detected '{intent}' (confidence {confidence:.0%}). Run {skill}? (y/n) ")
        if answer.strip().lower() != "y":
            print("Cancelled")
            return

    # 3. make sure the arm is idle
    _, status = call(server, "GET", "/status", token=token)
    if status.get("state") != "idle":
        print("The arm is busy, try again later")
        return

    # 4. run the skill. request_id makes network retries safe (the arm will not move twice)
    code, job = call(server, "POST", "/jobs", {"skill": skill, "request_id": uuid.uuid4().hex}, token=token)
    if code not in (200, 202):
        print(f"Could not start: {job}")
        return
    job_id = job["job_id"]
    print(f"Running {skill} (job {job_id}) ...")

    # 5. wait until it ends
    while True:
        time.sleep(1)
        _, job = call(server, "GET", f"/jobs/{job_id}", token=token)
        if job["state"] != "running":
            break

    print(f"Final state: {job['state']}")
    if job["state"] == "finished":
        print("The policy finished running. Note: this does not mean success; check with the camera that the object is in place.")
    else:
        print("Last log lines:")
        print("\n".join(job.get("log_tail", [])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a BCI intent and call a robot-arm skill")
    parser.add_argument("intent", help="intent name, e.g. cube")
    parser.add_argument("--confidence", type=float, default=1.0, help="BCI confidence 0-1")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--token", help="bearer token if the skill server was started with --token")
    args = parser.parse_args()
    run_intent(args.server, args.intent, args.confidence, args.token)


if __name__ == "__main__":
    main()
