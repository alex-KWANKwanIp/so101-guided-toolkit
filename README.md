# SO-101 Guided Toolkit

Tools that make **real-robot imitation learning with [LeRobot](https://github.com/huggingface/lerobot) and the SO-101 arm** more systematic: collect well-spread demonstrations, keep the scene identical between recording and testing, measure success rates fairly, and expose trained policies as skills that an AI agent or a brain-computer interface can call.

Everything runs on top of LeRobot's own `lerobot-record` / `lerobot-rollout`. The toolkit adds guidance windows and bookkeeping around them. It never modifies your recorded data.

> **Status:** built and used on one single-arm SO-101 setup (follower + leader, front + wrist USB cameras), Linux, LeRobot **0.6.1**, ACT and SmolVLA policies. Other setups will probably need small adjustments. Issues and PRs welcome.

<!-- TODO: add a GIF of the green-circle guidance and of the evaluation window here -->

## Why

Imitation policies (ACT, SmolVLA, π0.5...) learn *where things are in the camera image*. In practice this means:

- **Placement matters.** A policy works where you demonstrated and fails a few centimetres away. Placing the object "somewhere different each time" by hand leaves gaps and clusters.
- **The scene must not move.** A camera bumped by 2 cm, or a box shifted, silently breaks a policy that worked yesterday.
- **"It seems to work" is not a result.** You need the same test positions for every checkpoint and a per-condition success rate.
- **A policy is not yet a usable skill.** An agent needs a simple, safe API: one job at a time, idempotent requests, stop and cancel.

## What's inside

| Tool | What it does |
|---|---|
| `guided_record.py` | Wraps `lerobot-record`. Before each episode a window shows **where to place the object** (green circle) and detects where it actually is (arrow + distance). Seven position modes, balanced round-robin plans, held-out test positions, multi-object and distractor tasks, "object moved during the episode" demos. |
| `guided_eval.py` | Wraps `lerobot-rollout`. Uses the **same position plans to test a checkpoint**: place, run, press `s`/`f`, repeat. Writes per-point / per-condition / per-instruction success rates. Has a live **instruction box** to test whether a VLA follows different sentences. |
| `scene_check.py` | Compares the live camera with a saved **standard layout** (or a dataset's first frames): median feature shift in pixels, arrows showing what moved, colour-tone and brightness check, manual white balance / exposure keys. |
| `placement_guide.py` | Analyses where objects were placed in an existing dataset, plans new positions that fill the gaps, and can record one guided episode at a time. |
| `skill_server.py` | HTTP API that runs registered checkpoints as named skills (`POST /jobs`, `GET /jobs/<id>`, `/stop`...). One job at a time, idempotent `request_id`, optional bearer token. |
| `register_skill.py` | Registers a checkpoint as a skill: checks files and cameras, reads the task sentence, estimates a time limit, and adds the right rollout flags for VLA policies. |
| `intent_client.py` | Example client: intent + confidence → confirmation if unsure → skill call → wait for the result (what a BCI or agent would do). |
| `motor_check.py` | Read-only check that all six motors answer, with voltage, temperature and error flags. |
| `make_udev_rules.py` | Linux: give arms and cameras fixed names (`/dev/so101_follower`, `/dev/cam_front`...) by unplugging and replugging each one. |
| `policy_args.py`, `config.py` | Shared helpers: per-policy rollout flags, and your hardware / layout settings. |

## Requirements

- An SO-101 follower + leader arm, a **front** camera that sees the work area and a **wrist** camera. The camera names `front` and `wrist` are assumed throughout.
- **LeRobot 0.6.1** with Feetech support: `pip install "lerobot[feetech]==0.6.1"`. For SmolVLA add the `smolvla` extra (it needs `transformers`).
- Python packages normally present in a LeRobot environment: `opencv-python`, `numpy`, `pandas`, `pillow`, `pynput`, `pyyaml`, `av`; plus `tkinter` for the windows (`conda install tk` if missing).
- Optional (Linux): `v4l-utils` for white balance / exposure control in `scene_check.py`.

> The guidance windows hook into internal LeRobot functions (`lerobot_record.record_loop`, `EpisodicStrategy._policy_loop` / `_reset_loop`). They are tested on 0.6.1 and may need updating for other LeRobot versions.

## Setup

```bash
git clone https://github.com/alex-KWANKwanIp/so101-guided-toolkit.git
cd so101-guided-toolkit
cp config.example.json config.json      # macOS: cp config.example.mac.json config.json
```

Edit `config.json` (see the docstring in `config.py` for every key):

- `robot.port` / `teleop.port`: serial ports. On Linux prefer `/dev/serial/by-id/...` or run `python make_udev_rules.py`. On macOS use `lerobot-find-port`.
- `robot.id` / `teleop.id`: the calibration ids you used with `lerobot-calibrate`.
- `cameras.front` / `cameras.wrist`: `/dev/...` paths (Linux) or indices (macOS: `lerobot-find-cameras opencv`), and `fourcc` (`"MJPG"` on Linux, `null` on macOS).
- `layout.box_right_x`: x pixel in the 640×480 front image left of which no target is placed (where your box sits). Set it to `0` if the box is elsewhere.
- `hf_user`: leave `null` to use the account you are logged in with (`hf auth login`).

Check the motors and save the reference scene:

```bash
python motor_check.py
python scene_check.py --save-layout lab      # arm in start pose, no objects, nothing will move from now on
python guided_record.py --set-region         # click the area the arm can reach and the camera can see
```

## Typical workflow

```bash
# 1. every session: realign the scene (median shift < 8 px), then press q
python scene_check.py --layout

# 2. record 108 well-spread demonstrations over the whole reachable area (36 points × 3)
python guided_record.py --name my_pick_place --mode wide --episodes 108 --show-plan   # look at the plan first
python guided_record.py --name my_pick_place --mode wide --episodes 108

#    ...or split it over several sessions
python guided_record.py --name my_pick_place --mode wide --episodes 60 --plan-size 108
python guided_record.py --resume ~/.cache/huggingface/lerobot/<user>/my_pick_place_<date> --mode wide --episodes 48

# 3. train with lerobot-train as usual (e.g. SmolVLA from lerobot/smolvla_base)

# 4. evaluate on fixed positions, then on the never-recorded held-out positions
python guided_eval.py --policy ~/outputs/train/my_smolvla/checkpoints/040000/pretrained_model --mode wide --points 10 --tag v1
python guided_eval.py --policy ~/outputs/train/my_smolvla/checkpoints/040000/pretrained_model --holdout --rounds 2 --tag v1_holdout

# 5. turn the best checkpoint into a skill and serve it
python register_skill.py --name pick_cube_to_box --checkpoint ~/outputs/train/my_smolvla/checkpoints/040000 --intent cube --eval 8/10
python skill_server.py --host 0.0.0.0 --token "$(openssl rand -hex 16)"
python intent_client.py cube --server http://<robot-pc>:8000 --token <token>
```

## Position modes (`guided_record.py` / `guided_eval.py`)

| Mode | Positions | Use it for |
|---|---|---|
| `standard` | 20 points in the area of a previously recorded dataset (+~2 cm) | a first dataset around known positions (needs `placement_guide.py --analyze`) |
| `dense` | 40 closer points in the same area | filling in between points |
| `far` | 16 points in a ring 2–5 cm outside that area | testing / extending the reach |
| `wide` | 36 points over the whole clicked region, 5 positions held out | "works anywhere on the table" |
| `move` | 24 start points + 1–2 places to move the object to **during** the episode (two people) | teaching recovery: re-find and re-grasp a moved object |
| `multi` | several coloured objects (default red, blue, green), all picked in order 1 → 2 → 3 | multi-object sequences |
| `select` | 3 coloured objects, pick only `--targets` (default red, blue); the rest are distractors, 1/3 of the time placed 5–7 cm from a target | teaching the policy to tell objects apart by colour |

**How plans are built:**

- Pseudo-random with a fixed `--seed`, so the same seed always gives the same plan. Split sessions, re-runs and different checkpoints therefore all line up.
- Round-robin: every round visits every point once in shuffled order, with a small jitter. Point counts never differ by more than one.
- In `multi` / `select`, every colour, distractors included, cycles through the points independently. The planner also avoids repeating the same pair of positions. On a 192-episode select plan, no pair of positions repeats more than twice.
- Use an episode count that is a multiple of the number of points (`wide` 36 → 180 / 216, `select` 24 → 192).

Object detection uses background subtraction: press `b` in the window once with the objects removed. `multi` / `select` also classify colour from HSV hue. If a colour is not detected under your lighting, adjust `COLORS` in `guided_record.py`.

## Evaluation output

`eval_results/<tag>_<mode>_<time>/` contains:

- `results.csv`: one row per episode, with target, placed position, result, duration, instruction, and distractor near / far.
- `summary.txt`: overall success rate, then per point, per round, per distractor condition (select) and per instruction (when you changed it).
- `plan.json`: the exact test positions.

The rollouts themselves are saved as a LeRobot dataset (`rollout_eval_<tag>_...`), so you can watch every attempt.

## Skill server API

| Method | Path | Body / result |
|---|---|---|
| GET | `/skills` | registered skills with training info and measured success rate |
| GET | `/status` | `idle` or `running` + current job |
| POST | `/jobs` | `{"skill": "pick_cube_to_box", "request_id": "unique"}` → `202` + `job_id` (`409` if busy; the same `request_id` returns the same job) |
| GET | `/jobs/<job_id>` | `running` / `finished` / `failed` / `cancelled` / `timeout` + last log lines |
| POST | `/jobs/<job_id>/cancel` | stop that job (the arm returns to where it started) |
| POST | `/stop` | stop whatever is running |

`finished` means the policy ran without errors. **It does not mean the task succeeded:** verify with a camera.

## Safety

- Stay next to the arm when testing a new policy, with a hand near the power switch. `Esc` / `Ctrl+C` stop recording and rollouts.
- `skill_server.py --host 0.0.0.0` makes the arm controllable from your network. Always use `--token` (or `SKILL_SERVER_TOKEN`).
- Only one program can open a camera at a time: close `scene_check.py` before recording or serving skills.

## macOS notes

- Use `config.example.mac.json` (`fourcc: null`, camera indices, `policy_device: "mps"`).
- Give the terminal **Accessibility** and **Input Monitoring** permissions. Without them the arrow keys and `s`/`f` silently do nothing.
- `make_udev_rules.py` and the white balance / exposure keys are Linux-only.

## Files the tools create (git-ignored)

`config.json`, `skills.json`, `intents.json`, `registry_history.csv`, `scene_refs/` (your reference photos), `scene_checks/`, `placement/` (plans and logs), `eval_results/`, `skill_logs/`.

## License

Apache-2.0, see [LICENSE](LICENSE). Not affiliated with Hugging Face or LeRobot.
