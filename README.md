# YAM Bimanual Food Manipulation

Low-level control and data collection stack for bimanual food manipulation using [YAM](https://i2rt.com/) robot arms and [LeRobot](https://github.com/huggingface/lerobot) for policy training.

## Hardware

| Component | Qty | Role |
|-----------|-----|------|
| YAM arm + teaching handle | 2 | Leader — you hold and move these |
| YAM arm + crank_4310 gripper | 2 | Follower — copies your motion |
| CANable USB-CAN adapter | 4 | One per arm |
| RGB USB cameras | 1–4 | Scene observation for policy input |

CAN channel names (can0, can1, …) are resolved automatically at runtime from USB serial numbers — no manual renaming needed. See [`config/arms.yaml`](./config/arms.yaml).

---

## Install

### 1. Clone

```bash
git clone --recurse-submodules git@github.com:jefferzn2001/FoodManipLowLevel.git
cd FoodManipLowLevel
```

Already cloned without submodules?
```bash
git submodule update --init --depth 1
```

### 2. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11
source .venv/bin/activate
sudo apt update && sudo apt install build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

### 3. LeRobot (for recording + training)

```bash
pip install -e lerobot/
```

### 4. CAN auto-start on boot (one-time)

```bash
sudo sh devices/install_devices.sh
```

Installs a udev rule that brings up all `can*` interfaces at 1 Mbps on plug-in.

---

## Usage

### Check cameras

```bash
python scripts/cam.py
```

Shows all detected cameras tiled in a fullscreen window. Press Q or ESC to quit.

### Test arms — zero-gravity mode

```bash
python scripts/zero_grav.py            # both leaders
python scripts/zero_grav.py --arm Lleft
python scripts/zero_grav.py --arm Lright
```

### Teleop only (no recording)

```bash
python scripts/teleop.py              # both pairs: Lleft→Fleft + Lright→Fright
python scripts/teleop.py --left       # left pair only
python scripts/teleop.py --right      # right pair only
```

**Teaching handle controls:**
- **Top button** — press once to start sync (gravity comp on, bilateral feedback on); press again to stop
- Sync must be active before the follower mirrors your motion

### Record demonstrations

```bash
python scripts/record.py --name pick_place --task "pick up the block"
python scripts/record.py --name pick_place --task "..." --visualize   # live camera view
python scripts/record.py --name pick_place --task "..." --hz 15       # higher frequency
python scripts/record.py --name pick_place --task "..." --left        # one arm only
```

**Teaching handle controls during recording:**
- **Top button** — sync on/off (same as teleop)
- **Bottom button** — press to START an episode; press again to SAVE it

Data is saved to `data/<name>/` in LeRobot v3 format (parquet + mp4), ready for training.

---

## Training

No modifications to LeRobot needed — the recorded dataset is directly compatible.

```bash
# ACT (recommended starting point — most sample-efficient)
python lerobot/src/lerobot/scripts/lerobot_train.py \
  dataset.repo_id=yam/pick_place \
  dataset.root=data/pick_place \
  policy=act \
  output_dir=outputs/act_pick_place

# Diffusion Policy
python lerobot/src/lerobot/scripts/lerobot_train.py \
  dataset.repo_id=yam/pick_place \
  dataset.root=data/pick_place \
  policy=diffusion \
  output_dir=outputs/dp_pick_place
```

**How many episodes?**
- ACT: 50–200 (start here — designed for bimanual, very sample-efficient)
- Diffusion Policy: 100–500 (better generalisation, needs more data)

Start with 50 clean ACT demonstrations. Keep episodes short (5–15 s) and consistent.

---

## Configuration — `config/arms.yaml`

```yaml
leader_arms:
  Lleft:
    usb_serial: "canable.io_canable2_gs_usb_00330052594E501820313332"
    gripper_type: "yam_teaching_handle"
    gravity_comp_factor: 1.2   # increase if arm feels too heavy to move
  Lright:
    usb_serial: "canable.io_canable2_gs_usb_005F0056594E501820313332"
    gripper_type: "yam_teaching_handle"
    gravity_comp_factor: 1.2

follower_arms:
  Fleft:
    usb_serial: "canable.io_canable2_gs_usb_002A0064594E501820313332"
    gripper_type: "crank_4310"
  Fright:
    usb_serial: "canable.io_canable2_gs_usb_004C0053594E501820313332"
    gripper_type: "crank_4310"

teleop:
  bilateral_kp: 0.2   # how much the leader resists follower (0.1–0.3)
```

To find USB serials for new adapters:
```bash
udevadm info -q property -p /sys/class/net/can0 | grep ID_SERIAL
```

---

## Project Structure

```
FoodManipLowLevel/
├── config/
│   └── arms.yaml               # CAN serial → arm mapping, gains, bilateral_kp
├── scripts/
│   ├── teleop.py               # Bimanual leader-follower teleop
│   ├── record.py               # Record demonstrations → LeRobot dataset
│   ├── cam.py                  # Live multi-camera fullscreen viewer
│   ├── zero_grav.py            # Zero-gravity mode for arm testing
│   └── resolve_leader_can.py   # USB serial → CAN channel resolver
├── i2rt/                       # YAM robot SDK
│   ├── robots/                 # MotorChainRobot, get_robot
│   ├── motor_drivers/          # CAN / DM motor drivers
│   ├── robot_models/           # MuJoCo XMLs for gravity compensation
│   └── utils/
├── data/                       # Recorded datasets (gitignored)
│   └── <name>/                 # LeRobot v3 format
├── outputs/                    # Training checkpoints (gitignored)
├── lerobot/                    # HuggingFace LeRobot (submodule)
└── devices/
    └── install_devices.sh      # udev rule installer for CAN auto-start
```

---

## Gripper Types

| Name | Description |
|------|-------------|
| `yam_teaching_handle` | Teaching handle (trigger + 2 buttons) — leader arms |
| `crank_4310` | Zero-linkage crank gripper — follower arms |
| `linear_3507` | Linear gripper with DM3507 (requires calibration) |
| `linear_4310` | Linear gripper with DM4310 |
| `no_gripper` | No gripper |

---

## Advanced

- Set persistent CAN IDs: `python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1`
- Motor timeout: `python i2rt/motor_config_tool/set_timeout.py --channel can0`
- Full YAM docs: [I2RT upstream](https://github.com/i2rt-robotics/i2rt)

## Acknowledgments

- [I2RT](https://i2rt.com/) — YAM robot arms and SDK
- [LeRobot](https://github.com/huggingface/lerobot) — Policy training framework
- [GELLO](https://github.com/wuphilipp/gello_software) — Teleop inspiration
