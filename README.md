# FoodManipLowLevel

Bimanual food manipulation low-level control stack. Built on top of [I2RT](https://i2rt.com/) YAM robot arms with [LeRobot](https://github.com/huggingface/lerobot) integration.

## Hardware Setup

- **2× Leader arms** (YAM + teaching handle) — for teleoperation input
- **2× Follower arms** (YAM + crank_4310 gripper) — for manipulation
- **4× CANable USB-CAN adapters** — one per arm

Arm-to-CAN mapping is defined in [`config/leader_arms.yaml`](./config/leader_arms.yaml) by USB serial number, so CAN channel names (can0, can1, …) are resolved automatically at runtime regardless of plug order.

## Clone & Install (for a new machine)

### 1. Clone with lerobot submodule

```bash
git clone --recurse-submodules git@github.com:jefferzn2001/FoodManipLowLevel.git
cd FoodManipLowLevel
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --depth 1
```

### 2. Create the Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11
source .venv/bin/activate
```

```bash
sudo apt update
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

If pip is missing inside the venv:

```bash
python -m ensurepip --upgrade
```

### 3. Install lerobot (optional, for policy training)

```bash
cd lerobot
pip install -e .
cd ..
```

### 4. Auto-start CAN interfaces on boot (one-time)

```bash
sudo sh devices/install_devices.sh
```

This installs a udev rule to automatically bring up all `can*` interfaces at 1 Mbps on plug-in.

## Quick Start

### Bring up CAN interfaces (if not auto-started)

```bash
sh scripts/reset_all_can.sh
```

### Zero gravity mode (test a single arm)

```bash
# Both leader arms
python scripts/zero_grav.py

# Left leader only
python scripts/zero_grav.py --arm Lleft

# Right leader only
python scripts/zero_grav.py --arm Lright
```

CAN channels and gripper types are auto-resolved from `config/leader_arms.yaml`.

### Bimanual teleop (leader-follower)

```bash
# Both arm pairs (Lleft→Fleft + Lright→Fright)
python scripts/teleop.py

# Left pair only
python scripts/teleop.py --left

# Right pair only
python scripts/teleop.py --right

# Custom bilateral feedback strength
python scripts/teleop.py --bilateral_kp 0.15
```

Press the **top button** on the teaching handle to toggle sync on/off. `bilateral_kp` controls how much the leader arm resists (recommended 0.1–0.2).

### Camera calibration capture

```bash
python scripts/camera.py
```

SPACE to save a frame, ESC to quit. Images saved to `./camcal/`.

## Arm Configuration

All arm CAN mappings and gripper types are in [`config/leader_arms.yaml`](./config/leader_arms.yaml):

```yaml
leader_arms:
  Lleft:
    usb_serial: "canable.io_canable2_gs_usb_00330052594E501820313332"
    gripper_type: "yam_teaching_handle"
  Lright:
    usb_serial: "canable.io_canable2_gs_usb_005F0056594E501820313332"
    gripper_type: "yam_teaching_handle"

follower_arms:
  Fleft:
    usb_serial: "canable.io_canable2_gs_usb_002A0064594E501820313332"
    gripper_type: "crank_4310"
  Fright:
    usb_serial: "canable.io_canable2_gs_usb_004C0053594E501820313332"
    gripper_type: "crank_4310"
```

To find USB serials for new adapters:

```bash
ls -l /sys/class/net/can*
udevadm info -q property -p /sys/class/net/can0 | grep ID_SERIAL
```

## Gripper Types

| Gripper Name | Description |
|---|---|
| `crank_4310` | Zero-linkage crank gripper (follower arms) |
| `linear_3507` | Linear gripper with DM3507 motor (requires calibration) |
| `linear_4310` | Linear gripper with DM4310 motor |
| `yam_teaching_handle` | Teaching handle for leader arms (trigger + 2 buttons) |
| `no_gripper` | No gripper attached |

## Project Structure

```
FoodManipLowLevel/
├── config/
│   └── leader_arms.yaml        # Arm CAN serial → name mapping
├── scripts/
│   ├── teleop.py               # Bimanual leader-follower teleop
│   ├── zero_grav.py            # Zero gravity mode (test arms)
│   ├── resolve_leader_can.py   # USB serial → CAN channel resolver
│   ├── camera.py               # Camera preview + calibration capture
│   ├── reset_all_can.sh        # Reset all CAN interfaces
│   └── minimum_gello.py        # Single-pair leader-follower (original)
├── i2rt/                       # I2RT robot SDK
│   ├── robots/                 # Robot control (get_robot, motor_chain_robot)
│   ├── motor_drivers/          # CAN/DM motor drivers
│   ├── robot_models/           # URDF/MuJoCo XMLs
│   └── utils/                  # MuJoCo, encoder, gamepad utilities
├── lerobot/                    # HuggingFace LeRobot (submodule)
└── examples/                   # I2RT example scripts
```

## Git Setup (for maintainers)

### Change remote to private repo (one-time)

```bash
# Rename original i2rt remote to "upstream" for future pulls
git remote rename origin upstream

# Add your private repo as the new origin
git remote add origin git@github.com:jefferzn2001/FoodManipLowLevel.git

# Push everything to your private repo
git push -u origin main
```

To pull upstream i2rt updates later:

```bash
git fetch upstream
git merge upstream/main
```

### Add lerobot as a shallow submodule (one-time, already done)

```bash
git submodule add --depth 1 git@github.com:huggingface/lerobot.git lerobot
git commit -m "Add lerobot as shallow submodule"
git push
```

## Advanced: YAM Motor Configuration

See the [I2RT upstream docs](https://github.com/i2rt-robotics/i2rt) for:

- [Setting persistent CAN IDs](doc/set_persist_id_socket_can.md)
- [Teaching handle details](doc/yam_handle_readme.md)
- Motor timeout config: `python i2rt/motor_config_tool/set_timeout.py --channel can0`
- Motor zero offset: `python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1`

## Acknowledgments

- [I2RT](https://i2rt.com/) — YAM robot arm hardware and SDK
- [LeRobot](https://github.com/huggingface/lerobot) — Policy training framework
- [TidyBot++](https://github.com/jimmyyhwu/tidybot2) — Flow base inspiration
- [GELLO](https://github.com/wuphilipp/gello_software) — Teleop inspiration
