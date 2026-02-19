"""
Resolve arm CAN interfaces at runtime using USB serial numbers.

Reads config/arms.yaml and matches each USB serial to the
live socketcan interface (can0, can1, …) via udevadm.

Each arm entry also carries its gripper_type so callers don't need
to guess leader vs follower defaults.

Also provides ensure_can_up() to bring up CAN interfaces automatically.

Usage:
    python scripts/resolve_leader_can.py
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "arms.yaml"


@dataclass
class ArmInfo:
    """Resolved information for a single arm.

    Attributes:
        channel: Live CAN interface name (e.g. "can0").
        gripper_type: Gripper type string from the config (e.g. "yam_teaching_handle").
        gravity_comp_factor: Gravity compensation scale factor (leaders only).
    """

    channel: str
    gripper_type: str
    gravity_comp_factor: float = 1.2


def _get_usb_serial(can_iface: str) -> str:
    """Query udevadm for the ID_SERIAL of a CAN network interface.

    Args:
        can_iface (str): Name of the CAN interface (e.g. "can0").

    Returns:
        str: The ID_SERIAL string, or "" if not found.
    """
    p = Path("/sys/class/net") / can_iface
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "-q", "property", "-p", str(p)],
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""
    for line in out.splitlines():
        if line.startswith("ID_SERIAL="):
            return line.split("=", 1)[1]
    return ""


def _scan_can_serials() -> Dict[str, str]:
    """Scan all live can* interfaces and return {usb_serial: iface_name}.

    Returns:
        Dict[str, str]: Mapping of USB serial to interface name.
    """
    found: Dict[str, str] = {}
    for p in Path("/sys/class/net").glob("can*"):
        serial = _get_usb_serial(p.name)
        if serial:
            found[serial] = p.name
    return found


def resolve_arms(
    section: str = "leader_arms",
    config_path: Path = CONFIG_PATH,
) -> Dict[str, ArmInfo]:
    """Resolve arm names to live CAN interfaces and gripper types.

    Args:
        section (str): YAML top-level key to read (e.g. "leader_arms").
        config_path (Path): Path to the YAML config file.

    Returns:
        Dict[str, ArmInfo]: Mapping like
            {"Lleft": ArmInfo(channel="can0", gripper_type="yam_teaching_handle"), ...}.

    Raises:
        FileNotFoundError: If the config file is missing.
        RuntimeError: If any arm's USB serial cannot be matched.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    arms_cfg = cfg[section]
    found = _scan_can_serials()

    result: Dict[str, ArmInfo] = {}
    missing = []
    for arm_name, arm_props in arms_cfg.items():
        usb_serial = arm_props["usb_serial"]
        gripper_type = arm_props["gripper_type"]
        if usb_serial not in found:
            missing.append(f"  {arm_name}: {usb_serial}")
        else:
            gravity_comp_factor = float(arm_props.get("gravity_comp_factor", 1.2))
            result[arm_name] = ArmInfo(
                channel=found[usb_serial],
                gripper_type=gripper_type,
                gravity_comp_factor=gravity_comp_factor,
            )

    if missing:
        raise RuntimeError(
            f"Could not resolve the following arms:\n"
            + "\n".join(missing)
            + f"\n  found serials = {found}"
        )

    return result


def load_teleop_config(
    config_path: Path = CONFIG_PATH,
) -> dict:
    """Load the ``teleop`` section of the config (e.g. bilateral_kp).

    Args:
        config_path (Path): Path to the YAML config file.

    Returns:
        dict: Contents of the ``teleop`` key, or {} if absent.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("teleop", {})


def ensure_can_up(
    arms: Dict[str, ArmInfo],
    bitrate: int = 1_000_000,
) -> None:
    """Bring up CAN interfaces for all resolved arms.

    Runs ``ip link set … up type can bitrate …`` via sudo for each
    unique channel. Safe to call repeatedly — it resets the link first.

    Args:
        arms (Dict[str, ArmInfo]): Resolved arm map from resolve_arms().
        bitrate (int): CAN bitrate (default 1 000 000).
    """
    # Deduplicate channels (two arms might theoretically share one)
    channels_seen: List[str] = []
    for info in arms.values():
        if info.channel not in channels_seen:
            channels_seen.append(info.channel)

    for ch in channels_seen:
        logging.info(f"Bringing up CAN interface {ch} at {bitrate} bps")
        subprocess.run(
            ["sudo", "ip", "link", "set", ch, "down"],
            check=False,
        )
        subprocess.run(
            ["sudo", "ip", "link", "set", ch, "up", "type", "can", "bitrate", str(bitrate)],
            check=True,
        )
    logging.info(f"All CAN interfaces up: {channels_seen}")


if __name__ == "__main__":
    mapping = resolve_arms("leader_arms")
    for name, info in mapping.items():
        print(f"{name}  ->  channel={info.channel}  gripper={info.gripper_type}")

