#!/usr/bin/env python3
"""
Launch one or both YAM leader arms in zero-gravity (gravity compensation) mode.

CAN channels and gripper types are resolved automatically from
config/arms.yaml (per-arm properties).

Usage:
    # Both leader arms (default)
    python scripts/zero_grav.py

    # Only leader left arm
    python scripts/zero_grav.py --arm Lleft

    # Only leader right arm
    python scripts/zero_grav.py --arm Lright
"""

import argparse
import logging
import signal
import sys
import time
from typing import List

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType
from i2rt.utils.utils import override_log_level

# Resolve CAN channels + gripper types from USB serial mapping
from resolve_leader_can import ArmInfo, ensure_can_up, resolve_arms


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Launch YAM leader arm(s) in zero-gravity mode."
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["Lleft", "Lright"],
        default=None,
        help="Which leader arm to launch. Omit to launch both. L = Leader.",
    )
    return parser.parse_args()


def launch_arm(arm_info: ArmInfo, label: str) -> MotorChainRobot:
    """Initialise a single arm in zero-gravity mode.

    Args:
        arm_info (ArmInfo): Resolved arm info (channel + gripper_type).
        label (str): Human-readable label ("Lleft" / "Lright").

    Returns:
        MotorChainRobot: The initialised robot instance.
    """
    gripper_type = GripperType.from_string_name(arm_info.gripper_type)
    logging.info(
        f"[{label}] Initialising on {arm_info.channel} "
        f"with gripper={gripper_type.value}"
    )
    robot = get_yam_robot(
        channel=arm_info.channel,
        gripper_type=gripper_type,
        zero_gravity_mode=True,
        gravity_comp_factor=arm_info.gravity_comp_factor,
    )
    logging.info(f"[{label}] Ready — zero-gravity mode active")
    return robot


def main() -> None:
    """Entry point: resolve CAN channels, launch arm(s), loop until Ctrl-C."""
    args = parse_args()
    override_log_level(level=logging.INFO)

    # Auto-resolve CAN interfaces and gripper types from config
    logging.info("Resolving leader-arm CAN interfaces from USB serials...")
    arm_map = resolve_arms("leader_arms")
    for name, info in arm_map.items():
        logging.info(f"  {name} -> channel={info.channel}, gripper={info.gripper_type}")

    # Bring CAN interfaces up (bitrate 1 Mbps); required before opening the bus
    ensure_can_up(arm_map)

    # Decide which arms to launch
    arms_to_launch: List[str] = (
        [args.arm] if args.arm else list(arm_map.keys())
    )

    robots: List[MotorChainRobot] = []
    for arm_name in arms_to_launch:
        robot = launch_arm(arm_map[arm_name], label=arm_name)
        robots.append(robot)

    # Graceful shutdown on Ctrl-C / SIGTERM
    def _shutdown(sig: int, frame: object) -> None:
        logging.info("Shutting down...")
        for r in robots:
            r.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logging.info(
        f"Zero-gravity mode active for: {', '.join(arms_to_launch)}. "
        "Press Ctrl-C to exit."
    )

    # Keep the main thread alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
