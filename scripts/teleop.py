#!/usr/bin/env python3
"""
Bimanual leader-follower teleop for YAM arms.

Resolves CAN interfaces and gripper types from config/arms.yaml,
brings up CAN links, and runs leader-follower pairs.

Usage:
    # Both arm pairs (Lleft→Fleft + Lright→Fright)
    python scripts/teleop.py

    # Left pair only (Lleft → Fleft)
    python scripts/teleop.py --left

    # Right pair only (Lright → Fright)
    python scripts/teleop.py --right

    # Custom bilateral_kp
    python scripts/teleop.py --bilateral_kp 0.15
"""

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Dict, List, Tuple

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType
from i2rt.utils.utils import override_log_level

from resolve_leader_can import ArmInfo, ensure_can_up, load_teleop_config, resolve_arms

# Use different ports for left and right follower servers
PORT_LEFT = 11333
PORT_RIGHT = 11334

# Track all robot instances for clean shutdown
_all_robots: List[MotorChainRobot] = []


# ── Re-use the server/client/leader classes from minimum_gello ──────────

class ServerRobot:
    """Portal-based RPC server wrapping a follower robot."""

    def __init__(self, robot: MotorChainRobot, port: int) -> None:
        import portal

        self._robot = robot
        self._server = portal.Server(port)
        logging.info(f"Follower server binding to port {port}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Start serving (blocks)."""
        self._server.start()


class ClientRobot:
    """Portal-based RPC client to talk to a follower server."""

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        import portal

        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        """Get follower joint positions.

        Returns:
            np.ndarray: Current joint positions.
        """
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Send a position command to the follower.

        Args:
            joint_pos (np.ndarray): Target joint positions.
        """
        self._client.command_joint_pos(joint_pos)


class YAMLeaderRobot:
    """Thin wrapper around a MotorChainRobot with a teaching handle."""

    def __init__(self, robot: MotorChainRobot) -> None:
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> Tuple[np.ndarray, list]:
        """Read leader qpos (with gripper from encoder) and button states.

        Returns:
            Tuple[np.ndarray, list]: (qpos_with_gripper, io_inputs).
        """
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command arm joints (6-DOF, no gripper).

        Args:
            joint_pos (np.ndarray): 6-DOF target positions.
        """
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """Update PD gains.

        Args:
            kp (np.ndarray): Proportional gains.
            kd (np.ndarray): Derivative gains.
        """
        self._robot.update_kp_kd(kp, kd)


# ── Helpers ──────────────────────────────────────────────────────────────

def start_follower_server(
    arm_info: ArmInfo,
    port: int,
    label: str,
) -> threading.Thread:
    """Create a follower robot and start its server in a daemon thread.

    Args:
        arm_info (ArmInfo): Resolved follower arm info.
        port (int): Port for the portal server.
        label (str): Label for logging.

    Returns:
        threading.Thread: The daemon thread running the server.
    """
    gripper_type = GripperType.from_string_name(arm_info.gripper_type)
    logging.info(f"[{label}] Creating follower on {arm_info.channel} "
                 f"(gripper={gripper_type.value})")
    robot = get_yam_robot(
        channel=arm_info.channel,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
    )
    _all_robots.append(robot)
    server = ServerRobot(robot, port)

    thread = threading.Thread(
        target=server.serve,
        name=f"follower_{label}",
        daemon=True,
    )
    thread.start()
    logging.info(f"[{label}] Follower server started on port {port}")
    return thread


def run_leader_follower_loop(
    leader: YAMLeaderRobot,
    client: ClientRobot,
    bilateral_kp: float,
    gravity_comp_factor: float,
    label: str,
    stop_event: threading.Event,
) -> None:
    """Run the leader-follower sync loop for one arm pair.

    Args:
        leader (YAMLeaderRobot): The leader arm.
        client (ClientRobot): RPC client to the follower server.
        bilateral_kp (float): Bilateral PD gain factor.
        gravity_comp_factor (float): Gravity comp scale when synced (0 when not synced).
        label (str): Label for logging.
        stop_event (threading.Event): Set to signal shutdown.
    """
    leader_kp = leader._robot._kp.copy()

    current_joint_pos, _ = leader.get_info()
    current_follower_joint_pos = client.get_joint_pos()
    logging.info(f"[{label}] Leader qpos:   {current_joint_pos}")
    logging.info(f"[{label}] Follower qpos: {current_follower_joint_pos}")

    def slow_move(target: np.ndarray, start: np.ndarray, duration: float = 1.0) -> None:
        """Gradually move follower from start to target."""
        steps = 100
        for i in range(steps):
            if stop_event.is_set():
                return
            blend = i / steps
            cmd = (1 - blend) * start + blend * target
            client.command_joint_pos(cmd)
            time.sleep(duration / steps)

    synchronized = False
    while not stop_event.is_set():
        current_joint_pos, current_button = leader.get_info()

        # Button[0] toggles sync and gravity compensation together.
        # Gravity comp is only active while synced so the arm doesn't
        # fly up when the user is not holding it.
        if current_button[0] > 0.5:
            if not synchronized:
                leader._robot.gravity_comp_factor = gravity_comp_factor
                leader.update_kp_kd(
                    kp=leader_kp * bilateral_kp,
                    kd=np.zeros(6),
                )
                leader.command_joint_pos(current_joint_pos[:6])
                slow_move(current_joint_pos, current_follower_joint_pos)
                logging.info(f"[{label}] Synchronized (gravity_comp={gravity_comp_factor})")
            else:
                leader._robot.gravity_comp_factor = 0.0
                leader.update_kp_kd(
                    kp=np.zeros(6),
                    kd=np.zeros(6),
                )
                leader.command_joint_pos(current_follower_joint_pos[:6])
                logging.info(f"[{label}] Un-synchronized (gravity_comp off)")
            synchronized = not synchronized

            # Wait for button release
            while current_button[0] > 0.5 and not stop_event.is_set():
                time.sleep(0.03)
                current_joint_pos, current_button = leader.get_info()

        current_follower_joint_pos = client.get_joint_pos()

        if synchronized:
            client.command_joint_pos(current_joint_pos)
            # Reason: bilateral force feedback — leader feels follower resistance
            leader.command_joint_pos(current_follower_joint_pos[:6])

        time.sleep(0.01)


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Bimanual leader-follower teleop for YAM arms.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--left",
        action="store_true",
        help="Run only the left pair (Lleft → Fleft).",
    )
    group.add_argument(
        "--right",
        action="store_true",
        help="Run only the right pair (Lright → Fright).",
    )
    teleop_cfg = load_teleop_config()
    parser.add_argument(
        "--bilateral_kp",
        type=float,
        default=teleop_cfg.get("bilateral_kp", 0.2),
        help="Bilateral PD gain factor (default from config).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: resolve arms, bring up CAN, run teleop."""
    args = parse_args()
    override_log_level(level=logging.INFO)

    # Resolve all arms from config
    logging.info("Resolving arm CAN interfaces...")
    leaders = resolve_arms("leader_arms")
    followers = resolve_arms("follower_arms")

    # Bring up all CAN interfaces
    all_arms: Dict[str, ArmInfo] = {**leaders, **followers}
    ensure_can_up(all_arms)

    # Decide which pairs to run
    pairs: List[Tuple[str, str, int]] = []  # (leader_key, follower_key, port)
    if args.left:
        pairs = [("Lleft", "Fleft", PORT_LEFT)]
    elif args.right:
        pairs = [("Lright", "Fright", PORT_RIGHT)]
    else:
        pairs = [
            ("Lleft", "Fleft", PORT_LEFT),
            ("Lright", "Fright", PORT_RIGHT),
        ]

    stop_event = threading.Event()

    # Start follower servers
    for _, f_key, port in pairs:
        start_follower_server(followers[f_key], port, label=f_key)

    # Give servers a moment to bind
    time.sleep(1.0)

    # Create leaders and connect clients
    leader_threads: List[threading.Thread] = []
    for l_key, f_key, port in pairs:
        l_info = leaders[l_key]
        gripper_type = GripperType.from_string_name(l_info.gripper_type)
        logging.info(f"[{l_key}] Creating leader on {l_info.channel} "
                     f"(gripper={gripper_type.value})")
        robot = get_yam_robot(
            channel=l_info.channel,
            gripper_type=gripper_type,
            zero_gravity_mode=True,
            gravity_comp_factor=0.0,  # starts off; enabled when user presses sync button
        )
        _all_robots.append(robot)
        leader = YAMLeaderRobot(robot)
        client = ClientRobot(port)

        t = threading.Thread(
            target=run_leader_follower_loop,
            args=(leader, client, args.bilateral_kp, l_info.gravity_comp_factor, f"{l_key}→{f_key}", stop_event),
            name=f"teleop_{l_key}",
            daemon=True,
        )
        t.start()
        leader_threads.append(t)

    # Graceful shutdown
    _shutting_down = False

    def _shutdown(sig: int, frame: object) -> None:
        nonlocal _shutting_down
        if _shutting_down:
            # Second Ctrl-C: force exit immediately
            logging.info("Force exit.")
            import os
            os._exit(1)
        _shutting_down = True
        logging.info("Shutting down teleop (Ctrl-C again to force)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pair_labels = [f"{l}→{f}" for l, f, _ in pairs]
    logging.info(f"Teleop active: {', '.join(pair_labels)}. Press Ctrl-C to exit.")

    # Keep main thread alive until stop is signalled
    while not stop_event.is_set():
        time.sleep(0.5)

    # Wait for leader threads to finish
    for t in leader_threads:
        t.join(timeout=3.0)

    # Close all robots (stops motor chains and internal threads)
    logging.info("Closing all robots...")
    for robot in _all_robots:
        try:
            robot.close()
        except Exception as e:
            logging.warning(f"Error closing robot: {e}")

    logging.info("Teleop stopped.")


if __name__ == "__main__":
    main()

