#!/usr/bin/env python3
"""
Bimanual recording script for YAM arms — LeRobot dataset format.

Records follower joint states (observation), leader joint commands (action),
and RGB camera streams in LeRobot v3 format, ready for training ACT,
Diffusion Policy, or any other LeRobot-compatible policy.

Controls (teaching handle buttons):
  Top button    — toggle sync + gravity compensation (same as teleop.py)
  Bottom button — press once to START an episode, press again to SAVE and end it

Data is saved to:  data/<name>/
Format:            LeRobot v3 (parquet + mp4)

Usage:
    python scripts/record.py --name pick_place --task "pick up the block"
    python scripts/record.py --name pick_place --task "..." --visualize
    python scripts/record.py --name pick_place --task "..." --hz 15
    python scripts/record.py --name pick_place --task "..." --left
"""

import argparse
import logging
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType
from i2rt.utils.utils import override_log_level

from resolve_leader_can import ArmInfo, ensure_can_up, load_teleop_config, resolve_arms

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

PORT_LEFT = 11333
PORT_RIGHT = 11334
NUM_ARM_JOINTS = 6
NUM_JOINTS = 7  # 6 arm + 1 gripper

_all_robots: List[MotorChainRobot] = []


# ── Shared state ──────────────────────────────────────────────────────────

class ArmState:
    """Thread-safe snapshot of one arm pair's current state."""

    def __init__(self) -> None:
        self.leader_qpos: Optional[np.ndarray] = None
        self.follower_qpos: Optional[np.ndarray] = None
        self.synchronized: bool = False
        self._lock = threading.Lock()

    def update(self, leader_qpos: np.ndarray, follower_qpos: np.ndarray, synced: bool) -> None:
        with self._lock:
            self.leader_qpos = leader_qpos.copy()
            self.follower_qpos = follower_qpos.copy()
            self.synchronized = synced

    def snapshot(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        with self._lock:
            lq = self.leader_qpos.copy() if self.leader_qpos is not None else None
            fq = self.follower_qpos.copy() if self.follower_qpos is not None else None
            return lq, fq, self.synchronized


class RecordingState:
    """Controls episode recording lifecycle, safe to call from multiple threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_recording = False
        self.episode_idx = 0
        self._save_requested = False
        self._last_toggle_time = 0.0

    def toggle(self, label: str = "") -> Optional[str]:
        """Toggle recording. Returns 'started', 'stopped', or None (cooldown)."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_toggle_time < 1.0:
                return None
            self._last_toggle_time = now
            if not self.is_recording:
                self.is_recording = True
                logging.info(f"[{label}] ● Recording STARTED — episode {self.episode_idx}")
                return "started"
            else:
                self.is_recording = False
                self._save_requested = True
                logging.info(f"[{label}] ■ Recording STOPPED — saving episode {self.episode_idx}")
                return "stopped"

    def take_save_request(self) -> bool:
        with self._lock:
            if self._save_requested:
                self._save_requested = False
                self.episode_idx += 1
                return True
            return False


# ── Camera reader ─────────────────────────────────────────────────────────

class CameraReader:
    """Continuously reads frames from a camera in a background thread."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"cam_{index}"
        )

    def start(self) -> None:
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def detect_cameras() -> List[CameraReader]:
    """Auto-detect all available cameras, start readers, return them."""
    readers = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            reader = CameraReader(idx)
            reader.start()
            readers.append(reader)
            logging.info(f"Camera {idx}: {reader.width}×{reader.height}")
        else:
            cap.release()
    if not readers:
        logging.warning("No cameras detected — recording without images")
    return readers


# ── Visualisation helper ──────────────────────────────────────────────────

def tile_frames(
    frames: List[Optional[np.ndarray]],
    labels: List[str],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    n = len(frames)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    cell_w = target_w // ncols
    cell_h = target_h // nrows
    canvas = np.zeros((nrows * cell_h, ncols * cell_w, 3), dtype=np.uint8)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        row, col = i // ncols, i % ncols
        y0, x0 = row * cell_h, col * cell_w
        cell = cv2.resize(frame, (cell_w, cell_h)) if frame is not None else \
               np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell
    return canvas


# ── Robot classes ─────────────────────────────────────────────────────────

class ServerRobot:
    def __init__(self, robot: MotorChainRobot, port: int) -> None:
        import portal
        self._robot = robot
        self._server = portal.Server(port)
        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        self._server.start()


class ClientRobot:
    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        import portal
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)


class YAMLeaderRobot:
    def __init__(self, robot: MotorChainRobot) -> None:
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> Tuple[np.ndarray, list]:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        gripper_cmd = 1 - encoder_obs[0].position
        return np.concatenate([qpos, [gripper_cmd]]), encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == NUM_ARM_JOINTS
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


def start_follower_server(arm_info: ArmInfo, port: int, label: str) -> threading.Thread:
    gripper_type = GripperType.from_string_name(arm_info.gripper_type)
    logging.info(f"[{label}] Creating follower on {arm_info.channel}")
    robot = get_yam_robot(
        channel=arm_info.channel,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
    )
    _all_robots.append(robot)
    server = ServerRobot(robot, port)
    thread = threading.Thread(target=server.serve, name=f"follower_{label}", daemon=True)
    thread.start()
    return thread


# ── Teleop loop ───────────────────────────────────────────────────────────

def run_leader_follower_loop(
    leader: YAMLeaderRobot,
    client: ClientRobot,
    bilateral_kp: float,
    gravity_comp_factor: float,
    arm_state: ArmState,
    recording_state: RecordingState,
    label: str,
    stop_event: threading.Event,
) -> None:
    leader_kp = leader._robot._kp.copy()
    current_joint_pos, _ = leader.get_info()
    current_follower_joint_pos = client.get_joint_pos()

    def slow_move(target: np.ndarray, start: np.ndarray, duration: float = 1.0) -> None:
        steps = 100
        for i in range(steps):
            if stop_event.is_set():
                return
            blend = i / steps
            client.command_joint_pos((1 - blend) * start + blend * target)
            time.sleep(duration / steps)

    synchronized = False
    while not stop_event.is_set():
        current_joint_pos, current_button = leader.get_info()

        # Top button: toggle sync + gravity comp
        if current_button[0] > 0.5:
            if not synchronized:
                leader._robot.gravity_comp_factor = gravity_comp_factor
                leader.update_kp_kd(kp=leader_kp * bilateral_kp, kd=np.zeros(NUM_ARM_JOINTS))
                leader.command_joint_pos(current_joint_pos[:NUM_ARM_JOINTS])
                slow_move(current_joint_pos, current_follower_joint_pos)
                logging.info(f"[{label}] Synchronized")
            else:
                leader._robot.gravity_comp_factor = 0.0
                leader.update_kp_kd(kp=np.zeros(NUM_ARM_JOINTS), kd=np.zeros(NUM_ARM_JOINTS))
                leader.command_joint_pos(current_follower_joint_pos[:NUM_ARM_JOINTS])
                logging.info(f"[{label}] Un-synchronized")
            synchronized = not synchronized
            while current_button[0] > 0.5 and not stop_event.is_set():
                time.sleep(0.03)
                current_joint_pos, current_button = leader.get_info()

        # Bottom button: start/stop episode recording
        if current_button[1] > 0.5:
            if synchronized or recording_state.is_recording:
                recording_state.toggle(label=label)
            else:
                logging.info(f"[{label}] Enable sync first before recording")
            while current_button[1] > 0.5 and not stop_event.is_set():
                time.sleep(0.03)
                current_joint_pos, current_button = leader.get_info()

        current_follower_joint_pos = client.get_joint_pos()
        if synchronized:
            client.command_joint_pos(current_joint_pos)
            leader.command_joint_pos(current_follower_joint_pos[:NUM_ARM_JOINTS])

        arm_state.update(current_joint_pos, current_follower_joint_pos, synchronized)
        time.sleep(0.01)


# ── Recording thread ──────────────────────────────────────────────────────

def recording_loop(
    state_left: Optional[ArmState],
    state_right: Optional[ArmState],
    cameras: List[CameraReader],
    dataset: "LeRobotDataset",
    task: str,
    hz: float,
    recording_state: RecordingState,
    stop_event: threading.Event,
) -> None:
    period = 1.0 / hz
    frames_in_episode = 0

    while not stop_event.is_set():
        t_start = time.monotonic()

        if recording_state.is_recording:
            left_leader, left_follower = None, None
            right_leader, right_follower = None, None
            if state_left is not None:
                left_leader, left_follower, _ = state_left.snapshot()
            if state_right is not None:
                right_leader, right_follower, _ = state_right.snapshot()

            left_ready = state_left is None or left_leader is not None
            right_ready = state_right is None or right_leader is not None

            if left_ready and right_ready:
                action_parts, state_parts = [], []
                if state_left is not None:
                    action_parts.append(left_leader.astype(np.float32))
                    state_parts.append(left_follower.astype(np.float32))
                if state_right is not None:
                    action_parts.append(right_leader.astype(np.float32))
                    state_parts.append(right_follower.astype(np.float32))

                frame: dict = {
                    "task": task,
                    "action": np.concatenate(action_parts),
                    "observation.state": np.concatenate(state_parts),
                }
                for i, cam in enumerate(cameras):
                    bgr = cam.get_frame()
                    if bgr is not None:
                        frame[f"observation.images.cam_{i}"] = Image.fromarray(
                            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        )

                dataset.add_frame(frame)
                frames_in_episode += 1

        if recording_state.take_save_request():
            if frames_in_episode > 0:
                dataset.save_episode()
                logging.info(
                    f"Episode {recording_state.episode_idx - 1} saved "
                    f"({frames_in_episode} frames, {frames_in_episode / hz:.1f}s)"
                )
            else:
                logging.warning("Empty episode discarded")
            frames_in_episode = 0

        elapsed = time.monotonic() - t_start
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── Dataset features ──────────────────────────────────────────────────────

def build_features(
    arm_labels: List[str],
    cameras: List[CameraReader],
) -> dict:
    joint_names = []
    for arm in arm_labels:
        joint_names += [f"{arm}_j{i}" for i in range(NUM_ARM_JOINTS)]
        joint_names.append(f"{arm}_gripper")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
    }
    for i, cam in enumerate(cameras):
        features[f"observation.images.cam_{i}"] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    teleop_cfg = load_teleop_config()
    parser = argparse.ArgumentParser(
        description="Record YAM bimanual demonstrations in LeRobot format."
    )
    parser.add_argument("--name", required=True,
                        help="Dataset name. Data saved to data/<name>/.")
    parser.add_argument("--task", required=True,
                        help='Task description, e.g. "pick up the block".')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--left", action="store_true", help="Left arm only.")
    group.add_argument("--right", action="store_true", help="Right arm only.")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Recording frequency in Hz (default: 10).")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live camera feeds in a fullscreen window while recording.")
    parser.add_argument(
        "--bilateral_kp", type=float,
        default=teleop_cfg.get("bilateral_kp", 0.2),
        help="Bilateral PD gain factor.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not LEROBOT_AVAILABLE:
        sys.exit("LeRobot not found. Install: pip install -e lerobot/")

    args = parse_args()
    override_log_level(level=logging.INFO)

    # ── Cameras ──────────────────────────────────────────────────────────
    logging.info("Detecting cameras...")
    cameras = detect_cameras()
    time.sleep(0.3)  # warm-up
    logging.info(f"{len(cameras)} camera(s) active")

    # ── Resolve arms ─────────────────────────────────────────────────────
    logging.info("Resolving arm CAN interfaces...")
    leaders = resolve_arms("leader_arms")
    followers = resolve_arms("follower_arms")
    ensure_can_up({**leaders, **followers})

    if args.left:
        pairs = [("Lleft", "Fleft", PORT_LEFT)]
    elif args.right:
        pairs = [("Lright", "Fright", PORT_RIGHT)]
    else:
        pairs = [("Lleft", "Fleft", PORT_LEFT), ("Lright", "Fright", PORT_RIGHT)]

    arm_labels = [l_key for l_key, _, _ in pairs]

    # ── Dataset ───────────────────────────────────────────────────────────
    root = Path("data") / args.name
    features = build_features(arm_labels, cameras)
    dataset = LeRobotDataset.create(
        repo_id=f"yam/{args.name}",
        fps=args.hz,
        features=features,
        robot_type="yam_bimanual",
        root=root,
        use_videos=True,
        image_writer_threads=max(4 * len(cameras), 4),
    )
    logging.info(f"Dataset → {root.resolve()}")
    logging.info(f"State/action dim: {len(features['action']['names'])}, "
                 f"cameras: {len(cameras)}, hz: {args.hz}")

    # ── Shared state + stop event ─────────────────────────────────────────
    arm_states: Dict[str, ArmState] = {l_key: ArmState() for l_key, _, _ in pairs}
    recording_state = RecordingState()
    stop_event = threading.Event()

    # ── Follower servers ──────────────────────────────────────────────────
    for _, f_key, port in pairs:
        start_follower_server(followers[f_key], port, label=f_key)
    time.sleep(1.0)

    # ── Recording thread ──────────────────────────────────────────────────
    rec_thread = threading.Thread(
        target=recording_loop,
        args=(arm_states.get("Lleft"), arm_states.get("Lright"),
              cameras, dataset, args.task, args.hz, recording_state, stop_event),
        name="recording", daemon=True,
    )
    rec_thread.start()

    # ── Teleop threads ────────────────────────────────────────────────────
    for l_key, f_key, port in pairs:
        l_info = leaders[l_key]
        gripper_type = GripperType.from_string_name(l_info.gripper_type)
        robot = get_yam_robot(
            channel=l_info.channel,
            gripper_type=gripper_type,
            zero_gravity_mode=True,
            gravity_comp_factor=0.0,
        )
        _all_robots.append(robot)
        leader = YAMLeaderRobot(robot)
        client = ClientRobot(port)
        t = threading.Thread(
            target=run_leader_follower_loop,
            args=(leader, client, args.bilateral_kp, l_info.gravity_comp_factor,
                  arm_states[l_key], recording_state, f"{l_key}→{f_key}", stop_event),
            name=f"teleop_{l_key}", daemon=True,
        )
        t.start()

    # ── Shutdown handler ──────────────────────────────────────────────────
    _shutting_down = False

    def _shutdown(sig: int, frame: object) -> None:
        nonlocal _shutting_down
        if _shutting_down:
            import os; os._exit(1)
        _shutting_down = True
        logging.info("Shutting down (Ctrl-C again to force)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logging.info(
        f"\nRecorder ready — {len(pairs)} arm pair(s), {len(cameras)} camera(s) @ {args.hz} Hz\n"
        f"  Top button    → sync on/off + gravity comp\n"
        f"  Bottom button → start episode / save & end episode\n"
        f"  Ctrl-C        → finalize and quit\n"
    )

    # ── Main loop: optional live visualisation ────────────────────────────
    if args.visualize and cameras:
        cam_labels = [f"cam {c.index}" for c in cameras]
        win = "record.py — Q/ESC to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while not stop_event.is_set():
            frames = [c.get_frame() for c in cameras]
            rect = cv2.getWindowImageRect(win)
            disp_w, disp_h = max(rect[2], 640), max(rect[3], 480)
            canvas = tile_frames(frames, cam_labels, disp_w, disp_h)

            # Recording status overlay
            if recording_state.is_recording:
                label = f"  REC  ep {recording_state.episode_idx}"
                cv2.putText(canvas, label, (disp_w - 300, disp_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 4)
                cv2.putText(canvas, label, (disp_w - 300, disp_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
            else:
                label = f"ep {recording_state.episode_idx} saved"
                cv2.putText(canvas, label, (disp_w - 280, disp_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                cv2.putText(canvas, label, (disp_w - 280, disp_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                stop_event.set()

        cv2.destroyAllWindows()
    else:
        while not stop_event.is_set():
            time.sleep(0.5)

    # ── Cleanup ───────────────────────────────────────────────────────────
    if recording_state.is_recording:
        logging.info("Saving in-progress episode before exit...")
        dataset.save_episode()
        recording_state.episode_idx += 1

    dataset.finalize()
    logging.info(f"Done — {recording_state.episode_idx} episode(s) → {root.resolve()}")

    for cam in cameras:
        cam.stop()
    for robot in _all_robots:
        try:
            robot.close()
        except Exception as e:
            logging.warning(f"Error closing robot: {e}")


if __name__ == "__main__":
    main()
