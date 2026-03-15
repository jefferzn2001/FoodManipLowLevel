#!/usr/bin/env python3
"""
Deploy a VLA (Vision-Language-Action) policy on YAM bimanual arms.

Captures camera frames, sends them along with a goal image to a running
VLA inference server (Dexbotic OFT), and executes the predicted delta
actions on the follower arms. Leader arm top buttons act as e-stop.

Prerequisites:
    1. Start the VLA server (on the same or a remote machine):
       cd food_manipulation
       CUDA_VISIBLE_DEVICES=0 python playground/custom/blockincup_oft.py --task inference

    2. Capture a goal image (what the final scene should look like):
       python scripts/cam.py   # press 's' to save a snapshot, or use any image

    3. Run this script:
       python scripts/deploy_vla.py --goal_image goal.png
       python scripts/deploy_vla.py --goal_image goal.png --visualize
       python scripts/deploy_vla.py --goal_image goal.png --server http://remote:7891

Action format (from VLA):
    The server returns 16 timesteps × 14D actions.
    Per timestep: [left_j1..j6_delta, left_gripper_abs,
                   right_j1..j6_delta, right_gripper_abs]
    Joints 0-5 and 7-12 are DELTA positions (added to current).
    Joints 6 and 13 are ABSOLUTE gripper positions.
"""

import argparse
import contextlib
import glob
import io
import logging
import math
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import requests

os.makedirs(os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts"), exist_ok=True)


@contextlib.contextmanager
def _quiet_stderr():
    """Suppress C-level stderr (Qt/OpenCV noise)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType
from i2rt.utils.utils import override_log_level

from resolve_leader_can import ArmInfo, ensure_can_up, resolve_arms

PORT_LEFT = 11333
PORT_RIGHT = 11334
NUM_ARM_JOINTS = 6
NUM_JOINTS = 7  # 6 arm + 1 gripper
VLA_ACTION_DIM = 14  # 7 per arm (6 joints + 1 gripper)
VLA_CHUNK_SIZE = 16

_all_robots: List[MotorChainRobot] = []


# ── Camera reader ─────────────────────────────────────────────────────────

class CameraReader:
    """Continuously reads frames from a camera in a background thread."""

    def __init__(self, device: str, width: int = 640, height: int = 480) -> None:
        self.device = device
        self.index = int(device.replace("/dev/video", ""))
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            self._cap.release()
            self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"cam_{self.index}"
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
        """Returns latest BGR frame or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def detect_cameras() -> List[CameraReader]:
    """Auto-detect cameras, probing both index0 and index1 per USB port."""
    by_path_all = sorted(glob.glob("/dev/v4l/by-path/*-video-index*"))
    if by_path_all:
        ports: dict[str, list[str]] = {}
        for p in by_path_all:
            port_key = p.rsplit("-video-index", 1)[0]
            ports.setdefault(port_key, []).append(p)
        logging.info(f"Found {len(ports)} USB camera port(s)")
        readers: List[CameraReader] = []
        for port_key in sorted(ports):
            links = sorted(ports[port_key])
            for lnk in links:
                dev = os.path.realpath(lnk)
                try:
                    reader = CameraReader(dev)
                    reader.start()
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        if reader.get_frame() is not None:
                            break
                        time.sleep(0.05)
                    if reader.get_frame() is not None:
                        readers.append(reader)
                        logging.info(f"  Camera {dev}: {reader.width}×{reader.height}")
                        break
                    else:
                        reader.stop()
                except Exception:
                    pass
        return readers

    candidates = sorted(
        (p for p in glob.glob("/dev/video*") if p[len("/dev/video"):].isdigit()),
        key=lambda p: int(p[len("/dev/video"):]),
    )
    readers = []
    for path in candidates:
        try:
            reader = CameraReader(path)
            reader.start()
            time.sleep(0.3)
            if reader.get_frame() is not None:
                readers.append(reader)
                logging.info(f"  Camera {path}: {reader.width}×{reader.height}")
            else:
                reader.stop()
        except Exception:
            pass
    return readers


# ── Robot helpers ─────────────────────────────────────────────────────────

class ServerRobot:
    """Portal-based RPC server wrapping a follower robot."""

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
    """Portal-based RPC client for commanding a follower robot."""

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        import portal
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)


class YAMLeaderRobot:
    """Leader arm wrapper — only used for e-stop button reading."""

    def __init__(self, robot: MotorChainRobot) -> None:
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_buttons(self) -> list:
        """Read teaching handle button states. Returns [top_button, bottom_button]."""
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        return encoder_obs[0].io_inputs


def start_follower_server(arm_info: ArmInfo, port: int, label: str) -> threading.Thread:
    """Create a follower robot and start its RPC server in a daemon thread."""
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


# ── Button monitor thread ─────────────────────────────────────────────────

def button_monitor(
    leaders: List[YAMLeaderRobot],
    stop_event: threading.Event,
    running_event: threading.Event,
) -> None:
    """Poll leader arm buttons. Top button toggles running/paused."""
    while not stop_event.is_set():
        for leader in leaders:
            try:
                buttons = leader.get_buttons()
                if buttons[0] > 0.5:
                    if not running_event.is_set():
                        running_event.set()
                        logging.info("START — VLA policy running")
                    else:
                        running_event.clear()
                        logging.warning("E-STOP — robot paused (press top button to resume)")
                    while not stop_event.is_set():
                        time.sleep(0.03)
                        buttons = leader.get_buttons()
                        if buttons[0] < 0.5:
                            break
            except Exception:
                pass
        time.sleep(0.02)


# ── VLA client ────────────────────────────────────────────────────────────

def encode_frame_as_png(bgr_frame: np.ndarray) -> bytes:
    """Encode a BGR OpenCV frame to PNG bytes for HTTP upload."""
    ret, buf = cv2.imencode(".png", bgr_frame)
    return buf.tobytes()


def query_vla_server(
    server_url: str,
    current_frame: np.ndarray,
    goal_image_bytes: bytes,
    task_text: str = "What action should the robot take to complete the task?",
) -> List[List[float]]:
    """Send current frame + goal image to the VLA server.

    Args:
        server_url: Base URL of the VLA server (e.g. "http://localhost:7891").
        current_frame: BGR camera frame (numpy array).
        goal_image_bytes: Pre-encoded PNG bytes of the goal image.
        task_text: Text prompt (not really used for goal-conditioned models).

    Returns:
        List of chunk_size action vectors, each of length action_dim (14).
    """
    current_png = encode_frame_as_png(current_frame)
    files = [
        ("image", ("current.png", io.BytesIO(current_png), "image/png")),
        ("image", ("goal.png", io.BytesIO(goal_image_bytes), "image/png")),
    ]
    data = {"text": task_text}
    resp = requests.post(f"{server_url}/process_frame", files=files, data=data)
    resp.raise_for_status()
    actions = resp.json()["response"]
    return actions


def apply_delta_action(
    current_qpos: np.ndarray,
    delta_action: List[float],
) -> np.ndarray:
    """Convert a 7D delta action to absolute joint positions.

    Args:
        current_qpos: Current [j1..j6, gripper] (7D, absolute).
        delta_action: VLA output [dj1..dj6, grip_abs] (7D).

    Returns:
        New absolute joint positions (7D).
    """
    new_pos = np.copy(current_qpos)
    # Joints 0-5: apply delta
    new_pos[:6] += np.array(delta_action[:6])
    # Wrap rotation joints (3, 4, 5) to [-pi, pi]
    for i in range(3, 6):
        if new_pos[i] > math.pi:
            new_pos[i] -= 2 * math.pi
        elif new_pos[i] < -math.pi:
            new_pos[i] += 2 * math.pi
    # Gripper (dim 6): use absolute value from VLA
    new_pos[6] = delta_action[6]
    return new_pos


# ── Visualisation helper ──────────────────────────────────────────────────

def tile_frames(
    frames: List[Optional[np.ndarray]],
    labels: List[str],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Tile camera frames vertically into a single canvas."""
    n = len(frames)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    cell_w = target_w
    cell_h = target_h // n
    canvas = np.zeros((n * cell_h, cell_w, 3), dtype=np.uint8)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        y0 = i * cell_h
        cell = cv2.resize(frame, (cell_w, cell_h)) if frame is not None else \
               np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        canvas[y0:y0 + cell_h] = cell
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for VLA deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy VLA (Dexbotic OFT) policy on YAM bimanual arms."
    )
    parser.add_argument(
        "--goal_image", required=True,
        help="Path to the goal image (what the end state should look like).",
    )
    parser.add_argument(
        "--server", type=str, default="http://localhost:7891",
        help="VLA inference server URL (default: http://localhost:7891).",
    )
    parser.add_argument(
        "--task", type=str,
        default="What action should the robot take to complete the task?",
        help="Task description text prompt (optional for goal-conditioned models).",
    )
    parser.add_argument(
        "--hz", type=float, default=10.0,
        help="Control frequency in Hz (default: 10).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show live camera feeds while running.",
    )
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=None,
        help="Camera indices. Auto-detects if not specified.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: connect cameras + arms, run VLA control loop."""
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
    args = parse_args()
    override_log_level(level=logging.INFO)

    # ── Validate goal image ──────────────────────────────────────────────
    goal_path = Path(args.goal_image)
    if not goal_path.exists():
        raise FileNotFoundError(f"Goal image not found: {goal_path}")
    goal_image_bytes = goal_path.read_bytes()
    goal_bgr = cv2.imread(str(goal_path))
    logging.info(f"Goal image: {goal_path} ({goal_bgr.shape[1]}×{goal_bgr.shape[0]})")

    # ── Test VLA server connection ───────────────────────────────────────
    logging.info(f"Testing VLA server at {args.server}...")
    try:
        requests.get(args.server, timeout=3)
    except requests.ConnectionError:
        logging.error(
            f"Cannot reach VLA server at {args.server}.\n"
            f"Start it first:\n"
            f"  cd food_manipulation\n"
            f"  CUDA_VISIBLE_DEVICES=0 python playground/custom/blockincup_oft.py --task inference"
        )
        return
    logging.info("VLA server reachable")

    # ── Cameras ──────────────────────────────────────────────────────────
    if args.cameras is not None:
        devices = [f"/dev/video{i}" for i in args.cameras]
        logging.info(f"Opening cameras: {devices}")
        cameras: List[CameraReader] = []
        for dev in devices:
            r = CameraReader(dev)
            r.start()
            cameras.append(r)
            logging.info(f"  {dev}: {r.width}×{r.height}")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if all(c.get_frame() is not None for c in cameras):
                break
            time.sleep(0.05)
    else:
        logging.info("Auto-detecting cameras...")
        cameras = detect_cameras()
    logging.info(f"{len(cameras)} camera(s) active")
    if not cameras:
        logging.error("No cameras found — cannot run VLA without camera input")
        return

    # ── Resolve arms ─────────────────────────────────────────────────────
    logging.info("Resolving arm CAN interfaces...")
    leaders_info = resolve_arms("leader_arms")
    followers_info = resolve_arms("follower_arms")
    ensure_can_up({**leaders_info, **followers_info})

    follower_pairs = [("Fleft", PORT_LEFT), ("Fright", PORT_RIGHT)]
    leader_keys = ["Lleft", "Lright"]

    # ── Follower servers ─────────────────────────────────────────────────
    clients: Dict[str, ClientRobot] = {}
    for f_key, port in follower_pairs:
        start_follower_server(followers_info[f_key], port, label=f_key)
    time.sleep(1.0)
    for f_key, port in follower_pairs:
        clients[f_key] = ClientRobot(port)

    # ── Leader arms (for e-stop buttons only) ────────────────────────────
    leader_robots: List[YAMLeaderRobot] = []
    for l_key in leader_keys:
        l_info = leaders_info[l_key]
        gripper_type = GripperType.from_string_name(l_info.gripper_type)
        robot = get_yam_robot(
            channel=l_info.channel,
            gripper_type=gripper_type,
            zero_gravity_mode=True,
            gravity_comp_factor=0.0,
        )
        _all_robots.append(robot)
        leader_robots.append(YAMLeaderRobot(robot))
    logging.info("Leader arms connected for e-stop buttons")

    # ── Shutdown + run state events ──────────────────────────────────────
    stop_event = threading.Event()
    running_event = threading.Event()
    _force_count = 0

    def _shutdown(sig, frame):
        nonlocal _force_count
        _force_count += 1
        if _force_count >= 2:
            logging.warning("Force exit")
            os._exit(1)
        logging.info("Shutting down (Ctrl-C again to force)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    btn_thread = threading.Thread(
        target=button_monitor,
        args=(leader_robots, stop_event, running_event),
        name="button_monitor", daemon=True,
    )
    btn_thread.start()

    # ── Control loop ─────────────────────────────────────────────────────
    period = 1.0 / args.hz
    step_count = 0
    action_queue: deque[List[float]] = deque()

    logging.info(
        f"\nReady — VLA server: {args.server}\n"
        f"  Goal: {args.goal_image}\n"
        f"  Cameras: {len(cameras)}, Hz: {args.hz}\n"
        f"  Press top button  → start / e-stop toggle\n"
        f"  Ctrl-C            → stop and exit\n"
        f"  Waiting for top button press to start...\n"
    )

    win = None
    cam_labels = []
    if args.visualize and cameras:
        cam_labels = [f"cam {c.index}" for c in cameras]
        win = "deploy_vla.py — Q/ESC to quit"
        with _quiet_stderr():
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while not stop_event.is_set():
            t_start = time.monotonic()

            if not running_event.is_set():
                action_queue.clear()
                if win is not None:
                    frames = [c.get_frame() for c in cameras]
                    rect = cv2.getWindowImageRect(win)
                    disp_w, disp_h = max(rect[2], 640), max(rect[3], 480)
                    canvas = tile_frames(frames, cam_labels, disp_w, disp_h)
                    label = "PAUSED — press top button to start" if step_count == 0 \
                            else "E-STOP — press top button to resume"
                    cv2.putText(canvas, label, (20, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        stop_event.set()
                else:
                    time.sleep(0.05)
                continue

            # 1. If action queue is empty, query the VLA server
            if len(action_queue) == 0:
                current_frame = cameras[0].get_frame()
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                try:
                    t_query = time.monotonic()
                    actions = query_vla_server(
                        args.server, current_frame, goal_image_bytes, args.task
                    )
                    logging.info(
                        f"VLA query: {len(actions)} actions in "
                        f"{time.monotonic() - t_query:.3f}s"
                    )
                    action_queue.extend(actions)
                except Exception as e:
                    logging.error(f"VLA server error: {e}")
                    time.sleep(0.5)
                    continue

            # 2. Pop one action from the chunk
            vla_action = action_queue.popleft()

            # 3. Read current follower states and apply delta actions
            if running_event.is_set():
                left_qpos = clients["Fleft"].get_joint_pos().astype(np.float32)
                right_qpos = clients["Fright"].get_joint_pos().astype(np.float32)

                # VLA dims 0-6: left arm, dims 7-13: right arm
                left_cmd = apply_delta_action(left_qpos, vla_action[:7])
                right_cmd = apply_delta_action(right_qpos, vla_action[7:14])

                clients["Fleft"].command_joint_pos(left_cmd)
                clients["Fright"].command_joint_pos(right_cmd)

            step_count += 1
            if step_count % 50 == 0:
                logging.info(
                    f"Step {step_count}, queue: {len(action_queue)} remaining"
                )

            # 4. Visualisation
            if win is not None:
                frames = [c.get_frame() for c in cameras]
                rect = cv2.getWindowImageRect(win)
                disp_w, disp_h = max(rect[2], 640), max(rect[3], 480)
                canvas = tile_frames(frames, cam_labels, disp_w, disp_h)
                label = f"VLA RUNNING  step {step_count}  queue {len(action_queue)}"
                cv2.putText(canvas, label, (disp_w - 500, disp_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
                cv2.imshow(win, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    stop_event.set()

            elapsed = time.monotonic() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logging.error(f"Error in control loop: {e}", exc_info=True)
    finally:
        logging.info(f"Stopping after {step_count} steps")
        if win is not None:
            cv2.destroyAllWindows()
        for cam in cameras:
            try:
                cam.stop()
            except Exception:
                pass
        for robot in _all_robots:
            try:
                robot.close()
            except Exception as e:
                logging.warning(f"Error closing robot: {e}")
        logging.info("VLA deploy stopped.")


if __name__ == "__main__":
    main()
