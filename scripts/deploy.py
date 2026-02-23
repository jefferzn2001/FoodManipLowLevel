#!/usr/bin/env python3
"""
Deploy a trained policy (ACT / Diffusion) on YAM bimanual arms.

Reads camera images + follower joint states, runs the policy, and sends
predicted actions to the follower arms. Leader arm top buttons act as
e-stop: press either one to freeze the robot immediately.

Usage:
    # From local checkpoint
    python scripts/deploy.py --policy outputs/act_blockincup/checkpoints/last/pretrained_model

    # From HuggingFace Hub
    python scripts/deploy.py --policy Jefferzn/act_blockincup

    # With live camera view
    python scripts/deploy.py --policy Jefferzn/act_blockincup --visualize
"""

import argparse
import contextlib
import glob
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as transforms_v2

# Create Qt font directory to suppress QFontDatabase warnings
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

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

PORT_LEFT = 11333
PORT_RIGHT = 11334
NUM_ARM_JOINTS = 6
NUM_JOINTS = 7  # 6 arm + 1 gripper

_all_robots: List[MotorChainRobot] = []


# ── Camera reader ─────────────────────────────────────────────────────────

class CameraReader:
    """Continuously reads frames from a camera in a background thread."""

    def __init__(self, device: str, width: int = 640, height: int = 360) -> None:
        self.device = device
        self.index = int(device.replace("/dev/video", ""))
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def detect_cameras() -> List[CameraReader]:
    """Auto-detect cameras using stable USB path symlinks.

    Uses /dev/v4l/by-path/ so that camera order is determined by
    physical USB port, not by boot/plug order. Falls back to
    /dev/videoN if by-path is not available.
    """
    by_path = sorted(glob.glob("/dev/v4l/by-path/*-video-index0"))
    if by_path:
        # Resolve symlinks to /dev/videoN, keep sorted by USB path
        paths = [os.path.realpath(p) for p in by_path]
        logging.info(f"Using stable USB paths: {dict(zip(by_path, paths))}")
    else:
        paths = sorted(
            (p for p in glob.glob("/dev/video*") if p[len("/dev/video"):].isdigit()),
            key=lambda p: int(p[len("/dev/video"):]),
        )

    readers = []
    with _quiet_stderr():
        for path in paths:
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if ret and w > 0 and h > 0:
                    reader = CameraReader(path)
                    reader.start()
                    readers.append(reader)
                    logging.info(f"Camera {path}: {reader.width}x{reader.height}")
            else:
                cap.release()
    if not readers:
        logging.warning("No cameras detected")
    return readers


# ── Robot helpers ─────────────────────────────────────────────────────────

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
    """Leader arm wrapper — only used for e-stop button reading."""

    def __init__(self, robot: MotorChainRobot) -> None:
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_buttons(self) -> list:
        """Read teaching handle button states. Returns [top_button, bottom_button]."""
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        return encoder_obs[0].io_inputs


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


# ── Button monitor thread ─────────────────────────────────────────────────

def button_monitor(
    leaders: List[YAMLeaderRobot],
    stop_event: threading.Event,
    running_event: threading.Event,
) -> None:
    """Poll leader arm buttons. Top button toggles between running and paused.

    Starts in paused state. First press = start running. Second press = e-stop (pause).
    Third press = resume. And so on.
    """
    while not stop_event.is_set():
        for leader in leaders:
            try:
                buttons = leader.get_buttons()
                if buttons[0] > 0.5:
                    if not running_event.is_set():
                        running_event.set()
                        logging.info("START — policy running")
                    else:
                        running_event.clear()
                        logging.warning("E-STOP — robot paused (press top button to resume)")
                    # Wait for button release
                    while not stop_event.is_set():
                        time.sleep(0.03)
                        buttons = leader.get_buttons()
                        if buttons[0] < 0.5:
                            break
            except Exception:
                pass
        time.sleep(0.02)


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


# ── Auto-detect training resize ──────────────────────────────────────────

def detect_training_resize(policy_path: str) -> Optional[Tuple[int, int]]:
    """Read train_config.json to find if a resize was used during training."""
    # Try local path first
    candidates = [
        Path(policy_path) / "train_config.json",
        Path(policy_path).parent / "train_config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                resize = cfg.get("dataset", {}).get("image_transforms", {}).get("resize")
                if resize and len(resize) == 2:
                    return tuple(resize)
            except (json.JSONDecodeError, KeyError):
                pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy trained policy on YAM arms.")
    parser.add_argument("--policy", required=True,
                        help="Path to pretrained model dir or HuggingFace repo id.")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Control frequency in Hz (default: 10, should match training).")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live camera feeds while running.")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Camera indices. Auto-detects if not specified.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (default: cuda).")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

    args = parse_args()
    override_log_level(level=logging.INFO)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logging.warning("CUDA not available, falling back to CPU")

    # ── Load policy ──────────────────────────────────────────────────────
    logging.info(f"Loading policy from {args.policy}...")
    policy_cfg = json.loads((Path(args.policy) / "config.json").read_text())
    policy_type = policy_cfg.get("type", "act")
    if policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(args.policy)
    else:
        policy = ACTPolicy.from_pretrained(args.policy)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    logging.info(f"Policy loaded: {policy.config.type} on {device}")
    logging.info(f"  Input:  {list(policy.config.input_features.keys())}")
    logging.info(f"  Output: {list(policy.config.output_features.keys())}")

    # Build pre/post processors (handles normalization/unnormalization)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
    )

    # ── Auto-detect resize from training config ──────────────────────────
    training_resize = detect_training_resize(args.policy)
    resize_transform = None
    if training_resize:
        resize_transform = transforms_v2.Resize(list(training_resize), antialias=True)
        logging.info(f"Auto-detected training resize: {training_resize[0]}x{training_resize[1]}")
    else:
        logging.info("No resize detected in training config — using camera native resolution")

    # ── Cameras ──────────────────────────────────────────────────────────
    if args.cameras:
        devices = [f"/dev/video{i}" for i in args.cameras]
        logging.info(f"Opening cameras: {devices}")
        cameras = []
        for dev in devices:
            r = CameraReader(dev)
            r.start()
            cameras.append(r)
            logging.info(f"  {dev}: {r.width}x{r.height}")
    else:
        logging.info("Auto-detecting cameras...")
        cameras = detect_cameras()
    time.sleep(0.3)
    logging.info(f"{len(cameras)} camera(s) active")

    # ── Resolve arms ─────────────────────────────────────────────────────
    logging.info("Resolving arm CAN interfaces...")
    leaders_info = resolve_arms("leader_arms")
    followers_info = resolve_arms("follower_arms")
    ensure_can_up({**leaders_info, **followers_info})

    # Always use both arms — bimanual policy controls both
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
    logging.info(f"Leader arms connected for e-stop buttons")

    # ── Shutdown + run state events ──────────────────────────────────────
    stop_event = threading.Event()
    running_event = threading.Event()  # clear = paused, set = running
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

    # Start button monitor thread
    btn_thread = threading.Thread(
        target=button_monitor,
        args=(leader_robots, stop_event, running_event),
        name="button_monitor", daemon=True,
    )
    btn_thread.start()

    # ── Control loop ─────────────────────────────────────────────────────
    period = 1.0 / args.hz
    step_count = 0

    logging.info(
        f"\nReady — {len(follower_pairs)} follower(s), {len(cameras)} camera(s) @ {args.hz} Hz\n"
        f"  Press top button  → start policy / e-stop toggle\n"
        f"  Ctrl-C            → stop and exit\n"
        f"  Waiting for top button press to start...\n"
    )

    # Set up visualisation window if requested
    win = None
    cam_labels = []
    if args.visualize and cameras:
        cam_labels = [f"cam {c.index}" for c in cameras]
        win = "deploy.py — Q/ESC to quit"
        with _quiet_stderr():
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while not stop_event.is_set():
            t_start = time.monotonic()

            # Wait for button press to start / e-stop pause
            if not running_event.is_set():
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

            # 1. Read follower state (observation)
            state_parts = []
            for f_key, _ in follower_pairs:
                qpos = clients[f_key].get_joint_pos()
                state_parts.append(qpos.astype(np.float32))
            state = np.concatenate(state_parts)

            # 2. Read camera images
            observation: dict = {
                "observation.state": torch.from_numpy(state),
            }
            for i, cam in enumerate(cameras):
                bgr = cam.get_frame()
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    # Convert to (C, H, W) float tensor in [0, 1]
                    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    if resize_transform is not None:
                        img_tensor = resize_transform(img_tensor)
                    observation[f"observation.images.cam_{i}"] = img_tensor

            # 3. Run policy
            processed_obs = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(processed_obs)
            action = postprocessor(action)
            action_np = action.squeeze(0).cpu().numpy()

            # 4. Send actions to followers (skip if e-stopped during inference)
            if running_event.is_set():
                offset = 0
                for f_key, _ in follower_pairs:
                    clients[f_key].command_joint_pos(action_np[offset:offset + NUM_JOINTS])
                    offset += NUM_JOINTS

            step_count += 1
            if step_count % 100 == 0:
                logging.info(f"Step {step_count}")

            # 5. Visualisation
            if win is not None:
                frames = [c.get_frame() for c in cameras]
                rect = cv2.getWindowImageRect(win)
                disp_w, disp_h = max(rect[2], 640), max(rect[3], 480)
                canvas = tile_frames(frames, cam_labels, disp_w, disp_h)
                label = f"RUNNING  step {step_count}"
                cv2.putText(canvas, label, (disp_w - 350, disp_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
                cv2.imshow(win, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    stop_event.set()

            # Rate limiting
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
        for robot in _all_robots:
            try:
                robot.close()
            except Exception as e:
                logging.warning(f"Error closing robot: {e}")
        logging.info("Deploy stopped.")


if __name__ == "__main__":
    main()
