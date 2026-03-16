#!/usr/bin/env python3
"""
Test VLA inference without any robot hardware.

Captures live camera frames, sends them with a goal image to the VLA
server, and logs/saves the predicted actions. Optionally shows the
camera feed with action overlay.

Prerequisites:
    1. VLA server running (Terminal 1):
       conda activate dexbotic
       cd food_manipulation
       CUDA_VISIBLE_DEVICES=0 python playground/custom/blockincup_oft.py --task inference

    2. Run this script (Terminal 2, i2rt venv):
       python scripts/test_vla.py --goal_image goal.png
       python scripts/test_vla.py --goal_image goal.png --num_queries 5
       python scripts/test_vla.py --goal_image goal.png --save_dir vla_test_output
       python scripts/test_vla.py --goal_image goal.png --visualize

Output:
    Prints each action chunk (16×14D) and optionally saves them as JSON.
"""

import argparse
import contextlib
import glob
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

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
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def detect_camera() -> Optional[CameraReader]:
    """Find the first working camera."""
    by_path_all = sorted(glob.glob("/dev/v4l/by-path/*-video-index*"))
    if by_path_all:
        ports: dict[str, list[str]] = {}
        for p in by_path_all:
            port_key = p.rsplit("-video-index", 1)[0]
            ports.setdefault(port_key, []).append(p)
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
                            return reader
                        time.sleep(0.05)
                    reader.stop()
                except Exception:
                    pass
        return None

    for path in sorted(glob.glob("/dev/video*")):
        if not path[len("/dev/video"):].isdigit():
            continue
        try:
            reader = CameraReader(path)
            reader.start()
            time.sleep(0.5)
            if reader.get_frame() is not None:
                return reader
            reader.stop()
        except Exception:
            pass
    return None


# ── VLA query ─────────────────────────────────────────────────────────────

def query_vla(
    server_url: str,
    current_frame: np.ndarray,
    goal_image_bytes: bytes,
    task_text: str,
) -> dict:
    """Send current frame + goal image to the VLA server.

    Returns:
        dict with 'actions' (list of 16×14D), 'latency_s', 'frame_shape'.
    """
    _, buf = cv2.imencode(".png", current_frame)
    current_png = buf.tobytes()

    files = [
        ("image", ("current.png", io.BytesIO(current_png), "image/png")),
        ("image", ("goal.png", io.BytesIO(goal_image_bytes), "image/png")),
    ]
    data = {"text": task_text}

    t0 = time.monotonic()
    resp = requests.post(f"{server_url}/process_frame", files=files, data=data)
    resp.raise_for_status()
    latency = time.monotonic() - t0

    actions = resp.json()["response"]
    return {
        "actions": actions,
        "latency_s": latency,
        "frame_shape": list(current_frame.shape),
    }


def print_action_summary(query_idx: int, result: dict) -> None:
    """Print a human-readable summary of one VLA query result."""
    actions = result["actions"]
    latency = result["latency_s"]
    n_steps = len(actions)
    dim = len(actions[0]) if actions else 0

    print(f"\n{'='*70}")
    print(f"Query {query_idx + 1}: {n_steps} steps × {dim}D, latency={latency:.3f}s")
    print(f"{'='*70}")

    # Per-dimension stats across all timesteps
    arr = np.array(actions)
    print(f"\n  {'Dim':>4}  {'Min':>10}  {'Max':>10}  {'Mean':>10}  {'Description'}")
    print(f"  {'----':>4}  {'---':>10}  {'---':>10}  {'----':>10}  {'-----------'}")
    labels = [
        "L j0 delta", "L j1 delta", "L j2 delta",
        "L j3 delta", "L j4 delta", "L j5 delta", "L grip abs",
        "R j0 delta", "R j1 delta", "R j2 delta",
        "R j3 delta", "R j4 delta", "R j5 delta", "R grip abs",
    ]
    for d in range(dim):
        col = arr[:, d]
        label = labels[d] if d < len(labels) else ""
        print(f"  {d:>4}  {col.min():>10.6f}  {col.max():>10.6f}  {col.mean():>10.6f}  {label}")

    # Total accumulated delta for joints
    total_delta = arr.sum(axis=0)
    print(f"\n  Accumulated delta (sum of {n_steps} steps):")
    print(f"    Left joints:  [{', '.join(f'{v:.4f}' for v in total_delta[:6])}] rad")
    print(f"    Left gripper: {total_delta[6]:.4f} (absolute, last={arr[-1, 6]:.4f})")
    print(f"    Right joints: [{', '.join(f'{v:.4f}' for v in total_delta[7:13])}] rad")
    print(f"    Right gripper:{total_delta[13]:.4f} (absolute, last={arr[-1, 13]:.4f})")


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Test VLA inference without robot hardware."
    )
    parser.add_argument(
        "--goal_image", required=True,
        help="Path to the goal image.",
    )
    parser.add_argument(
        "--server", type=str, default="http://localhost:7891",
        help="VLA server URL (default: http://localhost:7891).",
    )
    parser.add_argument(
        "--num_queries", type=int, default=2,
        help="Number of VLA queries to run (default: 2).",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Seconds between queries (default: 1.0).",
    )
    parser.add_argument(
        "--task", type=str,
        default="What action should the robot take to complete the task?",
        help="Task text prompt.",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera /dev/videoN index. Auto-detects if not specified.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save results (actions JSON + captured frames).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show camera feed with action info overlay.",
    )
    return parser.parse_args()


def main() -> None:
    """Run VLA test queries and display results."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Validate goal image
    goal_path = Path(args.goal_image)
    if not goal_path.exists():
        raise FileNotFoundError(f"Goal image not found: {goal_path}")
    goal_image_bytes = goal_path.read_bytes()
    goal_bgr = cv2.imread(str(goal_path))
    print(f"Goal image: {goal_path} ({goal_bgr.shape[1]}×{goal_bgr.shape[0]})")

    # Test server
    print(f"Testing VLA server at {args.server}...")
    try:
        requests.get(args.server, timeout=3)
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach VLA server at {args.server}")
        print(f"Start it first:")
        print(f"  conda activate dexbotic")
        print(f"  cd food_manipulation")
        print(f"  CUDA_VISIBLE_DEVICES=0 python playground/custom/blockincup_oft.py --task inference")
        return
    print("VLA server reachable")

    # Open camera
    if args.camera is not None:
        dev = f"/dev/video{args.camera}"
        print(f"Opening camera: {dev}")
        camera = CameraReader(dev)
        camera.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if camera.get_frame() is not None:
                break
            time.sleep(0.05)
    else:
        print("Auto-detecting camera...")
        camera = detect_camera()

    if camera is None or camera.get_frame() is None:
        print("ERROR: No camera found")
        return
    print(f"Camera ready: {camera.device} ({camera.width}×{camera.height})")

    # Prepare save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "goal.png"), goal_bgr)
        print(f"Saving results to: {save_dir}")

    # Visualize window
    win = None
    if args.visualize:
        win = "test_vla.py — Q/ESC to quit"
        with _quiet_stderr():
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 800, 600)

    # Run queries
    all_results = []
    print(f"\nRunning {args.num_queries} VLA queries (interval={args.interval}s)...")
    print(f"Task: {args.task}")

    try:
        for i in range(args.num_queries):
            frame = camera.get_frame()
            if frame is None:
                print(f"Query {i+1}: no frame, skipping")
                continue

            result = query_vla(args.server, frame, goal_image_bytes, args.task)
            result["query_index"] = i
            all_results.append(result)

            print_action_summary(i, result)

            # Save frame and actions
            if args.save_dir:
                cv2.imwrite(str(save_dir / f"frame_{i:03d}.png"), frame)
                with open(save_dir / f"actions_{i:03d}.json", "w") as f:
                    json.dump(result, f, indent=2)

            # Show visualization
            if win is not None:
                display = frame.copy()
                actions = np.array(result["actions"])
                mean_delta = np.abs(actions[:, :6]).mean()
                info_lines = [
                    f"Query {i+1}/{args.num_queries}",
                    f"Latency: {result['latency_s']:.3f}s",
                    f"Mean |delta|: {mean_delta:.6f} rad",
                    f"L grip: {actions[-1, 6]:.3f}  R grip: {actions[-1, 13]:.3f}",
                ]
                for j, line in enumerate(info_lines):
                    cv2.putText(display, line, (10, 30 + j * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Show goal image in corner
                gh, gw = goal_bgr.shape[:2]
                scale = 150 / max(gh, gw)
                goal_small = cv2.resize(goal_bgr, (int(gw * scale), int(gh * scale)))
                gsh, gsw = goal_small.shape[:2]
                display[10:10 + gsh, display.shape[1] - gsw - 10:display.shape[1] - 10] = goal_small
                cv2.putText(display, "GOAL", (display.shape[1] - gsw - 5, 10 + gsh + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(win, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break

            if i < args.num_queries - 1:
                time.sleep(args.interval)

    finally:
        camera.stop()
        if win is not None:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Save combined results
    if args.save_dir:
        with open(save_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {save_dir}/")

    print(f"\nDone — {len(all_results)} queries completed.")


if __name__ == "__main__":
    main()
