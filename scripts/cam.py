#!/usr/bin/env python3
"""
Multi-camera fullscreen viewer.

Auto-detects all connected USB cameras and tiles them in a single
fullscreen OpenCV window. Useful for checking framing before recording.

Usage:
    python scripts/cam.py
    python scripts/cam.py --cameras 0 2    # specific /dev/videoN indices only
"""

import argparse
import contextlib
import glob
import math
import os
import sys
import threading
from typing import List, Optional

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np

# Create Qt font directory to suppress QFontDatabase warnings
os.makedirs(os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts"), exist_ok=True)


@contextlib.contextmanager
def _quiet_stderr():
    """Suppress C-level stderr (Qt/OpenCV noise that bypasses Python logging)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class CameraReader:
    """Continuously reads frames from a camera in a background thread."""

    def __init__(self, device: str, width: int = 1280, height: int = 720) -> None:
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


def detect_cameras() -> List[str]:
    """Return camera paths, using stable USB path symlinks when available."""
    by_path = sorted(glob.glob("/dev/v4l/by-path/*-video-index0"))
    if by_path:
        candidates = [os.path.realpath(p) for p in by_path]
        print(f"Using stable USB paths:")
        for sym, dev in zip(by_path, candidates):
            print(f"  {sym} -> {dev}")
    else:
        candidates = sorted(
            (p for p in glob.glob("/dev/video*") if p[len("/dev/video"):].isdigit()),
            key=lambda p: int(p[len("/dev/video"):]),
        )

    found = []
    with _quiet_stderr():
        for path in candidates:
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if ret and w > 0 and h > 0:
                    found.append(path)
            cap.release()
    return found


def tile_frames(
    frames: List[Optional[np.ndarray]],
    labels: List[str],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Tile camera frames into a single image that fills target_w × target_h."""
    n = len(frames)
    ncols = 1
    nrows = n
    cell_w = target_w // ncols
    cell_h = target_h // nrows

    canvas = np.zeros((nrows * cell_h, ncols * cell_w, 3), dtype=np.uint8)

    for i, (frame, label) in enumerate(zip(frames, labels)):
        row = i // ncols
        col = i % ncols
        y0, x0 = row * cell_h, col * cell_w

        if frame is not None:
            cell = cv2.resize(frame, (cell_w, cell_h))
        else:
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(cell, "No signal", (cell_w // 4, cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

        # Camera label in top-left corner
        cv2.putText(cell, label, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fullscreen multi-camera viewer."
    )
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=None,
        help="Camera /dev/videoN indices to show. Auto-detects all if not specified.",
    )
    return parser.parse_args()


def main() -> None:
    # Point Qt to system fonts to suppress QFontDatabase warnings
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")
    args = parse_args()

    if args.cameras is not None:
        devices = [f"/dev/video{i}" for i in args.cameras]
    else:
        print("Scanning for cameras...")
        devices = detect_cameras()

    if not devices:
        sys.exit("No cameras found.")

    print(f"Opening cameras: {[r.replace('/dev/video', '') for r in devices]}")
    readers = []
    for dev in devices:
        r = CameraReader(dev)
        r.start()
        readers.append(r)
        print(f"  {dev}: {r.width}×{r.height}")

    labels = [f"cam_{i}  ({r.device}  {r.width}x{r.height})" for i, r in enumerate(readers)]

    win = "Cameras — press Q or ESC to quit"
    with _quiet_stderr():
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow(win, dummy)
        cv2.waitKey(1)

    try:
        while True:
            frames = [r.get_frame() for r in readers]

            # Get current window size for tiling
            rect = cv2.getWindowImageRect(win)
            disp_w = max(rect[2], 640)
            disp_h = max(rect[3], 480)

            canvas = tile_frames(frames, labels, disp_w, disp_h)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                break
    finally:
        cv2.destroyAllWindows()
        for r in readers:
            r.stop()


if __name__ == "__main__":
    main()
