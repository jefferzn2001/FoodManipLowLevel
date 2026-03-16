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
        # Try MJPG first (less CPU); fall back to driver default (often YUYV) if no frame
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


def _reset_usb_device(dev_path: str) -> None:
    """Reset the USB device backing a /dev/videoN node.

    This clears stuck V4L2 state without needing to physically unplug
    the camera. Requires write access to the USB device (usually root).
    """
    import fcntl
    USBDEVFS_RESET = 0x5514
    try:
        sys_path = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(dev_path)}/device")
        # Walk up to find the USB device node
        while sys_path and not os.path.exists(os.path.join(sys_path, "busnum")):
            sys_path = os.path.dirname(sys_path)
        if not sys_path:
            return
        busnum = open(os.path.join(sys_path, "busnum")).read().strip()
        devnum = open(os.path.join(sys_path, "devnum")).read().strip()
        usb_dev = f"/dev/bus/usb/{int(busnum):03d}/{int(devnum):03d}"
        with open(usb_dev, "w") as f:
            fcntl.ioctl(f, USBDEVFS_RESET, 0)
        import time
        time.sleep(0.5)
    except (OSError, PermissionError, FileNotFoundError, ValueError):
        pass


def detect_cameras(retry_with_reset: bool = True) -> List[CameraReader]:
    """Detect cameras and return started CameraReader instances.

    Each physical USB camera typically creates two /dev/videoN nodes
    (index0 and index1). Only one is the real capture endpoint; the other
    is metadata. We try both and keep whichever actually delivers frames.
    Uses /dev/v4l/by-path/ for stable ordering by physical USB port.

    If no cameras are found on the first pass, attempts a USB reset and
    retries once (avoids needing to physically replug).
    """
    import time

    by_path_all = sorted(glob.glob("/dev/v4l/by-path/*-video-index*"))
    if by_path_all:
        ports: dict[str, list[str]] = {}
        for p in by_path_all:
            port_key = p.rsplit("-video-index", 1)[0]
            ports.setdefault(port_key, []).append(p)
        print(f"Found {len(ports)} USB camera port(s)...")
        # Build ordered candidate list: for each port, index0 first then index1
        candidates = []
        port_of: dict[str, str] = {}
        for port_key in sorted(ports):
            links = sorted(ports[port_key])
            for lnk in links:
                dev = os.path.realpath(lnk)
                candidates.append(dev)
                port_of[dev] = port_key
    else:
        candidates = sorted(
            (p for p in glob.glob("/dev/video*") if p[len("/dev/video"):].isdigit()),
            key=lambda p: int(p[len("/dev/video"):]),
        )
        port_of = {p: p for p in candidates}

    # Open each candidate as a CameraReader, wait for a frame, keep the first
    # working device per physical USB port.
    readers: List[CameraReader] = []
    seen_ports: set[str] = set()
    for dev in candidates:
        port = port_of[dev]
        if port in seen_ports:
            continue
        try:
            reader = CameraReader(dev)
        except Exception:
            print(f"  ✗ {dev} (cannot open)")
            continue
        reader.start()
        # Wait up to 2s for the first frame
        deadline = time.monotonic() + 2.0
        frame = None
        while time.monotonic() < deadline:
            frame = reader.get_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        if frame is not None:
            readers.append(reader)
            seen_ports.add(port)
            print(f"  ✓ {dev} ({reader.width}×{reader.height})")
        else:
            reader.stop()
            print(f"  ✗ {dev} (no frames)")

    # If nothing found, try USB reset and retry once
    if not readers and retry_with_reset and candidates:
        print("\nNo cameras responded. Attempting USB reset...")
        for dev in candidates:
            _reset_usb_device(dev)
        time.sleep(1.0)
        return detect_cameras(retry_with_reset=False)

    return readers


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

    import time

    if args.cameras is not None:
        devices = [f"/dev/video{i}" for i in args.cameras]
        print(f"Opening cameras: {devices}")
        readers = []
        for dev in devices:
            r = CameraReader(dev)
            r.start()
            readers.append(r)
            print(f"  {dev}: {r.width}×{r.height}")
        # Warm-up: wait for first frame from each camera
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if all(r.get_frame() is not None for r in readers):
                break
            time.sleep(0.05)
    else:
        print("Scanning for cameras...")
        readers = detect_cameras()

    if not readers:
        sys.exit("No cameras found.")

    labels = [f"cam_{i}  ({r.device}  {r.width}x{r.height})" for i, r in enumerate(readers)]

    # Ensure cameras are always released, even on Ctrl-C or crash
    import signal

    def _cleanup(signum=None, frame=None):
        for r in readers:
            try:
                r.stop()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if signum is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

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

            rect = cv2.getWindowImageRect(win)
            disp_w = max(rect[2], 640)
            disp_h = max(rect[3], 480)

            canvas = tile_frames(frames, labels, disp_w, disp_h)
            cv2.imshow(win, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
