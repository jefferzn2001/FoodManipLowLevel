#!/usr/bin/env python3
"""
Camera preview and checkerboard calibration image capture tool.

Opens available cameras, lets you preview and select one, then
runs a live view with checkerboard corner detection. Press SPACE
to save frames for calibration, ESC to quit.

Usage:
    python scripts/camera.py
"""

import os
import time

import cv2 as cv
import numpy as np

# ====== YOUR BOARD SETTINGS (from the A4 generator) ======
CHECKERBOARD = (10, 7)   # inner corners: 10 cols × 7 rows
SQUARE_SIZE  = 0.025     # meters per square (25 mm)
# =========================================================

os.makedirs("camcal", exist_ok=True)


def find_cameras(max_test: int = 5) -> list:
    """Find all available cameras by testing indices 0 to max_test.

    Args:
        max_test (int): Maximum camera index to test.

    Returns:
        list: Available camera indices.
    """
    available = []
    for i in range(max_test):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv.CAP_PROP_FPS))
                print(f"Camera {i}: {width}x{height} @ {fps}fps")
                available.append(i)
            cap.release()
    return available


def select_camera(cameras: list) -> int:
    """Auto-select or let user pick from multiple cameras.

    Args:
        cameras (list): List of available camera indices.

    Returns:
        int: Selected camera index.
    """
    if len(cameras) == 1:
        print(f"Using camera {cameras[0]}")
        return cameras[0]

    print(f"\nMultiple cameras found. Trying camera indices in order...")
    cam_index = cameras[0]  # default fallback
    for idx in cameras:
        print(f"Testing camera {idx}...")
        test_cap = cv.VideoCapture(idx)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret:
                cv.imshow(
                    f"Camera {idx} Preview - Press ENTER to use, ESC to try next",
                    frame,
                )
                key = cv.waitKey(0)
                cv.destroyAllWindows()
                test_cap.release()
                if key == 13:  # ENTER
                    cam_index = idx
                    break
            else:
                test_cap.release()
    else:
        print(f"Using default camera {cam_index}")

    return cam_index


def main() -> None:
    """Entry point: find cameras, open preview, capture calibration images."""
    print("Searching for cameras...")
    cameras = find_cameras(max_test=5)

    if not cameras:
        raise RuntimeError("No cameras found! Check connections/permissions.")

    print(f"\nFound {len(cameras)} camera(s): {cameras}")
    cam_index = select_camera(cameras)

    cap = cv.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_index}")

    # Set to 1080p resolution to use full wide field of view
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv.CAP_PROP_FPS))
    print(f"\nUsing camera {cam_index} at {actual_width}x{actual_height} @ {actual_fps}fps")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

        disp = frame.copy()
        msg = "Corners: NO"
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(disp, CHECKERBOARD, corners, ret)
            msg = f"Corners: YES  (press SPACE to save)"

        cv.putText(disp, msg, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv.putText(disp, msg, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv.imshow("Capture checkerboard", disp)

        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32 and ret:  # SPACE
            path = f"camcal/{int(time.time() * 1000)}.png"
            cv.imwrite(path, frame)
            saved += 1
            print(f"Saved: {path}")

    cap.release()
    cv.destroyAllWindows()
    print(f"Saved {saved} images to ./camcal (get 20-30 diverse views).")


if __name__ == "__main__":
    main()

