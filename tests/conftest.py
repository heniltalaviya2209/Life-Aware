"""Shared pytest helpers (fake data only: no camera, mic or model download)."""

import cv2
import numpy as np


def write_test_video(path, seconds=3, fps=30, size=(64, 48)):
    """Write a small MJPG .avi whose frames get brighter over time."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "OpenCV could not create the test video"
    for i in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
