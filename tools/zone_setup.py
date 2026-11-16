"""Set up zones for our fixed camera.

Shows one camera image (or a frame from a video / an image file) with the
zones from config.yaml drawn on top. Click to print pixel coordinates, then
copy them into the ``zones`` section of config.yaml.

    python tools/zone_setup.py                 # webcam 0
    python tools/zone_setup.py --source clip.mp4
    python tools/zone_setup.py --save zones.png   # just save the picture

Keys: c = clear clicked points, s = save picture, q / Esc = quit.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from lifeaware.camera import FrameSource  # noqa: E402
from lifeaware.config import load_config  # noqa: E402
from lifeaware.zones import ZoneMap  # noqa: E402


def grab_frame(source):
    """One frame from a webcam index, a video file or an image file."""
    if isinstance(source, str) and source.lower().endswith((".png", ".jpg", ".jpeg")):
        return cv2.imread(source)
    with FrameSource(source, max_fps=None) as cam:
        for _ in range(10):  # webcams often need a few frames to adjust exposure
            frame, _ = cam.read()
        return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="0", help="webcam index, video or image path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--save", help="save the picture to this file and exit")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    frame = grab_frame(source)
    if frame is None:
        sys.exit(f"Could not read a frame from {args.source!r}.")
    zones = ZoneMap(load_config(args.config).get("zones"))
    picture = zones.draw(frame)
    print(f"Image size: {frame.shape[1]} x {frame.shape[0]} px")

    if args.save:
        cv2.imwrite(args.save, picture)
        print(f"Saved {args.save}")
        return

    clicked: list[list[int]] = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])
            print(f"[{x}, {y}]   polygon so far: {clicked}   zone here: {zones.zone_of(x, y)}")

    cv2.namedWindow("zones")
    cv2.setMouseCallback("zones", on_click)
    while True:
        view = picture.copy()
        for x, y in clicked:
            cv2.circle(view, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("zones", view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            clicked.clear()
        if key == ord("s"):
            cv2.imwrite("zones.png", view)
            print("Saved zones.png")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
