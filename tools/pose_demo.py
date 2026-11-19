"""Run camera -> pose detector and print the posture of the person, with fps.

    python tools/pose_demo.py                      # webcam, 20 seconds
    python tools/pose_demo.py --source clip.mp4    # a recorded clip (to the end)
    python tools/pose_demo.py --show               # also open a window with skeleton + zones

Each analysed frame prints one line: time, posture, confidence, zone, movement.
At the end it prints how often each posture was seen and the speed:
- frames/s handed out by the camera (capped by camera.max_fps)
- detector speed (model calls per second, only every n-th frame is analysed)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifeaware.camera import FrameSource  # noqa: E402
from lifeaware.config import load_config  # noqa: E402
from lifeaware.pose import PoseDetector, draw_poses  # noqa: E402
from lifeaware.zones import ZoneMap  # noqa: E402


def show(frame, zones, observations) -> bool:
    """Draw zones and skeletons; return False when the user pressed q."""
    import cv2

    view = draw_poses(zones.draw(frame), observations)
    cv2.imshow("LifeAware pose demo (q = quit)", view)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None, help="webcam index or video path (default: config)")
    parser.add_argument("--seconds", type=float, default=20, help="stop a webcam run after this time")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    source = args.source if args.source is not None else str(config["camera"]["source"])
    source = int(source) if source.isdigit() else source
    zones = ZoneMap(config["zones"])
    detector = PoseDetector.from_config(config, zone_map=zones)

    print("Loading model ...")
    detector.load()  # load before timing starts
    frames, model_calls, model_time = 0, 0, 0.0
    seen = Counter()
    wall_start = time.time()
    with FrameSource(source, max_fps=config["camera"]["max_fps"],
                     max_failures=config["camera"]["max_failures"], start_time=wall_start) as cam:
        while cam.status == "ok" and (cam.is_file or time.time() - wall_start < args.seconds):
            frame, timestamp = cam.read()
            if frame is None:
                continue
            frames += 1
            found = detector.detect(frame, timestamp)
            if found is None:
                continue
            model_calls += 1
            model_time += detector.last_inference_s or 0.0
            if not found:
                seen["nobody"] += 1
                print(f"  {timestamp - wall_start:6.1f}s  nobody found")
            for obs in found:
                seen[obs.label] += 1
                movement = obs.data["movement"]
                moving = "-" if movement is None else f"{movement:.2f}/s"
                print(f"  {timestamp - wall_start:6.1f}s  {obs.label:16s} conf={obs.confidence:.2f} "
                      f"zone={obs.zone} movement={moving}")
            if args.show and not show(frame, zones, found):
                break
        for obs in cam.pop_observations():
            print(f"  camera: {obs.label} ({obs.data.get('reason')})")
        status = cam.status

    elapsed = time.time() - wall_start
    print(f"\nCamera status: {status}")
    print(f"Frames processed: {frames} in {elapsed:.1f} s  ->  {frames / max(elapsed, 1e-9):.1f} fps end to end")
    if model_calls:
        avg_ms = 1000 * model_time / model_calls
        print(f"Detector: {model_calls} calls, {avg_ms:.0f} ms each  ->  {1000 / avg_ms:.1f} model calls/s possible")
    print("Postures seen:", dict(seen) or "none")


if __name__ == "__main__":
    main()
