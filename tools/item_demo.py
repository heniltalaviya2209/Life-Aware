"""Run camera -> object detector -> ItemMemory -> UC2 answer, and measure fps.

    python tools/item_demo.py                      # webcam, 20 seconds
    python tools/item_demo.py --source clip.mp4    # a recorded clip (to the end)
    python tools/item_demo.py --show               # also open a window with boxes + zones

At the end it prints the UC2 answers for "phone" and "keys" plus timing:
- frames/s handed out by the camera (capped by camera.max_fps)
- detector speed (model calls per second, only every n-th frame is analysed)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifeaware.camera import FrameSource  # noqa: E402
from lifeaware.config import load_config  # noqa: E402
from lifeaware.memory import ItemMemory  # noqa: E402
from lifeaware.objects import ObjectDetector  # noqa: E402
from lifeaware.usecases.lost_items import handle_query  # noqa: E402
from lifeaware.zones import ZoneMap  # noqa: E402


def show(frame, zones, observations) -> bool:
    """Draw zones and boxes; return False when the user pressed q."""
    import cv2

    view = zones.draw(frame)
    for obs in observations:
        x1, y1, x2, y2 = (int(v) for v in obs.data["bbox"])
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{obs.label} {obs.confidence:.2f} {obs.zone or '?'}"
        cv2.putText(view, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("LifeAware item demo (q = quit)", view)
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
    detector = ObjectDetector.from_config(config, zone_map=zones)
    memory = ItemMemory.from_config(config)

    print("Loading model ...")
    detector._get_model()  # load before timing starts
    frames, model_calls, model_time = 0, 0, 0.0
    wall_start = time.time()
    # start_time=now: clip timestamps become "now + video time", so answers read naturally.
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
            for obs in found:
                memory.update(obs)
                print(f"  seen: {obs.label} conf={obs.confidence:.2f} zone={obs.zone}")
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
    for item in ("phone", "keys"):
        print(f"UC2 '{item}': {handle_query(item, memory).reason}")


if __name__ == "__main__":
    main()
