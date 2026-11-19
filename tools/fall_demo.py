"""Run camera -> pose -> fall check (UC3) end to end, with answers from the keyboard.

    python tools/fall_demo.py --show                       # webcam, until you press q
    python tools/fall_demo.py --source clip.mp4 --show     # a recorded clip
    python tools/fall_demo.py --show --still 4 --timeout 8 # shorter timers for a quick try

The window shows skeleton, zones and the current UC3 state. Keys:
    y = "yes, I'm okay"     n = "no"     h = "help"     u = reply not understood
    m = microphone lost/ok (toggle)      a = acknowledge (helper)     q = quit

Keys stand in for the speech recogniser until the audio part is connected,
so results from this tool are "simulated answers" and must be labelled so.
Every state change is printed and, with --log, appended to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifeaware.camera import FrameSource  # noqa: E402
from lifeaware.config import load_config  # noqa: E402
from lifeaware.pose import PoseDetector, draw_poses  # noqa: E402
from lifeaware.schema import Observation  # noqa: E402
from lifeaware.usecases.fall_check import FallCheck  # noqa: E402
from lifeaware.zones import ZoneMap  # noqa: E402

STATE_COLOURS = {  # BGR
    "normal": (0, 170, 0), "possible_event": (0, 200, 255), "checking": (0, 140, 255),
    "alarm": (0, 0, 255), "solved": (200, 150, 0), "unknown": (150, 150, 150),
}
KEY_ANSWERS = {"y": "answer_yes", "n": "answer_no", "h": "answer_help", "u": "answer_unknown"}


def print_events(events, wall_start, log_writer=None) -> None:
    for e in events:
        stamp = datetime.fromtimestamp(e.time).strftime("%H:%M:%S")
        answered = f" [answer: {e.answered}]" if e.answered else ""
        print(f"  {stamp} (+{e.time - wall_start:5.1f}s)  {e.old_state} -> {e.new_state.upper()}: "
              f"{e.reason}{answered}")
        if log_writer is not None:
            log_writer.writerow([stamp, e.time, e.use_case, e.old_state, e.new_state, e.reason, e.answered or ""])


def show(frame, zones, observations, check: FallCheck):
    """Draw everything; return the key pressed (or '')."""
    import cv2

    view = draw_poses(zones.draw(frame), observations or [])
    colour = STATE_COLOURS.get(check.state, (255, 255, 255))
    cv2.rectangle(view, (0, 0), (view.shape[1], 34), colour, -1)
    mic = "mic ok" if check.mic_ok else "MIC LOST"
    cv2.putText(view, f"UC3: {check.state.upper()}   ({mic})", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view, "y/n/h/u answer  m mic  a ack  q quit", (10, view.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("LifeAware fall check (simulated answers)", view)
    key = cv2.waitKey(1) & 0xFF
    return chr(key) if key != 255 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None, help="webcam index or video path (default: config)")
    parser.add_argument("--seconds", type=float, default=0, help="stop a webcam run after this time (0 = never)")
    parser.add_argument("--show", action="store_true", help="window with skeleton + keyboard answers")
    parser.add_argument("--still", type=float, default=None, help="override fall.still_s")
    parser.add_argument("--timeout", type=float, default=None, help="override fall.answer_timeout_s")
    parser.add_argument("--log", default=None, help="append events to this CSV file")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.still is not None:
        config["fall"]["still_s"] = args.still
    if args.timeout is not None:
        config["fall"]["answer_timeout_s"] = args.timeout
    source = args.source if args.source is not None else str(config["camera"]["source"])
    source = int(source) if source.isdigit() else source
    zones = ZoneMap(config["zones"])
    detector = PoseDetector.from_config(config, zone_map=zones)
    check = FallCheck.from_config(config, say=lambda text: print(f'  ROBOT SAYS: "{text}"'))
    if not args.show:
        print("Note: without --show there is no way to answer, so a long lie-still ends in an alarm.")

    log_file = open(args.log, "a", newline="", encoding="utf-8") if args.log else None
    log_writer = csv.writer(log_file) if log_file else None

    print("Loading model ...")
    detector.load()
    print(f"Fall-check timers: {config['fall']}")
    wall_start = time.time()
    last_obs = []
    with FrameSource(source, max_fps=config["camera"]["max_fps"],
                     max_failures=config["camera"]["max_failures"], start_time=wall_start) as cam:
        while cam.status == "ok":
            if not cam.is_file and args.seconds and time.time() - wall_start > args.seconds:
                break
            frame, timestamp = cam.read()
            for obs in cam.pop_observations():  # camera_lost
                print_events(check.feed(obs), wall_start, log_writer)
            if frame is None:
                print_events(check.tick(timestamp), wall_start, log_writer)
                continue
            result = detector.detect(frame, timestamp)
            if result is not None:
                last_obs = result
            print_events(check.feed_pose(result, timestamp), wall_start, log_writer)

            if args.show:
                key = show(frame, zones, last_obs, check)
                if key == "q":
                    break
                if key in KEY_ANSWERS:
                    obs = Observation(time=timestamp, source="audio", label=KEY_ANSWERS[key], confidence=1.0)
                    print_events(check.feed(obs), wall_start, log_writer)
                elif key == "m":
                    label = "mic_lost" if check.mic_ok else "mic_ok"
                    check.feed(Observation(time=timestamp, source="audio", label=label, confidence=1.0))
                    print(f"  (microphone {'lost' if label == 'mic_lost' else 'ok'})")
                elif key == "a":
                    print_events(check.acknowledge(timestamp), wall_start, log_writer)
        for obs in cam.pop_observations():
            print_events(check.feed(obs), wall_start, log_writer)
        status = cam.status

    if log_file:
        log_file.close()
    print(f"\nCamera status: {status}.  Final UC3 state: {check.state}.  Events: {len(check.events)}")


if __name__ == "__main__":
    main()
