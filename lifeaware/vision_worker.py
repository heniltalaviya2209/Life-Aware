"""Camera + pose model in their own process, for the dashboard.

Why a separate process: inside Streamlit the pose model ran ~15x slower than in
the terminal (about 1300 ms instead of 90 ms per check), because it shared
Python's global lock (GIL) with the web server. A separate process has its own
GIL, so the page can never slow the model down.

``run_vision`` is the whole loop. It sends plain messages to ``out`` (a queue):

    ("status", "loading model" | "opening camera")
    ("frame", timestamp, jpeg_bytes, pose_result, inference_s)
        pose_result is PoseDetector.detect's value: None = skipped, [] = nobody
    ("tick", timestamp)            no frame this time (keeps timers running)
    ("camera", Observation)        e.g. camera_lost
    ("end", "stopped" | "ended" | "lost")
    ("error", "SomeError: text")

The same function also runs in a thread (tests with a fake model, or if a
process cannot be started). It must stay importable WITHOUT Streamlit, because
Windows starts the new process by importing this module.
"""

from __future__ import annotations

import time


def run_vision(config: dict, source, out, stop) -> None:
    """Camera -> pose -> JPEG, until ``stop`` is set or the camera ends."""
    try:
        import cv2

        from lifeaware.camera import FrameSource
        from lifeaware.pose import PoseDetector, draw_poses
        from lifeaware.zones import ZoneMap

        out.put(("status", "loading model"))
        zones = ZoneMap(config["zones"])
        detector = PoseDetector.from_config(config, zone_map=zones)
        detector.load()

        out.put(("status", "opening camera"))
        camera = config["camera"]
        last_pose = []
        with FrameSource(source, max_fps=camera["max_fps"], max_failures=camera["max_failures"],
                         start_time=time.time()) as cam:
            while not stop.is_set() and cam.status == "ok":
                frame, timestamp = cam.read()
                for obs in cam.pop_observations():
                    out.put(("camera", obs))
                if frame is None:
                    out.put(("tick", timestamp))
                    continue
                if cam.is_file:
                    stop.wait(max(0.0, timestamp - time.time()))  # play clips at real speed
                result = detector.detect(frame, timestamp)
                if result is not None:
                    last_pose = result
                view = draw_poses(zones.draw(frame), last_pose)
                ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
                inference = detector.last_inference_s if result is not None else None
                out.put(("frame", timestamp, jpeg.tobytes() if ok else None, result, inference))
            for obs in cam.pop_observations():
                out.put(("camera", obs))
            out.put(("end", cam.status if cam.status != "ok" else "stopped"))
    except Exception as exc:  # report instead of dying silently
        out.put(("error", f"{type(exc).__name__}: {exc}"))
