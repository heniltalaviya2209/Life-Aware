"""Why is the pose model slower in the dashboard than in the terminal? Measure it.

    python tools/speed_check.py

Runs the pose model on webcam frames in three ways, 10 s each, and prints the
average time per check:
  A. main thread                         (like tools/pose_demo.py)
  B. background thread                   (like the dashboard worker)
  C. background thread + busy main thread (like the dashboard while the page refreshes)
  D. like C with a shorter Python switch interval (sys.setswitchinterval)
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifeaware.camera import FrameSource  # noqa: E402
from lifeaware.config import load_config  # noqa: E402
from lifeaware.pose import PoseDetector  # noqa: E402

SECONDS = 10


def measure(detector: PoseDetector, frames: list) -> float:
    """Average ms per model call over SECONDS, cycling through the frames."""
    times, start, i = [], time.time(), 0
    while time.time() - start < SECONDS:
        t = time.perf_counter()
        detector._get_model().predict(frames[i % len(frames)], imgsz=detector.image_size, verbose=False)
        times.append(time.perf_counter() - t)
        i += 1
    return 1000 * sum(times) / len(times)


def in_thread(fn, *args) -> float:
    result = {}
    thread = threading.Thread(target=lambda: result.update(ms=fn(*args)))
    thread.start()
    thread.join()
    return result["ms"]


def main() -> None:
    config = load_config()
    detector = PoseDetector.from_config(config)
    print(f"Model: {detector.model_path}")
    try:
        import torch

        print(f"torch {torch.__version__}, threads: {torch.get_num_threads()}, cpu cores: {os.cpu_count()}")
    except ImportError:
        pass

    print("Grabbing 20 webcam frames ...")
    frames = []
    with FrameSource(config["camera"]["source"], max_fps=None) as cam:
        while len(frames) < 20 and cam.status == "ok":
            frame, _ = cam.read()
            if frame is not None:
                frames.append(frame)
    if not frames:
        print("No camera frames - is another program using the webcam?")
        return
    print("Loading model ...")
    detector.load()

    a = measure(detector, frames)
    print(f"A. main thread:                        {a:6.0f} ms per check")
    b = in_thread(measure, detector, frames)
    print(f"B. background thread:                  {b:6.0f} ms per check")

    stop = threading.Event()

    def busy() -> None:  # pure-Python work, like a web server answering the page
        while not stop.is_set():
            sum(i * i for i in range(2000))

    busy_thread = threading.Thread(target=busy, daemon=True)
    busy_thread.start()
    c = in_thread(measure, detector, frames)
    print(f"C. background thread + busy main:      {c:6.0f} ms per check")

    default = sys.getswitchinterval()
    for interval in (0.0005, 0.0001):
        sys.setswitchinterval(interval)
        ms = in_thread(measure, detector, frames)
        print(f"D. same as C, switch interval {interval * 1000:.1f} ms: {ms:6.0f} ms per check")
    sys.setswitchinterval(default)
    stop.set()


if __name__ == "__main__":
    main()
