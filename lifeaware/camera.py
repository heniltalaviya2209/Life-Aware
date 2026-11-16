"""Camera input: frames from a webcam or a video file, with timestamps.

``FrameSource`` is the first link of the LifeAware chain. It delivers at most
``max_fps`` frames per second and always reports an honest status:

- ``"ok"``     frames are arriving
- ``"lost"``   the webcam failed (or could not be opened) -> downstream "unknown"
- ``"ended"``  the video file has no more frames

When the status turns to ``"lost"``, a ``camera_lost`` Observation is queued;
the pipeline collects it with ``pop_observations()``.

OpenCV is imported lazily, so importing this module never needs ``cv2``.
"""

from __future__ import annotations

import os
import time

from lifeaware.schema import Observation

STATUS_OK = "ok"
STATUS_LOST = "lost"
STATUS_ENDED = "ended"

# Used when a video file does not report its own frame rate.
FALLBACK_VIDEO_FPS = 30.0

# Small tolerance so 30 fps -> 10 fps keeps exactly every 3rd frame
# despite floating-point rounding.
_TIME_EPSILON = 1e-6


def open_webcam(index: int = 0):
    """Open a webcam, or return None if no camera can be opened.

    Tries the default backend first, then DirectShow (``CAP_DSHOW``), which
    works more reliably on some Windows laptops.
    """
    import cv2  # lazy import: the app must run without OpenCV

    for args in ((index,), (index, cv2.CAP_DSHOW)):
        try:
            cap = cv2.VideoCapture(*args)
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None


class FrameSource:
    """Read frames from a webcam (int index) or a video file (path).

    Usage::

        with FrameSource("clip.mp4", max_fps=10) as cam:
            while cam.status == "ok":
                frame, timestamp = cam.read()

    Timestamps: webcam -> ``time.time()``; video file -> ``start_time`` plus
    the position in the video, so replays of the same clip are repeatable.
    """

    def __init__(
        self,
        source: int | str | os.PathLike = 0,
        max_fps: float | None = 10.0,
        max_failures: int = 15,
        start_time: float = 0.0,
    ) -> None:
        self.source = source
        self.max_fps = max_fps
        self.max_failures = max_failures
        self.start_time = start_time
        self.is_file = not isinstance(source, int)
        self.status = STATUS_OK

        self._failures = 0  # consecutive failed webcam reads
        self._frame_index = 0  # frames read from the video file so far
        self._last_time: float | None = None  # timestamp of the last frame given out
        self._observations: list[Observation] = []
        self._cap = None
        self._video_fps = FALLBACK_VIDEO_FPS
        self._open()

    # ------------------------------------------------------------------ setup

    def _open(self) -> None:
        """Open the webcam or the video file; mark the source lost on failure."""
        if not self.is_file:
            self._cap = open_webcam(self.source)
        else:
            import cv2  # lazy import

            cap = cv2.VideoCapture(str(self.source))
            if cap.isOpened():
                self._cap = cap
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    self._video_fps = fps
            else:
                cap.release()

        if self._cap is None:
            self._mark_lost(self._now(), reason="could not open source")

    # ---------------------------------------------------------------- reading

    def read(self):
        """Return ``(frame, timestamp)``, or ``(None, timestamp)`` if no frame.

        Check ``status`` to tell "no frame yet" from "lost" or "ended".
        """
        if self.status != STATUS_OK:
            return None, self._now()
        if self.is_file:
            return self._read_video()
        return self._read_webcam()

    def _read_video(self):
        """Read the next video frame, skipping frames to respect ``max_fps``."""
        while True:
            ok, frame = self._cap.read()
            if not ok:
                self.status = STATUS_ENDED
                return None, self._now()
            timestamp = self.start_time + self._frame_index / self._video_fps
            self._frame_index += 1
            if self._is_due(timestamp):
                self._last_time = timestamp
                return frame, timestamp

    def _read_webcam(self):
        """Wait for the next fps slot, then read one webcam frame."""
        if self.max_fps and self._last_time is not None:
            wait = self._last_time + 1.0 / self.max_fps - time.time()
            if wait > 0:
                time.sleep(wait)

        ok, frame = self._cap.read()
        timestamp = time.time()
        self._last_time = timestamp  # also paces retries after a failed read
        if not ok:
            self._failures += 1
            if self._failures >= self.max_failures:
                self._mark_lost(timestamp, reason=f"{self._failures} failed reads in a row")
            return None, timestamp

        self._failures = 0
        return frame, timestamp

    def _is_due(self, timestamp: float) -> bool:
        """True if enough time passed since the last frame for ``max_fps``."""
        if not self.max_fps or self._last_time is None:
            return True
        return timestamp - self._last_time >= 1.0 / self.max_fps - _TIME_EPSILON

    # ----------------------------------------------------------- observations

    def _mark_lost(self, timestamp: float, reason: str) -> None:
        """Switch to "lost" once and queue a ``camera_lost`` Observation."""
        if self.status == STATUS_LOST:
            return
        self.status = STATUS_LOST
        self._observations.append(
            Observation(
                time=timestamp,
                source="camera",
                label="camera_lost",
                confidence=1.0,
                data={"source": str(self.source), "reason": reason},
            )
        )

    def pop_observations(self) -> list[Observation]:
        """Return the queued observations and clear the queue."""
        queued, self._observations = self._observations, []
        return queued

    def _now(self) -> float:
        """Current time: wall clock for a webcam, video time for a file."""
        if self.is_file:
            return self._last_time if self._last_time is not None else self.start_time
        return time.time()

    # ---------------------------------------------------------------- cleanup

    def release(self) -> None:
        """Close the camera or file. Safe to call more than once."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
