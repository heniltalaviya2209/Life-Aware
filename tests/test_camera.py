"""Tests for lifeaware.camera.FrameSource.

No real camera is used: video tests write a tiny synthetic clip into pytest's
``tmp_path``; webcam tests replace ``open_webcam`` with a fake capture.
"""

import numpy as np
import pytest
from conftest import write_test_video

from lifeaware import camera
from lifeaware.camera import FrameSource


def read_all(source):
    """Read until the source stops; return the list of timestamps."""
    timestamps = []
    while source.status == "ok":
        frame, timestamp = source.read()
        if frame is not None:
            timestamps.append(timestamp)
    return timestamps


class FakeCapture:
    """Stands in for cv2.VideoCapture: returns a fixed list of read results."""

    def __init__(self, results):
        self.results = list(results)
        self.released = False

    def read(self):
        if self.results:
            return self.results.pop(0)
        return False, None

    def release(self):
        self.released = True


# ------------------------------------------------------------------ video file


def test_video_is_capped_to_max_fps(tmp_path):
    video = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)

    with FrameSource(video, max_fps=10) as cam:
        timestamps = read_all(cam)

    assert cam.status == "ended"
    assert 29 <= len(timestamps) <= 31  # 3 s at 10 fps
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    assert timestamps[1] - timestamps[0] == pytest.approx(0.1, abs=1e-6)


def test_video_timestamps_follow_video_time_and_repeat(tmp_path):
    video = write_test_video(tmp_path / "clip.avi", seconds=1, fps=30)

    with FrameSource(video, max_fps=10, start_time=1000.0) as cam:
        first_run = read_all(cam)
    with FrameSource(video, max_fps=10, start_time=1000.0) as cam:
        second_run = read_all(cam)

    assert first_run[0] == pytest.approx(1000.0)
    assert first_run == second_run  # replays are repeatable


def test_video_without_fps_cap_returns_every_frame(tmp_path):
    video = write_test_video(tmp_path / "clip.avi", seconds=1, fps=30)

    with FrameSource(video, max_fps=None) as cam:
        assert len(read_all(cam)) == 30


def test_missing_video_file_is_lost_not_crash(tmp_path):
    cam = FrameSource(tmp_path / "does_not_exist.mp4")

    frame, _ = cam.read()

    assert frame is None
    assert cam.status == "lost"
    assert [o.label for o in cam.pop_observations()] == ["camera_lost"]
    cam.release()


def test_reading_after_end_keeps_returning_none(tmp_path):
    video = write_test_video(tmp_path / "clip.avi", seconds=1, fps=30)

    with FrameSource(video) as cam:
        read_all(cam)
        frame, _ = cam.read()

    assert frame is None
    assert cam.status == "ended"
    assert cam.pop_observations() == []  # the end of a clip is not a lost camera


# ---------------------------------------------------------------------- webcam


def test_webcam_lost_after_max_failures(monkeypatch):
    fake = FakeCapture([(False, None)] * 20)
    monkeypatch.setattr(camera, "open_webcam", lambda index=0: fake)
    cam = FrameSource(0, max_fps=None, max_failures=15)

    for _ in range(14):
        cam.read()
    assert cam.status == "ok"  # 14 failures: not lost yet

    cam.read()
    assert cam.status == "lost"

    observations = cam.pop_observations()
    assert len(observations) == 1
    assert observations[0].source == "camera"
    assert observations[0].label == "camera_lost"
    assert observations[0].confidence == 1.0

    cam.read()  # reading again does not queue a second observation
    assert cam.pop_observations() == []


def test_webcam_failure_counter_resets_after_a_good_frame(monkeypatch):
    good = (True, np.zeros((4, 4, 3), dtype=np.uint8))
    fake = FakeCapture([(False, None)] * 10 + [good] + [(False, None)] * 10)
    monkeypatch.setattr(camera, "open_webcam", lambda index=0: fake)
    cam = FrameSource(0, max_fps=None, max_failures=15)

    for _ in range(21):
        cam.read()

    assert cam.status == "ok"  # never 15 failures in a row


def test_webcam_that_cannot_open_is_lost(monkeypatch):
    monkeypatch.setattr(camera, "open_webcam", lambda index=0: None)

    cam = FrameSource(0)

    assert cam.status == "lost"
    assert cam.read()[0] is None
    assert len(cam.pop_observations()) == 1


def test_context_manager_releases_capture(monkeypatch):
    fake = FakeCapture([])
    monkeypatch.setattr(camera, "open_webcam", lambda index=0: fake)

    with FrameSource(0):
        pass

    assert fake.released


def test_webcam_respects_max_fps(monkeypatch):
    good = (True, np.zeros((4, 4, 3), dtype=np.uint8))
    fake = FakeCapture([good] * 3)
    monkeypatch.setattr(camera, "open_webcam", lambda index=0: fake)
    cam = FrameSource(0, max_fps=20)

    _, t1 = cam.read()
    _, t2 = cam.read()
    _, t3 = cam.read()

    assert t2 - t1 >= 0.045  # 20 fps -> at least ~50 ms apart
    assert t3 - t2 >= 0.045
