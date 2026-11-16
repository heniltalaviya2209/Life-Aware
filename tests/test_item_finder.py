"""Tests for the item finder (lifeaware.item_finder): worker, cards. Fake model only."""

import copy
import time

import numpy as np
from conftest import write_test_video
from test_objects import FakeModel

from lifeaware.item_finder import ItemWorker, _item_card
from lifeaware import objects
from lifeaware.config import DEFAULTS
from lifeaware.memory import ItemMemory
from lifeaware.objects import draw_detections
from lifeaware.schema import Observation


def make_config():
    config = copy.deepcopy(DEFAULTS)
    config["zones"] = {"table": [[0, 0], [64, 0], [64, 48], [0, 48]]}
    return config


def wait_until_done(worker, timeout=10.0):
    deadline = time.time() + timeout
    while worker.running and time.time() < deadline:
        time.sleep(0.05)


def test_worker_fills_memory_from_a_clip(tmp_path, monkeypatch):
    model = FakeModel([([10, 10, 30, 30], 0.9, 67)])  # a phone in the "table" zone
    monkeypatch.setattr(objects, "load_model", lambda path: model)
    clip = write_test_video(tmp_path / "clip.avi", seconds=1, fps=30)
    memory = ItemMemory()

    worker = ItemWorker(make_config(), str(clip), memory).start()
    wait_until_done(worker)

    assert worker.error is None
    assert worker.status == "ended"
    assert worker.latest_view is not None
    assert memory.last_seen("phone").zone == "table"
    assert "on the table" in memory.answer("phone")


def test_worker_stops_on_request(tmp_path, monkeypatch):
    monkeypatch.setattr(objects, "load_model", lambda path: FakeModel([]))
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)

    worker = ItemWorker(make_config(), str(clip), ItemMemory()).start()
    time.sleep(0.3)
    worker.stop()

    assert not worker.running
    assert worker.status == "stopped"


def test_worker_reports_a_lost_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(objects, "load_model", lambda path: FakeModel([]))

    worker = ItemWorker(make_config(), str(tmp_path / "missing.mp4"), ItemMemory()).start()
    wait_until_done(worker)

    assert worker.status == "lost"


def test_worker_shows_model_errors(monkeypatch):
    def broken_loader(path):
        raise RuntimeError("model file damaged")

    monkeypatch.setattr(objects, "load_model", broken_loader)

    worker = ItemWorker(make_config(), 0, ItemMemory()).start()
    wait_until_done(worker)

    assert worker.status == "error"
    assert "model file damaged" in worker.error


def test_draw_detections_marks_a_copy():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    obs = Observation(time=1.0, source="objects", label="phone", confidence=0.9,
                      zone="table", data={"bbox": [10, 10, 30, 30]})

    drawn = draw_detections(frame, [obs])

    assert drawn.any()
    assert not frame.any()


# ------------------------------------------------------------- dashboard cards


def test_item_card_for_a_seen_phone():
    memory = ItemMemory()
    memory.update(Observation(time=time.time(), source="objects", label="phone",
                              confidence=0.72, zone="table"))

    card = _item_card("phone", memory, detectable={"phone"})

    assert "table" in card and "72% sure" in card and "seen" in card


def test_item_card_for_a_detectable_item_not_seen_yet():
    assert "Not seen yet" in _item_card("phone", ItemMemory(), detectable={"phone"})


def test_item_card_for_keys_says_not_detectable_yet():
    assert "Cannot be detected yet" in _item_card("keys", ItemMemory(), detectable={"phone"})
