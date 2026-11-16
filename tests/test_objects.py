"""Tests for lifeaware.objects.ObjectDetector with a FAKE model (no download)."""

import numpy as np
import pytest

from lifeaware import objects
from lifeaware.objects import ObjectDetector
from lifeaware.zones import ZoneMap

COCO_NAMES = {0: "person", 56: "chair", 67: "cell phone"}


class FakeBoxes:
    """Mimics ultralytics ``result.boxes`` (numpy instead of torch tensors)."""

    def __init__(self, detections):
        self.xyxy = np.array([d[0] for d in detections], dtype=float).reshape(-1, 4)
        self.conf = np.array([d[1] for d in detections], dtype=float)
        self.cls = np.array([d[2] for d in detections], dtype=float)


class FakeResult:
    def __init__(self, detections):
        self.boxes = FakeBoxes(detections)
        self.names = COCO_NAMES


class FakeModel:
    """Returns the same detections for every frame and records the calls."""

    names = COCO_NAMES

    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.detections)]


@pytest.fixture
def fake_model(monkeypatch):
    """Install a fake model; tests set ``fake_model.detections``."""
    model = FakeModel([])
    monkeypatch.setattr(objects, "load_model", lambda path: model)
    return model


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
ZONES = ZoneMap({"sofa": [[0, 0], [320, 0], [320, 480], [0, 480]]})


def test_phone_becomes_an_observation_with_zone(fake_model):
    fake_model.detections = [([100, 200, 140, 260], 0.87, 67)]  # centre (120, 230) -> sofa
    detector = ObjectDetector(every_n_frames=1, zone_map=ZONES)

    [obs] = detector.detect(FRAME, timestamp=123.0)

    assert obs.source == "objects"
    assert obs.label == "phone"
    assert obs.confidence == pytest.approx(0.87)
    assert obs.zone == "sofa"
    assert obs.time == 123.0
    assert obs.data["bbox"] == [100, 200, 140, 260]


def test_phone_outside_zones_has_no_zone(fake_model):
    fake_model.detections = [([500, 200, 540, 260], 0.9, 67)]
    detector = ObjectDetector(every_n_frames=1, zone_map=ZONES)

    assert detector.detect(FRAME, 1.0)[0].zone is None


def test_unwanted_classes_and_low_confidence_are_dropped(fake_model):
    fake_model.detections = [
        ([0, 0, 50, 50], 0.95, 0),  # person: not a wanted item
        ([0, 0, 50, 50], 0.30, 67),  # phone below min_conf
    ]
    detector = ObjectDetector(every_n_frames=1, min_conf=0.4)

    assert detector.detect(FRAME, 1.0) == []


def test_only_every_nth_frame_is_analysed(fake_model):
    detector = ObjectDetector(every_n_frames=3)

    results = [detector.detect(FRAME, float(t)) for t in range(7)]

    assert [r is not None for r in results] == [True, False, False, True, False, False, True]
    assert len(fake_model.calls) == 3


def test_missing_frame_is_skipped_not_empty(fake_model):
    detector = ObjectDetector(every_n_frames=1)

    assert detector.detect(None, 1.0) is None
    assert fake_model.calls == []


def test_model_is_asked_only_for_wanted_classes(fake_model):
    detector = ObjectDetector(every_n_frames=1, min_conf=0.5, image_size=320)

    detector.detect(FRAME, 1.0)

    call = fake_model.calls[0]
    assert call["classes"] == [67]
    assert call["conf"] == 0.5
    assert call["imgsz"] == 320


def test_model_is_loaded_once(monkeypatch):
    loads = []
    model = FakeModel([])
    monkeypatch.setattr(objects, "load_model", lambda path: loads.append(path) or model)
    detector = ObjectDetector(every_n_frames=1)

    for t in range(5):
        detector.detect(FRAME, float(t))

    assert loads == ["models/yolo11n.pt"]


def test_from_config(fake_model):
    config = {"objects": {"model": "x.pt", "every_n_frames": 5, "min_conf": 0.6,
                          "classes": {"cell phone": "phone"}, "image_size": 480}}

    detector = ObjectDetector.from_config(config)

    assert (detector.model_path, detector.every_n_frames, detector.min_conf) == ("x.pt", 5, 0.6)
