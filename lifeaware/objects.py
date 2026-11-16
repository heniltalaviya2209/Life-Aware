"""Object detection with a pretrained YOLO model (no training).

For now only the COCO class "cell phone" is used (reported as "phone").
"Keys" are not a COCO class, so nothing is reported for keys yet.

``detect()`` returns:
- ``None``  when this frame was skipped (only every n-th frame is analysed),
- ``[]``    when the frame was analysed and no wanted object was found,
- a list of ``Observation(source="objects", ...)`` otherwise.

Skipped and "nothing found" are different: never treat a skipped frame as
evidence that an item is gone.
"""

from __future__ import annotations

import time

import numpy as np

from lifeaware.schema import Observation
from lifeaware.zones import ZoneMap

# Loaded models, one per model path, shared by all detectors in this process.
_MODEL_CACHE: dict = {}


def load_model(model_path: str):
    """Load a YOLO model once and reuse it (downloads pretrained weights on first use)."""
    if model_path not in _MODEL_CACHE:
        from ultralytics import YOLO  # lazy import: heavy (torch)

        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


class ObjectDetector:
    """Find wanted COCO objects in frames and turn them into Observations."""

    def __init__(
        self,
        model: str = "models/yolo11n.pt",
        every_n_frames: int = 3,
        min_conf: float = 0.4,
        classes: dict[str, str] | None = None,
        image_size: int = 640,
        zone_map: ZoneMap | None = None,
    ) -> None:
        self.model_path = model
        self.every_n_frames = max(1, int(every_n_frames))
        self.min_conf = min_conf
        self.classes = classes or {"cell phone": "phone"}  # COCO name -> our label
        self.image_size = image_size
        self.zone_map = zone_map or ZoneMap()
        self.last_inference_s: float | None = None  # duration of the last model call
        self._frame_count = 0
        self._model = None  # set by _get_model() on first real detection
        self._class_ids: list[int] | None = None

    @classmethod
    def from_config(cls, config: dict, zone_map: ZoneMap | None = None) -> "ObjectDetector":
        """Build a detector from the ``objects`` section of ``load_config()``."""
        section = config.get("objects", {})
        return cls(
            model=section.get("model", "models/yolo11n.pt"),
            every_n_frames=section.get("every_n_frames", 3),
            min_conf=section.get("min_conf", 0.4),
            classes=section.get("classes"),
            image_size=section.get("image_size", 640),
            zone_map=zone_map,
        )

    def load(self) -> None:
        """Load the model and run one warm-up check.

        PyTorch's first prediction is much slower than the rest (about 2 s on our
        laptop), so we pay that cost here instead of on the first real frame.
        """
        model = self._get_model()
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        model.predict(blank, imgsz=self.image_size, classes=self._class_ids, verbose=False)

    def _get_model(self):
        """Load the model lazily and look up the COCO ids of the wanted classes."""
        if self._model is None:
            self._model = load_model(self.model_path)
            names = self._model.names  # {class_id: "cell phone", ...}
            self._class_ids = [i for i, name in names.items() if name in self.classes]
        return self._model

    def detect(self, frame: np.ndarray | None, timestamp: float) -> list[Observation] | None:
        """Analyse every n-th frame; see the module docstring for return values."""
        if frame is None:
            return None
        self._frame_count += 1
        if (self._frame_count - 1) % self.every_n_frames != 0:
            return None

        model = self._get_model()
        start = time.perf_counter()
        results = model.predict(
            frame, conf=self.min_conf, imgsz=self.image_size,
            classes=self._class_ids, verbose=False,
        )
        self.last_inference_s = time.perf_counter() - start
        return self._to_observations(results[0], timestamp)

    def _to_observations(self, result, timestamp: float) -> list[Observation]:
        """Convert one YOLO result into Observations (zone = zone of the box centre)."""
        boxes = result.boxes
        observations = []
        for xyxy, conf, cls_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            class_name = result.names[int(cls_id)]
            label = self.classes.get(class_name)
            if label is None or conf < self.min_conf:
                continue
            x1, y1, x2, y2 = xyxy
            centre = ((x1 + x2) / 2, (y1 + y2) / 2)
            observations.append(
                Observation(
                    time=timestamp,
                    source="objects",
                    label=label,
                    confidence=float(conf),
                    zone=self.zone_map.zone_of(*centre),
                    data={"bbox": [x1, y1, x2, y2], "class_name": class_name},
                )
            )
        return observations


def draw_detections(frame: np.ndarray, observations: list[Observation]) -> np.ndarray:
    """Return a copy of ``frame`` with a red box and "label conf zone" per detection."""
    import cv2  # lazy import

    canvas = frame.copy()
    for obs in observations:
        x1, y1, x2, y2 = (int(v) for v in obs.data["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{obs.label} {obs.confidence:.2f} {obs.zone or '?'}"
        cv2.putText(canvas, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return canvas
