"""Pose detection: find people, read their body points, decide the posture.

Uses a pretrained YOLO *pose* model (Ultralytics, no training). For every
person it gets 17 body points (COCO keypoints: nose, shoulders, hips, knees,
...) and turns them into one of these labels:

- ``person_standing``  upright, knees well below the hips
- ``person_sitting``   upright, knees close to hip height
- ``person_lying``     the body is (close to) horizontal
- ``person_unclear``   a person is there, but the body points are too weak or
                       hidden to tell. This is NOT "no person" and NOT "lying".

The decision itself is plain geometry in ``classify_posture()``: it needs no
camera and no model, so the rules can be tested with fake numbers.

``PoseDetector.detect()`` returns (same contract as ``objects.ObjectDetector``):
- ``None``  when this frame was skipped (only every n-th frame is analysed),
- ``[]``    when the frame was analysed and nobody was found,
- a list of ``Observation(source="pose", ...)`` otherwise.

Skipped and "nobody found" are different: never treat a skipped frame as
evidence that the person left or stopped moving.

Limits (be honest about them in the report):
- One fixed camera. A person lying with their head toward the camera, or seen
  from straight above, can look upright.
- Bending down or doing exercises on the floor can look like lying. A single
  frame never proves a fall; the fall check must look at time (several frames).
- The confidence is "how strong the evidence is" (keypoint confidence times
  person confidence), not a calibrated probability.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple, Sequence

import numpy as np

from lifeaware.objects import load_model
from lifeaware.schema import Observation
from lifeaware.zones import ZoneMap

SOURCE = "pose"

LABEL_STANDING = "person_standing"
LABEL_SITTING = "person_sitting"
LABEL_LYING = "person_lying"
LABEL_UNCLEAR = "person_unclear"

# COCO keypoint order used by YOLO pose models.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
NUM_KEYPOINTS = 17

# Lines drawn between keypoints by ``draw_poses`` (pairs of indices).
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),          # shoulders and arms
    (5, 11), (6, 12), (11, 12),                        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),            # legs
    (0, 1), (0, 2), (1, 3), (2, 4),                    # head
]

# Evidence strength for the weak cases (see "Limits" above).
ASPECT_ONLY_CONFIDENCE = 0.5  # lying judged from the box shape only
UNCLEAR_CONFIDENCE = 0.3      # person found, posture cannot be told

# If two analysed frames are further apart than this, the person's movement
# between them is not computed (the person may have left and come back).
MOVEMENT_MAX_GAP_S = 2.0


class PostureResult(NamedTuple):
    """Outcome of ``classify_posture``."""

    label: str
    confidence: float  # evidence strength, 0.0 - 1.0
    features: dict  # torso_angle, knee_ratio, bbox_aspect, keypoints_used


# ------------------------------------------------------------------ geometry


def _midpoint(keypoints: np.ndarray, first: int, second: int, min_conf: float):
    """Average ``(x, y, conf)`` of the visible points among two keypoints, or None."""
    visible = [keypoints[i] for i in (first, second) if keypoints[i][2] >= min_conf]
    if not visible:
        return None
    return (
        float(np.mean([p[0] for p in visible])),
        float(np.mean([p[1] for p in visible])),
        float(np.mean([p[2] for p in visible])),
    )


def classify_posture(
    keypoints: Sequence[Sequence[float]],
    bbox: Sequence[float],
    min_keypoint_conf: float = 0.3,
    lying_torso_angle: float = 60.0,
    sit_knee_ratio: float = 0.6,
    lying_aspect: float = 1.4,
    min_torso_px: float = 8.0,
) -> PostureResult:
    """Decide the posture of ONE person from keypoints and bounding box.

    ``keypoints``: 17 rows of ``[x, y, confidence]`` (pixels, COCO order).
    ``bbox``: ``[x1, y1, x2, y2]`` of the person in pixels.

    Rules:
    1. Torso = line from the middle of the shoulders to the middle of the hips.
       Its angle to the vertical is ``torso_angle`` (0 = upright, 90 = flat).
       ``torso_angle >= lying_torso_angle``  ->  lying.
    2. Otherwise (upright): ``knee_ratio`` = (knees below hips) / torso length.
       ``knee_ratio >= sit_knee_ratio`` -> standing, below -> sitting.
       Knees not visible -> unclear (a table or the frame edge hides the legs).
    3. No usable torso: a very wide box (width / height >= ``lying_aspect``)
       counts as lying with weak evidence; anything else is unclear.
    """
    features: dict = {"torso_angle": None, "knee_ratio": None, "bbox_aspect": None, "keypoints_used": 0}

    points = np.asarray(keypoints, dtype=float)
    if points.ndim != 2 or points.shape[0] < NUM_KEYPOINTS or points.shape[1] < 3:
        return PostureResult(LABEL_UNCLEAR, UNCLEAR_CONFIDENCE, features)

    x1, y1, x2, y2 = (float(v) for v in bbox)
    width, height = x2 - x1, y2 - y1
    aspect = width / height if height > 0 else None
    features["bbox_aspect"] = aspect

    shoulders = _midpoint(points, L_SHOULDER, R_SHOULDER, min_keypoint_conf)
    hips = _midpoint(points, L_HIP, R_HIP, min_keypoint_conf)

    torso_len = 0.0
    if shoulders is not None and hips is not None:
        dx, dy = hips[0] - shoulders[0], hips[1] - shoulders[1]
        torso_len = math.hypot(dx, dy)

    if torso_len < min_torso_px:
        # No usable torso: fall back to the shape of the box.
        if aspect is not None and aspect >= lying_aspect:
            return PostureResult(LABEL_LYING, ASPECT_ONLY_CONFIDENCE, features)
        return PostureResult(LABEL_UNCLEAR, UNCLEAR_CONFIDENCE, features)

    torso_angle = math.degrees(math.atan2(abs(dx), abs(dy)))  # 0 = upright, 90 = flat
    features["torso_angle"] = torso_angle
    used = [shoulders[2], hips[2]]

    if torso_angle >= lying_torso_angle:
        features["keypoints_used"] = len(used)
        return PostureResult(LABEL_LYING, float(np.mean(used)), features)

    knees = _midpoint(points, L_KNEE, R_KNEE, min_keypoint_conf)
    if knees is None:
        features["keypoints_used"] = len(used)
        return PostureResult(LABEL_UNCLEAR, UNCLEAR_CONFIDENCE, features)

    knee_ratio = (knees[1] - hips[1]) / torso_len
    features["knee_ratio"] = knee_ratio
    used.append(knees[2])
    features["keypoints_used"] = len(used)
    label = LABEL_STANDING if knee_ratio >= sit_knee_ratio else LABEL_SITTING
    return PostureResult(label, float(np.mean(used)), features)


# ------------------------------------------------------------------ detector


def _as_list(values) -> list:
    """Torch tensors and numpy arrays have ``tolist()``; plain lists pass through."""
    return values.tolist() if hasattr(values, "tolist") else list(values)


class PoseDetector:
    """Find people in frames and turn them into posture Observations."""

    def __init__(
        self,
        model: str = "models/yolo11s-pose.pt",
        every_n_frames: int = 3,
        min_conf: float = 0.4,
        min_keypoint_conf: float = 0.3,
        image_size: int = 640,
        max_persons: int = 1,
        lying_torso_angle: float = 60.0,
        sit_knee_ratio: float = 0.6,
        lying_aspect: float = 1.4,
        zone_map: ZoneMap | None = None,
    ) -> None:
        self.model_path = model
        self.every_n_frames = max(1, int(every_n_frames))
        self.min_conf = min_conf
        self.min_keypoint_conf = min_keypoint_conf
        self.image_size = image_size
        self.max_persons = max(1, int(max_persons))
        self.lying_torso_angle = lying_torso_angle
        self.sit_knee_ratio = sit_knee_ratio
        self.lying_aspect = lying_aspect
        self.zone_map = zone_map or ZoneMap()
        self.last_inference_s: float | None = None  # duration of the last model call
        self._frame_count = 0
        self._model = None  # set by _get_model() on first real detection
        self._previous: tuple[float, float, float] | None = None  # (cx, cy, time) of the main person

    @classmethod
    def from_config(cls, config: dict, zone_map: ZoneMap | None = None) -> "PoseDetector":
        """Build a detector from the ``pose`` section of ``load_config()``."""
        section = config.get("pose", {})
        return cls(
            model=section.get("model", "models/yolo11s-pose.pt"),
            every_n_frames=section.get("every_n_frames", 3),
            min_conf=section.get("min_conf", 0.4),
            min_keypoint_conf=section.get("min_keypoint_conf", 0.3),
            image_size=section.get("image_size", 640),
            max_persons=section.get("max_persons", 1),
            lying_torso_angle=section.get("lying_torso_angle", 60.0),
            sit_knee_ratio=section.get("sit_knee_ratio", 0.6),
            lying_aspect=section.get("lying_aspect", 1.4),
            zone_map=zone_map,
        )

    def load(self) -> None:
        """Load the model and run one warm-up check (the first prediction is slow)."""
        model = self._get_model()
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        model.predict(blank, imgsz=self.image_size, verbose=False)

    def _get_model(self):
        """Load the model lazily (shared cache with the object detector)."""
        if self._model is None:
            self._model = load_model(self.model_path)
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
        results = model.predict(frame, conf=self.min_conf, imgsz=self.image_size, verbose=False)
        self.last_inference_s = time.perf_counter() - start
        return self._to_observations(results[0], timestamp)

    def _to_observations(self, result, timestamp: float) -> list[Observation]:
        """Convert one YOLO pose result into Observations (largest person first)."""
        boxes, keypoints = result.boxes, result.keypoints
        people = []
        if boxes is not None and keypoints is not None:
            for box, conf, kps in zip(
                _as_list(boxes.xyxy), _as_list(boxes.conf), _as_list(keypoints.data)
            ):
                if conf >= self.min_conf:
                    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                    people.append((area, box, float(conf), kps))

        if not people:
            self._previous = None  # nobody there: the next person has no "previous position"
            return []

        people.sort(key=lambda p: p[0], reverse=True)
        observations = []
        for rank, (_, box, person_conf, kps) in enumerate(people[: self.max_persons]):
            posture = classify_posture(
                kps, box,
                min_keypoint_conf=self.min_keypoint_conf,
                lying_torso_angle=self.lying_torso_angle,
                sit_knee_ratio=self.sit_knee_ratio,
                lying_aspect=self.lying_aspect,
            )
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            diagonal = math.hypot(box[2] - box[0], box[3] - box[1])
            movement = self._movement(centre, diagonal, timestamp) if rank == 0 else None
            data = {
                "bbox": [float(v) for v in box],
                "keypoints": [[round(float(v), 1) for v in row] for row in kps],
                "person_conf": person_conf,
                "movement": movement,  # body-diagonals per second; None = not known
                **posture.features,
            }
            observations.append(
                Observation(
                    time=timestamp,
                    source=SOURCE,
                    label=posture.label,
                    confidence=float(posture.confidence * person_conf),
                    zone=self.zone_map.zone_of(*centre),
                    data=data,
                )
            )
        return observations

    def _movement(self, centre: tuple[float, float], diagonal: float, timestamp: float) -> float | None:
        """Speed of the main person's box centre since the last analysed frame.

        Unit: body-diagonals per second (the box diagonal is the person's size),
        so it does not depend on how far the person is from the camera.
        Returns None for the first sighting or after a long gap.
        """
        previous, self._previous = self._previous, (centre[0], centre[1], timestamp)
        if previous is None or diagonal <= 0:
            return None
        elapsed = timestamp - previous[2]
        if elapsed <= 0 or elapsed > MOVEMENT_MAX_GAP_S:
            return None
        distance = math.hypot(centre[0] - previous[0], centre[1] - previous[1])
        return distance / diagonal / elapsed


def draw_poses(frame: np.ndarray, observations: list[Observation]) -> np.ndarray:
    """Return a copy of ``frame`` with box, skeleton and "label conf zone" per person."""
    import cv2  # lazy import

    colours = {
        LABEL_STANDING: (0, 200, 0),
        LABEL_SITTING: (0, 200, 200),
        LABEL_LYING: (0, 0, 255),
        LABEL_UNCLEAR: (160, 160, 160),
    }
    canvas = frame.copy()
    for obs in observations:
        colour = colours.get(obs.label, (255, 255, 255))
        x1, y1, x2, y2 = (int(v) for v in obs.data["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        points = obs.data["keypoints"]
        for a, b in SKELETON:
            if points[a][2] >= 0.3 and points[b][2] >= 0.3:
                cv2.line(canvas, (int(points[a][0]), int(points[a][1])),
                         (int(points[b][0]), int(points[b][1])), colour, 2)
        for x, y, c in points:
            if c >= 0.3:
                cv2.circle(canvas, (int(x), int(y)), 3, (255, 255, 255), -1)

        text = f"{obs.label} {obs.confidence:.2f} {obs.zone or '?'}"
        cv2.putText(canvas, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return canvas
