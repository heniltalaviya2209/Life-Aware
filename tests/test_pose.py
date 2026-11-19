"""Tests for lifeaware.pose with FAKE keypoints and a FAKE model (no download, no camera)."""

import numpy as np
import pytest

from lifeaware import pose
from lifeaware.config import load_config
from lifeaware.pose import (
    LABEL_LYING,
    LABEL_SITTING,
    LABEL_STANDING,
    LABEL_UNCLEAR,
    PoseDetector,
    classify_posture,
    draw_poses,
)
from lifeaware.zones import ZoneMap

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def keypoints(shoulders=None, hips=None, knees=None, conf=0.9):
    """17 x [x, y, conf]. Each argument is ((left_x, left_y), (right_x, right_y)) or None."""
    kps = np.zeros((17, 3), dtype=float)
    for pair, (l_idx, r_idx) in (
        (shoulders, (5, 6)), (hips, (11, 12)), (knees, (13, 14)),
    ):
        if pair is None:
            continue
        left, right = pair
        kps[l_idx] = [*left, conf]
        kps[r_idx] = [*right, conf]
    return kps


# A standing person seen from the front: torso 110 px tall, knees 100 px below the hips.
STANDING = keypoints(shoulders=((300, 150), (340, 150)), hips=((305, 260), (335, 260)),
                     knees=((305, 360), (335, 360)))
STANDING_BOX = [280, 100, 360, 460]
# Sitting: thighs foreshortened, knees only 30 px below the hips.
SITTING = keypoints(shoulders=((300, 150), (340, 150)), hips=((305, 260), (335, 260)),
                    knees=((305, 290), (335, 290)))
SITTING_BOX = [280, 100, 360, 330]
# Lying on the floor: torso almost horizontal.
LYING = keypoints(shoulders=((150, 400), (150, 430)), hips=((260, 410), (260, 440)),
                  knees=((360, 415), (360, 440)))
LYING_BOX = [120, 380, 420, 460]


# ------------------------------------------------------------ classify_posture


def test_standing():
    result = classify_posture(STANDING, STANDING_BOX)
    assert result.label == LABEL_STANDING
    assert result.features["torso_angle"] == pytest.approx(0, abs=1)
    assert result.features["knee_ratio"] == pytest.approx(100 / 110, abs=0.01)
    assert result.confidence == pytest.approx(0.9)


def test_sitting_knees_close_to_hip_height():
    assert classify_posture(SITTING, SITTING_BOX).label == LABEL_SITTING


def test_sitting_sideways_with_thighs_horizontal():
    # Seen from the side: knees are far to the side but NOT below the hips.
    side = keypoints(shoulders=((300, 150), (300, 150)), hips=((300, 260), (300, 260)),
                     knees=((380, 265), (380, 265)))
    assert classify_posture(side, [260, 100, 420, 330]).label == LABEL_SITTING


def test_lying():
    result = classify_posture(LYING, LYING_BOX)
    assert result.label == LABEL_LYING
    assert result.features["torso_angle"] > 80


def test_lying_is_decided_by_torso_angle_threshold():
    # The torso is 100 px tall; sideways shift dx tilts it: dx = 180 -> ~61 deg (lying),
    # dx = 160 -> ~58 deg (just below the 60 deg limit, so not lying).
    def tilted(dx):
        return keypoints(shoulders=((300, 100), (300, 100)), hips=((300 + dx, 200), (300 + dx, 200)),
                         knees=((300 + dx, 300), (300 + dx, 300)))

    assert classify_posture(tilted(180), [250, 90, 520, 320]).label == LABEL_LYING
    assert classify_posture(tilted(160), [250, 90, 520, 320]).label != LABEL_LYING


def test_weak_keypoints_are_ignored_and_result_is_unclear():
    weak = np.array(STANDING)
    weak[:, 2] = 0.1  # below min_keypoint_conf
    result = classify_posture(weak, STANDING_BOX)
    assert result.label == LABEL_UNCLEAR  # not "lying", not "no person"
    assert result.confidence < 0.5


def test_upright_without_visible_knees_is_unclear():
    no_knees = keypoints(shoulders=((300, 150), (340, 150)), hips=((305, 260), (335, 260)))
    assert classify_posture(no_knees, [280, 100, 360, 300]).label == LABEL_UNCLEAR


def test_one_side_only_is_enough():
    # Only the left shoulder / hip / knee are visible (person seen from the side).
    half = np.zeros((17, 3))
    half[5] = [300, 150, 0.9]
    half[11] = [305, 260, 0.9]
    half[13] = [305, 360, 0.9]
    assert classify_posture(half, STANDING_BOX).label == LABEL_STANDING


def test_no_torso_wide_box_is_lying_with_weak_evidence():
    nothing = np.zeros((17, 3))
    result = classify_posture(nothing, [100, 380, 500, 460])  # 400 x 80 -> aspect 5
    assert result.label == LABEL_LYING
    assert result.confidence == pytest.approx(pose.ASPECT_ONLY_CONFIDENCE)


def test_no_torso_wide_box_cut_off_by_the_image_edge_is_unclear():
    # Sitting close to a laptop webcam: only head and shoulders, box runs off the bottom.
    nothing = np.zeros((17, 3))
    result = classify_posture(nothing, [85, 148, 640, 480], frame_size=(640, 480))
    assert result.label == LABEL_UNCLEAR
    assert result.features["cut_off"] is True


def test_wide_box_inside_the_image_is_still_lying():
    nothing = np.zeros((17, 3))
    result = classify_posture(nothing, [100, 380, 500, 460], frame_size=(640, 480))
    assert result.label == LABEL_LYING
    assert result.features["cut_off"] is False


def test_detector_passes_the_frame_size(fake_model):
    close_up = [85, 148, 640, 480]  # wide, touches the right and bottom edge of FRAME
    fake_model.people = [(close_up, 0.9, np.zeros((17, 3)))]
    [obs] = PoseDetector(every_n_frames=1).detect(FRAME, 1.0)
    assert obs.label == LABEL_UNCLEAR


def test_no_torso_tall_box_is_unclear():
    nothing = np.zeros((17, 3))
    assert classify_posture(nothing, [280, 100, 360, 460]).label == LABEL_UNCLEAR


def test_wrong_shape_is_unclear_not_a_crash():
    assert classify_posture([[1, 2, 3]], [0, 0, 10, 10]).label == LABEL_UNCLEAR
    assert classify_posture([], [0, 0, 10, 10]).label == LABEL_UNCLEAR


def test_zero_height_box_does_not_crash():
    nothing = np.zeros((17, 3))
    assert classify_posture(nothing, [10, 10, 50, 10]).label == LABEL_UNCLEAR


# -------------------------------------------------------------- PoseDetector


class FakeBoxes:
    def __init__(self, people):
        self.xyxy = np.array([p[0] for p in people], dtype=float).reshape(-1, 4)
        self.conf = np.array([p[1] for p in people], dtype=float)


class FakeKeypoints:
    def __init__(self, people):
        self.data = np.array([p[2] for p in people], dtype=float).reshape(-1, 17, 3)


class FakeResult:
    def __init__(self, people):
        self.boxes = FakeBoxes(people)
        self.keypoints = FakeKeypoints(people)


class FakeModel:
    """Returns the same people for every frame and records the calls."""

    def __init__(self):
        self.people = []  # list of (bbox, person_conf, keypoints)
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.people)]


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(pose, "load_model", lambda path: model)
    return model


ZONES = ZoneMap({
    "sofa": [[0, 0], [320, 0], [320, 480], [0, 480]],
    "floor": [[320, 0], [640, 0], [640, 480], [320, 480]],
})


def test_person_becomes_an_observation_with_zone_and_details(fake_model):
    fake_model.people = [(STANDING_BOX, 0.8, STANDING)]  # centre (320, 280) -> on the zone edge, sofa wins
    detector = PoseDetector(every_n_frames=1, zone_map=ZONES)

    [obs] = detector.detect(FRAME, timestamp=100.0)

    assert obs.source == "pose"
    assert obs.label == LABEL_STANDING
    assert obs.time == 100.0
    assert obs.confidence == pytest.approx(0.9 * 0.8)
    assert obs.zone == "sofa"
    assert obs.data["bbox"] == STANDING_BOX
    assert len(obs.data["keypoints"]) == 17
    assert obs.data["person_conf"] == pytest.approx(0.8)
    assert obs.data["movement"] is None  # first sighting: nothing to compare with
    assert "torso_angle" in obs.data and "knee_ratio" in obs.data


def test_lying_person_in_the_floor_zone(fake_model):
    box = [400, 380, 600, 460]  # centre (500, 420) -> floor
    fake_model.people = [(box, 0.9, LYING)]
    [obs] = PoseDetector(every_n_frames=1, zone_map=ZONES).detect(FRAME, 1.0)
    assert obs.label == LABEL_LYING
    assert obs.zone == "floor"


def test_skipped_frames_return_none_and_do_not_call_the_model(fake_model):
    fake_model.people = [(STANDING_BOX, 0.8, STANDING)]
    detector = PoseDetector(every_n_frames=3)

    results = [detector.detect(FRAME, float(i)) for i in range(6)]

    assert [r is None for r in results] == [False, True, True, False, True, True]
    assert len(fake_model.calls) == 2


def test_nobody_found_is_an_empty_list_not_none(fake_model):
    fake_model.people = []
    assert PoseDetector(every_n_frames=1).detect(FRAME, 1.0) == []


def test_missing_frame_is_none_and_not_counted(fake_model):
    detector = PoseDetector(every_n_frames=2)
    assert detector.detect(None, 1.0) is None
    assert fake_model.calls == []
    assert detector.detect(FRAME, 2.0) is not None  # still the first real frame


def test_person_below_min_conf_is_ignored(fake_model):
    fake_model.people = [(STANDING_BOX, 0.3, STANDING)]
    assert PoseDetector(every_n_frames=1, min_conf=0.4).detect(FRAME, 1.0) == []


def test_only_the_largest_person_is_reported_by_default(fake_model):
    small_box = [10, 10, 50, 90]
    fake_model.people = [(small_box, 0.9, STANDING), (LYING_BOX, 0.9, LYING)]
    observations = PoseDetector(every_n_frames=1).detect(FRAME, 1.0)
    assert [o.label for o in observations] == [LABEL_LYING]

    both = PoseDetector(every_n_frames=1, max_persons=2).detect(FRAME, 1.0)
    assert len(both) == 2
    assert both[0].label == LABEL_LYING  # largest first


def test_movement_is_measured_between_analysed_frames(fake_model):
    detector = PoseDetector(every_n_frames=1)
    box_a = [280, 100, 360, 460]
    box_b = [330, 100, 410, 460]  # 50 px to the right
    diagonal = float(np.hypot(80, 360))

    fake_model.people = [(box_a, 0.9, STANDING)]
    first = detector.detect(FRAME, 10.0)[0]
    fake_model.people = [(box_b, 0.9, STANDING)]
    second = detector.detect(FRAME, 10.5)[0]  # half a second later

    assert first.data["movement"] is None
    assert second.data["movement"] == pytest.approx(50 / diagonal / 0.5)


def test_no_movement_for_a_person_who_stays_still(fake_model):
    detector = PoseDetector(every_n_frames=1)
    fake_model.people = [(LYING_BOX, 0.9, LYING)]
    detector.detect(FRAME, 1.0)
    [obs] = detector.detect(FRAME, 1.5)
    assert obs.data["movement"] == pytest.approx(0.0)


def test_movement_resets_after_nobody_or_a_long_gap(fake_model):
    detector = PoseDetector(every_n_frames=1)
    fake_model.people = [(STANDING_BOX, 0.9, STANDING)]
    detector.detect(FRAME, 1.0)

    fake_model.people = []
    detector.detect(FRAME, 1.5)  # nobody: forget the previous position
    fake_model.people = [(STANDING_BOX, 0.9, STANDING)]
    assert detector.detect(FRAME, 2.0)[0].data["movement"] is None

    # A gap longer than MOVEMENT_MAX_GAP_S also gives "not known".
    assert detector.detect(FRAME, 2.0 + pose.MOVEMENT_MAX_GAP_S + 1)[0].data["movement"] is None


def test_result_without_keypoints_is_treated_as_nobody(fake_model):
    class NoKeypoints:
        boxes = FakeBoxes([])
        keypoints = None

    fake_model.predict = lambda frame, **kw: [NoKeypoints()]
    assert PoseDetector(every_n_frames=1).detect(FRAME, 1.0) == []


def test_load_runs_a_warm_up_check(fake_model):
    PoseDetector(image_size=320).load()
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["imgsz"] == 320


# ----------------------------------------------------------------- config


def test_defaults_contain_a_pose_section(tmp_path):
    config = load_config(tmp_path / "missing.yaml")
    assert config["pose"]["every_n_frames"] == 3
    assert config["pose"]["lying_torso_angle"] == 60


def test_from_config_reads_the_pose_section_and_keeps_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("pose:\n  lying_torso_angle: 70\n  max_persons: 2\n", encoding="utf-8")
    config = load_config(path)

    detector = PoseDetector.from_config(config)

    assert detector.lying_torso_angle == 70
    assert detector.max_persons == 2
    assert detector.every_n_frames == 3  # default kept
    assert detector.model_path == "models/yolo11s-pose.pt"


def test_the_real_config_yaml_loads_for_pose():
    detector = PoseDetector.from_config(load_config("config.yaml"))
    assert detector.every_n_frames >= 1
    assert detector.sit_knee_ratio > 0


# ------------------------------------------------------------------ drawing


def test_draw_poses_returns_a_copy_and_does_not_touch_the_original(fake_model):
    fake_model.people = [(STANDING_BOX, 0.8, STANDING)]
    observations = PoseDetector(every_n_frames=1).detect(FRAME, 1.0)

    canvas = draw_poses(FRAME, observations)

    assert canvas.shape == FRAME.shape
    assert canvas.any()  # something was drawn
    assert not FRAME.any()  # original untouched
