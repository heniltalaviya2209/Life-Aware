"""Tests for lifeaware.usecases.fall_check (UC3) with FAKE observations only."""

import pytest

from lifeaware.config import load_config
from lifeaware.schema import Observation
from lifeaware.usecases import fall_check as fc
from lifeaware.usecases.fall_check import FallCheck

# Short timers keep the scenarios readable: lying 3 s -> possible, still 10 s -> ask, 15 s answer.
SETTINGS = dict(min_lying_s=3, still_s=10, answer_timeout_s=15, max_gap_s=3, cooldown_s=60)


def pose(t, label="person_lying", zone="floor", movement=0.0, conf=0.8):
    return [Observation(time=t, source="pose", label=label, confidence=conf, zone=zone,
                        data={"movement": movement})]


def audio(t, label):
    return Observation(time=t, source="audio", label=label, confidence=1.0)


def lie_still(check, start, end, step=0.5, **kw):
    """Person lies still on the floor from ``start`` to ``end`` (one analysed frame per step)."""
    events = []
    t = start
    while t <= end + 1e-9:
        events += check.feed_pose(pose(t, **kw), t)
        t += step
    return events


def states(events):
    return [e.new_state for e in events]


@pytest.fixture
def spoken():
    return []


@pytest.fixture
def check(spoken):
    return FallCheck(say=spoken.append, **SETTINGS)


# ------------------------------------------------------------ main flow


def test_lying_still_leads_to_question_then_alarm(check, spoken):
    events = lie_still(check, 0, 10)
    assert states(events) == ["possible_event", "checking"]
    assert events[0].time == pytest.approx(3)       # lying 3 s
    assert events[1].time == pytest.approx(10)      # still 10 s since first seen
    assert spoken == [fc.QUESTION]

    events = lie_still(check, 10.5, 25)             # nobody answers for 15 s
    assert states(events) == ["alarm"]
    assert events[0].time == pytest.approx(25)
    assert events[0].old_state == "checking"
    assert "No answer" in events[0].reason
    assert spoken[-1] == fc.ALARM_TEXT


def test_every_event_has_use_case_reason_and_old_state(check):
    events = lie_still(check, 0, 30)
    assert [(e.old_state, e.new_state) for e in events] == [
        ("normal", "possible_event"), ("possible_event", "checking"), ("checking", "alarm")]
    assert all(e.use_case == "UC3" and e.reason for e in events)
    assert check.events == events


def test_answer_yes_solves_and_cooldown_prevents_asking_again(check):
    lie_still(check, 0, 10)
    [event] = check.feed(audio(12, "answer_yes"))
    assert (event.new_state, event.answered) == ("solved", "yes")

    # Still lying (e.g. floor exercises): no new question during the cooldown.
    assert lie_still(check, 12.5, 70) == []
    # After the cooldown the check starts again from normal.
    events = lie_still(check, 72, 90)
    assert states(events)[:3] == ["normal", "possible_event", "checking"]


def test_answer_no_or_help_raises_alarm_at_once(check):
    lie_still(check, 0, 10)
    [event] = check.feed(audio(11, "answer_no"))
    assert (event.new_state, event.answered) == ("alarm", "no")

    other = FallCheck(**SETTINGS)
    lie_still(other, 0, 10)
    [event] = other.feed(audio(11, "answer_help"))
    assert (event.new_state, event.answered) == ("alarm", "help")


def test_help_raises_alarm_even_without_a_question(check):
    [event] = check.feed(audio(5, "answer_help"))
    assert event.old_state == "normal" and event.new_state == "alarm"


def test_yes_or_no_without_a_question_is_ignored(check):
    assert check.feed(audio(1, "answer_no")) == []
    assert check.feed(audio(2, "answer_yes")) == []
    assert check.state == "normal"


def test_getting_up_while_asked_solves(check):
    lie_still(check, 0, 10)
    [event] = check.feed_pose(pose(14, label="person_standing", zone=None), 14)
    assert event.new_state == "solved"
    assert "got up" in event.reason
    # Standing again after "solved" returns to normal immediately.
    [event] = check.feed_pose(pose(15, label="person_standing", zone=None), 15)
    assert event.new_state == "normal"


# ---------------------------------------------- normal look-alike activities


def test_standing_up_resets_the_lying_timer(check):
    lie_still(check, 0, 2)
    check.feed_pose(pose(2.5, label="person_standing", zone=None), 2.5)
    events = lie_still(check, 3, 5)  # lies down again: 2 s only, counted from 3.0
    assert events == []


def test_short_lying_then_getting_up_never_asks(check, spoken):
    events = lie_still(check, 0, 2)                                     # 2 s < min_lying_s
    events += check.feed_pose(pose(2.5, label="person_standing"), 2.5)
    events += lie_still(check, 3, 20, label="person_standing", zone=None)
    assert events == []
    assert spoken == []


def test_lying_on_the_sofa_is_normal(check):
    assert lie_still(check, 0, 60, zone="sofa") == []


def test_getting_up_during_possible_event_returns_to_normal(check):
    events = lie_still(check, 0, 5)
    events += check.feed_pose(pose(5.5, label="person_sitting", zone="floor"), 5.5)
    assert states(events) == ["possible_event", "normal"]
    assert "sitting" in events[-1].reason


def test_moving_on_the_floor_delays_the_question(check):
    # Lying but moving (e.g. exercising) for 20 s: possible_event, but never "checking".
    events = lie_still(check, 0, 20, movement=0.5)
    assert states(events) == ["possible_event"]
    # Then lying still: the 10 s stillness count starts only now.
    # Last movement was seen at 20.0 s, so the question comes 10 s later.
    events = lie_still(check, 20.5, 29.5)
    assert states(events) == []
    events = lie_still(check, 30, 31)
    assert states(events) == ["checking"]
    assert events[0].time == pytest.approx(30)


def test_unknown_movement_never_counts_as_still(check):
    events = lie_still(check, 0, 30, movement=None)
    assert states(events) == ["possible_event"]


def test_weak_pose_observations_are_ignored(check):
    assert lie_still(check, 0, 30, conf=0.1) == []


def test_unknown_zone_counts_as_floor_by_default_but_can_be_switched_off():
    events = lie_still(FallCheck(**SETTINGS), 0, 10, zone=None)
    assert states(events) == ["possible_event", "checking"]

    strict = FallCheck(unknown_zone_counts=False, **SETTINGS)
    assert lie_still(strict, 0, 30, zone=None) == []


# --------------------------------------------- missing input and uncertainty


def test_skipped_frames_are_no_evidence(check):
    lie_still(check, 0, 2)
    for t in (2.5, 3.0, 3.5):
        assert check.feed_pose(None, t) == []
    assert check.state == "normal"  # skipped frames cannot move the check forward
    # The next real sighting does.
    [event] = check.feed_pose(pose(4.0), 4.0)
    assert event.new_state == "possible_event"


def test_person_disappears_during_possible_event_gives_unknown(check):
    lie_still(check, 0, 5)
    events = []
    for t in (5.5, 6.5, 7.5, 8.5):
        events += check.feed_pose([], t)  # nobody found (hidden / walked out)
    assert states(events) == ["unknown"]
    assert "lost sight" in events[0].reason
    assert check.state != "alarm"


def test_camera_lost_gives_unknown_never_alarm(check):
    lie_still(check, 0, 10)                                   # already asking
    [event] = check.feed(Observation(time=11, source="camera", label="camera_lost", confidence=1.0))
    assert event.new_state == "unknown"
    assert check.tick(100) == []                              # timeout cannot raise an alarm now


def test_camera_lost_while_normal_gives_unknown_and_recovers(check):
    [event] = check.feed(Observation(time=1, source="camera", label="camera_lost", confidence=1.0))
    assert event.new_state == "unknown"
    # Lying is not enough to leave "unknown"; an upright person is.
    assert check.feed_pose(pose(2), 2) == []
    [event] = check.feed_pose(pose(3, label="person_standing", zone=None), 3)
    assert event.new_state == "normal"


def test_no_microphone_gives_unknown_instead_of_alarm():
    check = FallCheck(microphone=False, **SETTINGS)
    events = lie_still(check, 0, 30)
    assert states(events) == ["possible_event", "checking", "unknown"]
    assert "microphone" in events[-1].reason


def test_microphone_lost_during_question_gives_unknown(check):
    lie_still(check, 0, 10)
    check.feed(audio(11, "mic_lost"))
    events = lie_still(check, 11.5, 30)
    assert states(events) == ["unknown"]
    # ... and a microphone that comes back makes alarms possible again.
    check.feed(audio(31, "mic_ok"))
    assert check.mic_ok


def test_reply_not_understood_is_unknown_not_no_answer(check):
    lie_still(check, 0, 10)
    assert check.feed(audio(12, "answer_unknown")) == []      # keep waiting for a clear answer
    events = lie_still(check, 12.5, 30)
    assert states(events) == ["unknown"]
    assert events[0].answered == "unknown"


def test_clear_answer_after_an_unclear_one_still_counts(check):
    lie_still(check, 0, 10)
    check.feed(audio(12, "answer_unknown"))
    [event] = check.feed(audio(14, "answer_yes"))
    assert event.new_state == "solved"


# ------------------------------------------------------- alarm handling


def test_alarm_stays_until_acknowledged(check):
    lie_still(check, 0, 25)
    assert check.state == "alarm"
    # Even getting up does not silently clear an alarm: a human confirms.
    assert check.feed_pose(pose(30, label="person_standing"), 30) == []
    [event] = check.acknowledge(31, who="caregiver")
    assert event.new_state == "solved"
    assert "caregiver" in event.reason


def test_acknowledge_does_nothing_in_normal_state(check):
    assert check.acknowledge(1) == []


def test_alarm_is_raised_only_once(check, spoken):
    events = lie_still(check, 0, 60)
    assert states(events).count("alarm") == 1
    assert spoken.count(fc.ALARM_TEXT) == 1


def test_a_failing_voice_does_not_stop_the_check():
    def broken(text):
        raise RuntimeError("no speaker")

    check = FallCheck(say=broken, **SETTINGS)
    assert states(lie_still(check, 0, 25)) == ["possible_event", "checking", "alarm"]


# ------------------------------------------------------------- replay helper


def test_run_replays_a_labelled_scenario():
    steps = [(t * 0.5, pose(t * 0.5)) for t in range(0, 21)]   # lying still 0..10 s
    steps.append((12.0, audio(12.0, "answer_yes")))
    events = FallCheck(**SETTINGS).run(steps)
    assert states(events) == ["possible_event", "checking", "solved"]


# -------------------------------------------------------------- config


def test_unknown_setting_is_an_error():
    with pytest.raises(ValueError):
        FallCheck(answer_timout_s=5)  # typo


def test_defaults_contain_a_fall_section(tmp_path):
    config = load_config(tmp_path / "missing.yaml")
    assert config["fall"]["answer_timeout_s"] == fc.DEFAULTS["answer_timeout_s"]


def test_from_config_reads_the_fall_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("fall:\n  still_s: 4\n  floor_zones: [floor, hallway]\n", encoding="utf-8")
    check = FallCheck.from_config(load_config(path))
    assert check.settings["still_s"] == 4
    assert check.settings["floor_zones"] == ["floor", "hallway"]
    assert check.settings["answer_timeout_s"] == fc.DEFAULTS["answer_timeout_s"]


def test_the_real_config_yaml_loads_for_the_fall_check():
    check = FallCheck.from_config(load_config("config.yaml"))
    assert check.settings["still_s"] > 0
