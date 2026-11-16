"""Tests for the shared Observation / Event contract."""

from lifeaware.schema import Event, Observation


def test_observation_defaults():
    obs = Observation(time=1.0, source="camera", label="camera_lost", confidence=1.0)

    assert obs.zone is None
    assert obs.data == {}


def test_observation_data_is_not_shared_between_instances():
    first = Observation(time=1.0, source="objects", label="phone", confidence=0.9)
    second = Observation(time=2.0, source="objects", label="phone", confidence=0.8)

    first.data["bbox"] = (0, 0, 10, 10)

    assert second.data == {}


def test_event_defaults():
    event = Event(
        time=1.0,
        use_case="UC2",
        old_state="normal",
        new_state="solved",
        reason="Phone last seen on the sofa.",
    )

    assert event.answered is None
