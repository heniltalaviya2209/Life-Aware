"""Tests for lifeaware.memory.ItemMemory with fake observations."""

from datetime import datetime

from lifeaware.memory import ItemMemory
from lifeaware.schema import Observation


def at(hour, minute, day=16):
    """Local timestamp on Nov <day> 2026."""
    return datetime(2026, 11, day, hour, minute).timestamp()


def phone(time, zone="sofa", conf=0.9):
    return Observation(time=time, source="objects", label="phone", confidence=conf, zone=zone)


def test_seen_once():
    memory = ItemMemory()
    memory.update(phone(at(14, 2)))

    assert memory.last_seen("phone") == (at(14, 2), "sofa", 0.9)
    assert memory.answer("phone", now=at(15, 0)) == "Your phone was last seen on the sofa at 14:02."


def test_seen_again_in_another_zone_keeps_the_latest():
    memory = ItemMemory()
    memory.update(phone(at(14, 2), zone="sofa"))
    memory.update(phone(at(14, 30), zone="table"))

    assert memory.last_seen("phone").zone == "table"
    assert "on the table at 14:30" in memory.answer("phone", now=at(15, 0))


def test_older_observation_does_not_overwrite_newer():
    memory = ItemMemory()
    memory.update(phone(at(14, 30), zone="table"))

    assert memory.update(phone(at(14, 2), zone="sofa")) is False
    assert memory.last_seen("phone").zone == "table"


def test_never_seen():
    memory = ItemMemory()

    assert memory.last_seen("keys") is None
    assert memory.answer("keys") == "I have not seen your keys — I'm not sure where they are."
    assert memory.answer("phone") == "I have not seen your phone — I'm not sure where it is."


def test_low_confidence_is_ignored():
    memory = ItemMemory(min_conf=0.4)

    assert memory.update(phone(at(14, 2), conf=0.39)) is False
    assert memory.last_seen("phone") is None


def test_low_confidence_does_not_hide_a_good_sighting():
    memory = ItemMemory(min_conf=0.4)
    memory.update(phone(at(14, 2), zone="sofa", conf=0.9))
    memory.update(phone(at(14, 30), zone="floor", conf=0.2))

    assert memory.last_seen("phone").zone == "sofa"


def test_non_object_observations_are_ignored():
    memory = ItemMemory()
    camera_lost = Observation(time=at(14, 0), source="camera", label="camera_lost", confidence=1.0)

    assert memory.update(camera_lost) is False
    assert memory.items() == {}


def test_unsure_sighting_says_so():
    memory = ItemMemory(min_conf=0.4, sure_conf=0.6)
    memory.update(phone(at(14, 2), conf=0.5))

    assert memory.answer("phone", now=at(15, 0)) == (
        "I think your phone was last seen on the sofa at 14:02, but I'm not sure."
    )


def test_unknown_zone_is_said_honestly():
    memory = ItemMemory()
    memory.update(phone(at(14, 2), zone=None))

    answer = memory.answer("phone", now=at(15, 0))

    assert answer == "Your phone was last seen at 14:02, but I don't know in which part of the room."


def test_answer_mentions_yesterday_and_older_days():
    memory = ItemMemory()
    memory.update(phone(at(20, 15, day=15)))

    assert "yesterday at 20:15" in memory.answer("phone", now=at(9, 0, day=16))
    assert "on 15 Nov at 20:15" in memory.answer("phone", now=at(9, 0, day=18))


def test_from_config():
    memory = ItemMemory.from_config({"items": {"min_conf": 0.5, "sure_conf": 0.8}})

    assert (memory.min_conf, memory.sure_conf) == (0.5, 0.8)
