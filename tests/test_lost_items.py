"""Tests for UC2 (lifeaware.usecases.lost_items) with fake memory contents."""

from datetime import datetime

import pytest

from lifeaware.memory import ItemMemory
from lifeaware.schema import Observation
from lifeaware.usecases.lost_items import handle_query, handle_text_query, parse_item_query

NOW = datetime(2026, 11, 16, 15, 0).timestamp()
SEEN = datetime(2026, 11, 16, 14, 2).timestamp()


@pytest.fixture
def memory():
    mem = ItemMemory()
    mem.update(Observation(time=SEEN, source="objects", label="phone", confidence=0.9, zone="sofa"))
    return mem


@pytest.mark.parametrize(
    "text, item",
    [
        ("Where is my phone?", "phone"),
        ("where's my mobile", "phone"),
        ("Have you seen my cell phone?", "phone"),
        ("Where are my keys?", "keys"),
        ("I lost my key", "keys"),
        ("Where is my wallet?", None),
        ("", None),
        ("telephone bill", None),  # "phone" must be a whole word
    ],
)
def test_parse_item_query(text, item):
    assert parse_item_query(text) == item


def test_found_item_gives_solved_event(memory):
    event = handle_query("phone", memory, now=NOW)

    assert event.use_case == "UC2"
    assert event.new_state == "solved"
    assert event.reason == "Your phone was last seen on the sofa at 14:02."
    assert event.time == NOW


def test_unseen_item_gives_unknown_event(memory):
    event = handle_query("keys", memory, now=NOW)

    assert event.new_state == "unknown"
    assert "not sure" in event.reason


def test_answer_is_spoken(memory):
    spoken = []

    event = handle_query("phone", memory, now=NOW, say=spoken.append)

    assert spoken == [event.reason]


def test_broken_voice_output_does_not_crash(memory):
    def broken_say(text):
        raise RuntimeError("no speakers")

    event = handle_query("phone", memory, now=NOW, say=broken_say)

    assert event.new_state == "solved"


def test_text_query_for_known_item(memory):
    assert handle_text_query("where is my phone?", memory, now=NOW).new_state == "solved"


def test_text_query_for_unknown_item_is_honest(memory):
    spoken = []

    event = handle_text_query("where is my wallet?", memory, now=NOW, say=spoken.append)

    assert event.new_state == "unknown"
    assert "only look for your phone or your keys" in event.reason
    assert spoken == [event.reason]


def test_answer_card_escapes_html_and_marks_found_items(memory):
    from lifeaware.usecases.lost_items import _answer_card

    from lifeaware.schema import Event

    found = _answer_card(handle_query("phone", memory, now=NOW))
    tricky = _answer_card(Event(time=NOW, use_case="UC2", old_state="checking",
                                new_state="unknown", reason="<script>x</script>"))

    assert 'class="uc2-answer solved"' in found
    assert "on the sofa at 14:02" in found
    assert "<script>" not in tricky and "&lt;script&gt;" in tricky
