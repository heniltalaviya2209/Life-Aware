"""Memory of what the system has seen.

``ItemMemory`` keeps the LAST sighting of each item (phone, keys, ...).
A last sighting is not a promise that the item is still there, and the
answers say so in plain words.

(``ShortTermMemory`` for the other use cases will be added here later.)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import NamedTuple

from lifeaware.schema import Observation

# How to say "<preposition> the <zone>" for zones where "on the" sounds wrong.
_ZONE_PHRASES = {
    "kitchen": "in the kitchen",
    "doorway": "at the doorway",
    "shelf": "on the shelf",
}


class Sighting(NamedTuple):
    """Last sighting of one item. A tuple: ``(time, zone, confidence)``."""

    time: float
    zone: str | None
    confidence: float


class ItemMemory:
    """Remember where and when each item was last seen."""

    def __init__(self, min_conf: float = 0.4, sure_conf: float = 0.6) -> None:
        self.min_conf = min_conf  # sightings below this are ignored
        self.sure_conf = sure_conf  # below this, answers say "I'm not sure"
        self._last: dict[str, Sighting] = {}

    @classmethod
    def from_config(cls, config: dict) -> "ItemMemory":
        """Build from the ``items`` section of ``load_config()``."""
        section = config.get("items", {})
        return cls(section.get("min_conf", 0.4), section.get("sure_conf", 0.6))

    def update(self, obs: Observation) -> bool:
        """Store an object sighting. Returns True if it was stored."""
        if obs.source != "objects" or obs.confidence < self.min_conf:
            return False
        previous = self._last.get(obs.label)
        if previous is not None and obs.time < previous.time:
            return False  # an older observation never overwrites a newer one
        self._last[obs.label] = Sighting(obs.time, obs.zone, obs.confidence)
        return True

    def last_seen(self, item: str) -> Sighting | None:
        """``(time, zone, confidence)`` of the last sighting, or None."""
        return self._last.get(item)

    def items(self) -> dict[str, Sighting]:
        """All remembered items (copy), e.g. for a dashboard table."""
        return dict(self._last)

    def answer(self, item: str, now: float | None = None) -> str:
        """A plain-language answer to "where is my <item>?"."""
        sighting = self.last_seen(item)
        if sighting is None:
            return f"I have not seen your {item} — I'm not sure where {_pronoun(item)} {_is(item)}."

        when = _format_when(sighting.time, now if now is not None else time.time())
        if sighting.zone is None:
            where = "but I don't know in which part of the room"
            sentence = f"Your {item} {_was(item)} last seen {when}, {where}."
        else:
            where = _ZONE_PHRASES.get(sighting.zone, f"on the {sighting.zone}")
            sentence = f"Your {item} {_was(item)} last seen {where} {when}."

        if sighting.confidence < self.sure_conf:
            sentence = f"I think {sentence[0].lower()}{sentence[1:-1]}, but I'm not sure."
        return sentence


def _format_when(seen: float, now: float) -> str:
    """'at 14:02', 'yesterday at 14:02' or 'on 14 Nov at 14:02'."""
    seen_dt, now_dt = datetime.fromtimestamp(seen), datetime.fromtimestamp(now)
    clock = seen_dt.strftime("%H:%M")
    days_ago = (now_dt.date() - seen_dt.date()).days
    if days_ago == 0:
        return f"at {clock}"
    if days_ago == 1:
        return f"yesterday at {clock}"
    return f"on {seen_dt.day} {seen_dt.strftime('%b')} at {clock}"


def _is_plural(item: str) -> bool:
    return item.endswith("s") and not item.endswith("ss")


def _was(item: str) -> str:
    return "were" if _is_plural(item) else "was"


def _is(item: str) -> str:
    return "are" if _is_plural(item) else "is"


def _pronoun(item: str) -> str:
    return "they" if _is_plural(item) else "it"
