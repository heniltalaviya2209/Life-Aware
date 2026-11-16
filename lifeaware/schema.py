"""Shared data contract for the whole LifeAware chain.

Every module talks in these two types:
- ``Observation``: something a sensor or detector noticed at one moment.
- ``Event``: a use-case state change, with a plain-language reason.

Owner: Yash. Change the shape only after the team agrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """One thing noticed at one moment (e.g. "phone on the sofa, 0.82")."""

    time: float  # seconds since epoch (time.time())
    source: str  # "camera", "pose", "objects", "audio", "system"
    label: str  # e.g. "person_lying", "phone", "camera_lost", "answer_yes"
    confidence: float  # 0.0 - 1.0
    zone: str | None = None  # "sofa", "floor", "bed", ... or None if unknown
    data: dict = field(default_factory=dict)  # extra details (bbox, keypoints, movement)


@dataclass
class Event:
    """A use-case state change, shown on the dashboard and written to the log."""

    time: float
    use_case: str  # "UC1".."UC4" or "system"
    old_state: str
    new_state: str  # normal, possible_event, checking, alarm, solved, unknown
    reason: str  # plain-language explanation shown on the dashboard
    answered: str | None = None  # "yes", "no", "help", "unknown", None
