"""UC3 – Fall check.

A person lies on the floor and hardly moves for a while. The system asks
"Are you okay?" and listens. No answer (with working camera AND microphone)
-> local alarm. A device that does not work -> "unknown", never an alarm.
It does not diagnose injuries and does not call emergency services.

States (the ``new_state`` of every ``Event``)::

    normal ──lying on floor ≥ min_lying_s──▶ possible_event
    possible_event ──and still ≥ still_s──▶ checking   (asks "Are you okay?")
    possible_event ──gets up / lies on sofa──▶ normal
    possible_event ──person not visible > max_gap_s──▶ unknown
    checking ──"yes" / gets up──▶ solved
    checking ──"no" / "help"──▶ alarm
    checking ──timeout, no answer, mic ok──▶ alarm
    checking ──timeout, mic not ok or reply not understood──▶ unknown
    alarm, unknown ──acknowledge()──▶ solved
    unknown ──camera works again, person upright──▶ normal
    solved ──after cooldown_s or person upright──▶ normal
    any state ──camera_lost──▶ unknown;  any state ──"help"──▶ alarm

Inputs (all are ``Observation``s, timestamps decide, so replays are repeatable):

- ``feed_pose(result, now)`` with the return value of ``PoseDetector.detect``:
  ``None`` = frame skipped (ignored), ``[]`` = nobody found, else postures.
- ``feed(obs)`` for single observations: ``camera_lost`` (source "camera"),
  ``mic_ok`` / ``mic_lost`` and answers ``answer_yes`` / ``answer_no`` /
  ``answer_help`` / ``answer_unknown`` (source "audio").
- ``tick(now)`` lets the timers run out (call it every frame or every second).
  Timers only end things (answer timeout, lost sight, cooldown). Moving
  towards a question always needs a real sighting of the person lying.

Every method returns the list of ``Event``s it caused (usually empty).

"Nothing seen" is not "not moving": stillness only counts while the person is
actually seen lying. A missing ``movement`` value never counts as still.
"""

from __future__ import annotations

from typing import Callable, Iterable

from lifeaware.pose import LABEL_LYING, LABEL_SITTING, LABEL_STANDING
from lifeaware.schema import Event, Observation

USE_CASE = "UC3"

NORMAL = "normal"
POSSIBLE_EVENT = "possible_event"
CHECKING = "checking"
ALARM = "alarm"
SOLVED = "solved"
UNKNOWN = "unknown"

QUESTION = "Are you okay? Please say yes or no."
ALARM_TEXT = "I am raising an alarm so someone can come and help you."

UPRIGHT = {LABEL_STANDING, LABEL_SITTING}

DEFAULTS = {
    "floor_zones": ["floor"],      # lying in these zones is suspicious (sofa, bed are fine)
    "unknown_zone_counts": True,   # lying outside every zone counts as "on the floor"
    "min_conf": 0.3,               # ignore pose observations below this
    "min_lying_s": 3.0,            # lying on the floor this long -> possible_event
    "still_s": 10.0,               # ... and hardly moving this long -> ask
    "still_movement": 0.15,        # movement (body-diagonals/s) at or below this = still
    "max_gap_s": 3.0,              # person not seen this long during possible_event -> unknown
    "answer_timeout_s": 15.0,      # wait this long for an answer
    "cooldown_s": 60.0,            # after "yes, I'm okay": do not ask again for this long
    "microphone": True,            # is a microphone (or the answer buttons) connected?
}


class FallCheck:
    """State machine for UC3. Pure logic: no camera, model or sound needed."""

    def __init__(self, say: Callable[[str], None] | None = None, **settings) -> None:
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown fall-check settings: {sorted(unknown)}")
        self.settings = {**DEFAULTS, **settings}
        self.say = say  # voice output; a failing voice never stops the check
        self.mic_ok = bool(self.settings["microphone"])

        self.state = NORMAL
        self.events: list[Event] = []  # full history, newest last
        self._camera_ok = True
        self._lying_since: float | None = None   # start of the current lying-on-floor period
        self._moving_at: float | None = None     # last time the lying person moved
        self._last_lying: float | None = None    # last time the person was seen lying on the floor
        self._asked_at: float | None = None      # when "Are you okay?" was asked
        self._heard_unclear = False              # a reply came but was not understood
        self._solved_at: float | None = None

    @classmethod
    def from_config(cls, config: dict, say: Callable[[str], None] | None = None) -> "FallCheck":
        section = {k: v for k, v in config.get("fall", {}).items() if k in DEFAULTS}
        return cls(say=say, **section)

    # ------------------------------------------------------------- helpers

    def _set(self, now: float, new_state: str, reason: str, answered: str | None = None) -> list[Event]:
        if new_state == self.state:
            return []
        event = Event(time=now, use_case=USE_CASE, old_state=self.state,
                      new_state=new_state, reason=reason, answered=answered)
        self.state = new_state
        self.events.append(event)
        if new_state in (NORMAL, SOLVED, UNKNOWN, ALARM):
            self._lying_since = self._moving_at = self._last_lying = None
        if new_state != CHECKING:
            self._asked_at = None
            self._heard_unclear = False
        if new_state == SOLVED:
            self._solved_at = now
        if new_state == ALARM:
            self._speak(ALARM_TEXT)
        return [event]

    def _speak(self, text: str) -> None:
        if self.say is not None:
            try:
                self.say(text)
            except Exception:
                pass  # no sound is better than a crash; the dashboard still shows the state

    def _on_floor(self, zone: str | None) -> bool:
        if zone is None:
            return bool(self.settings["unknown_zone_counts"])
        return zone in self.settings["floor_zones"]

    # -------------------------------------------------------------- inputs

    def feed_pose(self, result: list[Observation] | None, now: float) -> list[Event]:
        """Feed one ``PoseDetector.detect`` result (None = skipped frame)."""
        if result is None:
            return self.tick(now)  # a skipped frame is no evidence at all
        events: list[Event] = []
        if not result:
            return events + self.tick(now)  # nobody found: only the gap timer can react
        for obs in result:
            events += self.feed(obs)
        return events + self.tick(now)

    def feed(self, obs: Observation) -> list[Event]:
        """Feed one observation (pose, camera status, microphone, answer)."""
        if obs.source == "camera" and obs.label == "camera_lost":
            self._camera_ok = False
            return self._set(obs.time, UNKNOWN, "The camera stopped working, so I cannot see the room.")
        if obs.source == "audio":
            return self._feed_audio(obs)
        if obs.source == "pose":
            return self._feed_pose_obs(obs)
        return []

    def _feed_audio(self, obs: Observation) -> list[Event]:
        now, label = obs.time, obs.label
        if label == "mic_ok":
            self.mic_ok = True
            return []
        if label == "mic_lost":
            self.mic_ok = False
            return []
        if label == "answer_help":
            return self._set(now, ALARM, "The person asked for help.", answered="help")
        if self.state != CHECKING:
            return []  # yes/no only mean something as an answer to our question
        if label == "answer_yes":
            return self._set(now, SOLVED, "The person said they are okay.", answered="yes")
        if label == "answer_no":
            return self._set(now, ALARM, "The person said they are not okay.", answered="no")
        if label == "answer_unknown":
            self._heard_unclear = True  # "not understood" is not "no answer"
        return []

    def _feed_pose_obs(self, obs: Observation) -> list[Event]:
        if obs.confidence < self.settings["min_conf"]:
            return []
        now = obs.time
        self._camera_ok = True
        upright = obs.label in UPRIGHT
        lying_on_floor = obs.label == LABEL_LYING and self._on_floor(obs.zone)

        if self.state == UNKNOWN:
            if upright:
                return self._set(now, NORMAL, f"The camera works again and the person is {_word(obs.label)}.")
            return []
        if self.state == SOLVED:
            if upright:
                return self._set(now, NORMAL, f"The person is {_word(obs.label)} again.")
            return []
        if self.state == ALARM:
            return []  # an alarm stays until a person acknowledges it

        if self.state == CHECKING:
            if upright:
                return self._set(now, SOLVED, f"The person got up ({_word(obs.label)}).")
            return []

        # normal / possible_event
        if not lying_on_floor:
            if upright or obs.label == LABEL_LYING:  # clearly not on the floor any more
                if self.state == POSSIBLE_EVENT:
                    where = f"is {_word(obs.label)}" if upright else f"is lying on the {obs.zone}"
                    return self._set(now, NORMAL, f"The person {where}, not on the floor.")
                self._lying_since = self._moving_at = self._last_lying = None
            return []  # "unclear" changes nothing

        if self._lying_since is None:
            self._lying_since = self._moving_at = now
        self._last_lying = now
        movement = obs.data.get("movement")
        if movement is None or movement > self.settings["still_movement"]:
            self._moving_at = now  # moving, or not known -> stillness starts again
        return self._advance(now)

    # -------------------------------------------------------------- timers

    def tick(self, now: float) -> list[Event]:
        """Let timers run out. Safe to call as often as you like."""
        s = self.settings
        if self.state == SOLVED and self._solved_at is not None and now - self._solved_at >= s["cooldown_s"]:
            return self._set(now, NORMAL, "Waiting time after the last check is over.")
        if self.state == POSSIBLE_EVENT and self._last_lying is not None \
                and now - self._last_lying > s["max_gap_s"]:
            return self._set(now, UNKNOWN, "I lost sight of the person who was lying on the floor.")
        if self.state == CHECKING and self._asked_at is not None \
                and now - self._asked_at >= s["answer_timeout_s"]:
            return self._timeout(now)
        return []  # moving forward needs a real sighting (see _feed_pose_obs)

    def _advance(self, now: float) -> list[Event]:
        """normal -> possible_event -> checking, based on how long the person lay still."""
        if self._lying_since is None or self.state not in (NORMAL, POSSIBLE_EVENT):
            return []
        s = self.settings
        lying_for = now - self._lying_since
        still_for = now - self._moving_at
        events: list[Event] = []
        if self.state == NORMAL and lying_for >= s["min_lying_s"]:
            events += self._set(now, POSSIBLE_EVENT,
                                f"The person has been lying on the floor for {lying_for:.0f} s.")
        if self.state == POSSIBLE_EVENT and lying_for >= s["min_lying_s"] and still_for >= s["still_s"]:
            events += self._set(now, CHECKING,
                                f"The person has been lying on the floor without moving for "
                                f"{still_for:.0f} s. Asking: \"{QUESTION}\"")
            self._asked_at = now
            self._speak(QUESTION)
        return events

    def _timeout(self, now: float) -> list[Event]:
        waited = now - self._asked_at
        if self._heard_unclear:
            return self._set(now, UNKNOWN, f"I heard a reply but could not understand it "
                                           f"(waited {waited:.0f} s). Please check on the person.",
                             answered="unknown")
        if not self.mic_ok:
            return self._set(now, UNKNOWN, "The microphone does not work, so I cannot hear an answer. "
                                           "Please check on the person.", answered="unknown")
        if not self._camera_ok:
            return self._set(now, UNKNOWN, "The camera does not work. Please check on the person.")
        return self._set(now, ALARM, f"No answer for {waited:.0f} s while the person lies still on the floor.")

    # ------------------------------------------------------------ helpers

    def acknowledge(self, now: float, who: str = "helper") -> list[Event]:
        """A person confirmed they took care of it (dashboard button)."""
        if self.state in (ALARM, UNKNOWN):
            return self._set(now, SOLVED, f"Acknowledged by the {who}.")
        return []

    def run(self, steps: Iterable[tuple[float, list[Observation] | Observation | None]]) -> list[Event]:
        """Replay a scenario: each step is ``(time, pose_result | single_obs | None)``.

        Handy for tests and the evaluation (labelled scenario -> expected events).
        """
        events: list[Event] = []
        for now, item in steps:
            if isinstance(item, Observation):
                events += self.feed(item) + self.tick(now)
            else:
                events += self.feed_pose(item, now)
        return events


def _word(label: str) -> str:
    return label.replace("person_", "")
