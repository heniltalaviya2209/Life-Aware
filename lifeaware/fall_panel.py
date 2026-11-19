"""Fall check (UC3) for Streamlit: live camera -> YOLO pose -> FallCheck -> dashboard.

One reusable section, shown in two places:
- ``app.py`` section H (the main dashboard),
- ``demo_fall.py`` (a standalone page, handy for testing and the mid-term demo).

A background thread runs the real chain at <= 10 fps. The page shows the latest
picture, the current UC3 state with its reason, answer buttons and the event list.

The answer buttons stand in for the speech recogniser until the audio part is
connected. Results from them must be reported as "simulated answers".
Every state change is appended to ``events/fall_events.csv`` (gitignored).
"""

from __future__ import annotations

import base64
import csv
import html
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable

import streamlit as st

from lifeaware.camera import FrameSource
from lifeaware.config import load_config
from lifeaware.pose import PoseDetector, draw_poses
from lifeaware.schema import Event, Observation
from lifeaware.usecases import fall_check as fc
from lifeaware.usecases.fall_check import FallCheck
from lifeaware.zones import ZoneMap

EVENT_LOG = os.path.join("events", "fall_events.csv")

# state -> (title shown on the card, css class, short explanation)
STATE_INFO = {
    fc.NORMAL: ("NORMAL", "ok", "Nothing unusual."),
    fc.POSSIBLE_EVENT: ("POSSIBLE EVENT", "warn", "Someone is lying on the floor. Watching for movement."),
    fc.CHECKING: ("CHECKING", "warn", "Asked “Are you okay?” – waiting for an answer."),
    fc.ALARM: ("ALARM", "bad", "Local alarm. A helper should check now and press Acknowledge."),
    fc.SOLVED: ("SOLVED", "ok", "Handled. Not asking again for a while."),
    fc.UNKNOWN: ("UNKNOWN", "grey", "The system cannot tell. A helper should take a look."),
}

_CSS = """
<style>
.fc-pill {display:inline-flex; align-items:center; gap:7px; padding:4px 12px;
  border-radius:999px; font-size:.8rem; font-weight:600; margin-bottom:8px;
  background:rgba(100,116,139,.12); border:1px solid rgba(100,116,139,.40); color:#64748b;}
.fc-pill.live {background:rgba(34,197,94,.12); border-color:rgba(34,197,94,.45); color:#16a34a;}
.fc-pill.bad  {background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.45); color:#dc2626;}
.fc-dot {width:8px; height:8px; border-radius:50%; background:currentColor;}
.fc-state {border-radius:14px; padding:14px 18px; margin:0 0 12px 0;
  background:rgba(148,163,184,.10); border:1px solid rgba(148,163,184,.35); border-left:8px solid #64748b;}
.fc-state.ok   {border-left-color:#22c55e;}
.fc-state.warn {border-left-color:#f59e0b; background:rgba(245,158,11,.08);}
.fc-state.bad  {border-left-color:#ef4444; background:rgba(239,68,68,.10);}
.fc-state .fc-title {font-size:1.35rem; font-weight:800; letter-spacing:.03em;}
.fc-state .fc-sub {font-size:.85rem; opacity:.75; margin-bottom:6px;}
.fc-state .fc-reason {font-size:1rem; font-weight:600; line-height:1.4;}
.fc-state .fc-meta {font-size:.78rem; opacity:.65; margin-top:6px;}
</style>
"""


def append_event_log(events: list[Event], path: str = EVENT_LOG) -> None:
    """Append events to a CSV file (created with a header if new)."""
    if not events:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["time", "timestamp", "use_case", "old_state", "new_state", "reason", "answered"])
        for e in events:
            writer.writerow([datetime.fromtimestamp(e.time).isoformat(timespec="seconds"), f"{e.time:.3f}",
                             e.use_case, e.old_state, e.new_state, e.reason, e.answered or ""])


class FallWorker:
    """Runs camera -> pose -> FallCheck in a background thread.

    All access to ``check`` goes through ``lock`` (the page thread presses
    buttons while this thread feeds frames).
    """

    def __init__(self, config: dict, source, check: FallCheck, log_path: str | None = EVENT_LOG) -> None:
        self.config = config
        self.source = source
        self.check = check
        self.log_path = log_path
        self.zones = ZoneMap(config["zones"])
        self.detector = PoseDetector.from_config(config, zone_map=self.zones)
        self.lock = threading.Lock()
        self.latest_view = None  # JPEG bytes of the picture with zones and skeleton, for the page
        self.latest_pose: list[Observation] = []
        self.status = "starting"
        self.fps = 0.0
        self.check_ms = 0.0
        self.error: str | None = None
        self.started_at = time.time()
        self._frame_times = deque(maxlen=20)
        self._check_times = deque(maxlen=5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "FallWorker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    # ---------------------------------------------------------------- inputs

    def _record(self, events: list[Event]) -> list[Event]:
        if events and self.log_path:
            try:
                append_event_log(events, self.log_path)
            except OSError:
                pass  # a full disk must not stop the fall check
        return events

    def answer(self, label: str) -> list[Event]:
        """An answer from the page (simulated speech): answer_yes/no/help/unknown."""
        obs = Observation(time=time.time(), source="audio", label=label, confidence=1.0)
        with self.lock:
            return self._record(self.check.feed(obs))

    def set_microphone(self, ok: bool) -> None:
        label = "mic_ok" if ok else "mic_lost"
        with self.lock:
            self.check.feed(Observation(time=time.time(), source="audio", label=label, confidence=1.0))

    def acknowledge(self) -> list[Event]:
        with self.lock:
            return self._record(self.check.acknowledge(time.time(), who="helper"))

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        try:
            self.status = "loading model"
            self.detector.load()
            self.status = "opening camera"
            self._loop()
        except Exception as exc:  # show the problem on the page instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"

    def _loop(self) -> None:
        camera = self.config["camera"]
        with FrameSource(self.source, max_fps=camera["max_fps"],
                         max_failures=camera["max_failures"], start_time=time.time()) as cam:
            while not self._stop.is_set() and cam.status == "ok":
                frame, timestamp = cam.read()
                self._feed_camera_status(cam)
                if frame is None:
                    with self.lock:
                        self._record(self.check.tick(timestamp))
                    continue
                if cam.is_file:
                    self._stop.wait(max(0.0, timestamp - time.time()))  # play clips at real speed
                result = self.detector.detect(frame, timestamp)
                if result is not None:
                    self.latest_pose = result
                    self._check_times.append(self.detector.last_inference_s or 0.0)
                with self.lock:
                    self._record(self.check.feed_pose(result, timestamp))
                self._publish(frame)
            self._feed_camera_status(cam)
            self.status = cam.status if cam.status != "ok" else "stopped"

    def _feed_camera_status(self, cam: FrameSource) -> None:
        for obs in cam.pop_observations():  # camera_lost -> UC3 "unknown"
            with self.lock:
                self._record(self.check.feed(obs))

    def _publish(self, frame) -> None:
        import cv2  # lazy import

        view = draw_poses(self.zones.draw(frame), self.latest_pose)
        # Encode once here (small JPEG) instead of sending a raw picture 4x per second:
        # much less work for the page and no broken images while it refreshes.
        ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._frame_times.append(time.time())
        span = self._frame_times[-1] - self._frame_times[0]
        with self.lock:
            if ok:
                self.latest_view = jpeg.tobytes()
            self.fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            if self._check_times:
                self.check_ms = 1000 * sum(self._check_times) / len(self._check_times)
            self.status = "ok"


# ------------------------------------------------------------- worker control


@st.cache_resource
def _active_workers() -> list:
    """Fall workers of ALL browser sessions: only one may hold the camera."""
    return []


def stop_fall_check() -> None:
    """Stop the fall check's camera (e.g. before another section opens the webcam)."""
    for worker in _active_workers():
        worker.stop()
    _active_workers().clear()


def _new_check(config: dict, speak: bool) -> FallCheck:
    say = None
    if speak:
        import reminders  # lazy: pyttsx3 helper from section F

        say = reminders.speak
    return FallCheck.from_config(config, say=say)


def _start_worker(config: dict, source, speak: bool) -> None:
    stop_fall_check()
    check = _new_check(config, speak)
    check.mic_ok = st.session_state.get("fall_mic", True)
    worker = FallWorker(config, source, check).start()
    _active_workers().append(worker)
    st.session_state.fall_worker = worker


def _parse_source(text: str):
    text = str(text).strip()
    return int(text) if text.isdigit() else text


# ---------------------------------------------------------------- page parts


def _render_controls(config: dict, show_source_input: bool, on_start: Callable | None) -> None:
    worker = st.session_state.get("fall_worker")
    running = worker is not None and worker.running
    source = config["camera"]["source"]
    if show_source_input:
        source = st.text_input("Camera", value=str(source), key="fall_source",
                               help="0 = built-in webcam, 1 = second camera, or a video file path")
    start, stop = st.columns(2)
    if start.button("▶ Start fall check", key="fall_start", type="primary",
                    disabled=running, width="stretch"):
        if on_start is not None:
            on_start()
        _start_worker(config, _parse_source(source), st.session_state.get("fall_speak", True))
        st.rerun()
    if stop.button("⏹ Stop", key="fall_stop", disabled=not running, width="stretch"):
        worker.stop()
        st.rerun()


def _status_pill(worker) -> str:
    if worker is None or (not worker.running and worker.status in ("stopped", "ended", "starting")):
        return '<span class="fc-pill"><span class="fc-dot"></span>Camera off</span>'
    if worker.error or worker.status == "lost":
        return '<span class="fc-pill bad"><span class="fc-dot"></span>Camera problem</span>'
    if worker.status == "ok":
        return '<span class="fc-pill live"><span class="fc-dot"></span>Live · watching for falls</span>'
    return f'<span class="fc-pill"><span class="fc-dot"></span>{html.escape(worker.status.capitalize())} …</span>'


@st.fragment(run_every=0.25)
def _live_view() -> None:
    worker = st.session_state.get("fall_worker")
    st.markdown(_status_pill(worker), unsafe_allow_html=True)
    if worker is None:
        st.info("Press **▶ Start fall check**. The camera should see the whole body "
                "(head to feet), e.g. from a shelf a few metres away.")
        return
    with worker.lock:
        view, status, fps = worker.latest_view, worker.status, worker.fps
    if worker.error:
        st.error(f"The fall check stopped because of an error: {worker.error}")
    elif status == "lost":
        st.error("📷 **Camera lost.** The fall check shows UNKNOWN (never an alarm). "
                 "Close other programs using the camera, then press Start.")
    elif status not in ("ok", "stopped", "ended"):
        waited = time.time() - worker.started_at
        if status == "loading model":
            st.caption(f"Loading the pose model … {waited:.0f} s (the first time it is downloaded).")
        else:
            st.caption(f"Opening the camera … {waited:.0f} s. On Windows this can take up to 30 s.")
        if waited > 40:
            st.warning("This is taking very long. Another program may be holding the webcam: "
                       "close other camera apps and terminals running a LifeAware demo, stop "
                       "sections C and G, then press Stop and Start again.")
    if view is not None:
        # The picture goes straight into the page (base64), not through Streamlit's
        # temporary media files: those are often deleted before the browser loads
        # them when the view refreshes 4x per second, which shows a broken image.
        picture = base64.b64encode(view).decode("ascii")
        st.markdown(f'<img src="data:image/jpeg;base64,{picture}" '
                    'style="width:100%; border-radius:8px;" alt="camera view">',
                    unsafe_allow_html=True)
        posture = ", ".join(f"{o.label.replace('person_', '')} ({o.zone or 'no zone'})"
                            for o in worker.latest_pose) or "nobody"
        n = worker.detector.every_n_frames
        every = "every frame" if n == 1 else f"every {_ordinal(n)} frame"
        st.caption(f"Seen: {posture} · {fps:.1f} frames/s · pose check "
                   f"{worker.check_ms:.0f} ms, {every}")


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th'."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def state_card(state: str, last: Event | None, mic_ok: bool) -> str:
    """HTML card for the current UC3 state and the reason of the last change."""
    title, css, sub = STATE_INFO.get(state, (state.upper(), "grey", ""))
    reason = last.reason if last is not None else "No events yet."
    meta = []
    if last is not None:
        meta.append(f"since {datetime.fromtimestamp(last.time):%H:%M:%S}")
        if last.answered:
            meta.append(f"answer: {last.answered}")
    meta.append("microphone ok" if mic_ok else "microphone NOT working")
    return (f'<div class="fc-state {css}"><div class="fc-title">{html.escape(title)}</div>'
            f'<div class="fc-sub">{html.escape(sub)}</div>'
            f'<div class="fc-reason">{html.escape(reason)}</div>'
            f'<div class="fc-meta">{html.escape(" · ".join(meta))}</div></div>')


def events_table(events: list[Event], limit: int = 12):
    """Newest-first rows for st.dataframe."""
    import pandas as pd

    rows = [{
        "Time": datetime.fromtimestamp(e.time).strftime("%H:%M:%S"),
        "Change": f"{e.old_state} → {e.new_state}",
        "Reason": e.reason,
        "Answer": e.answered or "",
    } for e in reversed(events[-limit:])]
    return pd.DataFrame(rows, columns=["Time", "Change", "Reason", "Answer"])


@st.fragment(run_every=0.5)
def _state_panel() -> None:
    worker = st.session_state.get("fall_worker")
    if worker is None:
        st.markdown(state_card(fc.NORMAL, None, st.session_state.get("fall_mic", True)),
                    unsafe_allow_html=True)
        st.caption("Answer buttons work while the fall check is running.")
        return

    with worker.lock:
        state, events, mic_ok = worker.check.state, list(worker.check.events), worker.check.mic_ok
    st.markdown(state_card(state, events[-1] if events else None, mic_ok), unsafe_allow_html=True)

    st.markdown("**Answer** · simulated speech until the microphone part is connected")
    yes, no, help_, unclear = st.columns(4)
    asking = state == fc.CHECKING
    if yes.button("✅ Yes, I'm okay", key="fall_yes", disabled=not asking, width="stretch"):
        worker.answer("answer_yes")
        st.rerun(scope="fragment")
    if no.button("❌ No", key="fall_no", disabled=not asking, width="stretch"):
        worker.answer("answer_no")
        st.rerun(scope="fragment")
    if help_.button("🆘 Help", key="fall_help", width="stretch"):
        worker.answer("answer_help")
        st.rerun(scope="fragment")
    if unclear.button("❔ Not understood", key="fall_unclear", disabled=not asking, width="stretch"):
        worker.answer("answer_unknown")
        st.rerun(scope="fragment")

    if st.button("🙋 Acknowledge (helper is here)", key="fall_ack",
                 disabled=state not in (fc.ALARM, fc.UNKNOWN), width="stretch"):
        worker.acknowledge()
        st.rerun(scope="fragment")

    st.markdown("**Event history** (also saved to `events/fall_events.csv`)")
    if events:
        st.dataframe(events_table(events), hide_index=True, width="stretch")
    else:
        st.caption("No state changes yet.")


def _on_mic_toggle() -> None:
    worker = st.session_state.get("fall_worker")
    if worker is not None:
        worker.set_microphone(st.session_state.fall_mic)


# --------------------------------------------------------------- public entry


def render_fall_check(
    config: dict | None = None,
    on_start: Callable[[], None] | None = None,
    show_source_input: bool = False,
) -> None:
    """Draw the whole fall check: camera on the left, state and answers on the right.

    ``on_start`` runs just before the fall check opens the webcam (e.g. to stop
    the camera preview or the item finder on the same page).
    """
    config = config or load_config()
    st.markdown(_CSS, unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_controls(config, show_source_input, on_start)
        _live_view()
    with right:
        a, b = st.columns(2)
        a.toggle("🔊 Speak aloud", value=True, key="fall_speak",
                 help="Takes effect at the next Start.")
        b.toggle("🎤 Microphone works", value=True, key="fall_mic", on_change=_on_mic_toggle,
                 help="Off = a missing answer gives UNKNOWN instead of an alarm.")
        _state_panel()
        timers = config["fall"]
        st.caption(f"Rules: lying on the floor ≥ {timers['min_lying_s']:g} s → possible event · "
                   f"still ≥ {timers['still_s']:g} s → ask · no answer in {timers['answer_timeout_s']:g} s "
                   f"→ alarm (only with a working camera and microphone). Not a medical device; "
                   f"it does not call emergency services.")
