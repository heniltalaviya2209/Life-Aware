"""Item finder (UC2) for Streamlit: live camera -> YOLO -> zones -> memory -> answers.

One reusable section, shown in two places:
- ``app.py`` section G (the main dashboard),
- ``demo_items.py`` (a standalone page, handy for testing).

A background thread runs the real chain at <= 10 fps; the page only shows the
latest picture and the memory, so the dashboard stays responsive.
"""

from __future__ import annotations

import html
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable

import streamlit as st

import reminders
from lifeaware.camera import FrameSource
from lifeaware.config import load_config
from lifeaware.memory import ItemMemory
from lifeaware.objects import ObjectDetector, draw_detections
from lifeaware.usecases.lost_items import render_lost_items_panel
from lifeaware.zones import ZoneMap

# Items shown as cards: label -> (icon, display name).
ITEM_CARDS = {"phone": ("📱", "Phone"), "keys": ("🔑", "Keys")}

_CSS = """
<style>
.lf-pill {display:inline-flex; align-items:center; gap:7px; padding:4px 12px;
  border-radius:999px; font-size:.8rem; font-weight:600; margin-bottom:8px;
  background:rgba(100,116,139,.12); border:1px solid rgba(100,116,139,.40); color:#64748b;}
.lf-pill.live {background:rgba(34,197,94,.12); border-color:rgba(34,197,94,.45); color:#16a34a;}
.lf-pill.bad  {background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.45); color:#dc2626;}
.lf-dot {width:8px; height:8px; border-radius:50%; background:currentColor;}
.lf-item {display:flex; align-items:center; gap:12px; padding:10px 14px; margin:0 0 8px 0;
  border-radius:12px; background:rgba(148,163,184,.10); border:1px solid rgba(148,163,184,.30);}
.lf-item .lf-icon {font-size:1.5rem;}
.lf-item b {display:block; font-size:.95rem;}
.lf-item span {font-size:.83rem; opacity:.8;}
.lf-item.seen {border-left:4px solid #22c55e;}
</style>
"""


class ItemWorker:
    """Runs camera -> detector -> ItemMemory in a background thread."""

    def __init__(self, config: dict, source, memory: ItemMemory) -> None:
        self.config = config
        self.source = source
        self.memory = memory
        self.zones = ZoneMap(config["zones"])
        self.detector = ObjectDetector.from_config(config, zone_map=self.zones)
        self.lock = threading.Lock()
        self.latest_view = None  # RGB picture with zones and boxes, for the page
        self.recent = deque(maxlen=15)  # latest stored sightings (newest first)
        self.status = "starting"
        self.fps = 0.0  # frames per second over the last few seconds
        self.check_ms = 0.0  # average duration of the recent model checks
        self.error: str | None = None
        self._frame_times = deque(maxlen=20)  # for the rolling fps
        self._check_times = deque(maxlen=5)  # for the rolling check duration
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "ItemWorker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.status = "loading model"
            self.detector.load()  # includes a warm-up check (the first one is slow)
            self.status = "opening camera"
            self._loop()
        except Exception as exc:  # show the problem on the page instead of dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"

    def _loop(self) -> None:
        camera = self.config["camera"]
        boxes = []
        with FrameSource(self.source, max_fps=camera["max_fps"],
                         max_failures=camera["max_failures"], start_time=time.time()) as cam:
            while not self._stop.is_set() and cam.status == "ok":
                frame, timestamp = cam.read()
                if frame is None:
                    continue
                if cam.is_file:
                    self._stop.wait(max(0.0, timestamp - time.time()))  # play clips at real speed
                found = self.detector.detect(frame, timestamp)
                if found is not None:  # None = frame skipped; keep showing the last boxes
                    boxes = found
                    self._check_times.append(self.detector.last_inference_s or 0.0)
                    self._remember(found)
                self._publish(frame, boxes)
            self.status = cam.status if cam.status != "ok" else "stopped"

    def _remember(self, observations) -> None:
        with self.lock:
            for obs in observations:
                if self.memory.update(obs):
                    self.recent.appendleft(obs)

    def _publish(self, frame, boxes) -> None:
        view = draw_detections(self.zones.draw(frame), boxes)
        self._frame_times.append(time.time())
        span = self._frame_times[-1] - self._frame_times[0]
        with self.lock:
            self.latest_view = view[:, :, ::-1]  # BGR (OpenCV) -> RGB (browser)
            self.fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
            if self._check_times:
                self.check_ms = 1000 * sum(self._check_times) / len(self._check_times)
            self.status = "ok"


# ------------------------------------------------------------- worker control


@st.cache_resource
def _active_workers() -> list:
    """Workers of ALL browser sessions: only one may hold the camera at a time."""
    return []


def stop_item_finder() -> None:
    """Stop the finder's camera (e.g. before another section opens the webcam)."""
    for worker in _active_workers():
        worker.stop()
    _active_workers().clear()


def _start_worker(config: dict, source) -> None:
    stop_item_finder()
    worker = ItemWorker(config, source, st.session_state.item_memory).start()
    _active_workers().append(worker)
    st.session_state.item_worker = worker


def _parse_source(text: str):
    """'0' -> webcam 0; anything else -> a video file path."""
    text = str(text).strip()
    return int(text) if text.isdigit() else text


# ---------------------------------------------------------------- page parts


def _render_controls(config: dict, show_source_input: bool, on_start: Callable | None) -> None:
    worker = st.session_state.get("item_worker")
    running = worker is not None and worker.running
    source = config["camera"]["source"]
    if show_source_input:
        source = st.text_input("Camera", value=str(source), key="item_source",
                               help="0 = built-in webcam, 1 = second camera, or a video file path")
    start, stop, forget = st.columns(3)
    if start.button("▶ Start finder", key="item_start", type="primary",
                    disabled=running, width="stretch"):
        if on_start is not None:
            on_start()
        _start_worker(config, _parse_source(source))
        st.rerun()
    if stop.button("⏹ Stop", key="item_stop", disabled=not running, width="stretch"):
        worker.stop()
        st.rerun()
    if forget.button("🗑 Forget items", key="item_forget", width="stretch"):
        st.session_state.item_memory = ItemMemory.from_config(config)
        if worker is not None:
            worker.memory = st.session_state.item_memory
        st.session_state.pop("uc2_last_event", None)
        st.rerun()


def _status_pill(worker) -> str:
    if worker is None or worker.status in ("stopped", "ended", "starting") and not worker.running:
        return '<span class="lf-pill"><span class="lf-dot"></span>Camera off</span>'
    if worker.error or worker.status == "lost":
        return '<span class="lf-pill bad"><span class="lf-dot"></span>Camera problem</span>'
    if worker.status == "ok":
        return '<span class="lf-pill live"><span class="lf-dot"></span>Live · watching for items</span>'
    return f'<span class="lf-pill"><span class="lf-dot"></span>{html.escape(worker.status.capitalize())} …</span>'


@st.fragment(run_every=0.25)
def _live_view() -> None:
    worker = st.session_state.get("item_worker")
    st.markdown(_status_pill(worker), unsafe_allow_html=True)
    if worker is None:
        st.info("Press **▶ Start finder**, then hold your phone up to the camera.")
        return
    with worker.lock:
        view, status, fps = worker.latest_view, worker.status, worker.fps
    if worker.error:
        st.error(f"The finder stopped because of an error: {worker.error}")
    elif status == "lost":
        st.error("📷 **Camera lost.** Close Teams/Zoom or the camera preview, then press Start. "
                 "Answers below may be out of date.")
    elif status not in ("ok", "stopped", "ended"):
        st.caption("Loading takes a few seconds the first time …")
    if view is not None:
        st.image(view, width="stretch")
        model_name = os.path.basename(worker.detector.model_path)
        st.caption(f"{fps:.1f} frames/s · {model_name} every {worker.detector.every_n_frames}th frame "
                   f"· {worker.check_ms:.0f} ms per check · not mirrored (room camera view)")


def _item_card(item: str, memory: ItemMemory, detectable: set[str]) -> str:
    icon, name = ITEM_CARDS.get(item, ("📦", item.capitalize()))
    sighting = memory.last_seen(item)
    if sighting is not None:
        where = sighting.zone or "unknown area"
        text = f"📍 {where} · {datetime.fromtimestamp(sighting.time):%H:%M:%S} · {sighting.confidence:.0%} sure"
        css = "lf-item seen"
    elif item in detectable:
        text, css = "Not seen yet", "lf-item"
    else:
        text, css = "Cannot be detected yet (planned)", "lf-item"
    return (f'<div class="{css}"><div class="lf-icon">{icon}</div>'
            f"<div><b>{html.escape(name)}</b><span>{html.escape(text)}</span></div></div>")


@st.fragment(run_every=1.0)
def _memory_cards(detectable: set[str]) -> None:
    memory: ItemMemory = st.session_state.item_memory
    st.markdown("**Last seen**")
    st.markdown("".join(_item_card(item, memory, detectable) for item in ITEM_CARDS),
                unsafe_allow_html=True)


# --------------------------------------------------------------- public entry


def render_item_finder(
    config: dict | None = None,
    on_start: Callable[[], None] | None = None,
    show_source_input: bool = False,
) -> None:
    """Draw the whole item finder: camera on the left, questions and memory on the right.

    ``on_start`` runs just before the finder opens the webcam (e.g. to stop
    another camera preview on the same page).
    """
    config = config or load_config()
    st.session_state.setdefault("item_memory", ItemMemory.from_config(config))
    st.markdown(_CSS, unsafe_allow_html=True)
    detectable = set(config["objects"]["classes"].values())

    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_controls(config, show_source_input, on_start)
        _live_view()
    with right:
        speak = st.toggle("🔊 Speak answers aloud", value=True, key="item_speak")
        render_lost_items_panel(st.session_state.item_memory,
                                say=reminders.speak if speak else None)
        _memory_cards(detectable)
