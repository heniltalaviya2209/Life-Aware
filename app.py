"""
SCENE-DETECT - Intelligent Scene Recognition
============================================

Streamlit UI prototype for a university project (OTH Amberg-Weiden).

IMPORTANT - WHAT THIS PROGRAM IS *NOT*
--------------------------------------
* It does NOT run a real AI model.
* It does NOT detect people, poses, emotions or falls.
* All demonstration results are pre-written simulations for fixed scenarios.
* The camera section shows a live video preview only (frames are not analysed).

The goal of this version is to explore the workflow and the user interface.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

import reminders
from lifeaware.camera import open_webcam
from lifeaware.fall_panel import render_fall_check, stop_fall_check
from lifeaware.item_finder import render_item_finder, stop_item_finder

# ---------------------------------------------------------------------------
# Configuration - all texts and simulated data live here (easy to change)
# ---------------------------------------------------------------------------

APP_TITLE = "SCENE-DETECT — Intelligent Scene Recognition"

APP_DESCRIPTION = (
    "Prototype dashboard for an AI-based situation-recognition system for the home. "
    "Pick a predefined scenario to see how a decision could be explained. "
    "No real detection runs in this version — everything shown is simulated."
)

DEFAULT_CAMERA_MESSAGE = (
    "info",
    "📷 Camera is off. Press **Start camera preview** to see your live webcam feed. "
    "The preview is video only — no analysis of any kind is performed.",
)

# The four situations the project is about.
SITUATIONS = {
    "NO PERSON": {
        "tone": "neutral",
        "tag": "EMPTY",
        "description": (
            "The scene looks empty. Nobody is visible to the camera, so the system "
            "simply reports that no person is present. An empty room is not an emergency."
        ),
    },
    "NORMAL": {
        "tone": "normal",
        "tag": "OK",
        "description": (
            "A person is visible and their activity looks ordinary — for example "
            "standing, sitting, walking or resting. Nothing needs to be done."
        ),
    },
    "CHECK PERSON": {
        "tone": "warning",
        "tag": "CHECK",
        "description": (
            "Something is unclear or unusual: a person is partly hidden, the view is "
            "bad, or the evidence is too weak for a confident decision. A human should "
            "take a look at the scene."
        ),
    },
    "ALERT": {
        "tone": "alert",
        "tag": "ALERT",
        "description": (
            "Strong simulated evidence of an emergency, such as a sudden fall. In the "
            "real system this state would trigger a helper alert. It is NOT a medical "
            "diagnosis and not a claim that a person is safe or unsafe."
        ),
    },
}

# The seven predefined demonstration scenarios.
# Every field below is SIMULATED — nothing is detected in real time.
SCENARIOS = {
    "Empty room": {
        "state": "NO PERSON",
        "score": "0.97",
        "features": "Nobody visible · Motion: none",
        "evidence": [
            "No person-shaped object anywhere in the frame.",
            "The scene has looked unchanged for a long time.",
        ],
        "explanation": (
            "The whole visible area was checked and no person was found, so NO PERSON "
            "is the honest answer. It does not mean anything is wrong."
        ),
    },
    "Person standing": {
        "state": "NORMAL",
        "score": "0.94",
        "features": "Orientation: upright · Motion: none",
        "evidence": [
            "One full-body person visible in the middle of the frame.",
            "Body is upright: head above the hips, feet on the floor.",
        ],
        "explanation": (
            "Standing is an ordinary pose. The person is upright and stable, "
            "so the label is NORMAL."
        ),
    },
    "Person sitting": {
        "state": "NORMAL",
        "score": "0.93",
        "features": "Orientation: upright, lowered · Motion: none",
        "evidence": [
            "Body is in a sitting shape: hips low, knees bent.",
            "Sitting is a normal everyday activity (chair, sofa, table).",
        ],
        "explanation": (
            "Sitting is never treated as an emergency by itself. The person is stable, "
            "so the label is NORMAL."
        ),
    },
    "Person walking": {
        "state": "NORMAL",
        "score": "0.92",
        "features": "Orientation: upright · Motion: continuous",
        "evidence": [
            "Body is upright and moving steadily across the frame.",
            "Movement looks regular, not a sudden drop.",
        ],
        "explanation": (
            "Walking is normal activity. Regular movement with an upright body "
            "gives the label NORMAL."
        ),
    },
    "Person lying on the floor": {
        "state": "CHECK PERSON",
        "score": "0.71",
        "features": "Orientation: horizontal · Motion: none",
        "evidence": [
            "The body lies horizontally along the floor.",
            "Lying down can be completely normal: sleeping, resting, stretching.",
            "A single moment in time cannot tell resting from a fall.",
        ],
        "explanation": (
            "Lying down alone is NOT treated as an emergency — people sleep and rest. "
            "Because this prototype cannot watch movement over time, it only asks a "
            "human to CHECK PERSON instead of raising an alert."
        ),
    },
    "Possible fall": {
        "state": "ALERT",
        "score": "0.88",
        "features": "Orientation: upright → horizontal · Motion: sudden",
        "evidence": [
            "The body changed from upright to horizontal very quickly.",
            "The person ended up on the floor in an unusual spot (not a bed or sofa).",
            "No movement was seen after the change.",
        ],
        "explanation": (
            "Several strong signals happen together: a sudden drop to the floor plus no "
            "movement afterwards. In the real system these signals must persist across "
            "many frames before ALERT is shown. This result is simulated for the demo."
        ),
    },
    "Uncertain or partially hidden person": {
        "state": "CHECK PERSON",
        "score": "0.48",
        "features": "Body partly visible · Key points missing",
        "evidence": [
            "Only part of a body is visible (blocked or cropped by the frame edge).",
            "Important body points (head, hips) cannot be seen clearly.",
        ],
        "explanation": (
            "When the evidence is too weak, the system should say so honestly. "
            "Instead of guessing, it shows CHECK PERSON so a human can review the scene."
        ),
    },
}

# Ideas for later versions — nothing here is implemented yet.
FUTURE_FEATURES = [
    (
        "🧍",
        "Real-time human pose detection",
        "Estimate body key points (shoulders, hips, knees, ankles) and classify "
        "standing, sitting, walking, bending and lying.",
    ),
    (
        "🧎",
        "Fall and inactivity detection",
        "Watch movement across several frames and only raise an alert when strong "
        "evidence persists.",
    ),
    (
        "🔊",
        "Voice interaction",
        "Ask the person «Are you okay?» and listen for an answer before escalating.",
    ),
    (
        "📞",
        "Human assistance alerts",
        "Notify a caregiver or emergency contact together with the scene details.",
    ),
    (
        "🎵",
        "Favourite music playback",
        "Play the person's preferred music on request.",
    ),
    (
        "💬",
        "Emotion-aware conversation (optional)",
        "Adapt the assistant's tone to the recognised mood — only if it proves reliable.",
    ),
]

# ---------------------------------------------------------------------------
# Styles — a simple colour scheme: normal / warning / alert
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SCENE-DETECT — Intelligent Scene Recognition",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .sd-header h1 {font-size: 1.9rem; margin: 0 0 4px 0;}
    .sd-header p  {margin: 0 0 10px 0; font-size: 0.95rem; opacity: 0.85; max-width: 920px;}

    .sd-status {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
        background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.45); color: #16a34a;
        margin-right: 8px;
    }
    .sd-status.idle {
        background: rgba(100,116,139,.12); border-color: rgba(100,116,139,.45); color: #64748b;
    }
    .sd-dot {
        width: 9px; height: 9px; border-radius: 50%; background: #22c55e;
        animation: sd-pulse 1.8s infinite;
    }
    .sd-status.idle .sd-dot {background: #94a3b8; animation: none;}
    @keyframes sd-pulse {
        0%   {box-shadow: 0 0 0 0 rgba(34,197,94,.55);}
        70%  {box-shadow: 0 0 0 9px rgba(34,197,94,0);}
        100% {box-shadow: 0 0 0 0 rgba(34,197,94,0);}
    }

    .chip {
        display: inline-block; font-size: 0.68rem; font-weight: 800;
        letter-spacing: 0.08em; text-transform: uppercase;
        padding: 3px 10px; border-radius: 999px; margin-bottom: 6px;
    }
    .chip.neutral {background: rgba(100,116,139,.20); color: #64748b;}
    .chip.normal  {background: rgba(34,197,94,.18);   color: #16a34a;}
    .chip.warning {background: rgba(245,158,11,.18);  color: #d97706;}
    .chip.alert   {background: rgba(239,68,68,.18);   color: #dc2626;}

    .sd-card {
        height: 100%; border-radius: 14px; padding: 14px 16px 16px 16px;
        background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.35);
        border-top: 4px solid #64748b; margin-bottom: 8px;
    }
    .sd-card.normal  {border-top-color: #22c55e;}
    .sd-card.warning {border-top-color: #f59e0b;}
    .sd-card.alert   {border-top-color: #ef4444;}
    .sd-card h4 {margin: 2px 0 6px 0; font-size: 1.02rem;}
    .sd-card p  {margin: 0; font-size: 0.88rem; line-height: 1.5; opacity: 0.9;}

    .sd-result {
        border-radius: 16px; padding: 16px 20px 18px 20px; margin: 4px 0 14px 0;
        background: rgba(148,163,184,.10); border: 1px solid rgba(148,163,184,.35);
        border-left: 6px solid #64748b;
    }
    .sd-result.normal  {border-left-color: #22c55e;}
    .sd-result.warning {border-left-color: #f59e0b;}
    .sd-result.alert   {border-left-color: #ef4444;}
    .sd-badge {
        display: inline-block; margin-bottom: 10px; padding: 4px 12px; border-radius: 8px;
        font-size: 0.75rem; font-weight: 800; letter-spacing: 0.04em;
        background: rgba(245,158,11,.14); color: #b45309;
        border: 1px dashed rgba(245,158,11,.65);
    }
    .sd-result h3 {margin: 0 0 2px 0; font-size: 1.35rem;}
    .sd-metrics {display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 4px 0;}
    .sd-metrics > div {
        background: rgba(148,163,184,.14); border-radius: 10px;
        padding: 8px 12px; min-width: 160px;
    }
    .sd-k {display: block; font-size: 0.68rem; text-transform: uppercase;
           letter-spacing: 0.06em; opacity: 0.7;}
    .sd-v {display: block; font-size: 0.95rem; font-weight: 600; margin-top: 2px;}
    .sd-why {margin: 10px 0 4px 0; font-size: 0.95rem; line-height: 1.55;}
    .sd-ev  {margin: 6px 0 2px 0; font-size: 0.88rem;}
    .sd-result ul {margin: 0 0 4px 0; padding-left: 20px; font-size: 0.88rem;
                   line-height: 1.55; opacity: 0.92;}

    .sd-future {
        height: 100%; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
        background: rgba(148,163,184,.08); border: 1px solid rgba(148,163,184,.30);
    }
    .sd-future h5 {margin: 0 0 4px 0; font-size: 0.95rem;}
    .sd-future p  {margin: 0; font-size: 0.84rem; line-height: 1.45; opacity: 0.85;}
    .sd-planned {
        display: inline-block; margin-top: 8px; padding: 2px 9px; border-radius: 999px;
        font-size: 0.64rem; font-weight: 800; letter-spacing: 0.1em;
        color: #7c3aed; background: rgba(124,58,237,.14);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state — remembers things while the app is open
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []          # rows for the event history table
if "last_result" not in st.session_state:
    st.session_state.last_result = None    # last demonstration result
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False     # is the preview running?
if "camera_cap" not in st.session_state:
    st.session_state.camera_cap = None     # the OpenCV capture object
if "camera_failures" not in st.session_state:
    st.session_state.camera_failures = 0   # consecutive failed frame reads
if "camera_message" not in st.session_state:
    st.session_state.camera_message = DEFAULT_CAMERA_MESSAGE
if "reminders" not in st.session_state:
    st.session_state.reminders = reminders.load_reminders()
if "reminder_audio_gen" not in st.session_state:
    st.session_state.reminder_audio_gen = 0
if "reminder_status" not in st.session_state:
    st.session_state.reminder_status = None


# ---------------------------------------------------------------------------
# Camera helpers (OpenCV is imported lazily, so the app works without it)
# ---------------------------------------------------------------------------


def _release_camera() -> None:
    """Close the webcam if it is open."""
    cap = st.session_state.get("camera_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.camera_cap = None


def _stop_camera(message: tuple) -> None:
    """Switch the preview off and remember a friendly message."""
    _release_camera()
    st.session_state.camera_on = False
    st.session_state.camera_message = message


def _show_camera_frame(preview_slot) -> None:
    """Show ONE camera frame, then schedule the next update.

    Rendering only one frame per script run keeps the rest of the page
    responsive — the Stop button is always processed between frames.
    """
    try:
        import cv2
    except ImportError:
        _stop_camera(
            (
                "error",
                "OpenCV is not installed, so the camera preview cannot start. "
                "Install it with `pip install opencv-python`, or simply keep using "
                "the simulated demonstrations — they work without a camera.",
            )
        )
        st.rerun()
        return

    cap = st.session_state.get("camera_cap")
    if cap is None:
        cap = open_webcam()  # shared with lifeaware.camera.FrameSource
        if cap is None:
            _stop_camera(
                (
                    "error",
                    "📷 **No camera could be opened.** Possible reasons: no webcam is "
                    "connected, another program is already using it, or Windows camera "
                    "privacy settings block this app (Settings → Privacy → Camera). "
                    "The simulated demonstrations below still work.",
                )
            )
            st.rerun()
            return
        st.session_state.camera_cap = cap

    ok, frame = cap.read()
    if not ok:
        st.session_state.camera_failures += 1
        if st.session_state.camera_failures >= 15:
            _stop_camera(
                (
                    "error",
                    "📷 **The camera stopped delivering images.** It may have been "
                    "unplugged or taken over by another program. Press start to try "
                    "again — or continue with the simulated demonstrations.",
                )
            )
            st.rerun()
            return
        time.sleep(0.1)
        st.rerun()
        return

    st.session_state.camera_failures = 0
    frame = cv2.flip(frame, 1)  # mirror image, like a normal selfie view
    cv2.putText(
        frame,
        "PREVIEW ONLY - NO AI DETECTION",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preview_slot.image(rgb, channels="RGB")
    time.sleep(0.05)  # ≈ 20 frames per second
    st.rerun()


# ---------------------------------------------------------------------------
# Helper: render the result panel of a demonstration
# ---------------------------------------------------------------------------


def show_result(result: dict) -> None:
    """Draw the (simulated) result panel for one scenario."""
    state = result["state"]
    tone = SITUATIONS[state]["tone"]
    evidence_html = "".join(f"<li>{item}</li>" for item in result["evidence"])
    st.markdown(
        f"""
        <div class="sd-result {tone}">
            <div class="sd-badge">SIMULATED DEMONSTRATION — not produced by a real AI model</div>
            <div class="chip {tone}">{state}</div>
            <h3>Scenario: {result['scenario']}</h3>
            <div class="sd-metrics">
                <div><span class="sd-k">Illustrative score (simulated)</span>
                     <span class="sd-v">{result['score']}</span></div>
                <div><span class="sd-k">Simulated body features</span>
                     <span class="sd-v">{result['features']}</span></div>
                <div><span class="sd-k">Recorded at</span>
                     <span class="sd-v">{result['time']}</span></div>
            </div>
            <p class="sd-why"><strong>Why this label was chosen:</strong><br>
               {result['explanation']}</p>
            <p class="sd-ev"><strong>Evidence used (simulated):</strong></p>
            <ul>{evidence_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE
# ===========================================================================

st.markdown(
    f"""
    <div class="sd-header">
        <h1>{APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
        <span class="sd-status"><span class="sd-dot"></span> Prototype running — simulation mode</span>
        <span class="sd-status idle"><span class="sd-dot"></span> AI model: only in section G (item finder)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# A. Main dashboard — the four situations
# ---------------------------------------------------------------------------

st.subheader("A · Main Dashboard")
st.caption(
    "These four cards are the states the SCENE-DETECT system can show. "
    "They describe software states, never a medical condition."
)

cards = st.columns(4)
for column, (name, meta) in zip(cards, SITUATIONS.items()):
    with column:
        st.markdown(
            f"""
            <div class="sd-card {meta['tone']}">
                <span class="chip {meta['tone']}">{meta['tag']}</span>
                <h4>{name}</h4>
                <p>{meta['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# B. Scene demonstration (simulated)
# ---------------------------------------------------------------------------

st.subheader("B · Scene Demonstration")
st.info(
    "Everything in this section is **simulated**. Selecting a scenario and pressing "
    "the button shows a pre-written illustrative result — no camera and no AI model "
    "are involved."
)

pick_col, button_col = st.columns([3, 1])
with pick_col:
    scenario_name = st.selectbox("Choose a predefined scenario", list(SCENARIOS.keys()))
with button_col:
    st.write("")  # small vertical offset so the button lines up with the box
    st.write("")
    if st.button("▶ Run demonstration", type="primary"):
        template = SCENARIOS[scenario_name]
        st.session_state.last_result = {
            "scenario": scenario_name,
            "state": template["state"],
            "score": template["score"],
            "features": template["features"],
            "evidence": template["evidence"],
            "explanation": template["explanation"],
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.history.append(
            [
                st.session_state.last_result["time"],
                scenario_name,
                template["state"],
                template["explanation"],
            ]
        )

result = st.session_state.last_result
if result is None:
    st.caption("👆 Select a scenario and click **Run demonstration** to see a result.")
else:
    show_result(result)

# ---------------------------------------------------------------------------
# C. Camera preview (optional, requires OpenCV + a webcam)
# ---------------------------------------------------------------------------

st.subheader("C · Camera Preview (optional)")
st.caption(
    "Live video preview only. Frames are shown as-is — no people, poses, emotions or "
    "falls are recognised in this prototype. If the camera is unavailable, everything "
    "else on this page keeps working."
)

button_col, preview_col = st.columns([1, 3])
preview_slot = preview_col.empty()

with button_col:
    if not st.session_state.camera_on:
        if st.button("📷 Start camera preview", type="primary"):
            stop_item_finder()  # only one section can use the webcam (see sections G, H)
            stop_fall_check()
            st.session_state.camera_on = True
            st.session_state.camera_failures = 0
            st.session_state.camera_message = ("info", "📷 Starting camera …")
            st.rerun()
    else:
        if st.button("⏹ Stop camera preview"):
            _stop_camera(
                (
                    "info",
                    "📷 Camera stopped. Press **Start camera preview** to see the live "
                    "feed again.",
                )
            )
            st.rerun()

if not st.session_state.camera_on:
    message_kind, message_text = st.session_state.camera_message
    getattr(preview_slot, message_kind)(message_text)

# ---------------------------------------------------------------------------
# D. Event history
# ---------------------------------------------------------------------------

st.subheader("D · Event History")

history_header = st.columns([4, 1])
with history_header[1]:
    if st.button("🗑 Clear history"):
        st.session_state.history = []
        st.rerun()

if not st.session_state.history:
    st.caption("No events recorded yet — run a demonstration in section B to add one.")
else:
    history_df = pd.DataFrame(
        st.session_state.history,
        columns=["Time", "Scenario", "State", "Explanation"],
    )
    st.dataframe(history_df, hide_index=True, height=240)

# ---------------------------------------------------------------------------
# E. Future features (ideas only — not implemented)
# ---------------------------------------------------------------------------

st.subheader("E · Future Features")
st.caption("Ideas for later versions of SCENE-DETECT. None of them is implemented yet.")

for start in range(0, len(FUTURE_FEATURES), 3):
    row = FUTURE_FEATURES[start : start + 3]
    cols = st.columns(3)
    for column, (emoji, title, text) in zip(cols, row):
        with column:
            st.markdown(
                f"""
                <div class="sd-future">
                    <h5>{emoji} {title}</h5>
                    <p>{text}</p>
                    <span class="sd-planned">PLANNED</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# F. Reminders (voice -> offline transcription -> schedule -> announcement)
# ---------------------------------------------------------------------------


@st.fragment(run_every="5s")
def _check_due_reminders() -> None:
    """Announce any pending reminder whose time has arrived, then mark it done."""
    now = datetime.now()
    changed = False
    for item in st.session_state.reminders:
        if item.get("status") != "pending":
            continue
        due = reminders.parse_iso(item.get("scheduled_at"))
        if due is None or due > now:
            continue
        message = f"Reminder: {item['task']}"
        # Speak through the machine's speakers. This bypasses browser autoplay
        # policies (and pyttsx3's macOS main-thread limitation) via `say`/subprocess.
        try:
            reminders.speak(message)
            st.toast(f"🔊 {message}", icon="🔊")
        except Exception:
            st.toast(f"⏰ Reminder due — {item['task']}", icon="⏰")
        # Optional browser replay control (no autoplay, so no double playback).
        try:
            audio, mime = reminders.text_to_audio(message)
            if audio:
                st.audio(audio, format=mime)
        except Exception:
            pass
        item["status"] = "completed"
        changed = True
    if changed:
        reminders.save_reminders(st.session_state.reminders)


st.subheader("F · Reminders")
st.caption(
    "Record a reminder such as “remind me to take my medicine at 4pm” or "
    "“remind me to stretch in 30 minutes”. Speech is transcribed offline (Vosk), "
    "the trigger time is parsed from the text (dateparser), and due reminders are "
    "announced out loud (pyttsx3)."
)

# One-shot status message produced by the workflow below (shown after rerun).
if st.session_state.reminder_status:
    kind, message = st.session_state.reminder_status
    getattr(st, kind)(message)
    st.session_state.reminder_status = None

audio_value = st.audio_input(
    "🎙️ Record a reminder",
    key=f"reminder_audio_{st.session_state.reminder_audio_gen}",
)
if audio_value is not None:
    spoken = None
    with st.spinner("Transcribing your reminder…"):
        try:
            pcm = reminders.decode_audio_to_pcm16_mono(audio_value)
            spoken = reminders.transcribe_pcm(pcm)
        except FileNotFoundError as exc:
            st.session_state.reminder_status = ("error", str(exc))
        except Exception as exc:
            st.session_state.reminder_status = (
                "error",
                f"Could not transcribe the recording: {exc}",
            )

    if spoken:
        task, scheduled = reminders.parse_reminder(spoken, now=datetime.now())
        item = {
            "id": reminders.new_id(),
            "task": task or spoken,
            "scheduled_at": reminders.to_iso(scheduled) if scheduled else None,
            "status": "pending",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.reminders.append(item)
        reminders.save_reminders(st.session_state.reminders)
        if scheduled:
            st.session_state.reminder_status = (
                "success",
                f"Reminder saved: “{item['task']}” — scheduled for "
                f"{reminders.format_scheduled(item['scheduled_at'])}.",
            )
        else:
            st.session_state.reminder_status = (
                "warning",
                f"Reminder saved: “{item['task']}” — no time detected, so it is "
                "unscheduled (say a time such as “at 4pm” or “in 30 minutes”).",
            )
    elif spoken == "" and not st.session_state.reminder_status:
        st.session_state.reminder_status = (
            "warning",
            "No speech was recognised in that recording — please try again, a little "
            "closer to the microphone.",
        )

    # Bump the widget key so the same recording isn't processed again on rerun.
    st.session_state.reminder_audio_gen += 1
    st.rerun()

if not st.session_state.reminders:
    st.caption("No reminders yet — record one above.")
else:
    rows = [
        {
            "Task": item.get("task", ""),
            "Scheduled": reminders.format_scheduled(item.get("scheduled_at")),
            "Status": "✅ completed" if item.get("status") == "completed" else "⏳ pending",
        }
        for item in st.session_state.reminders
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=240)

    if any(item.get("status") == "completed" for item in st.session_state.reminders):
        if st.button("🗑 Clear completed reminders"):
            st.session_state.reminders = [
                item
                for item in st.session_state.reminders
                if item.get("status") != "completed"
            ]
            reminders.save_reminders(st.session_state.reminders)
            st.rerun()

_check_due_reminders()

# ---------------------------------------------------------------------------
# G. Find my phone or keys (UC2) — REAL detection, see lifeaware/item_finder.py
# ---------------------------------------------------------------------------


def _free_webcam_for_finder() -> None:
    """Stop the camera preview (C) and the fall check (H) so the item finder can use the webcam."""
    stop_fall_check()
    if st.session_state.camera_on:
        _stop_camera(("info", "📷 Preview stopped — the item finder (section G) is using the webcam."))


st.subheader("G · Find My Phone or Keys")
st.caption(
    "**Real AI, running locally:** the webcam feeds a pretrained YOLO model that "
    "recognises a phone, notes which area of the room it is in, and remembers the last "
    "sighting. Ask, and the answer is shown and spoken. Keys cannot be detected yet, "
    "so the honest answer for keys is “not sure”."
)
render_item_finder(on_start=_free_webcam_for_finder)

# ---------------------------------------------------------------------------
# H. Fall check (UC3) — REAL pose detection, see lifeaware/fall_panel.py
# ---------------------------------------------------------------------------


def _free_webcam_for_fall_check() -> None:
    """Stop the camera preview (C) and the item finder (G) so the fall check can use the webcam."""
    stop_item_finder()
    if st.session_state.camera_on:
        _stop_camera(("info", "📷 Preview stopped — the fall check (section H) is using the webcam."))


st.subheader("H · Fall Check")
st.caption(
    "**Real AI, running locally:** a pretrained YOLO pose model finds the person's body "
    "points and decides standing / sitting / lying. If someone lies on the floor without "
    "moving, the system asks “Are you okay?”. No answer → local alarm; a broken camera or "
    "microphone → UNKNOWN, never an alarm. Answers come from the buttons for now "
    "(simulated speech)."
)
render_fall_check(on_start=_free_webcam_for_fall_check)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "SCENE-DETECT prototype · Sections A–B are simulated · Section G uses a real "
    "pretrained object detector (phone only) · Section H uses a real pretrained pose "
    "model for the fall check (answers simulated) · No emotion recognition · "
    "Not a medical device and not a medical diagnosis."
)

# --- The camera frame is rendered last so the whole page always updates first ---
if st.session_state.camera_on:
    _show_camera_frame(preview_slot)
