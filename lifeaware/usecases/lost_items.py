"""UC2 – Find lost items.

The person asks "where is my phone?" (typed or a button for now; voice later).
We answer from ``ItemMemory``: the LAST sighting, or an honest "not sure".

Flow: query -> ItemMemory.answer() -> Event(use_case="UC2") -> say(answer)
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Callable

from lifeaware.memory import ItemMemory
from lifeaware.schema import Event

USE_CASE = "UC2"

# Words people use -> the item label used in ItemMemory.
ITEM_SYNONYMS = {
    "phone": "phone",
    "mobile": "phone",
    "cell phone": "phone",
    "cellphone": "phone",
    "smartphone": "phone",
    "handy": "phone",
    "keys": "keys",
    "key": "keys",
}


def parse_item_query(text: str) -> str | None:
    """Find which item a question is about: "where is my phone?" -> "phone".

    Returns None when no known item is mentioned.
    """
    words = re.sub(r"[^a-z ]", " ", (text or "").lower())
    words = f" {' '.join(words.split())} "
    # Longest synonyms first, so "cell phone" wins over "phone".
    for synonym in sorted(ITEM_SYNONYMS, key=len, reverse=True):
        if f" {synonym} " in words:
            return ITEM_SYNONYMS[synonym]
    return None


def handle_query(
    item: str,
    memory: ItemMemory,
    now: float | None = None,
    say: Callable[[str], None] | None = None,
) -> Event:
    """Answer "where is my <item>?" and return the UC2 Event.

    ``say`` is the voice output (e.g. ``reminders.speak``); if it fails, the
    Event is still returned, so the dashboard can show the answer as text.
    """
    now = time.time() if now is None else now
    answer = memory.answer(item, now=now)
    found = memory.last_seen(item) is not None

    if say is not None:
        try:
            say(answer)
        except Exception:
            pass  # no sound is better than a crash; the text answer remains

    return Event(
        time=now,
        use_case=USE_CASE,
        old_state="checking",
        new_state="solved" if found else "unknown",
        reason=answer,
    )


def handle_text_query(
    text: str,
    memory: ItemMemory,
    now: float | None = None,
    say: Callable[[str], None] | None = None,
) -> Event:
    """Like ``handle_query`` but for free text; unknown items give an honest reply."""
    item = parse_item_query(text)
    if item is not None:
        return handle_query(item, memory, now=now, say=say)

    now = time.time() if now is None else now
    reply = "Sorry, I can only look for your phone or your keys at the moment."
    if say is not None:
        try:
            say(reply)
        except Exception:
            pass
    return Event(time=now, use_case=USE_CASE, old_state="checking", new_state="unknown", reason=reply)


# Question buttons of the panel: item label -> button text.
QUESTION_BUTTONS = {"phone": "📱 Where is my phone?", "keys": "🔑 Where are my keys?"}

_ANSWER_CSS = """
<style>
.uc2-answer {border-radius:14px; padding:14px 18px; margin:6px 0 14px 0;
  background:rgba(148,163,184,.10); border:1px solid rgba(148,163,184,.35);
  border-left:6px solid #64748b;}
.uc2-answer.solved {border-left-color:#22c55e;}
.uc2-answer .uc2-q {font-size:.72rem; font-weight:800; letter-spacing:.06em;
  text-transform:uppercase; opacity:.65;}
.uc2-answer .uc2-a {font-size:1.12rem; font-weight:600; margin:4px 0 6px 0; line-height:1.4;}
.uc2-answer .uc2-note {font-size:.78rem; opacity:.7;}
</style>
"""


def _answer_card(event: Event) -> str:
    """HTML card for the last UC2 answer (green edge when the item was found)."""
    import html

    asked = datetime.fromtimestamp(event.time).strftime("%H:%M:%S")
    return (
        f'<div class="uc2-answer {event.new_state}">'
        f'<div class="uc2-q">Answer · asked at {asked}</div>'
        f'<div class="uc2-a">{html.escape(event.reason)}</div>'
        '<div class="uc2-note">Last sighting only — not a guarantee the item is still there.</div>'
        "</div>"
    )


def render_lost_items_panel(
    memory: ItemMemory,
    say: Callable[[str], None] | None = None,
    with_text_box: bool = False,
) -> None:
    """Streamlit panel for UC2: question buttons (optional text box) and the answer card."""
    import streamlit as st  # lazy import: keeps this module testable without Streamlit

    event = None
    columns = st.columns(len(QUESTION_BUTTONS))
    for column, (item, label) in zip(columns, QUESTION_BUTTONS.items()):
        if column.button(label, key=f"uc2_ask_{item}", width="stretch"):
            event = handle_query(item, memory, say=say)

    if with_text_box:
        question = st.text_input("Or type a question", key="uc2_question",
                                 placeholder="Where is my phone?")
        if st.button("Ask", key="uc2_ask_text") and question.strip():
            event = handle_text_query(question, memory, say=say)

    if event is not None:
        st.session_state["uc2_last_event"] = event
    last = st.session_state.get("uc2_last_event")
    if last is not None:
        st.markdown(_ANSWER_CSS + _answer_card(last), unsafe_allow_html=True)
    else:
        st.caption("Ask a question — the answer is shown here and spoken aloud.")
