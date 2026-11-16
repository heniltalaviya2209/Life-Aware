"""UC2 – Find lost items.

The person asks "where is my phone?" (typed or a button for now; voice later).
We answer from ``ItemMemory``: the LAST sighting, or an honest "not sure".

Flow: query -> ItemMemory.answer() -> Event(use_case="UC2") -> say(answer)
"""

from __future__ import annotations

import re
import time
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


def render_lost_items_panel(memory: ItemMemory, say: Callable[[str], None] | None = None) -> None:
    """Streamlit panel for UC2: quick buttons, a text box and the last answer.

    Meant to be placed in ``app.py`` by the dashboard owner, e.g.::

        from lifeaware.usecases.lost_items import render_lost_items_panel
        render_lost_items_panel(st.session_state.item_memory, say=reminders.speak)
    """
    import streamlit as st  # lazy import: keeps this module testable without Streamlit

    event = None
    left, right, _ = st.columns([1, 1, 2])
    if left.button("📱 Where is my phone?"):
        event = handle_query("phone", memory, say=say)
    if right.button("🔑 Where are my keys?"):
        event = handle_query("keys", memory, say=say)

    question = st.text_input("Or type a question", placeholder="Where is my phone?")
    if st.button("Ask") and question.strip():
        event = handle_text_query(question, memory, say=say)

    if event is not None:
        st.session_state["uc2_last_event"] = event
    last = st.session_state.get("uc2_last_event")
    if last is not None:
        show = st.success if last.new_state == "solved" else st.info
        show(last.reason)
        st.caption("This is the last sighting, not a guarantee that the item is still there.")
