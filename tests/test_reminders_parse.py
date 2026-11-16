"""Tests for reminders.parse_reminder (text -> task + scheduled time).

All tests pass a fixed ``now`` so the results do not depend on the clock.
No microphone, Vosk model or network is needed: only ``dateparser``.
"""

from datetime import datetime

import pytest

from reminders import parse_reminder

# Monday 16 Nov 2026, 10:00 — the fixed "current time" for every test.
NOW = datetime(2026, 11, 16, 10, 0)


def test_time_later_today_is_parsed():
    """A clock time later today keeps today's date and removes the command words."""
    task, scheduled = parse_reminder("remind me to take my medicine at 4pm", now=NOW)

    assert task == "take my medicine"
    assert scheduled == datetime(2026, 11, 16, 16, 0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known bug: 'tomorrow at 9' loses the hour and keeps the current time "
        "of day (gives 10:00 instead of 09:00). Remove this marker once fixed."
    ),
)
def test_tomorrow_with_hour_keeps_the_hour():
    """'tomorrow at 9' should mean 09:00 on the next day."""
    task, scheduled = parse_reminder("remind me to call mom tomorrow at 9", now=NOW)

    assert task == "call mom"
    assert scheduled == datetime(2026, 11, 17, 9, 0)
