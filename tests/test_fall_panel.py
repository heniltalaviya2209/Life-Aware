"""Tests for the fall-check dashboard section (lifeaware.fall_panel). Fake pose model only."""

import copy
import csv
import time

import numpy as np
from conftest import write_test_video
from test_pose import FakeModel

from lifeaware import pose
from lifeaware.config import DEFAULTS
from lifeaware.fall_panel import FallWorker, _ordinal, append_event_log, events_table, state_card
from lifeaware.schema import Event
from lifeaware.usecases.fall_check import FallCheck

# A person lying flat inside the 64 x 48 test frame (box does not touch the edges).
LYING_BOX = [5, 28, 55, 42]
LYING_KPS = np.zeros((17, 3))
LYING_KPS[5], LYING_KPS[6] = [10, 33, 0.9], [10, 37, 0.9]      # shoulders
LYING_KPS[11], LYING_KPS[12] = [32, 34, 0.9], [32, 38, 0.9]    # hips
LYING_KPS[13], LYING_KPS[14] = [48, 34, 0.9], [48, 38, 0.9]    # knees


def make_config(**fall):
    config = copy.deepcopy(DEFAULTS)
    config["zones"] = {"floor": [[0, 0], [64, 0], [64, 48], [0, 48]]}
    config["pose"]["every_n_frames"] = 1
    config["fall"].update({"min_lying_s": 0.3, "still_s": 0.6, "answer_timeout_s": 0.6,
                           "max_gap_s": 1.0, **fall})
    return config


def fake_lying_model(monkeypatch):
    model = FakeModel()
    model.people = [(LYING_BOX, 0.9, LYING_KPS)]
    monkeypatch.setattr(pose, "load_model", lambda path: model)
    return model


def wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.05)


def states(check):
    return [e.new_state for e in check.events]


def test_worker_runs_the_whole_chain_to_an_alarm(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config()
    log = tmp_path / "events.csv"

    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=str(log)).start()
    wait_until(lambda: not worker.running)

    assert worker.error is None
    assert worker.status == "ended"
    assert worker.latest_view[:2] == b"\xff\xd8"  # a JPEG picture for the page
    assert worker.check_ms >= 0 and worker.fps > 0
    assert states(worker.check) == ["possible_event", "checking", "alarm"]
    rows = list(csv.reader(log.open(encoding="utf-8")))
    assert rows[0][0] == "time" and [r[4] for r in rows[1:]] == ["possible_event", "checking", "alarm"]


def test_answer_yes_from_the_page_solves(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config(answer_timeout_s=30)

    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None).start()
    wait_until(lambda: worker.check.state == "checking")
    events = worker.answer("answer_yes")
    worker.stop()

    assert [e.new_state for e in events] == ["solved"]
    assert events[0].answered == "yes"


def test_microphone_off_gives_unknown_instead_of_alarm(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config()
    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None)
    worker.set_microphone(False)
    worker.start()
    wait_until(lambda: not worker.running)

    assert states(worker.check)[-1] == "unknown"
    assert "alarm" not in states(worker.check)


def test_acknowledge_after_alarm(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config()
    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None).start()
    wait_until(lambda: worker.check.state == "alarm")
    worker.stop()

    [event] = worker.acknowledge()
    assert event.new_state == "solved"


def test_missing_video_gives_unknown_not_alarm(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    config = make_config()
    worker = FallWorker(config, str(tmp_path / "missing.mp4"), FallCheck.from_config(config),
                        log_path=None).start()
    wait_until(lambda: not worker.running)

    assert worker.status == "lost"
    assert states(worker.check) == ["unknown"]


def test_model_errors_are_shown(monkeypatch):
    def broken_loader(path):
        raise RuntimeError("pose model damaged")

    monkeypatch.setattr(pose, "load_model", broken_loader)
    config = make_config()
    worker = FallWorker(config, 0, FallCheck.from_config(config), log_path=None).start()
    wait_until(lambda: not worker.running)

    assert worker.status == "error"
    assert "pose model damaged" in worker.error


# ----------------------------------------------------------------- page parts


def test_state_card_shows_state_reason_and_microphone():
    event = Event(time=time.time(), use_case="UC3", old_state="checking", new_state="alarm",
                  reason="No answer for 15 s.", answered=None)
    card = state_card("alarm", event, mic_ok=False)
    assert "ALARM" in card and "No answer for 15 s." in card
    assert "fc-state bad" in card and "NOT working" in card


def test_state_card_escapes_html():
    event = Event(time=1.0, use_case="UC3", old_state="normal", new_state="unknown", reason="<b>x</b>")
    assert "<b>x</b>" not in state_card("unknown", event, mic_ok=True)


def test_events_table_is_newest_first():
    events = [Event(time=float(i), use_case="UC3", old_state="a", new_state=f"s{i}", reason="r")
              for i in range(3)]
    table = events_table(events)
    assert list(table["Change"]) == ["a → s2", "a → s1", "a → s0"]


def test_append_event_log_writes_header_once(tmp_path):
    path = tmp_path / "sub" / "log.csv"
    event = Event(time=1.0, use_case="UC3", old_state="normal", new_state="alarm", reason="x", answered="no")
    append_event_log([event], str(path))
    append_event_log([event], str(path))
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert len(rows) == 3 and rows[0][0] == "time" and rows[1][6] == "no"


def test_ordinal():
    assert [_ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]



def test_stop_ends_the_worker(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config()
    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None).start()
    time.sleep(0.3)
    worker.stop()
    assert not worker.running
    assert worker.status == "stopped"

