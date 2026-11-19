"""Tests for the fall-check dashboard section (lifeaware.fall_panel). Fake pose model only."""

import copy
import csv
import os
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

    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=str(log),
                        use_process=False).start()
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

    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None,
                        use_process=False).start()
    wait_until(lambda: worker.check.state == "checking")
    events = worker.answer("answer_yes")
    worker.stop()

    assert [e.new_state for e in events] == ["solved"]
    assert events[0].answered == "yes"


def test_microphone_off_gives_unknown_instead_of_alarm(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config()
    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None,
                        use_process=False)
    worker.set_microphone(False)
    worker.start()
    wait_until(lambda: not worker.running)

    assert states(worker.check)[-1] == "unknown"
    assert "alarm" not in states(worker.check)


def test_acknowledge_after_alarm(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config()
    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None,
                        use_process=False).start()
    wait_until(lambda: worker.check.state == "alarm")
    worker.stop()

    [event] = worker.acknowledge()
    assert event.new_state == "solved"


def test_missing_video_gives_unknown_not_alarm(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    config = make_config()
    worker = FallWorker(config, str(tmp_path / "missing.mp4"), FallCheck.from_config(config),
                        log_path=None, use_process=False).start()
    wait_until(lambda: not worker.running)

    assert worker.status == "lost"
    assert states(worker.check) == ["unknown"]


def test_model_errors_are_shown(monkeypatch):
    def broken_loader(path):
        raise RuntimeError("pose model damaged")

    monkeypatch.setattr(pose, "load_model", broken_loader)
    config = make_config()
    worker = FallWorker(config, 0, FallCheck.from_config(config), log_path=None,
                        use_process=False).start()
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



def test_the_separate_process_reports_errors_back_to_the_page():
    # A zone with only 2 points fails before any model is loaded, so this test
    # checks the real process plumbing (spawn, queue, error message) quickly.
    config = make_config()
    config["zones"] = {"broken": [[0, 0], [10, 10]]}
    worker = FallWorker(config, 0, FallCheck.from_config(config), log_path=None,
                        use_process=True).start()
    wait_until(lambda: not worker.running, timeout=60)

    assert worker.status == "error"
    assert "at least 3 points" in worker.error
    assert worker.check.state == "normal"  # a broken camera side never raises an alarm


def test_stop_ends_the_worker(tmp_path, monkeypatch):
    fake_lying_model(monkeypatch)
    clip = write_test_video(tmp_path / "clip.avi", seconds=3, fps=30)
    config = make_config()
    worker = FallWorker(config, str(clip), FallCheck.from_config(config), log_path=None,
                        use_process=False).start()
    time.sleep(0.3)
    worker.stop()
    assert not worker.running
    assert worker.status == "stopped"


def test_the_camera_process_does_not_rerun_the_main_script(tmp_path):
    import subprocess
    import sys
    import textwrap

    # A "main script" that leaves a marker file whenever it is executed.
    marker = tmp_path / "main_ran.txt"
    script = tmp_path / "fake_app.py"
    script.write_text(textwrap.dedent(f"""
        import multiprocessing, os, sys
        sys.path.insert(0, {os.getcwd()!r})
        with open({str(marker)!r}, "a") as fh:
            fh.write("x")
        from lifeaware.fall_panel import _start_without_main
        if __name__ == "__main__":
            ctx = multiprocessing.get_context("spawn")
            p = ctx.Process(target=print, args=("child ok",))
            _start_without_main(p)
            p.join(30)
            sys.exit(p.exitcode)
    """), encoding="utf-8")

    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "x"  # only the parent ran the script, not the child
