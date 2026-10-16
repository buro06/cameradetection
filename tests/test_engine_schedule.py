from datetime import datetime

import pytest

from camwatch.config import CameraConfig
from camwatch.engine import Engine

MON = 12  # 2026-10-12 is a Monday


def at(day, hhmm):
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, day, h, m)


@pytest.fixture
def eng(cfg, tmp_path):
    e = Engine.__new__(Engine)  # just arming and schedules, no cameras or models
    e.cfg = cfg
    e.armed, e.disarmed_cameras, e._sched_state = True, set(), {}
    e._state_path = tmp_path / "state.json"
    cfg.cameras = [CameraConfig("door", "0", schedule=["18:00-07:00"]), CameraConfig("desk", "1")]
    return e


def test_first_check_applies_the_current_state(eng):
    eng.apply_schedules(at(MON, "12:00"))
    assert not eng.is_armed("door") and eng.is_armed("desk")  # no schedule: untouched
    assert eng._state_path.exists()  # survives a restart


def test_window_start_arms_and_end_disarms(eng):
    eng.apply_schedules(at(MON, "12:00"))
    eng.apply_schedules(at(MON, "18:00"))
    assert eng.is_armed("door")
    eng.apply_schedules(at(MON + 1, "6:59"))
    assert eng.is_armed("door")
    eng.apply_schedules(at(MON + 1, "7:00"))
    assert not eng.is_armed("door")


def test_manual_change_holds_until_the_next_scheduled_change(eng):
    eng.apply_schedules(at(MON, "12:00"))
    eng.set_armed("door", True)  # e.g. /arm door while away for the day
    eng.apply_schedules(at(MON, "12:05"))
    assert eng.is_armed("door")
    eng.apply_schedules(at(MON, "19:00"))
    eng.set_armed("door", False)  # /disarm door for the evening
    eng.apply_schedules(at(MON, "23:00"))
    assert not eng.is_armed("door")
    eng.apply_schedules(at(MON + 1, "7:00"))
    eng.apply_schedules(at(MON + 1, "18:00"))  # back on schedule
    assert eng.is_armed("door")


def test_disarm_all_is_not_undone_by_the_schedule(eng):
    eng.apply_schedules(at(MON, "12:00"))
    eng.set_armed(None, False)
    eng.apply_schedules(at(MON, "18:00"))
    assert not eng.armed and not eng.is_armed("door") and not eng.is_armed("desk")
    eng.set_armed(None, True)
    assert eng.is_armed("door") and eng.is_armed("desk")


def test_edited_schedule_applies_right_away(eng):
    eng.apply_schedules(at(MON, "12:00"))
    eng.set_armed("door", True)
    eng.cfg.camera("door").schedule = ["20:00-06:00"]
    eng.reset_schedule("door")
    eng.apply_schedules(at(MON, "12:01"))
    assert not eng.is_armed("door")


def test_schedule_text(eng):
    assert eng.schedule_text("door", at(MON, "12:00")) == "arms 6:00 PM"
    assert eng.schedule_text("door", at(MON, "20:00")) == "disarms Tue 7:00 AM"
    assert eng.schedule_text("desk", at(MON, "12:00")) == ""
