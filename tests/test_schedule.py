from datetime import datetime

import pytest

from camwatch.schedule import armed_at, next_change, parse_schedule, parse_window, split_windows

# 2026-10-12 is a Monday
MON, FRI, SAT, SUN = 12, 16, 17, 18


def at(day, hhmm):
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, day, h, m)


def test_parse_days_and_times():
    assert parse_window("18:00-07:00").days == frozenset(range(7))
    assert parse_window("Mon-Fri 18:00-07:00").days == frozenset({0, 1, 2, 3, 4})
    assert parse_window("fri-mon 9:00-17:00").days == frozenset({4, 5, 6, 0})  # wraps over the weekend
    assert parse_window("Sat,Sun 00:00-24:00").days == parse_window("saturday sunday 0:00-24:00").days == {5, 6}
    w = parse_window("SAT 00:00-24:00")
    assert (w.start, w.end, w.overnight) == (0, 86400, False)


@pytest.mark.parametrize("text", ["25:00-01:00", "18:00-18:00", "Funday 1:00-2:00", "satsun 1:00-2:00",
                                  "18:00-24:30", "24:00-01:00", "evening", ""])
def test_bad_windows_are_rejected(text):
    with pytest.raises(ValueError):
        parse_window(text)


def test_split_windows_rejoins_comma_separated_days():
    assert split_windows("Mon-Fri 18:00-07:00, Sat, Sun 22:00-07:00") == ["Mon-Fri 18:00-07:00", "Sat,Sun 22:00-07:00"]
    assert split_windows("  ") == []
    with pytest.raises(ValueError):
        split_windows("18:00-07:00, Sat")


def test_same_day_window():
    w = parse_schedule(["09:00-17:00"])
    assert not armed_at(w, at(MON, "8:59")) and armed_at(w, at(MON, "9:00"))
    assert armed_at(w, at(MON, "16:59")) and not armed_at(w, at(MON, "17:00"))


def test_overnight_window_belongs_to_its_start_day():
    w = parse_schedule(["Mon-Fri 18:00-07:00"])
    assert armed_at(w, at(FRI, "23:00")) and armed_at(w, at(SAT, "6:59"))  # Friday night
    assert not armed_at(w, at(SAT, "7:00")) and not armed_at(w, at(SAT, "18:00"))
    assert not armed_at(w, at(SUN, "23:00")) and not armed_at(w, at(MON + 7, "6:00"))  # Sunday night isn't
    assert armed_at(w, at(MON, "18:00"))


def test_several_windows_and_empty_schedule():
    w = parse_schedule(["Mon-Fri 18:00-07:00", "Sat-Sun 22:00-07:00"])
    assert armed_at(w, at(SAT, "23:00")) and not armed_at(w, at(SAT, "20:00"))
    assert armed_at(w, at(MON + 7, "6:00"))  # Sunday night window
    assert not armed_at([], at(MON, "12:00"))


def test_next_change():
    w = parse_schedule(["Mon-Fri 18:00-07:00"])
    assert next_change(w, at(MON, "12:00")) == at(MON, "18:00")
    assert next_change(w, at(MON, "18:00")) == at(MON + 1, "7:00")
    assert next_change(w, at(SAT, "12:00")) == at(MON + 7, "18:00")  # nothing over the weekend
    # back-to-back windows don't count as a change
    assert next_change(parse_schedule(["18:00-24:00", "00:00-07:00"]), at(MON, "20:00")) == at(MON + 1, "7:00")
    assert next_change(parse_schedule(["00:00-24:00"]), at(MON, "12:00")) is None
