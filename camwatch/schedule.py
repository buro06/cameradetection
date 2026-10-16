"""Per-camera alert schedules: windows like "18:00-07:00" or "Mon-Fri 18:00-07:00" when a camera is armed.

A window that crosses midnight belongs to the day it starts on: "Mon-Fri 18:00-07:00" covers Friday night
until Saturday 07:00, but not Sunday night.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
EXAMPLE = "e.g. 18:00-07:00 or Mon-Fri 18:00-23:30"
_WINDOW = re.compile(r"^(?:(?P<days>[a-z,\- ]+?)\s+)?(?P<start>\d{1,2}:\d{2})\s*-\s*(?P<end>\d{1,2}:\d{2})$")


@dataclass(frozen=True)
class Window:
    days: frozenset[int]  # weekdays the window starts on (0 = Monday)
    start: int  # seconds after midnight
    end: int  # seconds after midnight, up to 24:00; <= start means it ends the next day

    @property
    def overnight(self) -> bool:
        return self.end <= self.start


def _seconds(hhmm: str, text: str, allow_24: bool) -> int:
    h, m = map(int, hhmm.split(":"))
    if m > 59 or h > 24 or (h == 24 and (m or not allow_24)):
        raise ValueError(f"Bad time {hhmm!r} in schedule {text!r} ({EXAMPLE})")
    return h * 3600 + m * 60


def _day(name: str, text: str) -> int:
    name = name.strip()
    if name in DAYS:
        return DAYS.index(name)
    if name in DAY_NAMES:
        return DAY_NAMES.index(name)
    raise ValueError(f"Unknown day {name!r} in schedule {text!r} (use Mon … Sun)")


def _days(spec: str | None, text: str) -> frozenset[int]:
    if not spec:
        return frozenset(range(7))
    days: set[int] = set()
    for part in re.split(r"[,\s]+", re.sub(r"\s*-\s*", "-", spec.strip())):
        if "-" in part:
            a, b = (_day(x, text) for x in part.split("-", 1))
            days.update((a + i) % 7 for i in range((b - a) % 7 + 1))  # Fri-Mon wraps over the weekend
        elif part:
            days.add(_day(part, text))
    return frozenset(days)


def parse_window(text: str) -> Window:
    m = _WINDOW.match(text.strip().lower())
    if not m:
        raise ValueError(f"Can't read schedule {text!r} ({EXAMPLE})")
    start, end = _seconds(m["start"], text, False), _seconds(m["end"], text, True)
    if start == end:
        raise ValueError(f"Schedule {text!r} starts and ends at the same time (use 00:00-24:00 for all day)")
    return Window(_days(m["days"], text), start, end)


def split_windows(text: str) -> list[str]:
    """Comma-separated windows as typed in the menu. Commas also separate days ("Sat,Sun 10:00-12:00"),
    so pieces without a time are joined onto the window that follows."""
    windows, days = [], []
    for piece in (p.strip() for p in text.split(",")):
        if not piece:
            continue
        if ":" in piece:
            windows.append(f"{','.join(days)},{piece}" if days else piece)
            days = []
        else:
            days.append(piece)
    if days:
        raise ValueError(f"{', '.join(days)!r} has no time range ({EXAMPLE})")
    return windows


def parse_schedule(items: list[str]) -> list[Window]:
    return [parse_window(s) for s in items]


def _secs(dt: datetime) -> int:
    return dt.hour * 3600 + dt.minute * 60 + dt.second


def armed_at(windows: list[Window], dt: datetime) -> bool:
    t, day = _secs(dt), dt.weekday()
    for w in windows:
        if not w.overnight:
            if day in w.days and w.start <= t < w.end:
                return True
        elif (day in w.days and t >= w.start) or ((day - 1) % 7 in w.days and t < w.end):
            return True
    return False


def next_change(windows: list[Window], dt: datetime) -> datetime | None:
    """When the schedule next flips between armed and disarmed (None if it never does)."""
    now = armed_at(windows, dt)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    edges = set()
    for offset in range(-1, 9):
        day = midnight + timedelta(days=offset)
        for w in windows:
            if day.weekday() in w.days:
                edges.add(day + timedelta(seconds=w.start))
                edges.add(day + timedelta(days=w.overnight, seconds=w.end))
    return next((e for e in sorted(edges) if e > dt and armed_at(windows, e) != now), None)
