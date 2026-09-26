"""Human-facing timestamps: the PC's local time zone on a 12-hour clock with AM/PM."""

from __future__ import annotations

from datetime import datetime

# Logging's strftime; zero-padded because %-I (POSIX) and %#I (Windows) aren't portable.
LOG_DATEFMT = "%Y-%m-%d %I:%M:%S %p"


def clock(dt: datetime | float) -> str:
    """'3:07:12 PM' — accepts a datetime or a Unix timestamp (converted to local time)."""
    if not isinstance(dt, datetime):
        dt = datetime.fromtimestamp(dt)
    return dt.strftime("%I:%M:%S %p").lstrip("0")


def stamp(dt: datetime | float) -> str:
    """'2026-09-24 3:07:12 PM'"""
    if not isinstance(dt, datetime):
        dt = datetime.fromtimestamp(dt)
    return f"{dt:%Y-%m-%d} {clock(dt)}"
