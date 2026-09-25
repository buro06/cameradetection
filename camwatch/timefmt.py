"""Human-facing timestamps: the PC's local time zone on a 12-hour clock with AM/PM."""

from __future__ import annotations

import time
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


def ago(ts: float) -> str:
    """'45s ago', '12m ago', '3h ago', '2d ago'."""
    s = int(time.time() - ts)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    return f"{s // 3600}h ago" if s < 86400 else f"{s // 86400}d ago"
