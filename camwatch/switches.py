"""HTTP GET on/off (outdoor lights, a relay, a webhook, ...): a GET to `on_url` when a camera sees someone, to
`off_url` once it doesn't.

A target ("switch" here) is on while any of its cameras saw a person or moving vehicle that arrived (the alert rules' arrival
check, so a brief false detection doesn't count) in the last `off_after_seconds`, and its schedule allows it.
Each switch runs on its own thread, so an unreachable device never holds up detection or the other switches.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING

import requests

from .config import SwitchConfig
from .schedule import armed_at, parse_schedule

if TYPE_CHECKING:
    from .engine import Engine

log = logging.getLogger(__name__)

STEP_SECONDS = 0.2  # how often the switch looks at its cameras
RETRY_SECONDS = 5.0  # a failed request is retried this long after
HTTP_TIMEOUT = 5.0


def get(url: str) -> None:
    """One GET; raises requests.RequestException if it fails or the device answers with an HTTP error."""
    requests.get(url, timeout=HTTP_TIMEOUT).raise_for_status()


class SwitchWorker(threading.Thread):
    def __init__(self, engine: Engine, cfg: SwitchConfig):
        super().__init__(name=f"http-get {cfg.name}", daemon=True)
        self.engine, self.cfg = engine, cfg
        # fixed while it runs: the menu edits `cfg` in place, then Engine.apply_switches starts a new worker
        self.urls = (cfg.on_url, cfg.off_url)
        self.windows = parse_schedule(cfg.schedule)
        self.state: bool | None = None  # what the device was last switched to (None: not yet, e.g. at startup)
        self.failing = False  # the last request didn't get through
        self._retry_at = float("-inf")
        self._halt = threading.Event()

    def last_activity(self) -> float:
        """Latest confirmed activity (monotonic) on any of the switch's running cameras."""
        eng = self.engine
        names = [cam.name for c in self.cfg.cameras if (cam := eng.cfg.camera(c))]
        with eng._cam_lock:
            monitors = [m for n in names if (m := eng.monitors.get(n))]
        return max((m.last_activity for m in monitors), default=float("-inf"))

    def wanted(self, now: float, wall: datetime) -> bool:
        if self.windows and not armed_at(self.windows, wall):
            return False
        return now - self.last_activity() <= self.cfg.off_after_seconds

    def step(self, now: float, wall: datetime) -> None:
        """Switch the device if what it should be differs from what it was last switched to."""
        want = self.wanted(now, wall)
        if want != self.state and now >= self._retry_at:
            self._send(want, now)

    def _send(self, on: bool, now: float) -> bool:
        what = "on" if on else "off"
        try:
            get(self.urls[0] if on else self.urls[1])
        except requests.RequestException as e:
            self._retry_at = now + RETRY_SECONDS
            if not self.failing:
                log.warning("[http-get %s] turning %s failed (%s); retrying every %gs", self.cfg.name, what, e,
                            RETRY_SECONDS)
            self.failing = True
            return False
        if self.failing:
            log.info("[http-get %s] reachable again", self.cfg.name)
        self.failing, self.state = False, on
        log.info("[http-get %s] %s", self.cfg.name, what)
        return True

    def run(self) -> None:
        while not self._halt.wait(STEP_SECONDS):
            try:
                self.step(time.monotonic(), datetime.now())
            except Exception:
                log.exception("[http-get %s] error", self.cfg.name)

    def stop(self) -> None:
        self._halt.set()

    def switch_off(self) -> None:
        """At shutdown, once the thread has stopped: leave the device off."""
        if self.state:
            self._send(False, time.monotonic())

    @property
    def status(self) -> str:
        state = {True: "on", False: "off", None: "unknown"}[self.state]
        return f"{state}, unreachable" if self.failing else state
