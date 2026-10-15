import queue
from collections import deque

import pytest

from camwatch import engine as engine_mod
from camwatch.engine import Engine
from camwatch.events import EventResult


class FakeBot:
    def __init__(self):
        self.sent = []

    def send_alert(self, result, clip):
        self.sent.append(result.camera)


@pytest.fixture
def eng(cfg, monkeypatch, tmp_path):
    e = Engine.__new__(Engine)  # just the alert path, no cameras or models
    e.cfg, e.data, e.clips_dir = cfg, tmp_path, tmp_path / "clips"
    e.events, e._alerts, e._limiter, e.bot = deque(maxlen=100), queue.Queue(), engine_mod.AlertLimiter(), FakeBot()
    e.ntfy = None
    clips = []
    monkeypatch.setattr(engine_mod, "write_clip", lambda result, path, _cfg: clips.append(result.camera) or path)
    e.clips = clips
    return e


def result(camera, decision="alert"):
    return EventResult(camera, 0.0, decision, [], [], 0.0, 5.0, 0.0)


def deliver(eng, now, monkeypatch, *results):
    monkeypatch.setattr(engine_mod.time, "monotonic", lambda: now)
    for r in results:
        eng._on_result(r)
    while not eng._alerts.empty():
        eng._handle_alert(*eng._alerts.get_nowait())


def test_second_alert_on_same_camera_within_30s_is_clipped_but_not_sent(eng, monkeypatch):
    assert eng.cfg.telegram.min_alert_interval_seconds == 30
    first, second, other = result("door"), result("door"), result("garage")
    deliver(eng, 100.0, monkeypatch, first)
    deliver(eng, 110.0, monkeypatch, second, other)
    assert (first.decision, second.decision, other.decision) == ("alert", "rate-limited", "alert")
    assert eng.bot.sent == ["door", "garage"]
    assert eng.clips == ["door", "door", "garage"]  # the muted event still has its clip
    assert [e.decision for e in eng.events] == ["alert", "rate-limited", "alert"]


def test_alerts_resume_after_the_hard_cooldown(eng, monkeypatch):
    deliver(eng, 100.0, monkeypatch, result("door"))
    deliver(eng, 120.0, monkeypatch, result("door"))  # muted: doesn't restart the 30 s
    late = result("door")
    deliver(eng, 130.5, monkeypatch, late)
    assert late.decision == "alert" and eng.bot.sent == ["door", "door"]


def test_non_alert_decisions_are_untouched(eng, monkeypatch):
    deliver(eng, 100.0, monkeypatch, result("door"))
    trusted = result("door", "trusted")
    deliver(eng, 101.0, monkeypatch, trusted)
    assert trusted.decision == "trusted" and eng.clips == ["door"]


class BrokenBot(FakeBot):
    def send_alert(self, result, clip):
        raise RuntimeError("boom")


def test_every_channel_gets_the_alert_and_rate_limited_ones_reach_none(eng, monkeypatch):
    eng.ntfy = FakeBot()
    deliver(eng, 100.0, monkeypatch, result("door"))
    deliver(eng, 105.0, monkeypatch, result("door"))  # rate-limited
    assert eng.bot.sent == eng.ntfy.sent == ["door"]


def test_a_failing_channel_does_not_stop_the_others(eng, monkeypatch):
    eng.bot, eng.ntfy = BrokenBot(), FakeBot()
    deliver(eng, 100.0, monkeypatch, result("door"))
    assert eng.ntfy.sent == ["door"]
