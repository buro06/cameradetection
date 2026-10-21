import threading
from datetime import datetime
from types import SimpleNamespace

import pytest
import requests

from camwatch import switches
from camwatch.config import CameraConfig, SwitchConfig
from camwatch.switches import SwitchWorker

NOON, NIGHT = datetime(2026, 10, 12, 12, 0), datetime(2026, 10, 12, 22, 0)
NEVER = float("-inf")


class FakeHttp:
    """Stands in for requests.get: records URLs, answers with `status` or raises `error`."""

    def __init__(self):
        self.urls, self.status, self.error = [], 200, None

    def __call__(self, url, timeout):
        self.urls.append(url)
        if self.error:
            raise self.error
        return self

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} Server Error")


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(switches.requests, "get", fake)
    return fake


@pytest.fixture
def eng(cfg):
    cfg.cameras = [CameraConfig("Backyard", "0"), CameraConfig("driveway", "1")]
    return SimpleNamespace(cfg=cfg, _cam_lock=threading.RLock(),
                           monitors={c.name: SimpleNamespace(last_activity=NEVER) for c in cfg.cameras})


def light(eng, cameras=("backyard",), **kw):
    return SwitchWorker(eng, SwitchConfig("lights", list(cameras), "http://relay/on", "http://relay/off", **kw))


def seen(eng, camera, t):
    eng.monitors[camera].last_activity = t


def test_first_step_switches_off_then_nothing_while_idle(eng, http):
    sw = light(eng)
    for t in range(0, 60, 1):
        sw.step(float(t), NOON)
    assert http.urls == ["http://relay/off"] and sw.status == "off"


def test_on_at_activity_and_off_120s_after_the_last(eng, http):
    sw = light(eng)
    sw.step(0.0, NOON)
    for t in range(10, 70):  # someone in view for a minute
        seen(eng, "Backyard", float(t))
        sw.step(float(t), NOON)
    sw.step(69.0 + 119, NOON)
    assert http.urls == ["http://relay/off", "http://relay/on"] and sw.status == "on"
    sw.step(69.0 + 121, NOON)
    sw.step(69.0 + 200, NOON)
    assert http.urls == ["http://relay/off", "http://relay/on", "http://relay/off"]


def test_a_shared_light_stays_on_while_either_camera_sees_someone(eng, http):
    sw = light(eng, cameras=["backyard", "driveway"])
    sw.step(0.0, NOON)
    seen(eng, "Backyard", 10.0)
    sw.step(10.0, NOON)
    seen(eng, "driveway", 100.0)
    sw.step(135.0, NOON)  # backyard quiet for 125 s, driveway for 35 s
    assert http.urls[-1] == "http://relay/on"
    sw.step(221.0, NOON)
    assert http.urls == ["http://relay/off", "http://relay/on", "http://relay/off"]


def test_schedule_keeps_it_off_outside_its_window_and_ends_it_inside(eng, http):
    sw = light(eng, schedule=["17:00-23:00"])
    seen(eng, "Backyard", 10.0)
    sw.step(10.0, NOON)
    assert http.urls == ["http://relay/off"]
    sw.step(11.0, NIGHT)  # window opens while someone is in view
    assert http.urls[-1] == "http://relay/on"
    sw.step(12.0, datetime(2026, 10, 12, 23, 0))  # window ends
    assert http.urls[-1] == "http://relay/off"


@pytest.mark.parametrize("failure", ["status", "exception"])
def test_failed_request_is_retried_after_5s(eng, http, failure):
    sw = light(eng)
    sw.step(0.0, NOON)
    seen(eng, "Backyard", 10.0)
    if failure == "status":
        http.status = 500
    else:
        http.error = requests.ConnectionError("unreachable")
    sw.step(10.0, NOON)
    sw.step(14.0, NOON)  # too soon to retry
    assert http.urls == ["http://relay/off", "http://relay/on"] and sw.status == "off, unreachable"
    http.status, http.error = 200, None
    seen(eng, "Backyard", 15.0)
    sw.step(15.0, NOON)
    assert http.urls == ["http://relay/off", "http://relay/on", "http://relay/on"] and sw.status == "on"


def test_stopped_or_unknown_cameras_are_ignored(eng, http):
    del eng.monitors["driveway"]
    sw = light(eng, cameras=["driveway", "garage", "backyard"])
    seen(eng, "Backyard", 5.0)
    sw.step(5.0, NOON)
    assert http.urls == ["http://relay/on"]


def test_switch_off_at_shutdown_only_if_on(eng, http):
    sw = light(eng)
    sw.step(0.0, NOON)
    sw.switch_off()
    assert http.urls == ["http://relay/off"]
    seen(eng, "Backyard", 10.0)
    sw.step(10.0, NOON)
    sw.switch_off()
    assert http.urls == ["http://relay/off", "http://relay/on", "http://relay/off"]


# ---- live changes from the menu --------------------------------------------------------------------
@pytest.fixture
def engine(eng, monkeypatch):
    from camwatch.engine import Engine
    e = Engine.__new__(Engine)  # just the switches, no cameras or models
    e.cfg, e.monitors, e._cam_lock, e.switches = eng.cfg, eng.monitors, eng._cam_lock, []
    monkeypatch.setattr(SwitchWorker, "start", lambda self: None)  # tests drive step() themselves
    return e


def add(engine, **kw):
    sw = SwitchConfig("lights", ["backyard"], "http://relay/on", "http://relay/off", **kw)
    engine.cfg.http_get.append(sw)
    engine.apply_switches()
    return sw


def test_editing_a_switch_keeps_its_state_without_resending(engine, http):
    sw = add(engine)
    seen(engine, "Backyard", 10.0)
    engine.switches[0].step(10.0, NOON)
    sw.off_after_seconds, sw.name = 300.0, "porch-lights"
    engine.apply_switches()
    assert http.urls == ["http://relay/on"] and engine.switches[0].state is True
    engine.switches[0].step(250.0, NOON)  # 240 s after: off_after_seconds is now 300
    assert http.urls == ["http://relay/on"]


def test_new_urls_turn_the_old_device_off_and_sync_the_new_one(engine, http):
    sw = add(engine)
    seen(engine, "Backyard", 10.0)
    engine.switches[0].step(10.0, NOON)
    sw.on_url, sw.off_url = "http://other/on", "http://other/off"
    engine.apply_switches()
    engine.switches[0].step(11.0, NOON)
    assert http.urls == ["http://relay/on", "http://relay/off", "http://other/on"]


@pytest.mark.parametrize("how", ["remove", "disable"])
def test_removed_or_disabled_switch_is_turned_off(engine, http, how):
    sw = add(engine)
    seen(engine, "Backyard", 10.0)
    engine.switches[0].step(10.0, NOON)
    if how == "remove":
        engine.cfg.http_get.remove(sw)
    else:
        sw.enabled = False
    engine.apply_switches()
    assert http.urls == ["http://relay/on", "http://relay/off"] and engine.switches == []


def test_resync_after_a_test_sends_what_it_should_be(engine, http):
    sw = add(engine)
    engine.switches[0].step(0.0, NOON)  # off
    http.urls.clear()
    engine.resync_switch(sw)  # the menu's test left the light in an unknown state
    engine.switches[0].step(1.0, NOON)
    assert http.urls == ["http://relay/off"]


@pytest.mark.parametrize("on, off", [
    ("http://10.0.0.5/relay/0?turn=on", "http://10.0.0.5/relay/0?turn=off"),
    ("http://10.0.0.5/rpc/Switch.Set?id=0&on=true", "http://10.0.0.5/rpc/Switch.Set?id=0&on=false"),
    ("http://10.0.0.6/cm?cmnd=Power%20On", "http://10.0.0.6/cm?cmnd=Power%20Off"),
    ("http://hub/lights/on", "http://hub/lights/off"),
    ("http://hub/webhook/abc", ""),
])
def test_off_url_is_guessed_from_the_on_url(on, off):
    from camwatch.menus import guess_off_url
    assert guess_off_url(on) == off
