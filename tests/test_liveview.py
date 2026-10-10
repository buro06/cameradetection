import cv2
import numpy as np
import pytest

from camwatch import liveview
from camwatch.liveview import LiveView, window_title


class FakeStream:
    status = "live"

    def __init__(self, live=True):
        self.live = live

    def is_live(self):
        return self.live


class FakeEngine:
    def __init__(self, *names):
        self.streams = {n: FakeStream() for n in names}
        self.frames = {n: np.zeros((1080, 1920, 3), np.uint8) for n in names}

    def annotated_frame(self, name):
        if name not in self.streams:
            raise KeyError(name)
        return self.frames[name]


class FakeWindows:
    """Stands in for OpenCV's window functions."""

    def __init__(self):
        self.open: dict[str, tuple] = {}
        self.sizes: dict[str, tuple] = {}
        self.key = -1
        self.broken = False

    def install(self, monkeypatch):
        mp = lambda name, fn: monkeypatch.setattr(liveview.cv2, name, fn)
        mp("namedWindow", lambda t, flags=0: self.open.setdefault(t, None))
        mp("resizeWindow", lambda t, w, h: self.sizes.__setitem__(t, (w, h)))
        mp("imshow", self.imshow)
        mp("waitKey", lambda ms=0: self.key)
        mp("getWindowProperty", lambda t, prop: 1.0 if t in self.open else -1.0)
        mp("destroyWindow", lambda t: self.open.pop(t, None))
        monkeypatch.setattr(liveview.time, "monotonic", self.clock)
        self.t = 0.0
        return self

    def clock(self):
        self.t += 1.0  # every pump is due
        return self.t

    def imshow(self, title, img):
        if self.broken:
            raise cv2.error("The function is not implemented. Rebuild the library with Windows support")
        self.open[title] = img.shape


@pytest.fixture
def win(monkeypatch):
    return FakeWindows().install(monkeypatch)


def test_toggle_opens_a_window_per_camera_and_closes_them(win):
    view = LiveView(FakeEngine("door", "den"))
    view.toggle(["door", "den"])
    view.pump()
    assert set(win.open) == {window_title("door"), window_title("den")}
    assert win.sizes[window_title("door")] == (960, 540)  # starts small, keeps the aspect ratio
    view.toggle(["door", "den"])
    assert not win.open and not view.active


def test_closing_one_window_keeps_the_others(win):
    view = LiveView(FakeEngine("door", "den"))
    view.toggle(["door", "den"])
    view.pump()
    del win.open[window_title("door")]  # user clicked the close button
    view.pump()
    assert view.cameras == ["den"] and set(win.open) == {window_title("den")}


def test_escape_closes_all_windows(win):
    view = LiveView(FakeEngine("door"))
    view.toggle(["door"])
    view.pump()
    win.key = 27
    view.pump()
    assert not win.open and not view.active


def test_windows_hide_during_menus_and_come_back(win):
    view = LiveView(FakeEngine("door"))
    view.toggle(["door"])
    view.pump()
    view.hide()
    assert not win.open and view.active
    view.pump()
    assert set(win.open) == {window_title("door")}


def test_removed_camera_drops_its_window(win):
    eng = FakeEngine("door", "den")
    view = LiveView(eng)
    view.toggle(["door", "den"])
    view.pump()
    del eng.streams["den"]
    view.pump()
    assert view.cameras == ["door"] and window_title("den") not in win.open


def test_offline_camera_shows_a_placeholder(win):
    eng = FakeEngine("door")
    eng.frames["door"] = None
    eng.streams["door"].live = False
    view = LiveView(eng)
    view.toggle(["door"])
    view.pump()
    assert win.open[window_title("door")] == (360, 640, 3)


def test_opencv_without_window_support_is_reported_not_raised(win, caplog):
    win.broken = True
    view = LiveView(FakeEngine("door"))
    view.toggle(["door"])
    view.pump()
    assert not view.active and "opencv-python-headless" in caplog.text
