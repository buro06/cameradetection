import logging

import cv2
import pytest

import numpy as np

import camwatch.camera as camera
import camwatch.menus as menus
import camwatch.usbcams as usbcams
from camwatch.config import CameraConfig, ClipConfig


class FakeCapture:
    """Records property sets; delivers `native` size unless `honor` is True."""

    def __init__(self, native=(640, 360), honor=True):
        self.calls, self.native, self.honor = [], native, honor
        self.size = list(native)

    def __call__(self, index, backend):
        self.opened_with = (index, backend)
        return self

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.calls.append((prop, value))
        if self.honor and prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.size[0] = int(value)
        if self.honor and prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.size[1] = int(value)
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.size[0], cv2.CAP_PROP_FRAME_HEIGHT: self.size[1]}.get(prop, 0)

    def read(self):
        return True, np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def release(self):
        pass


def open_usb(monkeypatch, platform, fake, **cam):
    monkeypatch.setattr(usbcams.sys, "platform", platform)
    monkeypatch.setattr(usbcams.cv2, "VideoCapture", fake)
    stream = camera.CameraStream(CameraConfig(name="usb", source="0", **cam), ClipConfig(), 2.0)
    stream._open()
    return fake


MJPG = cv2.VideoWriter_fourcc(*"MJPG")


def test_windows_requests_mjpg_before_resolution(monkeypatch):
    fake = open_usb(monkeypatch, "win32", FakeCapture(), width=1920, height=1080)
    assert fake.opened_with == (0, cv2.CAP_DSHOW)
    assert fake.calls == [(cv2.CAP_PROP_FOURCC, MJPG), (cv2.CAP_PROP_FRAME_WIDTH, 1920), (cv2.CAP_PROP_FRAME_HEIGHT, 1080)]


def test_auto_fourcc_is_left_alone_on_macos(monkeypatch):
    fake = open_usb(monkeypatch, "darwin", FakeCapture(), width=1920, height=1080)
    assert (cv2.CAP_PROP_FOURCC, MJPG) not in fake.calls


def test_linux_uses_v4l2_and_asks_for_mjpg_above_640x480(monkeypatch):
    fake = open_usb(monkeypatch, "linux", FakeCapture(), width=1920, height=1080)
    assert fake.opened_with == (0, cv2.CAP_V4L2)
    assert fake.calls[0] == (cv2.CAP_PROP_FOURCC, MJPG)
    small = open_usb(monkeypatch, "linux", FakeCapture(), width=640, height=480)
    assert (cv2.CAP_PROP_FOURCC, MJPG) not in small.calls


def test_fourcc_none_and_explicit(monkeypatch):
    assert not any(p == cv2.CAP_PROP_FOURCC for p, _ in open_usb(monkeypatch, "win32", FakeCapture(), fourcc="none").calls)
    yuy2 = cv2.VideoWriter_fourcc(*"YUY2")
    assert (cv2.CAP_PROP_FOURCC, yuy2) in open_usb(monkeypatch, "darwin", FakeCapture(), fourcc="yuy2").calls


def test_no_resolution_requested_by_default(monkeypatch):
    fake = open_usb(monkeypatch, "win32", FakeCapture())
    assert not any(p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) for p, _ in fake.calls)


class FormatDroppingCapture(FakeCapture):
    """Like DirectShow: changing the FPS restarts the device in YUY2; `sticky` = MJPG can be re-applied."""

    def __init__(self, sticky=True):
        super().__init__(native=(1920, 1080))
        self.fourcc, self.sticky = 0, sticky

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOURCC and (self.sticky or not self.fourcc):
            self.fourcc = int(value)
        if prop == cv2.CAP_PROP_FPS:
            self.fourcc = YUY2
        return super().set(prop, value)

    def get(self, prop):
        return self.fourcc if prop == cv2.CAP_PROP_FOURCC else super().get(prop)


YUY2 = cv2.VideoWriter_fourcc(*"YUY2")


def test_fps_is_requested_after_resolution(monkeypatch):
    fake = open_usb(monkeypatch, "win32", FakeCapture(), width=1920, height=1080, fps=30)
    assert fake.calls == [(cv2.CAP_PROP_FOURCC, MJPG), (cv2.CAP_PROP_FRAME_WIDTH, 1920),
                          (cv2.CAP_PROP_FRAME_HEIGHT, 1080), (cv2.CAP_PROP_FPS, 30)]


def test_no_fps_requested_by_default(monkeypatch):
    assert not any(p == cv2.CAP_PROP_FPS for p, _ in open_usb(monkeypatch, "win32", FakeCapture()).calls)


def test_mjpg_reapplied_when_fps_change_drops_it(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="camwatch.usbcams"):
        fake = open_usb(monkeypatch, "win32", FormatDroppingCapture(), width=1920, height=1080, fps=30)
    assert fake.calls[-1] == (cv2.CAP_PROP_FOURCC, MJPG) and fake.fourcc == MJPG
    assert "ignored pixel format" not in caplog.text and "1920x1080 MJPG" in caplog.text


def test_ignored_pixel_format_is_logged(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="camwatch.usbcams"):
        open_usb(monkeypatch, "win32", FormatDroppingCapture(sticky=False), width=1920, height=1080, fps=30)
    assert "camera ignored pixel format MJPG and uses YUY2" in caplog.text


def test_resolution_mismatch_is_logged_from_real_frames(monkeypatch, caplog):
    monkeypatch.setattr(usbcams.sys, "platform", "win32")
    monkeypatch.setattr(usbcams.cv2, "VideoCapture", FakeCapture(honor=False))
    stream = camera.CameraStream(CameraConfig(name="usb", source="0", width=1920, height=1080), ClipConfig(), 2.0)
    monkeypatch.setattr(stream, "_on_frame", lambda frame: stream.stop())  # one frame is enough
    with caplog.at_level(logging.WARNING, logger="camwatch.camera"):
        stream._run_stream()
    assert "requested 1920x1080 but camera delivers 640x360" in caplog.text


# ---- finding a setup that works ------------------------------------------------------------
class PickyCapture(FakeCapture):
    """Only delivers the requested size with one backend + format; otherwise stays at 640x480."""

    def __init__(self, works):
        super().__init__(native=(640, 480))
        self.works, self.tried = works, []

    def __call__(self, index, backend):
        fresh = PickyCapture(self.works)
        fresh.backend, fresh.fourcc, fresh.tried = backend, 0, self.tried
        return fresh

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOURCC:
            self.fourcc = int(value)
        ok = (self.backend, self.fourcc) == self.works
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) and not ok:
            return False
        return super().set(prop, value)

    def release(self):
        self.tried.append((self.backend, self.fourcc))


def test_finds_the_setup_that_really_delivers_1080p(monkeypatch):
    monkeypatch.setattr(usbcams.sys, "platform", "linux")
    fake = PickyCapture(works=(cv2.CAP_V4L2, cv2.VideoWriter_fourcc(*"YUY2")))
    monkeypatch.setattr(usbcams.cv2, "VideoCapture", fake)
    cam = CameraConfig(name="usb", source="usb:Cam", width=1920, height=1080)
    result = usbcams.find_working_setup(0, cam)
    assert (result.backend, result.fourcc, result.size) == ("v4l2", "YUY2", (1920, 1080))
    assert len(fake.tried) == 2  # v4l2+MJPG first, then v4l2+YUY2


def test_reports_the_best_size_when_nothing_reaches_the_request(monkeypatch):
    monkeypatch.setattr(usbcams.sys, "platform", "win32")
    monkeypatch.setattr(usbcams.cv2, "VideoCapture", PickyCapture(works=(None, None)))
    result = usbcams.find_working_setup(0, CameraConfig(name="usb", source="0", width=3840, height=2160))
    assert result.size == (640, 480) and result.frame is not None


def test_media_foundation_only_tried_with_a_single_camera(monkeypatch):
    monkeypatch.setattr(usbcams.sys, "platform", "win32")
    assert ("msmf", "MJPG") in usbcams.candidate_setups(single_camera=True)
    assert all(b != "msmf" for b, _ in usbcams.candidate_setups(single_camera=False))


def test_missing_named_camera_is_a_clear_connection_error(monkeypatch):
    monkeypatch.setattr(usbcams, "list_cameras", lambda modes=True: [usbcams.UsbCamera("Other", 0)])
    with pytest.raises(ConnectionError, match="USB camera 'Logitech BRIO' is not connected"):
        usbcams.resolve_index(CameraConfig(name="usb", source="usb:Logitech BRIO"))


# ---- menu pickers ------------------------------------------------------------------------
def script(monkeypatch, *answers):
    it = iter(answers)
    monkeypatch.setattr(menus, "_select", lambda msg, choices, **kw: next(it))
    monkeypatch.setattr(menus, "_text", lambda msg, default="", **kw: next(it))


def test_pick_preset_resolution(monkeypatch):
    cam = CameraConfig(name="usb", source="0")
    script(monkeypatch, (1920, 1080))
    assert menus.pick_resolution(cam) is True and (cam.width, cam.height) == (1920, 1080)


def test_pick_custom_resolution(monkeypatch):
    cam = CameraConfig(name="usb", source="0", width=1920, height=1080)
    script(monkeypatch, "custom", "1600 x 1200")
    assert menus.pick_resolution(cam) is True and (cam.width, cam.height) == (1600, 1200)


@pytest.mark.parametrize("answers", [(None,), ("custom", None)])
def test_cancel_leaves_resolution_unchanged(monkeypatch, answers):
    cam = CameraConfig(name="usb", source="0", width=1280, height=720)
    script(monkeypatch, *answers)
    assert menus.pick_resolution(cam) is None and (cam.width, cam.height) == (1280, 720)


def test_same_resolution_reports_unchanged(monkeypatch):
    cam = CameraConfig(name="usb", source="0", width=1280, height=720)
    script(monkeypatch, (1280, 720))
    assert menus.pick_resolution(cam) is False


def test_pick_preset_and_custom_fps(monkeypatch):
    cam = CameraConfig(name="usb", source="0")
    script(monkeypatch, 30.0)
    assert menus.pick_fps(cam) is True and cam.fps == 30
    script(monkeypatch, "custom", "20")
    assert menus.pick_fps(cam) is True and cam.fps == 20
    script(monkeypatch, 20.0)
    assert menus.pick_fps(cam) is False


def test_pick_usb_format(monkeypatch):
    cam = CameraConfig(name="usb", source="0")
    script(monkeypatch, "MJPG", "msmf")
    assert menus.pick_usb_format(cam) is True and (cam.fourcc, cam.backend) == ("MJPG", "msmf")


def test_resolution_text_flags_mismatch():
    cam = CameraConfig(name="usb", source="0", width=1920, height=1080)
    assert "asked 1920x1080" in menus._resolution_text(cam, (640, 360))
    assert menus._resolution_text(cam, (1920, 1080)) == "1920x1080"
    assert menus._resolution_text(CameraConfig(name="ip", source="rtsp://x"), (1280, 720)) == "1280x720"


class FakeSnapshotSession:
    """Serves one JPEG, then halts the stream so _run_snapshot returns."""

    def __init__(self, stream):
        self.stream = stream
        ok, jpg = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
        self.content = jpg.tobytes()

    def get(self, url, timeout):
        self.stream._halt.set()
        return self

    def raise_for_status(self):
        pass


def test_snapshot_camera_delivers_frames(monkeypatch):
    stream = camera.CameraStream(CameraConfig(name="garage", source="http://cam/snapshot.jpg"), ClipConfig(), 2.0)
    monkeypatch.setattr(camera.requests, "Session", lambda: FakeSnapshotSession(stream))
    stream._run_snapshot()
    fid, _, frame = stream.latest()
    assert (fid, frame.shape, stream.error) == (1, (480, 640, 3), "")
