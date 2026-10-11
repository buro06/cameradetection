import struct
import sys
import types

import pytest

import camwatch.menus as menus
import camwatch.usbcams as usbcams
from camwatch.config import CameraConfig
from camwatch.usbcams import Mode, UsbCamera

# ffmpeg 6/7 style (device kind on the name line)
DSHOW_DEVICES = r'''[dshow @ 000001d2] "HD Pro Webcam C920" (video)
[dshow @ 000001d2]   Alternative name "@device_pnp_\\?\usb#vid_046d&pid_082d&mi_00#6&1b0f7b0f&0&0000#{65e8773d}\global"
[dshow @ 000001d2] "OBS Virtual Camera" (none)
[dshow @ 000001d2]   Alternative name "@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
[dshow @ 000001d2] "HD Pro Webcam C920" (video)
[dshow @ 000001d2]   Alternative name "@device_pnp_\\?\usb#vid_046d&pid_082d&mi_00#7&2c1e8c2e&0&0000#{65e8773d}\global"
[dshow @ 000001d2] "Microphone (HD Pro Webcam C920)" (audio)
[dshow @ 000001d2]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{2C2A1C46}"
dummy: Immediate exit requested
'''

# ffmpeg 4 style (section headers)
DSHOW_DEVICES_OLD = '''[dshow @ 0000020f] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0000020f]  "Integrated Camera"
[dshow @ 0000020f]     Alternative name "@device_pnp_\\\\?\\usb#vid_04f2"
[dshow @ 0000020f] DirectShow audio devices
[dshow @ 0000020f]  "Microphone Array (Realtek(R) Audio)"
'''

DSHOW_OPTIONS = '''[dshow @ 0000021c] DirectShow video device options (from video devices)
[dshow @ 0000021c]  Pin "Capture" (alternative pin name "0")
[dshow @ 0000021c]   pixel_format=yuyv422  min s=640x480 fps=5 max s=640x480 fps=30
[dshow @ 0000021c]   pixel_format=yuyv422  min s=1920x1080 fps=5 max s=1920x1080 fps=5
[dshow @ 0000021c]   vcodec=mjpeg  min s=1920x1080 fps=5 max s=1920x1080 fps=30
[dshow @ 0000021c]   vcodec=mjpeg  min s=1920x1080 fps=5 max s=1920x1080 fps=30 (tv, bt470bg/bt709/unknown, topleft)
[dshow @ 0000021c]   vcodec=mjpeg  min s=3840x2160 fps=5 max s=3840x2160 fps=30
'''

AVFOUNDATION = '''[AVFoundation indev @ 0x14] AVFoundation video devices:
[AVFoundation indev @ 0x14] [0] FaceTime HD Camera
[AVFoundation indev @ 0x14] [1] Capture screen 0
[AVFoundation indev @ 0x14] AVFoundation audio devices:
[AVFoundation indev @ 0x14] [0] MacBook Pro Microphone
'''


def test_dshow_device_list_keeps_only_video_devices_in_order():
    assert usbcams.parse_dshow_devices(DSHOW_DEVICES) == ["HD Pro Webcam C920", "HD Pro Webcam C920"]
    assert usbcams.parse_dshow_devices(DSHOW_DEVICES_OLD) == ["Integrated Camera"]


def test_dshow_modes():
    modes = usbcams.parse_dshow_modes(DSHOW_OPTIONS)
    assert Mode("MJPG", 3840, 2160, 30.0) in modes and Mode("YUY2", 1920, 1080, 5.0) in modes
    assert len(modes) == 4  # the repeated MJPG 1080p line (with colour info) is listed once


def test_avfoundation_skips_screen_capture():
    assert usbcams.parse_avfoundation_devices(AVFOUNDATION) == [(0, "FaceTime HD Camera")]


def test_windows_listing_numbers_identical_cameras(monkeypatch):
    calls = []

    def ffmpeg(*args):
        calls.append(args)
        return DSHOW_DEVICES if "-list_devices" in args else DSHOW_OPTIONS

    monkeypatch.setattr(usbcams.sys, "platform", "win32")
    monkeypatch.setattr(usbcams, "_ffmpeg", ffmpeg)
    cams = usbcams.list_cameras()
    assert [(c.name, c.index) for c in cams] == [("HD Pro Webcam C920", 0), ("HD Pro Webcam C920 #2", 1)]
    assert calls[2][calls[2].index("-video_device_number") + 1] == "1"  # modes of the second one
    assert cams[1].describe() == "HD Pro Webcam C920 #2 — up to 3840x2160"


def test_sizes_prefer_mjpg_and_the_best_frame_rate():
    cam = UsbCamera("c", 0, usbcams.parse_dshow_modes(DSHOW_OPTIONS))
    assert cam.sizes() == [(3840, 2160, "MJPG", 30.0), (1920, 1080, "MJPG", 30.0), (640, 480, "YUY2", 30.0)]


# ---- Linux: V4L2 ioctls, stubbed -------------------------------------------------------------------
def fake_v4l2(monkeypatch, devices):
    """devices: {N: (card, is_capture, [(fourcc, [(w, h, fps)])])} for /dev/videoN."""
    fds = {}
    fake_os = types.SimpleNamespace(O_RDWR=2, O_NONBLOCK=4, close=lambda fd: None)

    def open_(path, flags):
        n = int(path.rsplit("video", 1)[1])
        if n not in devices:
            raise OSError("no such device")
        fds[100 + n] = devices[n]
        return 100 + n

    fake_os.open = open_

    def ioctl(fd, request, buf):
        card, capture, formats = fds[fd]
        if request == usbcams.VIDIOC_QUERYCAP:
            buf[16:16 + len(card)] = card.encode()
            struct.pack_into("=II", buf, 84, usbcams.V4L2_CAP_DEVICE_CAPS | 0x1, 0x1 if capture else 0x00800000)
            return
        index = struct.unpack_from("=I", buf, 0)[0]
        if request == usbcams.VIDIOC_ENUM_FMT:
            if index >= len(formats):
                raise OSError
            struct.pack_into("=I", buf, 44, struct.unpack("<I", formats[index][0].encode())[0])
            return
        pixfmt = struct.unpack_from("=I", buf, 4)[0]
        sizes = next(s for f, s in formats if struct.unpack("<I", f.encode())[0] == pixfmt)
        if request == usbcams.VIDIOC_ENUM_FRAMESIZES:
            if index >= len(sizes):
                raise OSError
            struct.pack_into("=3I", buf, 8, 1, sizes[index][0], sizes[index][1])
            return
        if request == usbcams.VIDIOC_ENUM_FRAMEINTERVALS:
            w, h = struct.unpack_from("=2I", buf, 8)
            fps = next(f for sw, sh, f in sizes if (sw, sh) == (w, h))
            if index >= 1:
                raise OSError
            struct.pack_into("=3I", buf, 16, 1, 1, fps)
            return
        raise OSError

    monkeypatch.setattr(usbcams, "os", fake_os)
    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(ioctl=ioctl))


def test_linux_lists_capture_nodes_with_their_modes(monkeypatch, tmp_path):
    for n in range(4):
        (tmp_path / f"video{n}").touch()
    fake_v4l2(monkeypatch, {
        0: ("HD Pro Webcam C920", True, [("YUYV", [(640, 480, 30), (1920, 1080, 5)]), ("MJPG", [(1920, 1080, 30)])]),
        1: ("HD Pro Webcam C920", False, []),  # metadata node
        2: ("HD Pro Webcam C920", True, [("MJPG", [(1280, 720, 30)])]),
        3: ("HD Pro Webcam C920", False, []),
    })
    cams = usbcams._unique_names(usbcams._linux_cameras(tmp_path, modes=True))
    assert [(c.name, c.index) for c in cams] == [("HD Pro Webcam C920", 0), ("HD Pro Webcam C920 #2", 2)]
    assert cams[0].sizes() == [(1920, 1080, "MJPG", 30.0), (640, 480, "YUY2", 30.0)]


# ---- names and menus -------------------------------------------------------------------------------
def test_find_and_usb_camera_of(monkeypatch):
    cams = [UsbCamera("Cam", 0), UsbCamera("Cam #2", 2)]
    monkeypatch.setattr(usbcams, "list_cameras", lambda modes=True: cams)
    assert usbcams.find("usb:Cam #2").index == 2 and usbcams.find("usb:Gone") is None
    assert menus.usb_camera_of(CameraConfig(name="a", source="2"), cams).name == "Cam #2"
    assert menus.usb_camera_of(CameraConfig(name="a", source="usb:Cam"), cams).index == 0
    assert usbcams.is_usb_source("usb:Cam") and usbcams.is_usb_source("1") and not usbcams.is_usb_source("rtsp://x")


@pytest.mark.parametrize("sizes, expected", [
    ([(3840, 2160), (1920, 1080), (1280, 720)], (1920, 1080)),
    ([(2592, 1944), (1280, 960), (640, 480)], (1280, 960)),  # largest up to 1080p
    ([], (1920, 1080)),
])
def test_default_size(sizes, expected):
    cam = UsbCamera("c", 0, [Mode("MJPG", w, h, 30) for w, h in sizes])
    assert menus.default_size(cam) == expected


def test_resolution_picker_offers_the_cameras_own_modes(monkeypatch):
    shown = {}

    def select(msg, choices, **kw):
        shown["values"] = [c.value for c in choices]
        return (2592, 1944)

    monkeypatch.setattr(menus, "_select", select)
    usb = UsbCamera("c", 0, [Mode("MJPG", 2592, 1944, 15), Mode("YUY2", 640, 480, 30)])
    cam = CameraConfig(name="c", source="usb:c")
    assert menus.pick_resolution(cam, usb=usb) is True and (cam.width, cam.height) == (2592, 1944)
    assert shown["values"] == [(0, 0), (2592, 1944), (640, 480), "custom"]
