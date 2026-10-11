"""USB cameras: list them by name with the modes they support, open them, and find a capture setup that
really delivers the requested resolution.

Sources: a plain index ("0") or a name ("usb:Logitech BRIO", "usb:HD Webcam #2" for a second identical one).
Names are looked up again on every (re)connect, so re-plugging or rebooting doesn't swap cameras.

Listing per OS:
  * Linux: V4L2 ioctls on /dev/video* (pure Python; metadata nodes are skipped).
  * Windows: DirectShow device and mode lists from the ffmpeg bundled with imageio-ffmpeg. Its device order is
    OpenCV's CAP_DSHOW index order.
  * macOS: AVFoundation names from ffmpeg (no modes).
"""

from __future__ import annotations

import logging
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)

USB_PREFIX = "usb:"
BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "ffmpeg": cv2.CAP_FFMPEG,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}
BACKEND_NAMES = {"dshow": "DirectShow", "msmf": "Media Foundation", "v4l2": "V4L2", "avfoundation": "AVFoundation"}
TEST_FRAMES = 8  # frames read when testing a capture setup (the first few can still be in the old mode)


@dataclass(frozen=True)
class Mode:
    fourcc: str  # MJPG, YUY2, H264, NV12, ...
    width: int
    height: int
    fps: float | None = None  # highest frame rate at this size, if known


@dataclass
class UsbCamera:
    name: str  # unique on this machine ("#2" added for a second identical camera)
    index: int  # OpenCV index for this OS's default backend
    modes: list[Mode] = field(default_factory=list)

    @property
    def source(self) -> str:
        return USB_PREFIX + self.name

    def sizes(self) -> list[tuple[int, int, str, float | None]]:
        """Distinct resolutions, largest first, each with its best format (MJPG preferred) and frame rate."""
        best: dict[tuple[int, int], Mode] = {}
        for m in self.modes:
            cur = best.get((m.width, m.height))
            rank = lambda x: (x.fps or 0, x.fourcc == "MJPG")  # noqa: E731
            if cur is None or rank(m) > rank(cur):
                best[(m.width, m.height)] = m
        return [(w, h, m.fourcc, m.fps) for (w, h), m in sorted(best.items(), key=lambda kv: -kv[0][0] * kv[0][1])]

    def describe(self) -> str:
        sizes = self.sizes()
        return f"{self.name} — up to {sizes[0][0]}x{sizes[0][1]}" if sizes else self.name


def is_usb_source(source: str) -> bool:
    s = source.strip()
    return s.isdigit() or s.startswith(USB_PREFIX)


def _unique_names(cams: list[UsbCamera]) -> list[UsbCamera]:
    seen: dict[str, int] = {}
    for c in sorted(cams, key=lambda c: c.index):
        seen[c.name] = seen.get(c.name, 0) + 1
        if seen[c.name] > 1:
            c.name = f"{c.name} #{seen[c.name]}"
    return sorted(cams, key=lambda c: c.index)


def list_cameras(modes: bool = True) -> list[UsbCamera]:
    """USB cameras plugged in now, in index order. Empty if this OS/driver can't be listed."""
    try:
        if sys.platform.startswith("linux"):
            return _unique_names(_linux_cameras(Path("/dev"), modes))
        if sys.platform == "win32":
            return _unique_names(_dshow_cameras(modes))
        if sys.platform == "darwin":
            return _unique_names(_avfoundation_cameras())
    except Exception:
        log.exception("Could not list USB cameras")
    return []


def find(source: str) -> UsbCamera | None:
    """The camera a `usb:<name>` source refers to, if it's plugged in."""
    name = source.strip()[len(USB_PREFIX):].strip()
    return next((c for c in list_cameras(modes=False) if c.name == name), None)


# ---- Linux: V4L2 ioctls ------------------------------------------------------------------------------
def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_QUERYCAP = (2 << 30) | (104 << 16) | (ord("V") << 8) | 0
VIDIOC_ENUM_FMT = _iowr(2, 64)
VIDIOC_ENUM_FRAMESIZES = _iowr(74, 44)
VIDIOC_ENUM_FRAMEINTERVALS = _iowr(75, 52)
V4L2_CAP_VIDEO_CAPTURE, V4L2_CAP_DEVICE_CAPS = 0x1, 0x80000000
BUF_TYPE_VIDEO_CAPTURE = 1
FRMSIZE_DISCRETE, FRMIVAL_DISCRETE = 1, 1


def _fourcc_str(code: int) -> str:
    text = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip()
    return {"YUYV": "YUY2"}.get(text, text)


def _ioctl_loop(fd: int, request: int, fmt: str, *head):
    """Yield the unpacked structs of an enumerating ioctl (index 0, 1, ... until it fails)."""
    import fcntl
    for i in range(256):
        buf = bytearray(struct.calcsize(fmt))
        struct.pack_into(f"={1 + len(head)}I", buf, 0, i, *head)  # index, then the fields that pick what to list
        try:
            fcntl.ioctl(fd, request, buf)
        except OSError:
            return
        yield struct.unpack(fmt, buf)


def _linux_modes(fd: int) -> list[Mode]:
    modes = []
    for fmt in _ioctl_loop(fd, VIDIOC_ENUM_FMT, "=3I32sI4I", BUF_TYPE_VIDEO_CAPTURE):
        pixfmt = fmt[4]
        for size in _ioctl_loop(fd, VIDIOC_ENUM_FRAMESIZES, "=3I6I2I", pixfmt):
            if size[2] == FRMSIZE_DISCRETE:
                w, h = size[3], size[4]
            else:  # stepwise / continuous: the largest it allows
                w, h = size[4], size[7]
            fps = None
            for iv in _ioctl_loop(fd, VIDIOC_ENUM_FRAMEINTERVALS, "=5I6I2I", pixfmt, w, h):
                if iv[4] == FRMIVAL_DISCRETE and iv[5]:
                    fps = max(fps or 0.0, iv[6] / iv[5])
            modes.append(Mode(_fourcc_str(pixfmt), w, h, round(fps, 1) if fps else None))
    return modes


def _linux_cameras(dev: Path, modes: bool) -> list[UsbCamera]:
    cams = []
    for path in sorted(dev.glob("video[0-9]*"), key=lambda p: int(p.name[5:])):
        try:
            fd = os.open(str(path), os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            import fcntl
            cap = bytearray(104)
            fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
            card = cap[16:48].split(b"\0", 1)[0].decode(errors="replace").strip()
            caps, device_caps = struct.unpack_from("=II", cap, 84)
            if not ((device_caps if caps & V4L2_CAP_DEVICE_CAPS else caps) & V4L2_CAP_VIDEO_CAPTURE):
                continue  # metadata node: every webcam has one next to its video node
            cams.append(UsbCamera(card or path.name, int(path.name[5:]), _linux_modes(fd) if modes else []))
        except OSError:
            continue
        finally:
            os.close(fd)
    return cams


# ---- Windows: DirectShow through ffmpeg --------------------------------------------------------------
_DSHOW_NAME = re.compile(r'\]\s+"(?P<name>[^"]+)"\s*(?:\((?P<kind>video|audio|none)\))?\s*$')
_DSHOW_MODE = re.compile(r"(?:vcodec|pixel_format)=(?P<fmt>\S+)\s+min s=\d+x\d+ fps=[\d.]+\s+"
                         r"max s=(?P<w>\d+)x(?P<h>\d+) fps=(?P<fps>[\d.]+)")
_FFMPEG_FORMATS = {"mjpeg": "MJPG", "yuyv422": "YUY2", "h264": "H264", "nv12": "NV12", "yuv420p": "I420"}


def parse_dshow_devices(text: str) -> list[str]:
    """Video device names from `ffmpeg -f dshow -list_devices true` output, in DirectShow order."""
    names, section = [], None
    for line in text.splitlines():
        if "DirectShow video devices" in line:
            section = "video"
        elif "DirectShow audio devices" in line:
            section = "audio"
        if "Alternative name" in line:
            continue
        m = _DSHOW_NAME.search(line)
        if m and (m["kind"] or section) == "video":
            names.append(m["name"])
    return names


def parse_dshow_modes(text: str) -> list[Mode]:
    """Modes from `ffmpeg -f dshow -list_options true` output."""
    out = []
    for m in _DSHOW_MODE.finditer(text):
        fmt = _FFMPEG_FORMATS.get(m["fmt"], m["fmt"].upper())
        mode = Mode(fmt, int(m["w"]), int(m["h"]), float(m["fps"]))
        if mode not in out:
            out.append(mode)
    return out


def _ffmpeg(*args: str) -> str:
    from .recorder import ffmpeg_exe
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    r = subprocess.run([exe, "-hide_banner", *args], capture_output=True, text=True, errors="replace",
                       timeout=20, creationflags=flags)
    return r.stderr


def _dshow_cameras(modes: bool) -> list[UsbCamera]:
    names = parse_dshow_devices(_ffmpeg("-f", "dshow", "-list_devices", "true", "-i", "dummy"))
    cams = []
    for i, name in enumerate(names):
        found = []
        if modes:
            number = names[:i].count(name)  # which of several identically named cameras
            found = parse_dshow_modes(_ffmpeg("-f", "dshow", "-list_options", "true",
                                              "-video_device_number", str(number), "-i", f"video={name}"))
        cams.append(UsbCamera(name, i, found))
    return cams


# ---- macOS ----------------------------------------------------------------------------------------------
def parse_avfoundation_devices(text: str) -> list[tuple[int, str]]:
    out, video = [], False
    for line in text.splitlines():
        if "video devices" in line:
            video = True
        elif "audio devices" in line:
            video = False
        elif video and (m := re.search(r"\]\s+\[(\d+)\]\s+(.+?)\s*$", line)) and not m[2].startswith("Capture screen"):
            out.append((int(m[1]), m[2]))
    return out


def _avfoundation_cameras() -> list[UsbCamera]:
    return [UsbCamera(name, i) for i, name in
            parse_avfoundation_devices(_ffmpeg("-f", "avfoundation", "-list_devices", "true", "-i", ""))]


# ---- opening -------------------------------------------------------------------------------------------
def resolve_index(cam: CameraConfig) -> int:
    src = cam.source.strip()
    if src.isdigit():
        return int(src)
    found = find(src)
    if found is None:
        raise ConnectionError(f"USB camera '{src[len(USB_PREFIX):]}' is not connected")
    return found.index


def default_backend() -> int:
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2  # CAP_ANY can pick GStreamer, which ignores size requests
    return cv2.CAP_ANY


def wanted_fourcc(cam: CameraConfig) -> str:
    fourcc = cam.fourcc.strip().upper()
    if fourcc != "AUTO":
        return "" if fourcc == "NONE" else fourcc
    # Over USB 2.0, HD only fits as MJPG; uncompressed YUYV tops out around 640x480 at full speed.
    # DirectShow also opens in the driver's default (often low-res) mode unless a format is asked for.
    if sys.platform == "win32" or (sys.platform.startswith("linux") and cam.width * cam.height > 640 * 480):
        return "MJPG"
    return ""


def fourcc_text(code: int) -> str:
    text = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    return text if code and text.isprintable() else "unknown format"


def open_usb(index: int, cam: CameraConfig, name: str = "") -> cv2.VideoCapture:
    """Open a USB camera with the configured backend, pixel format, size and frame rate."""
    name = name or cam.name
    backend = BACKENDS.get(cam.backend)
    cap = cv2.VideoCapture(index, default_backend() if backend is None else backend)
    fourcc = wanted_fourcc(cam)
    want = cv2.VideoWriter_fourcc(*fourcc) if len(fourcc) == 4 else 0
    if want:
        cap.set(cv2.CAP_PROP_FOURCC, want)  # must come before the size
    if cam.width and cam.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
    if cam.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, cam.fps)
    # DirectShow restarts the device on an FPS change and can fall back to uncompressed YUY2,
    # which USB 2.0 can only carry at ~5 fps in 1080p; ask for the format again if it was lost.
    have = max(0, int(cap.get(cv2.CAP_PROP_FOURCC)))
    if want and have and have != want:
        cap.set(cv2.CAP_PROP_FOURCC, want)
        have = max(0, int(cap.get(cv2.CAP_PROP_FOURCC)))
        if have != want:
            log.warning("[%s] camera ignored pixel format %s and uses %s (try Pixel format / backend in the menu)",
                        name, fourcc, fourcc_text(have))
    if cap.isOpened():
        log.info("[%s] USB camera opened: %dx%d %s @ %.0f fps (driver-reported)", name,
                 cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                 fourcc_text(have), cap.get(cv2.CAP_PROP_FPS))
    return cap


@dataclass
class SetupResult:
    backend: str
    fourcc: str
    size: tuple[int, int] | None  # what the frames really were (None: no frames)
    frame: np.ndarray | None = None

    def describe(self) -> str:
        if self.size is None:
            return "no picture"
        fmt = f" {self.fourcc}" if self.fourcc not in ("auto", "none") else ""
        return f"{self.size[0]}x{self.size[1]}{fmt} via {BACKEND_NAMES.get(self.backend, self.backend)}"


def try_setup(index: int, cam: CameraConfig) -> SetupResult:
    """Open the camera as configured and report the real size of the frames it delivers."""
    cap = open_usb(index, cam)
    frame = None
    try:
        if cap.isOpened():
            for _ in range(TEST_FRAMES):
                ok, img = cap.read()
                if ok and img is not None:
                    frame = img
    finally:
        cap.release()
    size = (frame.shape[1], frame.shape[0]) if frame is not None else None
    return SetupResult(cam.backend, cam.fourcc, size, frame)


def candidate_setups(single_camera: bool) -> list[tuple[str, str]]:
    """(backend, fourcc) combinations worth trying on this OS, most likely first."""
    if sys.platform == "win32":
        # Media Foundation numbers cameras in its own order: only safe to try when there's one camera
        return [("dshow", "MJPG"), *([("msmf", "MJPG")] if single_camera else []), ("dshow", "none")]
    if sys.platform.startswith("linux"):
        return [("v4l2", "MJPG"), ("v4l2", "YUY2"), ("v4l2", "none")]
    return [("auto", "none")]


def find_working_setup(index: int, cam: CameraConfig, single_camera: bool = True) -> SetupResult:
    """Try the candidate backend/format combinations until frames arrive at the requested size. Returns the
    first that does, else the one with the largest picture (`size` None if none gave a picture)."""
    wanted = (cam.width, cam.height) if cam.width and cam.height else None
    best = SetupResult(cam.backend, cam.fourcc, None)
    for backend, fourcc in candidate_setups(single_camera):
        result = try_setup(index, replace(cam, backend=backend, fourcc=fourcc))
        log.info("[%s] tried %s/%s: %s", cam.name, backend, fourcc, result.describe())
        if result.size and (wanted is None or result.size == wanted):
            return result
        if result.size and (best.size is None or result.size[0] * result.size[1] > best.size[0] * best.size[1]):
            best = result
    return best
