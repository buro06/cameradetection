"""Camera capture threads with a pre-roll ring buffer for clip recording."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from .config import CameraConfig, ClipConfig

log = logging.getLogger(__name__)

STALE_SECONDS = 10.0
OPEN_TIMEOUT_MS = 10000
READ_TIMEOUT_MS = 10000

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "ffmpeg": cv2.CAP_FFMPEG,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}


def set_rtsp_transport(transport: str) -> None:
    """Must run before any FFMPEG capture is opened (process-wide setting)."""
    if transport:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{transport}")


def describe_source(source: str) -> str:
    """Source string safe for display (credentials stripped)."""
    try:
        u = urlparse(source)
        if u.password or u.username:
            host = u.hostname or ""
            if u.port:
                host += f":{u.port}"
            return u._replace(netloc=f"***@{host}").geturl()
    except ValueError:
        pass
    return source


@dataclass
class Overlay:
    """Tracked person boxes (full-res coords) at a point in time, drawn onto clip frames."""

    ts: float
    boxes: list[tuple[int, tuple[float, float, float, float]]]  # (track_id, xyxy)


@dataclass
class BufferedFrame:
    ts: float  # time.monotonic()
    image: np.ndarray  # clip-resolution BGR
    scale: float  # clip px / full-res px
    overlay: Overlay | None


@dataclass
class Recording:
    since: float
    frames: list[BufferedFrame] = field(default_factory=list)


class CameraStream(threading.Thread):
    def __init__(self, cfg: CameraConfig, clip: ClipConfig, pre_seconds: float):
        super().__init__(name=f"cam-{cfg.name}", daemon=True)
        self.cfg = cfg
        self.clip = clip
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_ts = 0.0
        self.frame_id = 0
        self._ring: deque[BufferedFrame] = deque(maxlen=int((pre_seconds + 1.5) * clip.fps) + 5)
        self._recordings: list[Recording] = []
        self._overlay: Overlay | None = None
        self._last_buffered = 0.0
        self.status = "starting"
        self.error = ""
        self.fps = 0.0
        self.resolution = (0, 0)
        self.reconnects = 0

    # ---- public API -------------------------------------------------------
    def stop(self) -> None:
        self._halt.set()

    def latest(self) -> tuple[int, float, np.ndarray | None]:
        with self._lock:
            return self.frame_id, self._latest_ts, self._latest

    def is_live(self) -> bool:
        return self.status == "live" and time.monotonic() - self._latest_ts < STALE_SECONDS

    @property
    def overlay(self) -> Overlay | None:
        return self._overlay

    def set_overlay(self, overlay: Overlay | None) -> None:
        self._overlay = overlay

    def start_recording(self, since: float) -> Recording:
        rec = Recording(since=since)
        with self._lock:
            rec.frames = [f for f in self._ring if f.ts >= since]
            self._recordings.append(rec)
        return rec

    def stop_recording(self, rec: Recording) -> None:
        with self._lock:
            if rec in self._recordings:
                self._recordings.remove(rec)

    # ---- capture loop -----------------------------------------------------
    def run(self) -> None:
        backoff = 2.0
        while not self._halt.is_set():
            try:
                if self._is_snapshot():
                    self._run_snapshot()
                else:
                    self._run_stream()
                backoff = 2.0
            except Exception as e:  # never let a camera thread die
                log.exception("[%s] capture error", self.cfg.name)
                self.error = str(e)
            if self._halt.is_set():
                break
            self.status = "reconnecting"
            self.reconnects += 1
            self._halt.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        self.status = "stopped"

    def _is_snapshot(self) -> bool:
        if self.cfg.mode == "snapshot":
            return True
        if self.cfg.mode == "stream":
            return False
        path = urlparse(self.cfg.source).path.lower()
        return self.cfg.source.startswith(("http://", "https://")) and path.endswith((".jpg", ".jpeg", ".png"))

    def _open(self) -> tuple[cv2.VideoCapture, bool]:
        src = self.cfg.source.strip()
        if src.isdigit():
            backend = _BACKENDS.get(self.cfg.backend)
            if backend is None:
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(src), backend)
            if self.cfg.width and self.cfg.height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            return cap, False
        is_file = Path(src).exists()
        params = [] if is_file else [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS,
        ]
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG, params), is_file

    def _run_stream(self) -> None:
        self.status = "connecting"
        cap, is_file = self._open()
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"cannot open {describe_source(self.cfg.source)}")
        file_interval = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if is_file else 0.0
        failures = 0
        next_due = time.monotonic()
        try:
            while not self._halt.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_file:  # loop test videos
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    failures += 1
                    if failures >= 5:
                        raise ConnectionError("stream stopped delivering frames")
                    time.sleep(0.2)
                    continue
                failures = 0
                self._on_frame(frame)
                if is_file:
                    next_due += file_interval
                    delay = next_due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_due = time.monotonic()
        finally:
            cap.release()

    def _run_snapshot(self) -> None:
        self.status = "connecting"
        interval = 1.0 / max(self.cfg.snapshot_fps, 0.1)
        session = requests.Session()
        failures = 0
        while not self._halt.is_set():
            t0 = time.monotonic()
            try:
                r = session.get(self.cfg.source, timeout=5)
                r.raise_for_status()
                frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("response is not an image")
                failures = 0
                self._on_frame(frame)
            except Exception as e:
                failures += 1
                self.error = str(e)
                if failures >= 5:
                    raise ConnectionError(f"snapshot failed: {e}") from e
            self._halt.wait(max(0.0, interval - (time.monotonic() - t0)))

    def _on_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        h, w = frame.shape[:2]
        with self._lock:
            if self._latest_ts:
                dt = now - self._latest_ts
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            self._latest = frame
            self._latest_ts = now
            self.frame_id += 1
        self.status = "live"
        self.error = ""
        self.resolution = (w, h)

        if now - self._last_buffered < 0.9 / self.clip.fps:
            return
        self._last_buffered = now
        scale = min(1.0, self.clip.max_width / w) if self.clip.max_width else 1.0
        small = frame if scale >= 1.0 else cv2.resize(frame, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
        overlay = self._overlay
        if overlay is not None and now - overlay.ts > 0.6:
            overlay = None
        bf = BufferedFrame(ts=now, image=small, scale=small.shape[1] / w, overlay=overlay)
        with self._lock:
            self._ring.append(bf)
            for rec in self._recordings:
                rec.frames.append(bf)
