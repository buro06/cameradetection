"""Orchestrates cameras, the shared YOLO worker, face recognition, events and notifications."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import cv2

from .camera import STALE_SECONDS, CameraStream, describe_source, set_rtsp_transport
from .config import AppConfig, CameraConfig
from .detector import PersonDetector
from .events import CameraMonitor, EventResult
from .faces import FaceDB, FaceEngine
from .recorder import clip_path, draw_boxes, draw_stamp, purge_old_clips, write_clip
from .timefmt import clock, stamp

log = logging.getLogger(__name__)


@dataclass
class EventLogEntry:
    wall_time: float
    camera: str
    decision: str
    summary: str
    clip: str | None = None


class Engine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.data = cfg.data_path
        self.data.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.data / "clips"
        set_rtsp_transport(cfg.rtsp_transport)

        self.db = FaceDB(self.data, cfg.face.match_threshold)
        self.faces = FaceEngine(self.data / "models", cfg.face)
        self.detector = PersonDetector(cfg.detection, self.data / "models")

        self.streams: dict[str, CameraStream] = {}
        self.monitors: dict[str, CameraMonitor] = {}
        self.events: deque[EventLogEntry] = deque(maxlen=100)
        self._alerts: queue.Queue[EventResult] = queue.Queue(maxsize=20)
        self._halt = threading.Event()
        self._cam_lock = threading.RLock()
        self.started = time.time()

        self._state_path = self.data / "state.json"
        self.armed = True
        self.disarmed_cameras: set[str] = set()
        self._load_state()

        self.bot = None
        if cfg.telegram.enabled and cfg.telegram_token():
            from .telegram import TelegramBot
            self.bot = TelegramBot(self)

    # ---- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        for cam in self.cfg.cameras:
            if cam.enabled:
                self._start_camera(cam)
        threading.Thread(target=self._detect_loop, name="detector", daemon=True).start()
        threading.Thread(target=self._alert_loop, name="alerts", daemon=True).start()
        threading.Thread(target=self._housekeeping_loop, name="housekeeping", daemon=True).start()
        if self.bot:
            self.bot.start()
        log.info("camwatch started: %d camera(s), detector on %s", len(self.streams), self.detector.device)

    def stop(self) -> None:
        self._halt.set()
        if self.bot:
            self.bot.stop()
        with self._cam_lock:
            for s in self.streams.values():
                s.stop()

    # ---- cameras ---------------------------------------------------------------
    def _start_camera(self, cam: CameraConfig) -> None:
        with self._cam_lock:
            stream = CameraStream(cam, self.cfg.clip, self.cfg.events.pre_seconds)
            self.streams[cam.name] = stream
            self.monitors[cam.name] = CameraMonitor(self.cfg, stream, self.faces, self.db, self.is_armed)
            stream.start()
            log.info("[%s] started (%s)", cam.name, describe_source(cam.source))

    def stop_camera(self, name: str) -> None:
        with self._cam_lock:
            self.monitors.pop(name, None)
            s = self.streams.pop(name, None)
        if s:
            s.stop()

    def apply_camera(self, cam: CameraConfig) -> None:
        """(Re)start a camera after it was added or edited."""
        self.stop_camera(cam.name)
        if cam.enabled:
            self._start_camera(cam)

    def apply_telegram(self) -> None:
        """Pick up Telegram config changes made at runtime."""
        if self.cfg.telegram.enabled and self.cfg.telegram_token():
            if self.bot is None:
                from .telegram import TelegramBot
                self.bot = TelegramBot(self)
                self.bot.start()
            else:
                self.bot.bot_name = ""  # re-validates the (possibly new) token
        elif self.bot is not None:
            self.bot.stop()
            self.bot = None

    # ---- arming ------------------------------------------------------------------
    def is_armed(self, camera: str) -> bool:
        return self.armed and camera not in self.disarmed_cameras

    def set_armed(self, camera: str | None, armed: bool) -> None:
        if camera:
            cam = self.cfg.camera(camera)
            if cam is None:
                raise KeyError(camera)
            (self.disarmed_cameras.discard if armed else self.disarmed_cameras.add)(cam.name)
            if armed:
                self.armed = True
        else:
            self.armed = armed
            if armed:
                self.disarmed_cameras.clear()
        self._save_state()
        log.info("Alerts %s%s", "armed" if armed else "disarmed", f" for {camera}" if camera else "")

    def arm_text(self) -> str:
        if not self.armed:
            return "🔕 Alerts are <b>disarmed</b> on all cameras."
        if self.disarmed_cameras:
            return "🔔 Armed, except: " + ", ".join(sorted(self.disarmed_cameras))
        return "🔔 Alerts are <b>armed</b> on all cameras."

    def _load_state(self) -> None:
        try:
            st = json.loads(self._state_path.read_text())
            self.armed = bool(st.get("armed", True))
            self.disarmed_cameras = set(st.get("disarmed_cameras", []))
        except (FileNotFoundError, ValueError):
            pass

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps({"armed": self.armed, "disarmed_cameras": sorted(self.disarmed_cameras)}))

    # ---- detection worker -------------------------------------------------------
    def _detect_loop(self) -> None:
        last_run: dict[str, float] = {}
        last_fid: dict[str, int] = {}
        while not self._halt.is_set():
            try:
                interval = 1.0 / max(self.cfg.detection.detect_fps, 0.1)
                now = time.monotonic()
                with self._cam_lock:
                    monitors = list(self.monitors.values())
                batch = []
                for mon in monitors:
                    fid, ts, frame = mon.stream.latest()
                    if frame is None or fid == last_fid.get(mon.name) or now - ts > STALE_SECONDS:
                        continue
                    if now - last_run.get(mon.name, 0.0) < interval:
                        continue
                    batch.append((mon, frame))
                    last_fid[mon.name], last_run[mon.name] = fid, now
                if batch:
                    dets = self.detector.detect([f for _, f in batch])
                    now = time.monotonic()
                    for (mon, frame), d in zip(batch, dets):
                        mon.process(frame, d, now)
                for mon in monitors:
                    if not mon.stream.is_live():
                        mon.visible = []
                    result = mon.tick(time.monotonic())
                    if result:
                        self._on_result(result)
                if not batch:
                    time.sleep(0.01)
            except Exception:
                log.exception("Detection loop error")
                time.sleep(1)

    def _on_result(self, result: EventResult) -> None:
        entry = EventLogEntry(result.wall_time, result.camera, result.decision, result.summary())
        self.events.appendleft(entry)
        if result.decision == "alert":
            try:
                self._alerts.put_nowait((result, entry))
            except queue.Full:
                log.error("Alert queue full, dropping alert for %s", result.camera)
                result.frames = []
                self._append_event_log(entry)
        else:
            result.frames = []  # clip only rendered for alerts; free memory now
            self._append_event_log(entry)

    def _append_event_log(self, entry: EventLogEntry) -> None:
        try:
            with open(self.data / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": datetime.fromtimestamp(entry.wall_time).astimezone().isoformat(timespec="seconds"),
                                    "camera": entry.camera, "decision": entry.decision,
                                    "people": entry.summary, "clip": entry.clip}) + "\n")
        except OSError:
            log.exception("Could not write event log")

    def _alert_loop(self) -> None:
        while not self._halt.is_set():
            try:
                result, entry = self._alerts.get(timeout=1)
            except queue.Empty:
                continue
            path = None
            try:
                path = write_clip(result, clip_path(self.clips_dir, result), self.cfg.clip)
                entry.clip = str(path)
                log.info("[%s] clip saved: %s", result.camera, path.name)
            except Exception:
                log.exception("[%s] clip encoding failed", result.camera)
            self._append_event_log(entry)
            if self.bot:
                self.bot.send_alert(result, path)
            result.frames = []  # free memory

    def _housekeeping_loop(self) -> None:
        while not self._halt.wait(5):
            try:
                n = purge_old_clips(self.clips_dir, self.cfg.clip.retention_days)
                if n:
                    log.info("Deleted %d clip(s) older than %d days", n, self.cfg.clip.retention_days)
            except Exception:
                log.exception("Housekeeping failed")
            self._halt.wait(3600)

    # ---- queries -------------------------------------------------------------------
    def snapshot(self, camera: str) -> bytes | None:
        cam = self.cfg.camera(camera)
        if cam is None or cam.name not in self.streams:
            raise KeyError(camera)
        stream, mon = self.streams[cam.name], self.monitors[cam.name]
        _, _, frame = stream.latest()
        if frame is None:
            return None
        img = frame.copy()
        trusted_map = self.db.trusted_map()
        overlay = stream.overlay
        if overlay and time.monotonic() - overlay.ts < 1.0:
            labels = {tid: mon._label(t, trusted_map) for tid, t in dict(mon.tracker.tracks).items()}
            draw_boxes(img, overlay.boxes, labels, 1.0)
        draw_stamp(img, f"{cam.name}  {stamp(datetime.now())}")
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return jpg.tobytes() if ok else None

    def camera_rows(self) -> list[dict]:
        rows = []
        for cam in self.cfg.cameras:
            s, m = self.streams.get(cam.name), self.monitors.get(cam.name)
            last = m.last_result if m else None
            rows.append({
                "name": cam.name, "source": describe_source(cam.source), "enabled": cam.enabled,
                "status": (s.status if s.is_live() or s.status != "live" else "stale") if s else "disabled",
                "error": s.error if s else "", "fps": s.fps if s else 0.0, "infer_fps": m.infer_fps if m else 0.0,
                "resolution": s.resolution if s else (0, 0), "armed": self.is_armed(cam.name),
                "visible": list(m.visible) if m else [], "recording": bool(m and m.event),
                "last_event": last, "reconnects": s.reconnects if s else 0,
            })
        return rows

    def status_text(self) -> str:
        up = int(time.time() - self.started)
        lines = [f"<b>camwatch</b> · up {up // 86400}d {up % 86400 // 3600}h {up % 3600 // 60}m · {self.detector.device}",
                 self.arm_text(), ""]
        for r in self.camera_rows():
            icon = {"live": "🟢", "disabled": "⚪"}.get(r["status"], "🔴")
            line = f"{icon} <b>{r['name']}</b> {r['status']}"
            if r["status"] == "live":
                line += f" · {r['resolution'][0]}x{r['resolution'][1]} · {r['fps']:.0f} fps"
            if r["visible"]:
                line += f" · now: {', '.join(r['visible'])}"
            if r["last_event"]:
                le = r["last_event"]
                line += f"\n   last: {datetime.fromtimestamp(le.wall_time):%d %b} {clock(le.wall_time)} {le.decision} – {le.summary()}"
            lines.append(line)
        return "\n".join(lines)
