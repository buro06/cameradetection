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
from typing import Callable

import cv2
import numpy as np

from .camera import STALE_SECONDS, CameraStream, describe_source, set_rtsp_transport
from .config import AppConfig, CameraConfig, SwitchConfig
from .detector import PersonDetector
from .events import CameraMonitor, EventResult
from .faces import FaceDB, FaceEngine
from .recorder import clip_path, draw_boxes, draw_stamp, purge_old_clips, write_clip
from .schedule import armed_at, next_change, parse_schedule
from .switches import HTTP_TIMEOUT, SwitchWorker
from .timefmt import clock, hm, now_text, stamp

log = logging.getLogger(__name__)


@dataclass
class EventLogEntry:
    wall_time: float
    camera: str
    decision: str
    summary: str
    clip: str | None = None


class AlertLimiter:
    """Hard per-camera cooldown for all alert channels: at most one alert per `interval` seconds on each camera."""

    def __init__(self):
        self._last: dict[str, float] = {}

    def allow(self, camera: str, now: float, interval: float) -> bool:
        if now - self._last.get(camera, float("-inf")) < interval:
            return False
        self._last[camera] = now
        return True


class Engine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.data = cfg.data_path
        self.data.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.data / "clips"
        set_rtsp_transport(cfg.rtsp_transport)

        self.db = FaceDB(self.data, cfg.face.match_threshold)
        self.faces = FaceEngine(self.data / "models", cfg.face)
        self.db.migrate_embeddings(self.faces)
        self.detector = PersonDetector(cfg.detection, self.data / "models")

        self.streams: dict[str, CameraStream] = {}
        self.monitors: dict[str, CameraMonitor] = {}
        self.events: deque[EventLogEntry] = deque(maxlen=100)
        self._alerts: queue.Queue[EventResult] = queue.Queue(maxsize=20)
        self._limiter = AlertLimiter()
        self._halt = threading.Event()
        self._workers: list[threading.Thread] = []
        self._cam_lock = threading.RLock()
        self.started = time.time()

        self._state_path = self.data / "state.json"
        self.armed = True
        self.disarmed_cameras: set[str] = set()
        self._sched_state: dict[str, bool] = {}  # what each camera's schedule last set it to
        self._load_state()

        self.bot = None
        if cfg.telegram.enabled and cfg.telegram_token():
            from .telegram import TelegramBot
            self.bot = TelegramBot(self)
        self.ntfy = None
        if cfg.ntfy.enabled and cfg.ntfy.topic:
            from .ntfy import NtfyNotifier
            self.ntfy = NtfyNotifier(self)
        self.switches = [SwitchWorker(self, sw) for sw in cfg.http_get if sw.enabled]

    # ---- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        self.apply_schedules()
        for cam in self.cfg.cameras:
            if cam.enabled:
                self._start_camera(cam)
        self._workers = [threading.Thread(target=self._detect_loop, name="detector", daemon=True),
                         threading.Thread(target=self._alert_loop, name="alerts", daemon=True),
                         threading.Thread(target=self._housekeeping_loop, name="housekeeping", daemon=True)]
        for t in self._workers:
            t.start()
        if self.bot:
            self.bot.start()
        if self.ntfy:
            self.ntfy.start()
        for sw in self.switches:
            sw.start()
        log.info("camwatch started: %d camera(s), detector on %s", len(self.streams), self.detector.device)

    def stop(self, say: Callable[[str], None] = lambda msg: None, timeout: float = 5.0) -> None:
        """Signal all threads to stop, then wait up to `timeout` for each camera and worker thread,
        reporting every step through `say`."""
        self._halt.set()
        if self.bot:
            say("stopping Telegram bot")
            self.bot.stop()
        if self.ntfy:
            self.ntfy.stop()
        with self._cam_lock:
            streams = list(self.streams.values())
        for s in [*streams, *self.switches]:
            s.stop()
        for t in [*streams, *self._workers, *self.switches]:
            say(f"waiting for {t.name} …")
            t0 = time.monotonic()
            t.join(timeout)
            state = "still running, giving up" if t.is_alive() else "stopped"
            say(f"  {t.name}: {state} ({time.monotonic() - t0:.1f}s)")
        for sw in self.switches:
            if sw.state and not sw.is_alive():
                say(f"turning {sw.name} off")
                sw.switch_off()

    # ---- cameras ---------------------------------------------------------------
    def _start_camera(self, cam: CameraConfig) -> None:
        with self._cam_lock:
            # the clip starts when someone appeared, which is known `arrive_seconds` later
            stream = CameraStream(cam, self.cfg.clip, self.cfg.events.pre_seconds + self.cfg.events.arrive_seconds)
            self.streams[cam.name] = stream
            self.monitors[cam.name] = CameraMonitor(self.cfg, stream, self.faces, self.db, self.is_armed)
            stream.start()
            log.info("[%s] started (%s)", cam.name, describe_source(cam.source))

    def stop_camera(self, name: str, wait: bool = False) -> None:
        """Stop a camera; with `wait`, until its device is released (so it can be opened again right away)."""
        with self._cam_lock:
            self.monitors.pop(name, None)
            s = self.streams.pop(name, None)
        if s:
            s.stop()
            if wait:
                s.join(timeout=12.0)

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

    def apply_ntfy(self) -> None:
        """Pick up ntfy config changes made at runtime (credentials and topic are read on every send)."""
        if self.cfg.ntfy.enabled and self.cfg.ntfy.topic:
            if self.ntfy is None:
                from .ntfy import NtfyNotifier
                self.ntfy = NtfyNotifier(self)
                self.ntfy.start()
        elif self.ntfy is not None:
            self.ntfy.stop()
            self.ntfy = None

    def apply_switches(self) -> None:
        """Restart the HTTP GET workers after targets were added, edited or removed. One whose URLs didn't
        change keeps its known state (no extra OFF); one that was removed, disabled or pointed elsewhere is
        turned off if it was on."""
        old = {id(sw.cfg): sw for sw in self.switches}
        for sw in old.values():
            sw.stop()
        for sw in old.values():
            if sw.is_alive():
                sw.join(HTTP_TIMEOUT + 1)  # let a request in flight finish
        self.switches = []
        for cfg in self.cfg.http_get:
            if not cfg.enabled:
                continue
            sw = SwitchWorker(self, cfg)
            prev = old.get(id(cfg))
            if prev and prev.urls == sw.urls:
                sw.state, sw.failing = prev.state, prev.failing
                del old[id(cfg)]
            self.switches.append(sw)
        for prev in old.values():
            prev.switch_off()
        for sw in self.switches:
            sw.start()

    def resync_switch(self, cfg: SwitchConfig) -> None:
        """The device may have been switched behind the worker's back (e.g. a test): send what it should be."""
        for sw in self.switches:
            if sw.cfg is cfg:
                sw.state = None

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

    # ---- schedules ---------------------------------------------------------------
    def apply_schedules(self, now: datetime | None = None) -> None:
        """Arm/disarm cameras whose schedule changed state since the last check (or that weren't checked yet).
        A manual arm/disarm in between is left alone until the schedule's next change. Only the camera's own
        switch is flipped, so "disarm all" stays in force."""
        now = now or datetime.now()
        changed = False
        for cam in self.cfg.cameras:
            if not cam.schedule:
                self._sched_state.pop(cam.name, None)
                continue
            windows = parse_schedule(cam.schedule)
            want = armed_at(windows, now)
            if self._sched_state.get(cam.name) == want:
                continue
            self._sched_state[cam.name] = want
            if want == (cam.name not in self.disarmed_cameras):
                continue
            (self.disarmed_cameras.discard if want else self.disarmed_cameras.add)(cam.name)
            changed = True
            nxt = next_change(windows, now)
            log.info("[%s] schedule: %s%s", cam.name, "armed" if want else "disarmed",
                     f" until {self._when(nxt, now)}" if nxt else "")
        if changed:
            self._save_state()

    def reset_schedule(self, camera: str) -> None:
        """Apply the camera's (edited) schedule on the next check, overriding a manual arm/disarm."""
        self._sched_state.pop(camera, None)

    @staticmethod
    def _when(dt: datetime, now: datetime) -> str:
        return hm(dt) if dt.date() == now.date() else f"{dt:%a} {hm(dt)}"

    def schedule_text(self, camera: str, now: datetime | None = None) -> str:
        """The schedule's next change, e.g. 'arms 6:00 PM' ('' without a schedule)."""
        cam = self.cfg.camera(camera)
        if not cam or not cam.schedule:
            return ""
        now = now or datetime.now()
        windows = parse_schedule(cam.schedule)
        nxt = next_change(windows, now)
        if nxt is None:
            return "always armed" if armed_at(windows, now) else "never armed"
        return f"{'disarms' if armed_at(windows, now) else 'arms'} {self._when(nxt, now)}"

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
        if result.decision == "alert" and not self._limiter.allow(
                result.camera, time.monotonic(), self.cfg.telegram.min_alert_interval_seconds):
            result.decision = "rate-limited"  # still clipped and logged, just not sent
            log.info("[%s] alert not sent: another one went out less than %gs ago", result.camera,
                     self.cfg.telegram.min_alert_interval_seconds)
        entry = EventLogEntry(result.wall_time, result.camera, result.decision, result.summary())
        self.events.appendleft(entry)
        if result.decision in ("alert", "rate-limited"):
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
            self._handle_alert(result, entry)

    def _handle_alert(self, result: EventResult, entry: EventLogEntry) -> None:
        path = None
        try:
            path = write_clip(result, clip_path(self.clips_dir, result), self.cfg.clip)
            entry.clip = str(path)
            log.info("[%s] clip saved: %s", result.camera, path.name)
        except Exception:
            log.exception("[%s] clip encoding failed", result.camera)
        self._append_event_log(entry)
        if result.decision == "alert":
            for notifier in (self.bot, self.ntfy):
                if notifier:
                    try:
                        notifier.send_alert(result, path)
                    except Exception:
                        log.exception("[%s] alert delivery failed", result.camera)
        result.frames = []  # free memory

    def _housekeeping_loop(self) -> None:
        next_purge = 0.0
        while not self._halt.wait(5):
            try:
                self.apply_schedules()
            except Exception:
                log.exception("Applying alert schedules failed")
            if time.monotonic() < next_purge:
                continue
            next_purge = time.monotonic() + 3600
            try:
                n = purge_old_clips(self.clips_dir, self.cfg.clip.retention_days)
                if n:
                    log.info("Deleted %d clip(s) older than %d days", n, self.cfg.clip.retention_days)
            except Exception:
                log.exception("Housekeeping failed")

    # ---- queries -------------------------------------------------------------------
    def annotated_frame(self, camera: str) -> np.ndarray | None:
        """The camera's latest frame with the current detections and a time stamp drawn on it."""
        cam = self.cfg.camera(camera)
        with self._cam_lock:
            stream, mon = (self.streams.get(cam.name), self.monitors.get(cam.name)) if cam else (None, None)
        if stream is None or mon is None:
            raise KeyError(camera)
        _, _, frame = stream.latest()
        if frame is None:
            return None
        img = frame.copy()
        overlay = stream.overlay
        if overlay and time.monotonic() - overlay.ts < 1.0:
            draw_boxes(img, overlay.boxes, mon.live_labels(), 1.0)
        draw_stamp(img, f"{cam.name}  {stamp(datetime.now())}")
        return img

    def snapshot(self, camera: str) -> bytes | None:
        img = self.annotated_frame(camera)
        if img is None:
            return None
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
                "schedule": self.schedule_text(cam.name),
                "visible": list(m.visible) if m else [], "recording": bool(m and m.event),
                "last_event": last, "reconnects": s.reconnects if s else 0,
            })
        return rows

    def status_text(self) -> str:
        up = int(time.time() - self.started)
        lines = [f"<b>camwatch</b> · up {up // 86400}d {up % 86400 // 3600}h {up % 3600 // 60}m · {self.detector.device}",
                 f"🕒 {now_text()}", self.arm_text(), ""]
        for r in self.camera_rows():
            icon = {"live": "🟢", "disabled": "⚪"}.get(r["status"], "🔴")
            line = f"{icon} <b>{r['name']}</b> {r['status']}"
            if r["status"] == "live":
                line += f" · {r['resolution'][0]}x{r['resolution'][1]} · {r['fps']:.0f} fps"
            if r["schedule"]:
                cam = self.cfg.camera(r["name"])
                line += f" · {'armed' if r['armed'] else 'disarmed'}, schedule {', '.join(cam.schedule)} ({r['schedule']})"
            if r["visible"]:
                line += f" · now: {', '.join(r['visible'])}"
            if r["last_event"]:
                le = r["last_event"]
                line += f"\n   last: {datetime.fromtimestamp(le.wall_time):%d %b} {clock(le.wall_time)} {le.decision} – {le.summary()}"
            lines.append(line)
        lines += [f"🌐 <b>{sw.cfg.name}</b> {sw.status}" for sw in self.switches]
        return "\n".join(lines)
