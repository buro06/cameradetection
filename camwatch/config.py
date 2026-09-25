"""Configuration loading/saving (YAML <-> dataclasses)."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(os.environ.get("CAMWATCH_CONFIG", "config.yaml"))


@dataclass
class CameraConfig:
    name: str
    source: str  # "0" (USB index), rtsp://..., http(s)://..., or a video file path
    enabled: bool = True
    # auto | stream | snapshot (poll a JPEG URL)
    mode: str = "auto"
    # auto | dshow | msmf | ffmpeg | v4l2 | avfoundation  (USB cameras only use this)
    backend: str = "auto"
    width: int = 0  # requested capture size for USB cameras (0 = driver default)
    height: int = 0
    fps: float = 0.0  # requested frame rate for USB cameras (0 = driver default)
    # USB pixel format: auto (MJPG on Windows) | MJPG | YUY2 | none
    fourcc: str = "auto"
    snapshot_fps: float = 2.0


@dataclass
class DetectionConfig:
    model: str = "yolo11s.pt"
    device: str = "auto"  # auto | cuda:0 | cpu | mps
    confidence: float = 0.5
    imgsz: int = 640
    detect_fps: float = 5.0  # inference rate per camera
    half: bool = False  # Maxwell GPUs (e.g. Quadro M4000) have no fast FP16
    min_box_height: float = 0.0  # ignore people smaller than this fraction of frame height


@dataclass
class FaceConfig:
    detector_score: float = 0.85
    # Measured on SFace: faces under ~60 px wide or very blurry rarely match even the right person.
    min_face_px: int = 64  # narrower faces are ignored
    # Blurry or turned-away faces are skipped too: they don't vote on who someone is, and aren't saved or enrolled.
    quality_min_sharpness: float = 25.0  # detail left in the aligned face (Laplacian variance); lower = blurrier
    quality_max_turn: float = 0.45  # how far the face is turned from the camera (0 = straight on, ~0.5 = 45°)
    match_threshold: float = 0.40  # SFace cosine similarity (OpenCV suggests 0.363)
    trusted_min_matches: int = 2  # face matches needed before a trusted person suppresses alerts
    save_unknowns: bool = True
    max_unknown_images: int = 5  # face crops kept per unknown cluster


@dataclass
class EventConfig:
    pre_seconds: float = 0.0  # clip starts this long before the person was first detected
    post_seconds: float = 5.0  # ...and runs this long after (identification continues meanwhile)
    cooldown_seconds: float = 60.0  # same person leaving and coming back within this time doesn't re-alert
    track_ttl: float = 2.0  # seconds a track survives without detections
    lost_memory_seconds: float = 30.0  # a still person hidden this long and reappearing in place isn't "new"


@dataclass
class ClipConfig:
    fps: float = 15.0
    max_width: int = 1280
    crf: int = 28
    annotate: bool = True
    retention_days: int = 14


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_ids: list[int] = field(default_factory=list)  # chats that receive alerts
    # command whitelist, applies in alert groups too (empty = anyone in an alert chat may send commands)
    allowed_user_ids: list[int] = field(default_factory=list)
    send_unknown_faces: bool = True


@dataclass
class AppConfig:
    data_dir: str = "data"
    rtsp_transport: str = "tcp"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    events: EventConfig = field(default_factory=EventConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    cameras: list[CameraConfig] = field(default_factory=list)

    # not serialized
    _path: Path = field(default=DEFAULT_CONFIG_PATH, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        if not p.is_absolute():
            p = self._path.resolve().parent / p
        return p

    def telegram_token(self) -> str:
        return os.environ.get("CAMWATCH_TELEGRAM_TOKEN") or self.telegram.bot_token

    def camera(self, name: str) -> CameraConfig | None:
        for c in self.cameras:
            if c.name.lower() == name.lower():
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            v = getattr(self, f.name)
            if is_dataclass(v):
                v = asdict(v)
            elif isinstance(v, list):
                v = [asdict(x) if is_dataclass(x) else x for x in v]
            out[f.name] = v
        return out

    def save(self, path: Path | None = None) -> None:
        path = Path(path or self._path)
        with self._lock:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
            os.replace(tmp, path)


def _build(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:  # typo, or a setting removed in a newer version: ignored, and dropped on the next save
        log.warning("Ignoring unknown %s keys in config: %s", cls.__name__, ", ".join(sorted(unknown)))
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    path = Path(path)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig(
        data_dir=raw.get("data_dir", "data"),
        rtsp_transport=raw.get("rtsp_transport", "tcp"),
        detection=_build(DetectionConfig, raw.get("detection")),
        face=_build(FaceConfig, raw.get("face")),
        events=_build(EventConfig, raw.get("events")),
        clip=_build(ClipConfig, raw.get("clip")),
        telegram=_build(TelegramConfig, raw.get("telegram")),
        cameras=[_build(CameraConfig, {**c, "source": str(c.get("source", ""))}) for c in raw.get("cameras") or []],
    )
    cfg._path = path
    return cfg
