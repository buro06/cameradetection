"""Configuration loading/saving (YAML <-> dataclasses)."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(os.environ.get("CAMWATCH_CONFIG", "config.yaml"))


@dataclass
class CameraConfig:
    name: str
    source: str  # "usb:<camera name>" or "0" (USB index), rtsp://..., http(s)://..., or a video file path
    enabled: bool = True
    # auto | stream | snapshot (poll a JPEG URL)
    mode: str = "auto"
    # auto (dshow on Windows, v4l2 on Linux) | dshow | msmf | ffmpeg | v4l2 | avfoundation  (USB cameras only)
    backend: str = "auto"
    width: int = 0  # requested capture size for USB cameras (0 = driver default)
    height: int = 0
    fps: float = 0.0  # requested frame rate for USB cameras (0 = driver default)
    # USB pixel format: auto (MJPG on Windows, and on Linux above 640x480) | MJPG | YUY2 | none
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
    # people smaller than this fraction of frame height don't count (twice that if cut off at the frame edge)
    person_min_height: float = 0.15


@dataclass
class FaceConfig:
    model: str = "buffalo_l"  # InsightFace model pack: SCRFD-10G face detector + ArcFace R50 recognizer
    device: str = "auto"  # auto (GPU if onnxruntime can use it) | cuda | cpu
    detector_score: float = 0.5  # SCRFD face confidence
    min_face_px: int = 40
    check_fps: float = 2.0  # face scans per second per camera, inside person boxes, only while someone is in view
    # face detector input for whole-frame scans (5+ people, enrollment); 960/1280 find farther faces, cost more
    detect_size: int = 640
    match_threshold: float = 0.35  # ArcFace cosine similarity; raise for fewer false matches
    # Blurry or turned-away faces don't vote on who someone is, and aren't saved or enrolled.
    quality_min_sharpness: float = 25.0  # detail left in the aligned face (Laplacian variance); lower = blurrier
    quality_max_turn: float = 0.45  # how far the face is turned from the camera (0 = straight on, ~0.5 = 45°)
    min_sightings: int = 2  # clear sightings a face needs during an event to count (trusted or not)
    save_unknowns: bool = True
    max_unknown_images: int = 5  # face crops kept per unknown cluster


@dataclass
class EventConfig:
    pre_seconds: float = 0.0  # clip starts this long before someone arrived
    post_seconds: float = 5.0  # ...and runs this long after (identification continues meanwhile)
    # someone must be gone this long before coming back alerts again (a face unseen this long is new;
    # a camera that saw nobody this long is empty, and the next person in it is an arrival)
    away_seconds: float = 30.0
    arrive_seconds: float = 2.0  # someone must be in view this long in an empty camera to count as arrived


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
    allowed_user_ids: list[int] = field(default_factory=list)  # users allowed to send commands
    send_unknown_faces: bool = True
    # hard cooldown per camera: alerts within this time of the last one sent are logged and clipped, not sent
    min_alert_interval_seconds: float = 30.0


@dataclass
class NtfyConfig:
    enabled: bool = False
    server: str = "https://ntfy.sh"
    topic: str = ""
    username: str = ""
    password: str = ""  # or env CAMWATCH_NTFY_PASSWORD (preferred)
    priority: int = 4  # 1 (min) … 5 (max/urgent)
    attach_clip: bool = True  # upload the clip and link it from the alert (tap → plays)
    clip_topic: str = ""  # where clips are uploaded; don't subscribe to it (empty = <topic>-clips)

    @property
    def clips_topic(self) -> str:
        return self.clip_topic or f"{self.topic}-clips"


@dataclass
class AppConfig:
    data_dir: str = "data"
    rtsp_transport: str = "tcp"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    events: EventConfig = field(default_factory=EventConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    ntfy: NtfyConfig = field(default_factory=NtfyConfig)
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

    def ntfy_password(self) -> str:
        return os.environ.get("CAMWATCH_NTFY_PASSWORD") or self.ntfy.password

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
            tmp.write_text(self.to_yaml(), encoding="utf-8")
            os.replace(tmp, path)

    def to_yaml(self) -> str:
        """YAML where every value that differs from its default carries a `# default: …` comment."""
        lines = ["# camwatch configuration. Changed settings show their default in a comment."]
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            v = getattr(self, f.name)
            if is_dataclass(v) or f.name == "cameras":
                lines.append("")
            if is_dataclass(v):
                lines.append(f"{f.name}:")
                lines += _yaml_fields(v, "  ")
            elif f.name == "cameras":
                lines.append("cameras:" if v else "cameras: []")
                for cam in v:
                    first, *rest = _yaml_fields(cam, "    ")
                    lines += ["  - " + first.lstrip(), *rest]
            else:
                lines.append(_yaml_line(f.name, v, field_default(type(self), f.name), ""))
        return "\n".join(lines).rstrip() + "\n"


_NO_DEFAULT = object()


def field_default(cls, name: str) -> Any:
    """Default value of dataclass field `name`, or _NO_DEFAULT if it has none (e.g. a camera's source)."""
    f = next(f for f in fields(cls) if f.name == name)
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return _NO_DEFAULT


def has_default(value: Any) -> bool:
    return value is not _NO_DEFAULT


def yaml_scalar(v: Any) -> str:
    """One-line YAML for a scalar or a flat list, e.g. 0.4, true, '0', [1, 2]."""
    s = yaml.safe_dump(v, default_flow_style=True, width=1 << 30).rstrip("\n")
    return s.removesuffix("\n...")


def _yaml_line(key: str, value: Any, default: Any, indent: str) -> str:
    line = f"{indent}{key}: {yaml_scalar(value)}"
    if has_default(default) and value != default:
        line += f"  # default: {yaml_scalar(default)}"
    return line


def _yaml_fields(obj: Any, indent: str) -> list[str]:
    return [_yaml_line(f.name, getattr(obj, f.name), field_default(type(obj), f.name), indent) for f in fields(obj)]


def _build(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {', '.join(sorted(unknown))}")
    return cls(**data)


LEGACY_FACE_KEYS = ("detector_score", "match_threshold")  # tuned for YuNet + SFace; wrong for SCRFD + ArcFace


def _migrate_face(face: dict | None) -> dict | None:
    """A face section without `model` was written for YuNet + SFace: its thresholds don't fit the new models."""
    if not face or "model" in face:
        return face
    dropped = [k for k in LEGACY_FACE_KEYS if k in face]
    if dropped:
        log.warning("Face recognition now uses InsightFace %s: resetting %s to the new defaults",
                    FaceConfig.model, ", ".join(dropped))
    return {k: v for k, v in face.items() if k not in LEGACY_FACE_KEYS}


# settings of the per-person tracker that people counting replaced; saved configs still contain them
REMOVED_KEYS = {
    "detection": ("min_hits", "min_box_height", "overlap_confirm_seconds"),
    "events": ("track_ttl", "lost_memory_seconds", "handoff_seconds", "cooldown_seconds", "leave_seconds"),
    "cameras": ("leave_seconds",),
    "face": ("trusted_min_matches",),
}


def _drop_removed(raw: dict[str, Any]) -> None:
    dropped = []
    face = raw.get("face") or {}
    if "trusted_min_matches" in face and "min_sightings" not in face:
        face["min_sightings"] = face["trusted_min_matches"]
    for section, keys in REMOVED_KEYS.items():
        found = raw.get(section) or {}
        for sec in found if isinstance(found, list) else [found]:  # cameras: a list of sections
            for k in keys:
                if isinstance(sec, dict) and k in sec:
                    del sec[k]
                    dropped.append(f"{section}.{k}")
    dropped = list(dict.fromkeys(dropped))
    if dropped:
        log.warning("Ignoring settings that no longer exist (they are removed on the next save): %s", ", ".join(dropped))


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    path = Path(path)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    _drop_removed(raw)
    cfg = AppConfig(
        data_dir=raw.get("data_dir", "data"),
        rtsp_transport=raw.get("rtsp_transport", "tcp"),
        detection=_build(DetectionConfig, raw.get("detection")),
        face=_build(FaceConfig, _migrate_face(raw.get("face"))),
        events=_build(EventConfig, raw.get("events")),
        clip=_build(ClipConfig, raw.get("clip")),
        telegram=_build(TelegramConfig, raw.get("telegram")),
        ntfy=_build(NtfyConfig, raw.get("ntfy")),
        cameras=[_build(CameraConfig, {**c, "source": str(c.get("source", ""))}) for c in raw.get("cameras") or []],
    )
    cfg._path = path
    return cfg
