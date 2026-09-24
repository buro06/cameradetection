"""Render event clips to H.264 MP4 (Telegram plays these inline) and annotate frames."""

from __future__ import annotations

import bisect
import logging
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import ClipConfig
from .events import EventResult

log = logging.getLogger(__name__)

GREEN, ORANGE, RED, WHITE, BLACK = (80, 200, 60), (0, 165, 255), (40, 40, 230), (255, 255, 255), (0, 0, 0)


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def label_color(label: str) -> tuple[int, int, int]:
    if "(trusted)" in label or label.endswith("✓"):
        return GREEN
    if label.lower().startswith(("unknown", "person")):
        return RED
    return ORANGE


def draw_boxes(img: np.ndarray, boxes, labels: dict[int, str], scale: float) -> None:
    for tid, (x1, y1, x2, y2) in boxes:
        label = labels.get(tid, "Person")
        color = label_color(label)
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        thick = max(2, img.shape[1] // 500)
        cv2.rectangle(img, p1, p2, color, thick)
        text = label.replace("✓", "").strip()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(th + 6, p1[1])
        cv2.rectangle(img, (p1[0], y - th - 6), (p1[0] + tw + 6, y), color, -1)
        cv2.putText(img, text, (p1[0] + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2, cv2.LINE_AA)


def draw_stamp(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)


def write_clip(result: EventResult, path: Path, cfg: ClipConfig) -> Path:
    frames = sorted(result.frames, key=lambda f: f.ts)
    if not frames:
        raise ValueError("no frames recorded for this event")
    h, w = frames[0].image.shape[:2]
    stamps = [f.ts for f in frames]
    labels = result.labels
    n_out = max(1, int(round((result.end - result.start) * cfg.fps)))

    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg not found (pip install imageio-ffmpeg)")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{cfg.fps:g}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", str(cfg.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(n_out):
            t = result.start + i / cfg.fps
            # resample by timestamp so the clip always has real-time duration
            f = frames[max(0, bisect.bisect_right(stamps, t) - 1)]
            img = f.image if f.image.shape[:2] == (h, w) else cv2.resize(f.image, (w, h))
            if cfg.annotate:
                img = img.copy()
                if f.overlay:
                    draw_boxes(img, f.overlay.boxes, labels, f.scale)
                wall = result.wall_time + (t - result.anchor)
                draw_stamp(img, f"{result.camera}  {datetime.fromtimestamp(wall):%Y-%m-%d %H:%M:%S}")
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait(timeout=60) != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")
    except Exception:
        proc.kill()
        path.unlink(missing_ok=True)
        raise
    return path


def clip_path(clips_dir: Path, result: EventResult) -> Path:
    ts = datetime.fromtimestamp(result.wall_time)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in result.camera)
    return clips_dir / f"{ts:%Y-%m-%d}" / f"{safe}_{ts:%H%M%S}.mp4"


def purge_old_clips(clips_dir: Path, retention_days: int) -> int:
    if retention_days <= 0 or not clips_dir.exists():
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for f in clips_dir.rglob("*.mp4"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            removed += 1
    for d in sorted(clips_dir.iterdir(), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return removed
