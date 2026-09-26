"""Disk usage of the camwatch folder compared with free space on its disk."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig

LOW_FREE_FRACTION = 0.10
LOW_FREE_BYTES = 5 * 1024**3

# data sub-folders reported individually, in this order
DATA_PARTS = [("clips", "Clips"), ("snapshots", "Snapshots"), ("faces", "Known faces"), ("unknown", "Unknown faces"),
              ("models", "Models"), ("logs", "Logs")]


@dataclass
class Usage:
    bytes: int = 0
    files: int = 0

    def __iadd__(self, other: Usage) -> Usage:
        self.bytes += other.bytes
        self.files += other.files
        return self


def folder_usage(path: Path) -> Usage:
    """Total size of all files below `path` (symlinks not followed, unreadable entries skipped)."""
    total = Usage()
    stack = [path]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total.bytes += entry.stat(follow_symlinks=False).st_size
                            total.files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def measure(cfg: AppConfig) -> dict:
    """Sizes of the camwatch folder (the config file's folder) and its parts, plus disk totals."""
    base = cfg._path.resolve().parent
    data = cfg.data_path.resolve()
    parts: list[tuple[str, Usage]] = []

    data_total = Usage()
    for sub, label in DATA_PARTS:
        u = folder_usage(data / sub)
        data_total += u
        parts.append((label, u))
    data_all = folder_usage(data)
    parts.append(("Face DB, event log & other data", Usage(data_all.bytes - data_total.bytes, data_all.files - data_total.files)))

    venv = base / ".venv"
    venv_usage = folder_usage(venv) if venv.is_dir() else Usage()
    if venv.is_dir():
        parts.append(("Python environment (.venv)", venv_usage))

    base_all = folder_usage(base)
    other = Usage(base_all.bytes - venv_usage.bytes, base_all.files - venv_usage.files)
    if _inside(data, base):
        other.bytes -= data_all.bytes
        other.files -= data_all.files
        folder_total = base_all
    else:  # data folder lives elsewhere: count it too
        folder_total = Usage(base_all.bytes + data_all.bytes, base_all.files + data_all.files)
    parts.append(("Program & other files", other))

    disk = shutil.disk_usage(base)
    data_disk = shutil.disk_usage(data) if data.exists() and os.stat(data).st_dev != os.stat(base).st_dev else None
    return {"base": base, "data": data, "total": folder_total, "parts": parts, "disk": disk, "data_disk": data_disk,
            "retention_days": cfg.clip.retention_days}


def disk_report(cfg: AppConfig, esc=lambda s: s) -> str:
    m = measure(cfg)
    total, disk = m["total"], m["disk"]
    free_frac = disk.free / disk.total if disk.total else 0.0
    lines = [f"💾 <b>Disk usage</b>",
             f"camwatch folder: <b>{human(total.bytes)}</b> ({total.files:,} files)",
             f"<code>{esc(str(m['base']))}</code>"]
    for label, u in m["parts"]:
        if u.bytes > 0:
            lines.append(f"  • {label}: {human(u.bytes)}")
    lines += ["",
              f"Disk <code>{esc(m['base'].anchor or str(m['base']))}</code>: <b>{human(disk.free)} free</b> of "
              f"{human(disk.total)} ({free_frac:.0%} free)",
              f"camwatch uses {total.bytes / disk.total:.1%} of the disk "
              f"(its size equals {total.bytes / max(disk.free, 1):.1%} of the free space)"]
    if m["data_disk"] is not None:
        dd = m["data_disk"]
        lines.append(f"Data folder is on another disk (<code>{esc(str(m['data']))}</code>): "
                     f"{human(dd.free)} free of {human(dd.total)} ({dd.free / dd.total:.0%} free)")
    if disk.free < LOW_FREE_BYTES or free_frac < LOW_FREE_FRACTION:
        lines.append("⚠️ <b>Low disk space</b> — lower <code>clip.retention_days</code> or delete old clips.")
    days = m["retention_days"]
    lines.append(f"<i>Clips are kept {days} days.</i>" if days > 0 else "<i>Clips are kept forever (retention_days: 0).</i>")
    return "\n".join(lines)
