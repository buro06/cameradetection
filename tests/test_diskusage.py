import os
import shutil
from collections import namedtuple

import pytest

import camwatch.diskusage as du
from camwatch.config import AppConfig

KB = 1024


def write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


@pytest.fixture
def install(tmp_path):
    base = tmp_path / "cameradetection"
    write(base / "config.yaml", 1 * KB)
    write(base / "camwatch" / "engine.py", 3 * KB)
    write(base / ".venv" / "lib" / "torch.dll", 50 * KB)
    write(base / "data" / "clips" / "2026-09-23" / "front_120000.mp4", 20 * KB)
    write(base / "data" / "clips" / "2026-09-23" / "front_120500.mp4", 20 * KB)
    write(base / "data" / "snapshots" / "a.jpg", 2 * KB)
    write(base / "data" / "models" / "yolo11s.pt", 10 * KB)
    write(base / "data" / "faces.db", 4 * KB)
    cfg = AppConfig()
    cfg._path = base / "config.yaml"
    return cfg, base


def parts(m):
    return {label: u.bytes for label, u in m["parts"]}


def test_breakdown_adds_up(install):
    cfg, base = install
    m = du.measure(cfg)
    p = parts(m)
    assert m["total"].bytes == 110 * KB and m["total"].files == 8
    assert p["Clips"] == 40 * KB and p["Snapshots"] == 2 * KB and p["Models"] == 10 * KB
    assert p["Face DB, event log & other data"] == 4 * KB
    assert p["Python environment (.venv)"] == 50 * KB
    assert p["Program & other files"] == 4 * KB
    assert sum(p.values()) == m["total"].bytes


def test_data_folder_outside_install_is_counted(install, tmp_path):
    cfg, base = install
    shutil.move(base / "data", tmp_path / "elsewhere")
    cfg.data_dir = str(tmp_path / "elsewhere")
    m = du.measure(cfg)
    assert m["total"].bytes == 110 * KB
    assert parts(m)["Program & other files"] == 4 * KB


def test_symlinks_are_not_followed(install, tmp_path):
    cfg, base = install
    big = tmp_path / "big"
    write(big / "movie.mkv", 500 * KB)
    try:
        os.symlink(big, base / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")
    assert du.measure(cfg)["total"].bytes == 110 * KB


def test_report_text_and_low_space_warning(install, monkeypatch):
    cfg, base = install
    DU = namedtuple("DU", "total used free")
    monkeypatch.setattr(du.shutil, "disk_usage", lambda p: DU(100 * 1024**3, 60 * 1024**3, 40 * 1024**3))
    text = du.disk_report(cfg)
    assert "camwatch folder: <b>110 KB</b> (8 files)" in text
    assert "40.0 GB free</b> of 100.0 GB (40% free)" in text
    assert "Clips: 40 KB" in text and "Low disk space" not in text
    assert "another disk" not in text  # data folder is on the same disk
    monkeypatch.setattr(du.shutil, "disk_usage", lambda p: DU(100 * 1024**3, 97 * 1024**3, 3 * 1024**3))
    assert "Low disk space" in du.disk_report(cfg)


@pytest.mark.parametrize("n,text", [(512, "512 B"), (2048, "2 KB"), (5 * 1024**2, "5.0 MB"), (3 * 1024**3, "3.0 GB")])
def test_human(n, text):
    assert du.human(n) == text
