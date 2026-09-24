"""Face detection (YuNet), recognition (SFace) and the face database (SQLite).

Both models come from the OpenCV Model Zoo (Apache-2.0 / MIT) and run through cv2.dnn.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import FaceConfig

log = logging.getLogger(__name__)

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
YUNET_MAX_SIDE = 640
MAX_EMBEDDINGS_PER_PERSON = 60
MAX_EMBEDDINGS_PER_UNKNOWN = 20


def ensure_models(models_dir: Path) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = models_dir / name
        if not path.exists() or path.stat().st_size < 100_000:
            log.info("Downloading %s", name)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 100_000:  # git-lfs pointer or error page
                tmp.unlink()
                raise RuntimeError(f"Download of {name} failed (file too small). Get it manually from {url}")
            tmp.replace(path)
        paths[name] = path
    return paths


@dataclass
class FaceObs:
    box: tuple[int, int, int, int]  # xyxy in frame coords
    score: float
    quality: float
    embedding: np.ndarray  # (128,) L2-normalised
    crop: np.ndarray  # face with context, for humans to look at


@dataclass
class Match:
    person_id: int
    name: str
    trusted: bool
    score: float


class FaceEngine:
    def __init__(self, models_dir: Path, cfg: FaceConfig):
        self.cfg = cfg
        paths = ensure_models(models_dir)
        self._det = cv2.FaceDetectorYN.create(str(paths["face_detection_yunet_2023mar.onnx"]), "", (320, 320), 0.5, 0.3, 50)
        self._rec = cv2.FaceRecognizerSF.create(str(paths["face_recognition_sface_2021dec.onnx"]), "")
        self._lock = threading.Lock()  # cv2.dnn nets are not thread-safe

    def analyze(self, frame: np.ndarray, region: tuple[float, float, float, float] | None = None,
                max_faces: int = 1, min_score: float | None = None) -> list[FaceObs]:
        """Find faces in `frame` (optionally only inside region xyxy) and embed them, best first."""
        fh, fw = frame.shape[:2]
        if region is None:
            x0, y0, x1, y1 = 0, 0, fw, fh
        else:
            bw, bh = region[2] - region[0], region[3] - region[1]
            x0 = int(max(0, region[0] - 0.1 * bw))
            y0 = int(max(0, region[1] - 0.1 * bh))
            x1 = int(min(fw, region[2] + 0.1 * bw))
            y1 = int(min(fh, region[3] + 0.05 * bh))
        crop = frame[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        if ch < self.cfg.min_face_px or cw < self.cfg.min_face_px:
            return []
        s = min(1.0, YUNET_MAX_SIDE / max(ch, cw))
        small = crop if s >= 1.0 else cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))))
        min_score = self.cfg.detector_score if min_score is None else min_score

        with self._lock:
            self._det.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self._det.detect(small)
            if rows is None:
                return []
            rows = rows.copy()
            rows[:, :14] /= s  # back to crop coordinates
            rows = rows[rows[:, 14] >= min_score]
            rows = rows[rows[:, 2] >= self.cfg.min_face_px]
            rows = rows[np.argsort(-rows[:, 14] * rows[:, 2])][:max_faces]
            out = []
            for row in rows:
                chip = self._rec.alignCrop(crop, row)
                emb = self._rec.feature(chip).reshape(-1).astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-9
                x, y, w, h = row[:4]
                box = (int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + h))
                out.append(FaceObs(box=box, score=float(row[14]), quality=float(row[14]) * min(1.0, w / 112.0),
                                   embedding=emb, crop=_context_crop(frame, box)))
        return out


def _context_crop(frame: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.5) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    fh, fw = frame.shape[:2]
    cx1, cy1 = max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h))
    cx2, cy2 = min(fw, int(x2 + pad * w)), min(fh, int(y2 + pad * h))
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.shape[0] < 160:  # upscale tiny faces so they are viewable in Telegram
        f = 160 / max(1, crop.shape[0])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return crop


def _vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class FaceDB:
    """Known people, their embeddings, and clusters of unknown faces."""

    def __init__(self, data_dir: Path, match_threshold: float):
        self.dir = data_dir
        self.threshold = match_threshold
        (data_dir / "faces").mkdir(parents=True, exist_ok=True)
        (data_dir / "unknown").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(data_dir / "faces.db", check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE COLLATE NOCASE NOT NULL,
                trusted INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                vec BLOB NOT NULL, image TEXT, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS unknowns (
                id INTEGER PRIMARY KEY, created REAL NOT NULL, last_seen REAL NOT NULL,
                camera TEXT, sightings INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS unknown_embeddings (
                id INTEGER PRIMARY KEY, unknown_id INTEGER NOT NULL REFERENCES unknowns(id) ON DELETE CASCADE,
                vec BLOB NOT NULL, image TEXT, quality REAL NOT NULL DEFAULT 0, created REAL NOT NULL);
            """
        )
        self._db.commit()
        self._reload()

    # ---- caches -------------------------------------------------------------
    def _reload(self) -> None:
        with self._lock:
            self._persons = {pid: (name, bool(tr)) for pid, name, tr in self._db.execute("SELECT id, name, trusted FROM persons")}
            rows = self._db.execute("SELECT person_id, vec FROM embeddings").fetchall()
            self._known_ids = np.array([r[0] for r in rows], dtype=np.int64)
            self._known = np.stack([_vec(r[1]) for r in rows]) if rows else np.zeros((0, 128), np.float32)
            rows = self._db.execute("SELECT unknown_id, vec FROM unknown_embeddings").fetchall()
            self._unk_ids = np.array([r[0] for r in rows], dtype=np.int64)
            self._unk = np.stack([_vec(r[1]) for r in rows]) if rows else np.zeros((0, 128), np.float32)

    @staticmethod
    def _best(matrix: np.ndarray, ids: np.ndarray, emb: np.ndarray) -> tuple[int, float] | None:
        if not len(ids):
            return None
        sims = matrix @ emb
        best_id, best = -1, -1.0
        for uid in np.unique(ids):
            s = float(sims[ids == uid].max())
            if s > best:
                best_id, best = int(uid), s
        return best_id, best

    # ---- matching -------------------------------------------------------------
    def match(self, emb: np.ndarray) -> Match | None:
        with self._lock:
            hit = self._best(self._known, self._known_ids, emb)
            if hit is None or hit[1] < self.threshold:
                return None
            name, trusted = self._persons[hit[0]]
            return Match(hit[0], name, trusted, hit[1])

    def record_unknown(self, emb: np.ndarray, crop: np.ndarray, quality: float, camera: str,
                       max_images: int = 5) -> int:
        """Add an unknown face to the closest unknown cluster (or a new one). Returns the cluster id."""
        now = time.time()
        with self._lock:
            hit = self._best(self._unk, self._unk_ids, emb)
            if hit is not None and hit[1] >= self.threshold:
                uid = hit[0]
                self._db.execute("UPDATE unknowns SET last_seen=?, sightings=sightings+1, camera=? WHERE id=?", (now, camera, uid))
            else:
                uid = self._db.execute("INSERT INTO unknowns (created, last_seen, camera) VALUES (?,?,?)", (now, now, camera)).lastrowid
            count = self._db.execute("SELECT COUNT(*) FROM unknown_embeddings WHERE unknown_id=?", (uid,)).fetchone()[0]
            if count < MAX_EMBEDDINGS_PER_UNKNOWN:
                image = None
                with_img = self._db.execute("SELECT COUNT(*) FROM unknown_embeddings WHERE unknown_id=? AND image IS NOT NULL", (uid,)).fetchone()[0]
                if with_img < max_images:
                    image = self._save_image(self.dir / "unknown" / str(uid), crop)
                self._db.execute(
                    "INSERT INTO unknown_embeddings (unknown_id, vec, image, quality, created) VALUES (?,?,?,?,?)",
                    (uid, emb.astype(np.float32).tobytes(), image, quality, now))
            self._db.commit()
            self._reload()
            return uid

    # ---- people ---------------------------------------------------------------
    def list_persons(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT p.id, p.name, p.trusted, COUNT(e.id) FROM persons p LEFT JOIN embeddings e ON e.person_id=p.id "
                "GROUP BY p.id ORDER BY p.name").fetchall()
        return [{"id": r[0], "name": r[1], "trusted": bool(r[2]), "samples": r[3]} for r in rows]

    def trusted_map(self) -> dict[str, bool]:
        with self._lock:
            return {name: trusted for name, trusted in self._persons.values()}

    def get_person(self, name: str) -> dict | None:
        return next((p for p in self.list_persons() if p["name"].lower() == name.lower()), None)

    def add_person(self, name: str, trusted: bool = False) -> int:
        name = name.strip()
        if not name:
            raise ValueError("name must not be empty")
        with self._lock:
            existing = self.get_person(name)
            if existing:
                return existing["id"]
            pid = self._db.execute("INSERT INTO persons (name, trusted, created) VALUES (?,?,?)", (name, int(trusted), time.time())).lastrowid
            self._db.commit()
            self._reload()
            return pid

    def add_embedding(self, person_id: int, emb: np.ndarray, crop: np.ndarray | None) -> None:
        with self._lock:
            image = self._save_image(self.dir / "faces" / str(person_id), crop) if crop is not None else None
            self._db.execute("INSERT INTO embeddings (person_id, vec, image, created) VALUES (?,?,?,?)",
                             (person_id, emb.astype(np.float32).tobytes(), image, time.time()))
            self._trim(person_id)
            self._db.commit()
            self._reload()

    def _trim(self, person_id: int) -> None:
        rows = self._db.execute("SELECT id, image FROM embeddings WHERE person_id=? ORDER BY created DESC", (person_id,)).fetchall()
        for eid, image in rows[MAX_EMBEDDINGS_PER_PERSON:]:
            self._db.execute("DELETE FROM embeddings WHERE id=?", (eid,))
            if image:
                Path(image).unlink(missing_ok=True)

    def set_trusted(self, person_id: int, trusted: bool) -> None:
        with self._lock:
            self._db.execute("UPDATE persons SET trusted=? WHERE id=?", (int(trusted), person_id))
            self._db.commit()
            self._reload()

    def rename(self, person_id: int, name: str) -> None:
        with self._lock:
            self._db.execute("UPDATE persons SET name=? WHERE id=?", (name.strip(), person_id))
            self._db.commit()
            self._reload()

    def delete_person(self, person_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM persons WHERE id=?", (person_id,))
            self._db.commit()
            shutil.rmtree(self.dir / "faces" / str(person_id), ignore_errors=True)
            self._reload()

    # ---- unknowns -------------------------------------------------------------
    def list_unknowns(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT id, created, last_seen, camera, sightings FROM unknowns ORDER BY last_seen DESC").fetchall()
        return [{"id": r[0], "created": r[1], "last_seen": r[2], "camera": r[3], "sightings": r[4]} for r in rows]

    def unknown_exists(self, uid: int) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM unknowns WHERE id=?", (uid,)).fetchone() is not None

    def unknown_images(self, uid: int) -> list[Path]:
        with self._lock:
            rows = self._db.execute("SELECT image FROM unknown_embeddings WHERE unknown_id=? AND image IS NOT NULL "
                                    "ORDER BY quality DESC", (uid,)).fetchall()
        return [Path(r[0]) for r in rows if Path(r[0]).exists()]

    def assign_unknown(self, uid: int, name: str, trusted: bool | None = None) -> dict:
        """Move an unknown cluster's faces to a (new or existing) person."""
        with self._lock:
            if not self.unknown_exists(uid):
                raise KeyError(f"unknown face #{uid} not found")
            pid = self.add_person(name, trusted=bool(trusted))
            if trusted is not None:
                self.set_trusted(pid, trusted)
            dest = self.dir / "faces" / str(pid)
            dest.mkdir(parents=True, exist_ok=True)
            for vec, image in self._db.execute("SELECT vec, image FROM unknown_embeddings WHERE unknown_id=?", (uid,)).fetchall():
                new_image = None
                if image and Path(image).exists():
                    new_image = str(dest / Path(image).name)
                    shutil.move(image, new_image)
                self._db.execute("INSERT INTO embeddings (person_id, vec, image, created) VALUES (?,?,?,?)",
                                 (pid, vec, new_image, time.time()))
            self._trim(pid)
            self._db.execute("DELETE FROM unknowns WHERE id=?", (uid,))
            self._db.commit()
            shutil.rmtree(self.dir / "unknown" / str(uid), ignore_errors=True)
            self._reload()
            return self.get_person(name)

    def delete_unknown(self, uid: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM unknowns WHERE id=?", (uid,))
            self._db.commit()
            shutil.rmtree(self.dir / "unknown" / str(uid), ignore_errors=True)
            self._reload()

    @staticmethod
    def _save_image(folder: Path, crop: np.ndarray) -> str:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(path)
