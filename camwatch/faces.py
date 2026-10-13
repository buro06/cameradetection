"""Face detection (SCRFD), recognition (ArcFace) and the face database (SQLite).

Both models come from InsightFace's `buffalo_l` pack (free for non-commercial use, e.g. at home) and run
through onnxruntime: on the GPU when onnxruntime can use it, otherwise on the CPU.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import FaceConfig

log = logging.getLogger(__name__)

MODEL_PACKS = {  # pack -> (zip URL, detector file, recognizer file)
    "buffalo_l": ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
                  "det_10g.onnx", "w600k_r50.onnx"),
}
DET_SIZES = (320, 480, 640)  # SCRFD input side: the smallest that fits the image, up to face.detect_size
DET_NMS = 0.4
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)  # 5 points in a 112x112 chip
EMBEDDING_DIM = 512
MAX_EMBEDDINGS_PER_PERSON = 60
MAX_EMBEDDINGS_PER_UNKNOWN = 20
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REBUILD_DETECTOR_SCORE = 0.4  # photos in a person's folder were chosen by a human, so detect more leniently
REBUILD_SECOND_FACE_RATIO = 0.6  # a second face at least this wide (vs the main one) makes a photo ambiguous


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread that also works for non-ASCII paths on Windows."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ensure_models(models_dir: Path, pack: str = "buffalo_l") -> dict[str, Path]:
    """Download the model pack once and keep only the two ONNX files camwatch uses."""
    if pack not in MODEL_PACKS:
        raise ValueError(f"Unknown face model {pack!r} (available: {', '.join(MODEL_PACKS)})")
    url, det, rec = MODEL_PACKS[pack]
    folder = models_dir / pack
    paths = {"det": folder / det, "rec": folder / rec}
    if all(p.exists() and p.stat().st_size > 1_000_000 for p in paths.values()):
        return paths
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{pack}.zip.part"
    log.info("Downloading face models %s (~280 MB, once)", pack)
    urllib.request.urlretrieve(url, tmp)
    try:
        with zipfile.ZipFile(tmp) as z:
            for key, path in paths.items():
                member = next((n for n in z.namelist() if n.rsplit("/", 1)[-1] == path.name), None)
                if member is None:
                    raise RuntimeError(f"{path.name} missing from {url}")
                part = path.with_suffix(".part")
                with z.open(member) as src, open(part, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                part.replace(path)
    except zipfile.BadZipFile:
        raise RuntimeError(f"Download of {pack} failed (not a zip). Get it manually from {url} "
                           f"and put {det} and {rec} into {folder}") from None
    finally:
        tmp.unlink(missing_ok=True)
    return paths


@dataclass
class FaceObs:
    box: tuple[int, int, int, int]  # xyxy in frame coords
    score: float
    quality: float
    embedding: np.ndarray  # (512,) L2-normalised
    crop: np.ndarray  # face with context, for humans to look at
    issue: str = ""  # why the face is too poor to identify from ("" = good)

    @property
    def good(self) -> bool:
        return not self.issue


@dataclass
class Match:
    person_id: int
    name: str
    trusted: bool
    score: float


def onnxruntime_cuda_hint() -> str:
    """How to fix onnxruntime-gpu being built for another CUDA major version than torch, or "" if they match.
    onnxruntime-gpu 1.27+ is CUDA 13 only, which drops Maxwell/Pascal/Volta GPUs (e.g. the Quadro M4000)."""
    try:
        import torch
        from onnxruntime.capi import build_and_package_info as ort_build
    except ImportError:
        return ""
    ort_cuda, torch_cuda = getattr(ort_build, "cuda_version", None), torch.version.cuda
    if not ort_cuda or not torch_cuda or ort_cuda.split(".")[0] == torch_cuda.split(".")[0]:
        return ""
    fix = 'pip install "onnxruntime-gpu<1.27"' if torch_cuda.startswith("12.") else "install the onnxruntime-gpu build for it"
    return (f"onnxruntime-gpu {getattr(ort_build, '__version__', '?')} is built for CUDA {ort_cuda.split('.')[0]} "
            f"but torch uses CUDA {torch_cuda.split('.')[0]}: {fix}")


def _sessions(paths: dict[str, Path], device: str):
    """onnxruntime sessions for detector and recognizer; GPU only if it actually runs (old GPUs may lack kernels)."""
    if device in ("auto", "cuda"):
        try:
            import torch  # loads CUDA/cuDNN DLLs that onnxruntime can reuse
            if device == "auto" and not torch.cuda.is_available():
                # onnxruntime-gpu lists CUDA even without a GPU and prints missing-library errors when tried
                device = "cpu"
        except ImportError:
            pass
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # SCRFD's ONNX declares 640x640 shapes; other input sizes work but warn every call
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            det = ort.InferenceSession(str(paths["det"]), opts, providers=providers)
            rec = ort.InferenceSession(str(paths["rec"]), opts, providers=providers)
            if det.get_providers()[0] == "CUDAExecutionProvider":
                det.run(None, {det.get_inputs()[0].name: np.zeros((1, 3, 320, 320), np.float32)})
                rec.run(None, {rec.get_inputs()[0].name: np.zeros((1, 3, 112, 112), np.float32)})
                return det, rec, "cuda"
            log.warning("onnxruntime could not start CUDA for face models; using the CPU. %s", onnxruntime_cuda_hint())
        except Exception as e:
            log.warning("Face models can't run on the GPU (%s); using the CPU. %s", str(e).splitlines()[0][:200],
                        onnxruntime_cuda_hint())
    elif device == "cuda":
        log.warning("onnxruntime has no CUDA support installed (pip install onnxruntime-gpu); using the CPU")
    cpu = ["CPUExecutionProvider"]
    return (ort.InferenceSession(str(paths["det"]), opts, providers=cpu),
            ort.InferenceSession(str(paths["rec"]), opts, providers=cpu), "cpu")


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 least-squares similarity transform (Umeyama) mapping src points onto dst, as InsightFace aligns."""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / n
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    r = u @ np.diag(d) @ vt
    scale = (sig * d).sum() / s_c.var(0).sum()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_face(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """112x112 face chip with eyes, nose and mouth corners at ArcFace's template positions."""
    m = _similarity_transform(kps.astype(np.float64), ARCFACE_TEMPLATE.astype(np.float64))
    return cv2.warpAffine(img, m, (112, 112), borderValue=0.0)


class FaceEngine:
    def __init__(self, models_dir: Path, cfg: FaceConfig):
        self.cfg = cfg
        self.model_id = cfg.model
        paths = ensure_models(models_dir, cfg.model)
        self._det, self._rec, self.device = _sessions(paths, cfg.device)
        self._det_in = self._det.get_inputs()[0].name
        self._rec_in = self._rec.get_inputs()[0].name
        self._anchors: dict[tuple[int, int, int], np.ndarray] = {}
        self._lock = threading.Lock()
        log.info("Face models %s on %s", cfg.model, self.device)

    # ---- SCRFD -------------------------------------------------------------------
    def _centres(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._anchors:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            self._anchors[key] = np.repeat(c, 2, axis=0)  # 2 anchors per position
        return self._anchors[key]

    def _detect(self, img: np.ndarray, min_score: float, det_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Faces in `img`: (N, 5) boxes x1, y1, x2, y2, score and (N, 5, 2) landmarks, in image coords.
        The detector input is at most `det_size` (default `face.detect_size`) pixels on its long side."""
        ih, iw = img.shape[:2]
        largest = max(32, (det_size or self.cfg.detect_size) // 32 * 32)
        side = next((s for s in (*(d for d in DET_SIZES if d < largest), largest) if max(ih, iw) <= s), largest)
        scale = side / max(ih, iw)
        nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
        canvas = np.zeros((side, side, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh)) if (nw, nh) != (iw, ih) else img
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (side, side), (127.5, 127.5, 127.5), swapRB=True)
        outs = [o[0] if o.ndim == 3 else o for o in self._det.run(None, {self._det_in: blob})]
        boxes, kpss = [], []
        for i, stride in enumerate((8, 16, 32)):
            scores, dist, kd = outs[i].reshape(-1), outs[i + 3] * stride, outs[i + 6] * stride
            centres = self._centres(side // stride, side // stride, stride)
            keep = np.flatnonzero(scores >= min_score)
            if not len(keep):
                continue
            c, d = centres[keep], dist[keep]
            boxes.append(np.column_stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2],
                                          c[:, 1] + d[:, 3], scores[keep]]))
            kpss.append(c[:, None, :] + kd[keep].reshape(-1, 5, 2))
        if not boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes, kpss = np.vstack(boxes), np.vstack(kpss)
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        idx = np.array(cv2.dnn.NMSBoxes(xywh, boxes[:, 4].tolist(), min_score, DET_NMS), int).reshape(-1)
        boxes, kpss = boxes[idx], kpss[idx]
        boxes[:, :4] /= scale
        return boxes.astype(np.float32), (kpss / scale).astype(np.float32)

    # ---- ArcFace -----------------------------------------------------------------
    def _embed(self, chip: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(chip, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self._rec.run(None, {self._rec_in: blob})[0].reshape(-1).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def analyze(self, frame: np.ndarray, region: tuple[float, float, float, float] | None = None,
                max_faces: int = 1, min_score: float | None = None, det_size: int | None = None) -> list[FaceObs]:
        """Find faces in `frame` (optionally only inside region xyxy) and embed them, best first.
        `det_size` caps the face detector's input (smaller is faster; see `_detect`)."""
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
        min_score = self.cfg.detector_score if min_score is None else min_score

        with self._lock:
            boxes, kpss = self._detect(crop, min_score, det_size)
            widths = boxes[:, 2] - boxes[:, 0]
            order = [i for i in np.argsort(-boxes[:, 4] * widths) if widths[i] >= self.cfg.min_face_px][:max_faces]
            out = []
            for i in order:
                b, kps = boxes[i], kpss[i]
                chip = align_face(crop, kps)
                emb = self._embed(chip)
                sharp, turn = sharpness(chip), head_turn(kps)
                box = (int(x0 + b[0]), int(y0 + b[1]), int(x0 + b[2]), int(y0 + b[3]))
                quality = float(b[4]) * min(1.0, widths[i] / 112.0) * min(1.0, sharp / 100.0)
                out.append(FaceObs(box=box, score=float(b[4]), quality=quality, embedding=emb,
                                   crop=_context_crop(frame, box), issue=face_issue(self.cfg, sharp, turn)))
        return out


def face_issue(cfg: FaceConfig, sharp: float, turn: float) -> str:
    """Why a face is too poor to identify someone from, or "" if it is good enough (small faces are never
    detected in the first place: `min_face_px`)."""
    if sharp < cfg.quality_min_sharpness:
        return "blurry"
    if turn > cfg.quality_max_turn:
        return "turned away"
    return ""


def sharpness(chip: np.ndarray) -> float:
    """Variance of the Laplacian of the aligned 112x112 face: high for crisp detail, low for blur."""
    return float(cv2.Laplacian(cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def head_turn(kps: np.ndarray) -> float:
    """How far the nose sits off the centre line between the eyes, in eye distances (0 = facing the camera)."""
    eye_a, eye_b, nose = kps[0], kps[1], kps[2]
    axis = eye_b - eye_a
    dist = float(np.linalg.norm(axis))
    if dist < 1:
        return 1.0
    return abs(float(np.dot(nose - (eye_a + eye_b) / 2, axis / dist))) / dist


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


@dataclass
class RebuildPlan:
    """Result of scanning data/faces/<person id>/ — applied with FaceDB.apply_rebuild()."""
    samples: dict[int, list[tuple[np.ndarray, str, float]]]  # person id -> (embedding, image path, file mtime)
    before: dict[int, int]  # person id -> sample count before the rebuild
    no_face: list[Path] = field(default_factory=list)
    multi_face: list[Path] = field(default_factory=list)
    unreadable: list[Path] = field(default_factory=list)
    stray: list[Path] = field(default_factory=list)  # files/folders not belonging to a known person
    last_embedding_id: int = 0  # detects samples added while scanning


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
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
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
            self._known = np.stack([_vec(r[1]) for r in rows]) if rows else np.zeros((0, EMBEDDING_DIM), np.float32)
            rows = self._db.execute("SELECT unknown_id, vec FROM unknown_embeddings").fetchall()
            self._unk_ids = np.array([r[0] for r in rows], dtype=np.int64)
            self._unk = np.stack([_vec(r[1]) for r in rows]) if rows else np.zeros((0, EMBEDDING_DIM), np.float32)

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
    def match(self, emb: np.ndarray, min_score: float | None = None) -> Match | None:
        """Closest known person scoring at least `min_score` (default: the match threshold)."""
        with self._lock:
            hit = self._best(self._known, self._known_ids, emb)
            if hit is None or hit[1] < (self.threshold if min_score is None else min_score):
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

    # ---- rebuild from photos -------------------------------------------------
    def plan_rebuild(self, faces: FaceEngine, progress: Callable[[int, int], None] | None = None) -> RebuildPlan:
        """Re-embed every photo in faces/<person id>/. Slow, so it runs without holding the DB lock."""
        with self._lock:
            before = {p["id"]: p["samples"] for p in self.list_persons()}
            last_id = self._db.execute("SELECT COALESCE(MAX(id), 0) FROM embeddings").fetchone()[0]
        plan = RebuildPlan(samples={pid: [] for pid in before}, before=before, last_embedding_id=last_id)
        todo: list[tuple[int, Path]] = []
        for entry in sorted((self.dir / "faces").iterdir()):
            if entry.is_dir() and entry.name.isdigit() and int(entry.name) in before:
                todo += [(int(entry.name), p) for p in image_files(entry)]
            elif entry.is_dir() or entry.suffix.lower() in IMAGE_EXTS:
                plan.stray.append(entry)
        min_score = min(faces.cfg.detector_score, REBUILD_DETECTOR_SCORE)
        for i, (pid, path) in enumerate(todo):
            if progress:
                progress(i, len(todo))
            img = read_image(path)
            if img is None:
                plan.unreadable.append(path)
                continue
            found = faces.analyze(img, None, max_faces=2, min_score=min_score)
            if not found:
                plan.no_face.append(path)
                continue
            if len(found) > 1:
                w0, w1 = (f.box[2] - f.box[0] for f in found[:2])
                if w1 >= REBUILD_SECOND_FACE_RATIO * w0:
                    plan.multi_face.append(path)
                    continue
            plan.samples[pid].append((found[0].embedding, str(path), path.stat().st_mtime))
        return plan

    def apply_rebuild(self, plan: RebuildPlan) -> None:
        """Replace all known-person embeddings with the ones from `plan`. Photos are never deleted."""
        with self._lock:
            last_id = self._db.execute("SELECT COALESCE(MAX(id), 0) FROM embeddings").fetchone()[0]
            if last_id != plan.last_embedding_id:
                raise RuntimeError("Face samples were added while scanning — run the rebuild again.")
            existing = set(self._persons)
            self._db.execute("DELETE FROM embeddings")
            self._db.executemany(
                "INSERT INTO embeddings (person_id, vec, image, created) VALUES (?,?,?,?)",
                [(pid, emb.astype(np.float32).tobytes(), image, mtime)
                 for pid, rows in plan.samples.items() if pid in existing for emb, image, mtime in rows])
            self._db.commit()
            self._reload()

    # ---- model changes ----------------------------------------------------------
    def embedding_model(self) -> str | None:
        """Face model the stored embeddings came from (None = empty DB). Before this was recorded: SFace."""
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
            if row:
                return row[0]
            used = self._db.execute("SELECT 1 FROM embeddings UNION ALL SELECT 1 FROM unknown_embeddings LIMIT 1").fetchone()
            return "sface" if used else None

    def set_embedding_model(self, model: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_model', ?)", (model,))
            self._db.commit()

    def migrate_embeddings(self, faces: FaceEngine) -> dict | None:
        """Embeddings from another face model can't be compared with new ones: re-embed every stored photo of
        known people and unknown clusters with `faces`. Returns a summary, or None if nothing had to change."""
        old = self.embedding_model()
        if old == faces.model_id:
            return None
        if old is None:
            self.set_embedding_model(faces.model_id)
            return None
        log.warning("Face model changed (%s -> %s): re-embedding stored face photos …", old, faces.model_id)
        plan = self.plan_rebuild(faces)
        self.apply_rebuild(plan)
        empty = [p["name"] for p in self.list_persons() if p["samples"] == 0]
        kept = dropped = 0
        with self._lock:
            min_score = min(faces.cfg.detector_score, REBUILD_DETECTOR_SCORE)
            for eid, image in self._db.execute("SELECT id, image FROM unknown_embeddings").fetchall():
                img = read_image(Path(image)) if image else None
                found = faces.analyze(img, None, max_faces=1, min_score=min_score) if img is not None else []
                if found:
                    self._db.execute("UPDATE unknown_embeddings SET vec=? WHERE id=?",
                                     (found[0].embedding.astype(np.float32).tobytes(), eid))
                    kept += 1
                else:
                    self._db.execute("DELETE FROM unknown_embeddings WHERE id=?", (eid,))
                    dropped += 1
            gone = [r[0] for r in self._db.execute(
                "SELECT id FROM unknowns WHERE id NOT IN (SELECT unknown_id FROM unknown_embeddings)").fetchall()]
            for uid in gone:
                self._db.execute("DELETE FROM unknowns WHERE id=?", (uid,))
                shutil.rmtree(self.dir / "unknown" / str(uid), ignore_errors=True)
            self._db.commit()
        self.set_embedding_model(faces.model_id)
        self._reload()
        people = {p["name"]: p["samples"] for p in self.list_persons()}
        log.warning("Face database re-embedded: %s; unknown faces kept %d, dropped %d (%d clusters removed)",
                    ", ".join(f"{n} {c}" for n, c in people.items()) or "no people", kept, dropped, len(gone))
        if empty:
            log.warning("No usable photos left for: %s. Re-enroll them (live camera or Telegram).", ", ".join(empty))
        return {"people": people, "empty": empty, "unknowns_kept": kept, "unknowns_dropped": dropped,
                "clusters_removed": len(gone)}

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
