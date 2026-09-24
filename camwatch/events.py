"""Per-camera person tracking, face voting, event lifecycle and alert decisions.

Event lifecycle:
  1. A confirmed person track needs attention (new, identity changed, or cooldown expired).
  2. Recording starts `pre_seconds` before the person was first detected (default 0: at detection; the
     ring buffer still supplies the frames between first detection and confirmation).
  3. For `post_seconds` detection + face recognition keep running and faces vote on each track.
  4. The event is finalized: no alert if every person is a confirmed trusted face, otherwise
     alert unless every non-trusted identity is still inside its per-camera cooldown.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .camera import BufferedFrame, CameraStream, Overlay, Recording
from .config import AppConfig
from .faces import FaceDB, FaceEngine, FaceObs

log = logging.getLogger(__name__)

LOCKED_VOTES = 5  # once a track has this many consistent votes, only re-check its face occasionally
RECHECK_SECONDS = 2.0


@dataclass
class Track:
    id: int
    box: np.ndarray
    first_seen: float
    last_seen: float
    hits: int = 1
    votes: dict[str, list[float]] = field(default_factory=dict)
    unknown_faces: int = 0
    best_unknown: FaceObs | None = None
    best_known: dict[str, FaceObs] = field(default_factory=dict)
    last_face_check: float = 0.0
    unknown_id: int | None = None
    evaluated_key: str | None = None
    evaluated_at: float = 0.0

    def resolve(self, trusted_map: dict[str, bool], trusted_min: int) -> tuple[str | None, bool]:
        """(name or None, trusted-and-confirmed). A name wins only with a majority of face observations."""
        if self.votes:
            name, scores = max(self.votes.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
            others = sum(len(v) for k, v in self.votes.items() if k != name) + self.unknown_faces
            if len(scores) > others and name in trusted_map:
                return name, trusted_map[name] and len(scores) >= trusted_min
        return None, False

    def key(self, trusted_map: dict[str, bool], trusted_min: int) -> str:
        name, _ = self.resolve(trusted_map, trusted_min)
        if name:
            return f"person:{name}"
        return f"unknown:{self.unknown_id}" if self.unknown_id else "unknown"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


AMBIGUITY_MARGIN = 0.08


def _head_cost(face: tuple[int, int, int, int], box: np.ndarray) -> float | None:
    """How well a face sits at the head position of a person box (lower is better; None = impossible)."""
    fcx, fcy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw <= 0 or bh <= 0 or not (box[0] <= fcx <= box[2]) or not (box[1] - 0.1 * bh <= fcy <= box[1] + 0.6 * bh):
        return None
    return abs(fcx - (box[0] + box[2]) / 2) / bw + abs(face[1] - box[1]) / bh


def assign_faces(faces: list[FaceObs], tracks: list[Track]) -> list[tuple[Track, FaceObs]]:
    """Give each face to at most one person. Person boxes overlap (arms, crowds), so a face goes to the
    box whose head position it fits best; faces that fit two people similarly well are dropped rather than
    risk a trusted face vouching for someone else."""
    best_for_track: dict[int, tuple[float, Track, FaceObs]] = {}
    for f in faces:
        costs = sorted((c, t.id, t) for t in tracks if (c := _head_cost(f.box, t.box)) is not None)
        if not costs or (len(costs) > 1 and costs[1][0] - costs[0][0] < AMBIGUITY_MARGIN):
            continue
        c, tid, t = costs[0]
        if tid not in best_for_track or c < best_for_track[tid][0]:
            best_for_track[tid] = (c, t, f)
    return [(t, f) for _, t, f in best_for_track.values()]


class Tracker:
    """Greedy IoU tracker; good enough at ~5 inference FPS for people walking."""

    def __init__(self, ttl: float):
        self.ttl = ttl
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, dets: np.ndarray, now: float) -> list[Track]:
        pairs = []
        for tid, t in self.tracks.items():
            for di, d in enumerate(dets):
                iou = _iou(t.box, d[:4])
                if iou < 0.2:  # fall back to centre distance for fast movers
                    diag = np.hypot(t.box[2] - t.box[0], t.box[3] - t.box[1])
                    dist = np.hypot((t.box[0] + t.box[2] - d[0] - d[2]) / 2, (t.box[1] + t.box[3] - d[1] - d[3]) / 2)
                    iou = 0.1 if dist < 0.5 * diag else 0.0
                if iou > 0:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True)
        used_t, used_d, seen = set(), set(), []
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            t = self.tracks[tid]
            t.box, t.last_seen, t.hits = dets[di][:4].copy(), now, t.hits + 1
            seen.append(t)
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(id=self._next, box=d[:4].copy(), first_seen=now, last_seen=now)
                self._next += 1
                self.tracks[t.id] = t
                seen.append(t)
        for tid in [tid for tid, t in self.tracks.items() if now - t.last_seen > self.ttl]:
            del self.tracks[tid]
        return seen


@dataclass
class PersonInfo:
    track_id: int
    name: str | None
    trusted: bool
    key: str
    unknown_id: int | None
    face: FaceObs | None

    @property
    def label(self) -> str:
        if self.name:
            return f"{self.name} (trusted)" if self.trusted else self.name
        if self.unknown_id:
            return f"Unknown #{self.unknown_id}"
        return "Unknown person" if self.face else "Unknown person (face not visible)"


@dataclass
class EventResult:
    camera: str
    wall_time: float
    decision: str  # alert | trusted | cooldown | disarmed
    people: list[PersonInfo]
    frames: list[BufferedFrame]
    start: float
    end: float
    anchor: float  # monotonic time the person appeared (== wall_time)

    @property
    def labels(self) -> dict[int, str]:
        return {p.track_id: p.label for p in self.people}

    def summary(self) -> str:
        return ", ".join(p.label for p in self.people) or "nobody"


@dataclass
class _Event:
    start: float
    end: float
    anchor: float
    wall: float
    recording: Recording
    tracks: dict[int, Track] = field(default_factory=dict)


class CameraMonitor:
    def __init__(self, cfg: AppConfig, stream: CameraStream, faces: FaceEngine, db: FaceDB,
                 is_armed: Callable[[str], bool]):
        self.cfg = cfg
        self.stream = stream
        self.faces = faces
        self.db = db
        self.is_armed = is_armed
        self.tracker = Tracker(cfg.events.track_ttl)
        self.event: _Event | None = None
        self.cooldowns: dict[str, float] = {}
        self.visible: list[str] = []
        self.last_result: EventResult | None = None
        self.infer_fps = 0.0
        self._last_infer = 0.0

    @property
    def name(self) -> str:
        return self.stream.cfg.name

    def process(self, frame: np.ndarray, dets: np.ndarray, now: float) -> None:
        """Called by the detection worker for every inference on this camera."""
        if self._last_infer:
            dt = now - self._last_infer
            self.infer_fps = 0.8 * self.infer_fps + 0.2 / dt if self.infer_fps else 1.0 / dt
        self._last_infer = now

        det_cfg, face_cfg = self.cfg.detection, self.cfg.face
        if det_cfg.min_box_height > 0 and len(dets):
            dets = dets[(dets[:, 3] - dets[:, 1]) >= det_cfg.min_box_height * frame.shape[0]]
        seen = self.tracker.update(dets, now)
        trusted_map = self.db.trusted_map()

        check = []
        for t in seen:
            name, _ = t.resolve(trusted_map, face_cfg.trusted_min_matches)
            locked = name is not None and len(t.votes[name]) >= LOCKED_VOTES
            if not locked or now - t.last_face_check >= RECHECK_SECONDS:
                t.last_face_check = now
                check.append(t)
        candidates: list[FaceObs] = []
        for t in check:
            for obs in self.faces.analyze(frame, tuple(t.box), max_faces=3):
                if all(_iou(np.array(obs.box, float), np.array(c.box, float)) < 0.5 for c in candidates):
                    candidates.append(obs)
        for t, obs in assign_faces(candidates, seen):
            if t not in check:
                continue
            m = self.db.match(obs.embedding)
            if m:
                t.votes.setdefault(m.name, []).append(m.score)
                best = t.best_known.get(m.name)
                if best is None or obs.quality > best.quality:
                    t.best_known[m.name] = obs
            else:
                t.unknown_faces += 1
                if t.best_unknown is None or obs.quality > t.best_unknown.quality:
                    t.best_unknown = obs

        confirmed = [t for t in seen if t.hits >= det_cfg.min_hits]
        self.stream.set_overlay(Overlay(now, [(t.id, tuple(t.box)) for t in confirmed]))
        self.visible = [self._label(t, trusted_map) for t in confirmed]

        if self.event is None:
            trigger = [t for t in confirmed if self._needs_event(t, trusted_map, now)]
            if trigger:
                anchor = max(min(t.first_seen for t in trigger), now - 1.0)
                since = anchor - self.cfg.events.pre_seconds
                self.event = _Event(start=since, anchor=anchor, end=anchor + self.cfg.events.post_seconds,
                                    wall=time.time() - (now - anchor), recording=self.stream.start_recording(since))
                log.debug("[%s] event started (tracks %s)", self.name, [t.id for t in trigger])
        if self.event is not None:
            for t in confirmed:
                self.event.tracks[t.id] = t

    def tick(self, now: float) -> EventResult | None:
        """Finish the active event once its clip window has been recorded."""
        if self.event is None or now < self.event.end:
            return None
        ev, self.event = self.event, None
        self.stream.stop_recording(ev.recording)
        return self._finalize(ev, now)

    def _label(self, t: Track, trusted_map: dict[str, bool]) -> str:
        name, trusted = t.resolve(trusted_map, self.cfg.face.trusted_min_matches)
        return f"{name}{' ✓' if trusted else ''}" if name else "unknown"

    def _needs_event(self, t: Track, trusted_map: dict[str, bool], now: float) -> bool:
        key = t.key(trusted_map, self.cfg.face.trusted_min_matches)
        if key != t.evaluated_key:
            return True
        _, trusted = t.resolve(trusted_map, self.cfg.face.trusted_min_matches)
        if trusted:
            return False
        last = max(t.evaluated_at, self.cooldowns.get(key, 0.0))
        return now - last >= self.cfg.events.cooldown_seconds

    def _finalize(self, ev: _Event, now: float) -> EventResult:
        face_cfg = self.cfg.face
        trusted_map = self.db.trusted_map()
        people = []
        for t in ev.tracks.values():
            name, trusted = t.resolve(trusted_map, face_cfg.trusted_min_matches)
            if name is None and t.best_unknown is not None and face_cfg.save_unknowns:
                b = t.best_unknown
                if t.unknown_id is None or not self.db.unknown_exists(t.unknown_id):
                    t.unknown_id = self.db.record_unknown(b.embedding, b.crop, b.quality, self.name, face_cfg.max_unknown_images)
            key = t.key(trusted_map, face_cfg.trusted_min_matches)
            face = t.best_known.get(name) if name else t.best_unknown
            people.append(PersonInfo(t.id, name, trusted, key, None if name else t.unknown_id, face))
            t.evaluated_key, t.evaluated_at = key, now

        cooldown = self.cfg.events.cooldown_seconds
        untrusted = [p for p in people if not p.trusted]
        if not untrusted:
            decision = "trusted"
        elif not self.is_armed(self.name):
            decision = "disarmed"
        elif all(now - self.cooldowns.get(p.key, -1e9) < cooldown for p in untrusted):
            decision = "cooldown"
        else:
            decision = "alert"
            for p in untrusted:
                self.cooldowns[p.key] = now

        result = EventResult(self.name, ev.wall, decision, people, ev.recording.frames, ev.start, ev.end, ev.anchor)
        self.last_result = result
        log.info("[%s] %s: %s", self.name, decision, result.summary())
        return result
