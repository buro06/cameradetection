"""Per-camera person tracking, face voting, event lifecycle and alert decisions.

Alerts are about *changes*, not presence: a person alerts when they arrive (or when their identity
becomes known as someone new), never again just for staying in view.

Event lifecycle:
  1. A confirmed person track needs attention: it is new, or its identity changed to one not yet evaluated.
  2. Recording starts `pre_seconds` before the person was first detected (default 0: at detection; the
     ring buffer still supplies the frames between first detection and confirmation).
  3. For `post_seconds` detection + face recognition keep running and faces vote on each track.
  4. The event is finalized, looking only at the people who were new in it: no alert if they are all
     confirmed trusted, otherwise alert unless each of them is the same identity (name or unknown-face
     cluster) that already alerted on this camera within the cooldown (i.e. left and came back).

ByteTrack keeps each person on one track while they move; faces too small, blurry or turned away to
identify reliably are ignored, so they can neither name the wrong person nor spawn extra unknowns.

People who are briefly hidden (occluded on a couch, detection flicker) are re-attached to their old
track instead of counting as new arrivals — but only if they were standing/sitting still when lost.
Someone who was moving when lost is assumed to have left, so a newcomer walking into the same spot or
through the same doorway is never mistaken for them.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np

from .camera import BufferedFrame, CameraStream, Overlay, Recording
from .config import AppConfig
from .detector import LOW_CONFIDENCE
from .faces import FaceDB, FaceEngine, FaceObs

log = logging.getLogger(__name__)

LOCKED_VOTES = 5  # once a track has this many consistent votes, only re-check its face occasionally
RECHECK_SECONDS = 2.0
REVIVE_IOU = 0.5  # how closely a reappearing person must overlap where a lost track was
STATIONARY_WINDOW = 2.0  # seconds of history used to decide whether a lost track was still
STATIONARY_MOVE = 0.15  # max centre movement (fraction of box height) to count as still
OVERLAP_CONTAINMENT = 0.6  # a new box this much inside/around someone already tracked is the same person


@dataclass
class Track:
    id: int
    box: np.ndarray
    first_seen: float
    last_seen: float
    confirmed: bool = False  # counts as a real person (stays true for the life of the track)
    votes: dict[str, list[float]] = field(default_factory=dict)
    unknown_faces: int = 0
    best_unknown: FaceObs | None = None
    last_face_check: float = 0.0
    unknown_id: int | None = None
    seen_keys: set[str] = field(default_factory=set)  # identities already evaluated for this track
    confirmed_trusted: str | None = None
    path: deque = field(default_factory=lambda: deque(maxlen=30))  # (t, cx, cy)

    def resolve(self, trusted_map: dict[str, bool], trusted_min: int) -> tuple[str | None, bool]:
        """(name or None, trusted-and-confirmed). A name wins only with a majority of face observations.
        Once confirmed trusted, a track stays trusted while tracked (faces turned away on a couch fail to
        match often), unless another known person's face outvotes them."""
        sticky = self.confirmed_trusted
        if sticky and trusted_map.get(sticky):
            if all(len(v) <= len(self.votes[sticky]) for k, v in self.votes.items() if k != sticky):
                return sticky, True
        if self.votes:
            name, scores = max(self.votes.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
            others = sum(len(v) for k, v in self.votes.items() if k != name) + self.unknown_faces
            if len(scores) > others and name in trusted_map:
                trusted = trusted_map[name] and len(scores) >= trusted_min
                if trusted:
                    self.confirmed_trusted = name
                return name, trusted
        return None, False

    def key(self, trusted_map: dict[str, bool], trusted_min: int) -> str:
        """Identity used for cooldowns: a name, an unknown-face cluster, or this track itself when no
        face has been seen (so two faceless strangers never share a cooldown)."""
        name, _ = self.resolve(trusted_map, trusted_min)
        if name:
            return f"person:{name}"
        return f"unknown:{self.unknown_id}" if self.unknown_id else f"track:{self.id}"

    def was_still(self) -> bool:
        recent = [(t, x, y) for t, x, y in self.path if t >= self.last_seen - STATIONARY_WINDOW]
        if len(recent) < 2:
            return True
        (_, x0, y0), (_, x1, y1) = recent[0], recent[-1]
        return math.hypot(x1 - x0, y1 - y0) <= STATIONARY_MOVE * (self.box[3] - self.box[1])

    def observe(self, box: np.ndarray, now: float) -> None:
        self.box, self.last_seen = box, now
        self.path.append((now, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


def _containment(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the smaller box that lies inside the other (catches nested and same-size duplicates)."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


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


class _Detections:
    """The part of ultralytics' Boxes interface that BYTETracker reads."""

    def __init__(self, dets: np.ndarray):
        self.dets = dets

    def __len__(self) -> int:
        return len(self.dets)

    def __getitem__(self, mask) -> _Detections:
        return _Detections(self.dets[mask])

    @property
    def conf(self) -> np.ndarray:
        return self.dets[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return np.zeros(len(self.dets), np.float32)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.dets[:, :4].T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


@functools.cache
def _byte_tracker_class():
    from ultralytics.trackers.byte_tracker import BYTETracker

    class CameraByteTracker(BYTETracker):
        @staticmethod
        def reset_id() -> None:
            """ByteTrack's id counter is global: resetting it for a new camera would reuse ids live on others."""

    return CameraByteTracker


class Tracker:
    """ByteTrack assigns detections to people; this class keeps camwatch's per-person state on top of it.

    ByteTrack predicts each person's motion (Kalman filter) and makes a second matching pass with
    low-confidence boxes, so someone walking fast, turning or partly hidden keeps one track instead of being
    split into several "people". It holds a missed person for `ttl` seconds. Still people are remembered for
    `memory` seconds beyond that and revived if someone reappears in the same place (occlusion on a couch,
    leaning out of view)."""

    def __init__(self, ttl: float, memory: float = 30.0, fps: float = 5.0, confidence: float = 0.5):
        self.ttl, self.memory, self.fps = ttl, memory, fps
        self.tracks: dict[int, Track] = {}
        self.lost: dict[int, Track] = {}
        args = SimpleNamespace(track_high_thresh=confidence, track_low_thresh=LOW_CONFIDENCE,
                               new_track_thresh=confidence, track_buffer=self._buffer(), match_thresh=0.8,
                               fuse_score=True)
        self._bt = _byte_tracker_class()(args)
        # ByteTrack reports a new person on their 2nd consecutive detection (filters one-frame false positives),
        # except on its very first frame; starting the count at 1 removes that exception.
        self._bt.frame_id = 1
        self._by_bt: dict[int, Track] = {}  # ByteTrack id -> our track
        self._last_update: float | None = None
        self._next = 1

    def _buffer(self) -> int:
        return max(1, round(self.ttl * self.fps))

    def set_confidence(self, confidence: float) -> None:
        self._bt.args.track_high_thresh = self._bt.args.new_track_thresh = confidence

    def update(self, dets: np.ndarray, now: float) -> list[Track]:
        self._bt.max_frames_lost = self._buffer()  # live edits of ttl / detect_fps
        out = self._bt.update(_Detections(dets))
        seen = []
        for row in out:
            bt_id, box = int(row[4]), dets[int(row[7]), :4].copy()  # the detection, not the smoothed box
            t = self._by_bt.get(bt_id)
            if t is not None and t.id in self.lost:  # ByteTrack re-found someone we had set aside
                self.tracks[t.id] = self.lost.pop(t.id)
            if t is None:
                t = self._revive(box, now)
            if t is None:  # a new person; their first detection was in the previous update
                t = Track(id=self._next, box=box, first_seen=now if self._last_update is None else self._last_update,
                          last_seen=now)
                self._next += 1
            self._by_bt[bt_id] = t
            t.observe(box, now)
            self.tracks[t.id] = t
            seen.append(t)
        # ByteTrack waits for a second detection before reporting a new box, but a still person reappearing in
        # their spot is known already: revive them on the first one, so a one-frame glimpse keeps them remembered.
        for s in self._bt.tracked_stracks:
            if not s.is_activated and s.start_frame == self._bt.frame_id and s.track_id not in self._by_bt:
                box = dets[int(s.idx), :4].copy()
                if t := self._revive(box, now):
                    self._by_bt[s.track_id] = t
                    t.observe(box, now)
                    self.tracks[t.id] = t
                    seen.append(t)
        for tid in [tid for tid, t in self.tracks.items() if now - t.last_seen > self.ttl]:
            t = self.tracks.pop(tid)
            if self.memory > 0 and t.was_still():
                self.lost[tid] = t
        for tid in [tid for tid, t in self.lost.items() if now - t.last_seen > self.memory]:
            del self.lost[tid]
        self._by_bt = {k: t for k, t in self._by_bt.items() if t.id in self.tracks or t.id in self.lost}
        self._last_update = now
        return seen

    def _revive(self, box: np.ndarray, now: float) -> Track | None:
        """A still person who went missing (set aside, or not yet expired) and reappears in the same place."""
        missing = [t for t in self.tracks.values() if t.last_seen < now and t.was_still()]
        best, best_iou = None, REVIVE_IOU
        for t in [*self.lost.values(), *missing]:
            iou = _iou(t.box, box)
            if iou >= best_iou:
                best, best_iou = t, iou
        if best is not None:
            self.lost.pop(best.id, None)
            log.debug("revived track %d (lost %.1fs, IoU %.2f)", best.id, now - best.last_seen, best_iou)
        return best


@dataclass
class PersonInfo:
    track_id: int
    name: str | None
    trusted: bool
    key: str
    unknown_id: int | None
    face: FaceObs | None  # best unknown face (shown for labeling); None for named people
    new: bool = True  # arrived / newly identified in this event (vs. already present)

    @property
    def label(self) -> str:
        if self.name:
            return f"{self.name} (trusted)" if self.trusted else self.name
        if self.unknown_id:
            return f"Unknown #{self.unknown_id}"
        return "Unknown person" if self.face else "Unknown person (no clear face)"


@dataclass
class EventResult:
    camera: str
    wall_time: float
    decision: str  # alert | trusted | cooldown | disarmed | unchanged
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
        self.tracker = Tracker(cfg.events.track_ttl, cfg.events.lost_memory_seconds, cfg.detection.detect_fps,
                               cfg.detection.confidence)
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
        tr = self.tracker  # live edits
        tr.ttl, tr.memory, tr.fps = self.cfg.events.track_ttl, self.cfg.events.lost_memory_seconds, det_cfg.detect_fps
        tr.set_confidence(det_cfg.confidence)
        seen = self.tracker.update(dets, now)
        trusted_map = self.db.trusted_map()
        pending = self._confirm(seen)
        people = [t for t in seen if t.id not in pending]  # duplicate boxes can't claim faces

        check = []
        for t in people:
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
        for t, obs in assign_faces(candidates, people):
            if t not in check or not obs.good:  # small/blurry/turned faces mismatch people and spawn unknowns
                continue
            m = self.db.match(obs.embedding)
            if m:
                t.votes.setdefault(m.name, []).append(m.score)
            else:
                t.unknown_faces += 1
                if t.best_unknown is None or obs.quality > t.best_unknown.quality:
                    t.best_unknown = obs

        confirmed = [t for t in seen if t.confirmed]
        self.stream.set_overlay(Overlay(now, [(t.id, tuple(t.box)) for t in confirmed]))
        self.visible = [self._label(t, trusted_map) for t in confirmed]

        if self.event is None:
            trigger = [t for t in confirmed if self._needs_event(t, trusted_map)]
            if trigger:
                anchor = max(min(t.first_seen for t in trigger), now - 1.0)
                since = anchor - self.cfg.events.pre_seconds
                self.event = _Event(start=since, anchor=anchor, end=anchor + self.cfg.events.post_seconds,
                                    wall=time.time() - (now - anchor), recording=self.stream.start_recording(since))
                log.debug("[%s] event started (tracks %s)", self.name, [t.id for t in trigger])
        if self.event is not None:
            for t in confirmed:
                self.event.tracks[t.id] = t

    def _confirm(self, seen: list[Track]) -> set[int]:
        """Decide which tracks count as real people. Returns ids of tracks held back as duplicates.

        A new person standing apart confirms as soon as ByteTrack reports them (2nd detection), so quick visits
        still alert. A new box mostly on top of (or around) someone already tracked is YOLO boxing the same person
        twice (torso + whole body, person + chair), so it never counts while it overlaps them; if the other person
        leaves and it remains, it becomes a person then."""
        established = [t for t in seen if t.confirmed]
        pending: set[int] = set()
        # bigger boxes first, so when a person arrives with two boxes (whole body + torso) the real one wins
        candidates = sorted((t for t in seen if not t.confirmed),
                            key=lambda t: -(t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        for t in candidates:
            if any(_containment(t.box, o.box) >= OVERLAP_CONTAINMENT for o in established):
                pending.add(t.id)
                continue
            t.confirmed = True
            established.append(t)
        return pending

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

    def _needs_event(self, t: Track, trusted_map: dict[str, bool]) -> bool:
        """New people and identities not yet evaluated need an event; people who stay never re-trigger."""
        return t.key(trusted_map, self.cfg.face.trusted_min_matches) not in t.seen_keys

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
            new = key not in t.seen_keys
            t.seen_keys.add(key)
            people.append(PersonInfo(t.id, name, trusted, key, None if name else t.unknown_id,
                                     None if name else t.best_unknown, new))

        # Only people who are new in this event can cause an alert; the others are listed as context.
        cooldown = self.cfg.events.cooldown_seconds
        new_untrusted = [p for p in people if p.new and not p.trusted]
        if not any(p.new for p in people):
            decision = "unchanged"
        elif not new_untrusted:
            decision = "trusted"
        elif not self.is_armed(self.name):
            decision = "disarmed"
        elif all(now - self.cooldowns.get(p.key, -1e9) < cooldown for p in new_untrusted):
            decision = "cooldown"  # same identities left and came back within the cooldown
        else:
            decision = "alert"
            for p in new_untrusted:
                self.cooldowns[p.key] = now

        result = EventResult(self.name, ev.wall, decision, people, ev.recording.frames, ev.start, ev.end, ev.anchor)
        self.last_result = result
        log.info("[%s] %s: %s", self.name, decision, result.summary())
        return result
