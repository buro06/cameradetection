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

ByteTrack keeps each person on one track while they move (Kalman motion prediction plus a second matching
pass with low-confidence boxes). On top of that:
  * People who are briefly hidden while still (occluded on a couch, detection flicker) are re-attached to
    their old track instead of counting as new arrivals.
  * Someone lost while *moving* who is re-found within `handoff_seconds` a plausible step further along is
    handed back their old track ("handoff"), so walking across the room never makes a second person.
  * A walker re-found after missed frames (by ByteTrack or a handoff) is on probation until their face
    matches again: two clear faces that don't match split the track off as a new person, so a stranger is
    never covered by the trusted person they were confused with.
  * Someone who was moving when lost and is not re-found that quickly, or who left through the side of the
    frame, is assumed to have left; a newcomer in the same spot is never mistaken for them.
Blurry or turned-away faces don't vote on who someone is.
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
HEAD_REGION = 0.6  # faces are searched in this top share of a person box (see _head_cost)
REVIVE_IOU = 0.5  # how closely a reappearing person must overlap where a lost track was
STATIONARY_WINDOW = 2.0  # seconds of history used to decide whether a lost track was still
STATIONARY_MOVE = 0.15  # max centre movement (fraction of box height) to count as still
OVERLAP_CONTAINMENT = 0.6  # box this much inside someone already tracked = possible duplicate detection
OVERLAP_PRESENCE = 0.6  # ...which must then be detected in this share of frames during the hold time
HANDOFF_EDGE = 0.03  # a moving person last seen this close (fraction of frame width) to the side walked out
HANDOFF_REACH = 1.5  # how far past their recent speed x gap a lost walker may be re-found
PROBATION_MISMATCHES = 2  # clear non-matching faces that split a handed-off track into a new person


@dataclass
class Track:
    id: int
    box: np.ndarray
    first_seen: float
    last_seen: float
    hits: int = 0  # detections so far
    confirmed: bool = False  # counts as a real person (stays true for the life of the track)
    votes: dict[str, list[float]] = field(default_factory=dict)
    unknown_faces: int = 0
    best_unknown: FaceObs | None = None
    best_known: dict[str, FaceObs] = field(default_factory=dict)
    last_face_check: float = float("-inf")
    unknown_id: int | None = None
    seen_keys: set[str] = field(default_factory=set)  # identities already evaluated for this track
    confirmed_trusted: str | None = None
    path: deque = field(default_factory=lambda: deque(maxlen=30))  # (t, cx, cy)
    poor_faces: int = 0  # faces seen but too blurry / turned away to identify from
    probation_since: float | None = None  # set while on probation after being re-found mid-walk
    mismatches: int = 0  # clear faces not matching this track's identity since then

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

    def speed(self) -> float:
        """Recent speed of the box centre in px/s."""
        recent = [(t, x, y) for t, x, y in self.path if t >= self.last_seen - 1.0]
        if len(recent) < 2 or recent[-1][0] <= recent[0][0]:
            return 0.0
        (t0, x0, y0), (t1, x1, y1) = recent[0], recent[-1]
        return math.hypot(x1 - x0, y1 - y0) / (t1 - t0)

    def observe(self, box: np.ndarray, now: float) -> None:
        self.box, self.last_seen = box, now
        self.hits += 1
        self.path.append((now, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


def _centre(box: np.ndarray) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


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

    ByteTrack holds a missed person for `ttl` seconds. Still people are remembered for `memory` seconds
    beyond that and revived if someone reappears in the same place; moving people re-found within
    `handoff` seconds nearby keep their track (see the module docstring)."""

    def __init__(self, ttl: float, memory: float = 30.0, fps: float = 5.0, confidence: float = 0.5,
                 handoff: float = 1.5):
        self.ttl, self.memory, self.fps, self.handoff = ttl, memory, fps, handoff
        self.tracks: dict[int, Track] = {}
        self.lost: dict[int, Track] = {}
        self.frame_width: int | None = None
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

    def _new_track(self, box: np.ndarray, first_seen: float, now: float) -> Track:
        t = Track(id=self._next, box=box, first_seen=first_seen, last_seen=now)
        self._next += 1
        return t

    def update(self, dets: np.ndarray, now: float) -> list[Track]:
        self._bt.max_frames_lost = self._buffer()  # live edits of ttl / detect_fps
        out = self._bt.update(_Detections(dets))
        # tracks ByteTrack is still following this frame can't be handed to anyone else
        claimed = {self._by_bt[int(r[4])].id for r in out if int(r[4]) in self._by_bt}
        seen: list[Track] = []
        for row in out:
            bt_id, box = int(row[4]), dets[int(row[7]), :4].copy()  # the detection, not the smoothed box
            t = self._by_bt.get(bt_id)
            if t is not None and t.id in self.lost:  # ByteTrack re-found someone we had set aside
                self.tracks[t.id] = self.lost.pop(t.id)
            if t is not None and now - t.last_seen > 1.5 / max(self.fps, 0.1) and not t.was_still():
                if self._at_side(t.box):  # they walked out through the side; this is someone coming in
                    del self._by_bt[bt_id]
                    t = None
                else:  # re-found mid-walk after missed frames: maybe someone else
                    t.probation_since, t.mismatches = now, 0
            if t is not None and any(o is t for o in seen):  # two ByteTrack ids for one person: keep the first
                continue
            if t is None:
                t = self._revive(box, now)
            if t is None:
                t = self._take_handoff(box, now, claimed)
            if t is None:  # a new person; their first detection was in the previous update
                t = self._new_track(box, now if self._last_update is None else self._last_update, now)
                t.hits = 1
            claimed.add(t.id)
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
            self._forget_bt(best)
            log.debug("revived track %d (lost %.1fs, IoU %.2f)", best.id, now - best.last_seen, best_iou)
        return best

    def _at_side(self, box: np.ndarray) -> bool:
        if not self.frame_width:
            return False
        margin = HANDOFF_EDGE * self.frame_width
        return box[0] <= margin or box[2] >= self.frame_width - margin

    def _take_handoff(self, box: np.ndarray, now: float, claimed: set[int]) -> Track | None:
        """A walker ByteTrack lost a moment ago, re-found a plausible step further along. Only when exactly one
        such person fits and nobody visible right now is closer to the new box."""
        window = min(self.handoff, self.ttl)
        if window <= 0:
            return None
        cx, cy = _centre(box)
        fits = []
        for t in self.tracks.values():
            gap = now - t.last_seen
            if t.id in claimed or not 0 < gap <= window or t.was_still() or self._at_side(t.box):
                continue
            tx, ty = _centre(t.box)
            dist = math.hypot(cx - tx, cy - ty)
            if dist <= max(t.box[3] - t.box[1], t.speed() * gap * HANDOFF_REACH):
                fits.append((dist, t))
        if len(fits) != 1:
            return None
        dist, t = fits[0]
        for o in self.tracks.values():
            if o is not t and o.last_seen == now and math.hypot(cx - _centre(o.box)[0], cy - _centre(o.box)[1]) < dist:
                return None
        self._forget_bt(t)
        t.probation_since, t.mismatches = now, 0
        log.debug("handoff to track %d (gap %.1fs, %.0f px)", t.id, now - t.last_seen, dist)
        return t

    def _forget_bt(self, t: Track) -> None:
        """Drop ByteTrack's old ids for `t`, so a late re-find of the old id can't make a second copy of them."""
        old = {k for k, v in self._by_bt.items() if v is t}
        if old:
            self._bt.lost_stracks = [s for s in self._bt.lost_stracks if s.track_id not in old]
            self._bt.tracked_stracks = [s for s in self._bt.tracked_stracks if s.track_id not in old]
            for k in old:
                del self._by_bt[k]

    def split(self, t: Track, now: float) -> Track:
        """`t` was handed off to the wrong person: give the newcomer (now under `t`'s ByteTrack id) a fresh
        track, and let `t` go as someone who left."""
        new = self._new_track(t.box.copy(), t.probation_since or now, now)
        new.hits, new.confirmed = t.hits, t.confirmed
        new.path.extend(p for p in t.path if p[0] >= new.first_seen)
        for k, v in list(self._by_bt.items()):
            if v is t:
                self._by_bt[k] = new
        self.tracks.pop(t.id, None)
        self.tracks[new.id] = new
        t.probation_since = None
        log.debug("track %d split off from %d (face mismatch after handoff)", new.id, t.id)
        return new


@dataclass
class PersonInfo:
    track_id: int
    name: str | None
    trusted: bool
    key: str
    unknown_id: int | None
    face: FaceObs | None
    new: bool = True  # arrived / newly identified in this event (vs. already present)
    score: float | None = None  # mean face-match similarity for `name` on this track
    poor_face: bool = False  # faces were seen, but none clear enough to identify
    track_ids: tuple[int, ...] = ()  # every track merged into this person (same name / unknown cluster)

    @property
    def label(self) -> str:
        if self.name:
            return f"{self.name} (trusted)" if self.trusted else self.name
        if self.unknown_id:
            return f"Unknown #{self.unknown_id}"
        if self.face:
            return "Unknown person"
        return "Unknown person (no clear face)" if self.poor_face else "Unknown person (face not visible)"

    @property
    def scored_label(self) -> str:
        """Label with the match score as a percent, for clips and alerts."""
        return f"{self.label} {self.score:.0%}" if self.name and self.score is not None else self.label


@dataclass
class EventResult:
    camera: str
    wall_time: float
    decision: str  # alert | rate-limited | trusted | cooldown | disarmed | unchanged
    people: list[PersonInfo]
    frames: list[BufferedFrame]
    start: float
    end: float
    anchor: float  # monotonic time the person appeared (== wall_time)

    @property
    def labels(self) -> dict[int, str]:
        return {tid: p.scored_label for p in self.people for tid in (p.track_ids or (p.track_id,))}

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
                               cfg.detection.confidence, cfg.events.handoff_seconds)
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
        tr, ev_cfg = self.tracker, self.cfg.events  # live edits
        tr.ttl, tr.memory, tr.handoff = ev_cfg.track_ttl, ev_cfg.lost_memory_seconds, ev_cfg.handoff_seconds
        tr.fps = det_cfg.detect_fps
        tr.set_confidence(det_cfg.confidence)
        tr.frame_width = frame.shape[1]
        seen = tr.update(dets, now)
        trusted_map = self.db.trusted_map()
        pending = self._confirm(seen, now)
        people = [t for t in seen if t.id not in pending]  # split-second duplicates can't claim faces

        check = []
        for t in people:
            name, _ = t.resolve(trusted_map, face_cfg.trusted_min_matches)
            locked = name is not None and len(t.votes[name]) >= LOCKED_VOTES and t.probation_since is None
            every = RECHECK_SECONDS if locked else 1.0 / max(face_cfg.check_fps, 0.1)
            if now - t.last_face_check >= every - 1e-6:
                t.last_face_check = now
                check.append(t)
        candidates: list[FaceObs] = []
        for t in check:
            x1, y1, x2, y2 = (float(v) for v in t.box)
            for obs in self.faces.analyze(frame, (x1, y1, x2, y1 + HEAD_REGION * (y2 - y1)), max_faces=3):
                if all(_iou(np.array(obs.box, float), np.array(c.box, float)) < 0.5 for c in candidates):
                    candidates.append(obs)
        for t, obs in assign_faces(candidates, people):
            if t not in check:
                continue
            if not obs.good:  # blurry / turned away: can't tell who it is either way
                t.poor_faces += 1
                continue
            m = self.db.match(obs.embedding)
            if t.probation_since is not None:
                t = self._check_probation(t, m, trusted_map, now, seen)
            if m:
                t.votes.setdefault(m.name, []).append(m.score)
                best = t.best_known.get(m.name)
                if best is None or obs.quality > best.quality:
                    t.best_known[m.name] = obs
            else:
                t.unknown_faces += 1
                if t.best_unknown is None or obs.quality > t.best_unknown.quality:
                    t.best_unknown = obs

        confirmed = [t for t in seen if t.confirmed]
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

    def _check_probation(self, t: Track, m, trusted_map: dict[str, bool], now: float, seen: list[Track]) -> Track:
        """A walker re-found after missed frames whose face doesn't match who they were may be someone else: after
        PROBATION_MISMATCHES such faces it splits into a new person (replacing it in `seen`)."""
        name, _ = t.resolve(trusted_map, self.cfg.face.trusted_min_matches)
        if name is None or (m is not None and m.name == name):
            t.probation_since = None  # nothing to protect, or it is them
            return t
        t.mismatches += 1
        if t.mismatches < PROBATION_MISMATCHES:
            return t
        new = self.tracker.split(t, now)
        seen[seen.index(t)] = new
        return new

    def _confirm(self, seen: list[Track], now: float) -> set[int]:
        """Decide which tracks count as real people. Returns ids of tracks held back as likely duplicates.

        A new person standing apart confirms after `min_hits` detections, so quick visits still alert.
        A box on top of someone already tracked is usually a split-second duplicate detection (YOLO boxing
        the same person twice), so it must be seen in most frames for `overlap_confirm_seconds` first."""
        det_cfg = self.cfg.detection
        established = [t for t in seen if t.confirmed]
        pending: set[int] = set()
        # bigger / steadier boxes first, so when a person arrives with two boxes the real one wins
        candidates = sorted((t for t in seen if not t.confirmed and t.hits >= det_cfg.min_hits),
                            key=lambda t: (-t.hits, -(t.box[2] - t.box[0]) * (t.box[3] - t.box[1])))
        for t in candidates:
            if any(_containment(t.box, o.box) >= OVERLAP_CONTAINMENT for o in established) \
                    and not self._persisted(t, now):
                pending.add(t.id)
                continue
            t.confirmed = True
            established.append(t)
        return pending

    def _persisted(self, t: Track, now: float) -> bool:
        hold = self.cfg.detection.overlap_confirm_seconds
        if hold <= 0:
            return True
        # seen in most frames of the window AND across all of it (a 3-frame flicker spans only ~0.4 s)
        frame = 1.0 / max(self.cfg.detection.detect_fps, 0.1)
        recent = [ts for ts, _, _ in t.path if ts >= now - hold - frame / 2]
        needed = max(self.cfg.detection.min_hits, OVERLAP_PRESENCE * hold * self.cfg.detection.detect_fps)
        return len(recent) >= needed and now - recent[0] >= hold - frame

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
        """New people and identities not yet evaluated need an event; people who stay never re-trigger."""
        return t.key(trusted_map, self.cfg.face.trusted_min_matches) not in t.seen_keys

    def _finalize(self, ev: _Event, now: float) -> EventResult:
        face_cfg = self.cfg.face
        trusted_map = self.db.trusted_map()
        people: list[PersonInfo] = []
        by_key: dict[str, PersonInfo] = {}
        for t in ev.tracks.values():
            name, trusted = t.resolve(trusted_map, face_cfg.trusted_min_matches)
            if name is None and t.best_unknown is not None and face_cfg.save_unknowns:
                b = t.best_unknown
                if t.unknown_id is None or not self.db.unknown_exists(t.unknown_id):
                    t.unknown_id = self.db.record_unknown(b.embedding, b.crop, b.quality, self.name, face_cfg.max_unknown_images)
            key = t.key(trusted_map, face_cfg.trusted_min_matches)
            face = t.best_known.get(name) if name else t.best_unknown
            new = key not in t.seen_keys
            t.seen_keys.add(key)
            score = float(np.mean(t.votes[name])) if name in t.votes else None
            p = PersonInfo(t.id, name, trusted, key, None if name else t.unknown_id, face, new, score,
                           poor_face=t.poor_faces > 0, track_ids=(t.id,))
            same = by_key.get(key) if key.startswith(("person:", "unknown:")) else None
            if same is None:  # (faceless tracks are never merged: two strangers must both count)
                by_key[key] = p
                people.append(p)
            else:  # one person on two tracks (re-found after a gap): list them once
                same.new, same.trusted = same.new or p.new, same.trusted or p.trusted
                same.track_ids += p.track_ids
                if p.score is not None and (same.score is None or p.score > same.score):
                    same.score = p.score
                if p.face is not None and (same.face is None or p.face.quality > same.face.quality):
                    same.face = p.face

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
