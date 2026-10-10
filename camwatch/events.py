"""Per-camera arrivals, face identification, event lifecycle and alert decisions.

Alerts are about *arrivals*, not presence. Nothing follows individual bodies from frame to frame, so walking
around, sitting down, turning away or a missed detection can't turn one person into two. Two signals:

  * People count (YOLO). Only solid person boxes count: tall enough, and not a sliver cut off at the frame
    edge. A stable `level` rises once more people have been in view for `arrive_seconds`, and falls only
    once fewer have been for `leave_seconds`. Each rise is an arrival. While the room stays occupied
    (an activity session) nobody re-alerts; an outdoor camera with a short `leave_seconds` makes every
    passer-by a short session of their own.
  * Faces (SCRFD + ArcFace) on the whole frame. Every clear face is identified wherever it is: a known
    person, or grouped with similar unknown faces seen on this camera. Blurry or turned-away faces are ignored.

Event lifecycle:
  1. Starts on an arrival, or when a face shows up that this camera hasn't seen within `cooldown_seconds`.
  2. Records for `post_seconds` while faces keep being identified.
  3. Finalizes: faces seen at least `min_sightings` times are the people in it. Arrivals no newly seen face
     accounts for become "Person (face not visible)" (mask, hood, dark room, back turned).
  4. Decides: no alert if every new person is trusted, otherwise alert unless each of them already alerted
     on this camera within the cooldown.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .camera import BufferedFrame, CameraStream, Overlay, Recording
from .config import AppConfig
from .faces import FaceDB, FaceEngine, FaceObs

log = logging.getLogger(__name__)

EDGE_MARGIN = 0.02  # a box this close (fraction of frame size) to the left/right/bottom edge is cut off
DUPLICATE_CONTAINMENT = 0.8  # a box this much inside a bigger one is the same person boxed twice
ARRIVE_SHARE = 0.8  # share of detections during `arrive_seconds` that must show the higher count
MAX_FACES = 6
FACE_MEMORY = 10  # embeddings kept per face seen on this camera
SIGHTING_MEMORY = 64  # sighting times kept per face (to tell whether it was here just before an event)
SOFT_MARGIN = 0.08  # a face this far below the match threshold still counts as someone already here
RECENT_FACE_SECONDS = 10.0  # a face seen this recently before an event was already here: it explains no arrival
VISIBLE_SECONDS = 2.0  # dashboard "in view" shows faces seen this recently
PERSON_BOX_ID, UNCLEAR_FACE_ID = 0, -1  # overlay ids of person boxes and unidentifiable faces (faces use 1, 2, ...)
PERSON_LABEL, UNCLEAR_LABEL, FACE_LABEL = "person", "unclear face", "face"


def _containment(a: np.ndarray, b: np.ndarray) -> float:
    """Share of box a's area inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def solid_people(dets: np.ndarray, width: int, height: int, min_height: float) -> list[np.ndarray]:
    """Person boxes that count: tall enough (twice as tall if cut off at the frame edge), one per person."""
    out: list[np.ndarray] = []
    mx, my = EDGE_MARGIN * width, EDGE_MARGIN * height
    for b in sorted(dets[:, :4], key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        cut = b[0] <= mx or b[2] >= width - mx or b[3] >= height - my
        if b[3] - b[1] < min_height * height * (2 if cut else 1):
            continue
        if any(_containment(b, o) >= DUPLICATE_CONTAINMENT for o in out):
            continue
        out.append(b)
    return out


@dataclass
class SeenFace:
    """Someone whose face was seen on this camera: a known person, or a stranger identified by appearance."""

    id: int
    name: str | None
    last_seen: float
    embeddings: deque = field(default_factory=lambda: deque(maxlen=FACE_MEMORY))
    times: deque = field(default_factory=lambda: deque(maxlen=SIGHTING_MEMORY))
    unknown_id: int | None = None  # stranger's cluster in the face DB, once recorded

    def similarity(self, emb: np.ndarray) -> float:
        return max(float(e @ emb) for e in self.embeddings)

    def seen_before(self, t: float) -> float:
        """Last sighting before time t (-inf: none)."""
        return max((s for s in self.times if s < t - 1e-6), default=float("-inf"))


@dataclass
class _Sightings:
    """One face's sightings during an event."""

    face: SeenFace
    count: int = 0
    scores: list[float] = field(default_factory=list)
    best: FaceObs | None = None


@dataclass
class PersonInfo:
    track_id: int  # overlay id of their face in the clip (0 = no face)
    name: str | None
    trusted: bool
    key: str
    unknown_id: int | None
    face: FaceObs | None
    new: bool = True  # arrived / newly identified in this event (vs. already present)
    score: float | None = None  # mean face-match similarity for `name`
    track_ids: tuple[int, ...] = ()  # every face id merged into this person (same name / unknown cluster)

    @property
    def label(self) -> str:
        if self.name:
            return f"{self.name} (trusted)" if self.trusted else self.name
        if self.unknown_id:
            return f"Unknown #{self.unknown_id}"
        if self.face:
            return "Unknown person"
        return "Person (face not visible)"

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
    box_labels: dict[int, str] = field(default_factory=dict)  # labels for overlay boxes that aren't people

    @property
    def labels(self) -> dict[int, str]:
        people = {tid: p.scored_label for p in self.people for tid in (p.track_ids or (p.track_id,)) if tid > 0}
        return {**self.box_labels, **people}

    def summary(self) -> str:
        return ", ".join(p.label for p in self.people) or "nobody"


@dataclass
class _Event:
    start: float
    end: float
    anchor: float
    wall: float
    recording: Recording
    session: float  # when the activity session this event belongs to began
    arrivals: int = 0
    faces: dict[int, _Sightings] = field(default_factory=dict)


class CameraMonitor:
    def __init__(self, cfg: AppConfig, stream: CameraStream, faces: FaceEngine, db: FaceDB,
                 is_armed: Callable[[str], bool]):
        self.cfg = cfg
        self.stream = stream
        self.faces = faces
        self.db = db
        self.is_armed = is_armed
        self.level = 0  # people in view, smoothed
        self.session: float | None = None  # start of the current activity session (someone in view)
        self._last_face = float("-inf")
        self._counts: deque[tuple[float, int]] = deque()
        self.seen: dict[int, SeenFace] = {}
        self._ids = itertools.count(1)
        self._face_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        self._last_face_scan = float("-inf")
        self._faceless = itertools.count(1)
        self.event: _Event | None = None
        self.cooldowns: dict[str, float] = {}
        self.visible: list[str] = []
        self.last_result: EventResult | None = None
        self.infer_fps = 0.0
        self._last_infer = 0.0

    @property
    def name(self) -> str:
        return self.stream.cfg.name

    def leave_seconds(self) -> float:
        own = self.stream.cfg.leave_seconds
        return self.cfg.events.leave_seconds if own is None else own

    # ---- per inference --------------------------------------------------------------------
    def process(self, frame: np.ndarray, dets: np.ndarray, now: float) -> None:
        """Called by the detection worker for every inference on this camera."""
        if self._last_infer:
            dt = now - self._last_infer
            self.infer_fps = 0.8 * self.infer_fps + 0.2 / dt if self.infer_fps else 1.0 / dt
        self._last_infer = now

        h, w = frame.shape[:2]
        people = solid_people(dets, w, h, self.cfg.detection.person_min_height)
        arrived, since = self._count(len(people), now)

        sightings, new_faces = [], []
        if now - self._last_face_scan >= 1.0 / max(self.cfg.face.check_fps, 0.1) - 1e-6:
            self._last_face_scan = now
            sightings, new_faces = self._scan_faces(frame, now)
        self._update_session(since if arrived else now, now, bool(sightings))

        if self.event is None and (arrived or new_faces):
            anchor = since if arrived else now
            start = anchor - self.cfg.events.pre_seconds
            self.event = _Event(start=start, anchor=anchor, end=anchor + self.cfg.events.post_seconds,
                                wall=time.time() - (now - anchor), recording=self.stream.start_recording(start),
                                session=self.session)
            log.debug("[%s] event started (arrivals %d, new faces %s)", self.name, arrived, new_faces)
        if self.event is not None:
            self.event.arrivals += arrived
            for s in sightings:
                ev_s = self.event.faces.setdefault(s.face.id, _Sightings(s.face))
                ev_s.count += s.count
                ev_s.scores += s.scores
                if s.best is not None and (ev_s.best is None or s.best.quality > ev_s.best.quality):
                    ev_s.best = s.best

        self.stream.set_overlay(Overlay(now, [(PERSON_BOX_ID, tuple(float(v) for v in b)) for b in people]
                                        + self._face_boxes))
        trusted_map = self.db.trusted_map()
        labels = [f"{f.name}{' ✓' if trusted_map.get(f.name) else ''}" if f.name else "unknown"
                  for f in self.seen.values() if now - f.last_seen <= VISIBLE_SECONDS]
        self.visible = ([f"{self.level} {'person' if self.level == 1 else 'people'}"] if self.level else []) + labels

    def _update_session(self, began: float, now: float, faces: bool) -> None:
        """A session lasts while anyone is in view; it ends once nobody was for `leave_seconds`."""
        if faces:
            self._last_face = now
        if self.level > 0 or faces:
            if self.session is None:
                self.session = began
                log.debug("[%s] activity session started", self.name)
        elif self.session is not None and now - self._last_face > self.leave_seconds() and self.event is None:
            self.session = None
            log.debug("[%s] activity session ended", self.name)

    def _is_new(self, f: SeenFace, t: float) -> bool:
        """Not seen earlier in this activity session, nor within the cooldown before it."""
        before = f.seen_before(t)
        return (self.session is None or before < self.session) and t - before > self.cfg.events.cooldown_seconds

    def _count(self, n: int, now: float) -> tuple[int, float]:
        """Update the smoothed people count; returns (people who just arrived, when they appeared)."""
        arrive, leave = self.cfg.events.arrive_seconds, self.leave_seconds()
        if self._counts and now - self._counts[-1][0] > leave:  # camera was offline: start over
            self._counts.clear()
            self.level = 0
        self._counts.append((now, n))
        while self._counts and self._counts[0][0] < now - max(arrive, leave) - 1.0:
            self._counts.popleft()
        frame = 1.0 / max(self.cfg.detection.detect_fps, 0.1)
        oldest = self._counts[0][0]

        recent = sorted(c for ts, c in self._counts if ts >= now - arrive)
        if now - oldest >= arrive - frame / 2:  # enough history to judge
            k = recent[int((1 - ARRIVE_SHARE) * len(recent))]  # count shown in >= ARRIVE_SHARE of detections
            if k > self.level:
                arrived, self.level = k - self.level, k
                since = next(ts for ts, c in self._counts if ts >= now - arrive and c >= k)
                log.debug("[%s] people in view: %d (+%d)", self.name, k, arrived)
                return arrived, since
        if now - oldest >= leave - frame / 2:
            most = max(c for ts, c in self._counts if ts >= now - leave)
            if most < self.level:
                log.debug("[%s] people in view: %d (-%d)", self.name, most, self.level - most)
                self.level = most
        return 0, now

    def _scan_faces(self, frame: np.ndarray, now: float) -> tuple[list[_Sightings], list[int]]:
        """Identify every face in the frame. Returns the sightings, and ids of faces that are new here
        (not seen earlier in this activity session, nor within the cooldown)."""
        self._face_boxes, newly = [], []
        sightings: dict[int, _Sightings] = {}
        for obs in self.faces.analyze(frame, None, max_faces=MAX_FACES):
            if not obs.good:  # blurry / turned away: can't tell who it is either way
                self._face_boxes.append((UNCLEAR_FACE_ID, obs.box))
                continue
            f, score = self._identify(obs.embedding, now)
            if f.id not in sightings:  # (the same person twice in one frame is a duplicate detection)
                f.last_seen = now
                f.times.append(now)
                if self._is_new(f, now):
                    newly.append(f.id)
            f.embeddings.append(obs.embedding)
            s = sightings.setdefault(f.id, _Sightings(f))
            s.count = 1
            if score is not None:
                s.scores.append(score)
            if s.best is None or obs.quality > s.best.quality:
                s.best = obs
            self._face_boxes.append((f.id, obs.box))
        self._forget(now)
        return list(sightings.values()), newly

    def _identify(self, emb: np.ndarray, now: float) -> tuple[SeenFace, float | None]:
        """Known person, else a stranger already seen here, else someone new. Returns the face and match score."""
        m = self.db.match(emb)
        if m is None:
            # a household member at a bad angle just under the threshold, while they're known to be here
            near = self.db.match(emb, self.db.threshold - SOFT_MARGIN)
            if near is not None and any(f.name == near.name and now - f.last_seen <= self.cfg.events.cooldown_seconds
                                        for f in self.seen.values()):
                m = near
        if m is not None:
            f = next((f for f in self.seen.values() if f.name == m.name), None)
            return f or self._new_face(m.name), m.score
        strangers = [(f.similarity(emb), f) for f in self.seen.values() if f.name is None and f.embeddings]
        best = max(strangers, key=lambda x: x[0], default=None)
        if best is not None and best[0] >= self.db.threshold:
            return best[1], None
        return self._new_face(None), None

    def _new_face(self, name: str | None) -> SeenFace:
        f = SeenFace(next(self._ids), name, float("-inf"))
        self.seen[f.id] = f
        return f

    def _forget(self, now: float) -> None:
        keep = 2 * max(self.cfg.events.cooldown_seconds, self.leave_seconds(), RECENT_FACE_SECONDS)
        in_event = set(self.event.faces) if self.event else set()
        for fid, f in list(self.seen.items()):
            in_session = self.session is not None and f.last_seen >= self.session  # still here, maybe turned away
            if now - f.last_seen > keep and fid not in in_event and not in_session:
                del self.seen[fid]

    # ---- events --------------------------------------------------------------------------
    def tick(self, now: float) -> EventResult | None:
        """Finish the active event once its clip window has been recorded."""
        if self.event is None or now < self.event.end:
            return None
        ev, self.event = self.event, None
        self.stream.stop_recording(ev.recording)
        return self._finalize(ev, now)

    def live_labels(self) -> dict[int, str]:
        """Overlay labels for a snapshot of what the camera sees right now."""
        trusted_map = self.db.trusted_map()
        labels = {PERSON_BOX_ID: PERSON_LABEL, UNCLEAR_FACE_ID: UNCLEAR_LABEL}
        for f in self.seen.values():
            labels[f.id] = f"{f.name}{' (trusted)' if trusted_map.get(f.name) else ''}" if f.name else "Unknown person"
        return labels

    def _finalize(self, ev: _Event, now: float) -> EventResult | None:
        face_cfg, cooldown = self.cfg.face, self.cfg.events.cooldown_seconds
        trusted_map = self.db.trusted_map()
        box_labels = {PERSON_BOX_ID: PERSON_LABEL, UNCLEAR_FACE_ID: UNCLEAR_LABEL}
        people: list[PersonInfo] = []
        by_key: dict[str, PersonInfo] = {}
        explained = 0
        for s in ev.faces.values():
            f = s.face
            if s.count < face_cfg.min_sightings:  # a fleeting face: too little to go on
                box_labels[f.id] = FACE_LABEL
                continue
            if f.name is None and face_cfg.save_unknowns and s.best is not None:
                if f.unknown_id is None or not self.db.unknown_exists(f.unknown_id):
                    f.unknown_id = self.db.record_unknown(s.best.embedding, s.best.crop, s.best.quality, self.name,
                                                          face_cfg.max_unknown_images)
            key = f"person:{f.name}" if f.name else (f"unknown:{f.unknown_id}" if f.unknown_id else f"face:{self.name}:{f.id}")
            before = f.seen_before(ev.anchor)
            if ev.anchor - before > RECENT_FACE_SECONDS:
                explained += 1  # wasn't here just before: this face can be (one of) the arrivals
            new = before < ev.session and ev.anchor - before > cooldown
            score = float(np.mean(s.scores)) if s.scores else None
            p = PersonInfo(f.id, f.name, bool(f.name and trusted_map.get(f.name)), key,
                           None if f.name else f.unknown_id, s.best, new, score, track_ids=(f.id,))
            same = by_key.get(key)
            if same is None:
                by_key[key] = p
                people.append(p)
            else:  # two faces on this camera turned out to be one unknown cluster: list them once
                same.new = same.new or p.new
                same.track_ids += p.track_ids
                if p.face is not None and (same.face is None or p.face.quality > same.face.quality):
                    same.face = p.face
        for _ in range(max(0, ev.arrivals - explained)):  # arrivals whose face was never seen clearly
            people.append(PersonInfo(PERSON_BOX_ID, None, False, f"faceless:{self.name}:{next(self._faceless)}",
                                     None, None, True))
        if not people:
            log.debug("[%s] event dropped: no face seen clearly enough and nobody arrived", self.name)
            return None

        # Only people who are new in this event can cause an alert; the others are listed as context.
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

        result = EventResult(self.name, ev.wall, decision, people, ev.recording.frames, ev.start, ev.end, ev.anchor,
                             box_labels)
        self.last_result = result
        log.info("[%s] %s: %s", self.name, decision, result.summary())
        return result
