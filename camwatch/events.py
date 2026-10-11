"""Per-camera arrivals, face identification, event lifecycle and alert decisions.

Three rules, all on one time span, `events.away_seconds` (30 s):

  * People (YOLO). A camera is empty once it has seen no person for `away_seconds`. Someone appearing in an
    empty camera (in most detections for `arrive_seconds`) is an arrival. Only solid boxes count: tall enough,
    and not a sliver cut off at the frame edge.
  * Faces (SCRFD + ArcFace), looked for only inside person boxes. Each clear face is identified: a known person,
    or grouped with similar unknown faces seen on this camera. A face is new if this camera hasn't seen it for
    `away_seconds` and not at all since the camera was last empty (someone who never left the room but kept
    their face turned away isn't new). Blurry or turned-away faces are ignored.
  * An arrival or a new face starts an event: a `post_seconds` clip while faces keep being identified. Faces seen
    at least `min_sightings` times are the people in it; if someone arrived and no face was clear enough, they
    are "Person (face not visible)". No alert if every new person is trusted.

So someone stepping out and back within `away_seconds` never re-alerts, and sitting, turning away or walking
around can't make one person into two: nothing follows bodies from frame to frame.
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
ARRIVE_SHARE = 0.8  # share of detections during `arrive_seconds` that must show someone
MAX_FACES = 6
MAX_FACE_REGIONS = 4  # more person boxes than this: one full-frame face scan is cheaper than one per box
CROP_DET_SIZE = 320  # face detector input for one person's box (a face is ~1/7 of a body: still plenty)
SAME_FACE_IOU = 0.5  # one face found through two overlapping person boxes
FACE_MEMORY = 10  # embeddings kept per face seen on this camera
VISIBLE_SECONDS = 2.0  # dashboard "in view" shows faces seen this recently
PERSON_BOX_ID, UNCLEAR_FACE_ID = 0, -1  # overlay ids of person boxes and unidentifiable faces (faces use 1, 2, ...)
PERSON_LABEL, UNCLEAR_LABEL, FACE_LABEL = "person", "unclear face", "face"


def solid_people(dets: np.ndarray, width: int, height: int, min_height: float) -> list[np.ndarray]:
    """Person boxes that count: tall enough, or twice that if cut off at the frame edge."""
    mx, my = EDGE_MARGIN * width, EDGE_MARGIN * height
    out = []
    for b in dets[:, :4]:
        cut = b[0] <= mx or b[2] >= width - mx or b[3] >= height - my
        if b[3] - b[1] >= min_height * height * (2 if cut else 1):
            out.append(b)
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class SeenFace:
    """Someone whose face was seen on this camera: a known person, or a stranger identified by appearance."""

    id: int
    name: str | None
    last_seen: float
    visit: float = float("-inf")  # when this visit began (first sighting after `away_seconds` unseen)
    embeddings: deque = field(default_factory=lambda: deque(maxlen=FACE_MEMORY))
    unknown_id: int | None = None  # stranger's cluster in the face DB, once recorded

    def similarity(self, emb: np.ndarray) -> float:
        return max(float(e @ emb) for e in self.embeddings)


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
    new: bool = True  # arrived in this event (vs. already here)
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
    decision: str  # alert | rate-limited | trusted | disarmed | unchanged
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
    arrival: bool = False  # someone appeared in the empty camera
    faces: dict[int, _Sightings] = field(default_factory=dict)


class CameraMonitor:
    def __init__(self, cfg: AppConfig, stream: CameraStream, faces: FaceEngine, db: FaceDB,
                 is_armed: Callable[[str], bool]):
        self.cfg = cfg
        self.stream = stream
        self.faces = faces
        self.db = db
        self.is_armed = is_armed
        self.occupied_since: float | None = None  # when someone appeared in the empty camera (None: empty)
        self.last_person = float("-inf")  # last detection with a person in it
        self._hits: deque[tuple[float, bool]] = deque()  # (time, person seen) for the arrival window
        self.seen: dict[int, SeenFace] = {}
        self._ids = itertools.count(1)
        self._face_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        self._last_face_scan = float("-inf")
        self._faceless = itertools.count(1)
        self.event: _Event | None = None
        self.visible: list[str] = []
        self.last_result: EventResult | None = None
        self.infer_fps = 0.0
        self._last_infer = 0.0

    @property
    def name(self) -> str:
        return self.stream.cfg.name

    # ---- per inference --------------------------------------------------------------------
    def process(self, frame: np.ndarray, dets: np.ndarray, now: float) -> None:
        """Called by the detection worker for every inference on this camera."""
        if self._last_infer:
            dt = now - self._last_infer
            self.infer_fps = 0.8 * self.infer_fps + 0.2 / dt if self.infer_fps else 1.0 / dt
        self._last_infer = now

        h, w = frame.shape[:2]
        people = solid_people(dets, w, h, self.cfg.detection.person_min_height)
        arrived = self._update_presence(bool(people), now)

        sightings, new_faces = [], []
        if not len(dets):
            self._face_boxes = []
        elif now - self._last_face_scan >= 1.0 / max(self.cfg.face.check_fps, 0.1) - 1e-6:
            self._last_face_scan = now
            sightings, new_faces = self._scan_faces(frame, dets[:, :4], now)

        if self.event is None and (arrived is not None or new_faces):
            anchor = now if arrived is None else arrived
            start = anchor - self.cfg.events.pre_seconds
            self.event = _Event(start=start, anchor=anchor, end=anchor + self.cfg.events.post_seconds,
                                wall=time.time() - (now - anchor), recording=self.stream.start_recording(start))
            log.debug("[%s] event started (arrival: %s, new faces: %s)", self.name, arrived is not None, new_faces)
        if self.event is not None:
            self.event.arrival |= arrived is not None
            for s in sightings:
                ev_s = self.event.faces.setdefault(s.face.id, _Sightings(s.face))
                ev_s.count += s.count
                ev_s.scores += s.scores
                if s.best is not None and (ev_s.best is None or s.best.quality > ev_s.best.quality):
                    ev_s.best = s.best
        self._forget(now)

        self.stream.set_overlay(Overlay(now, [(PERSON_BOX_ID, tuple(float(v) for v in b)) for b in people]
                                        + self._face_boxes))
        trusted_map = self.db.trusted_map()
        labels = [f"{f.name}{' ✓' if trusted_map.get(f.name) else ''}" if f.name else "unknown"
                  for f in self.seen.values() if now - f.last_seen <= VISIBLE_SECONDS]
        self.visible = ([f"{len(people)} {'person' if len(people) == 1 else 'people'}"] if people else []) + labels

    def _update_presence(self, person: bool, now: float) -> float | None:
        """Track whether anyone is here. Returns when someone appeared in the empty camera, else None."""
        away, arrive = self.cfg.events.away_seconds, self.cfg.events.arrive_seconds
        if self.occupied_since is not None and now - self.last_person > away:  # also after a camera outage
            self.occupied_since = None
            log.debug("[%s] nobody here for %gs", self.name, away)
        if person:
            self.last_person = now
        self._hits.append((now, person))
        while self._hits[0][0] < now - arrive - 1.0:
            self._hits.popleft()
        if self.occupied_since is not None:
            return None
        frame = 1.0 / max(self.cfg.detection.detect_fps, 0.1)
        if now - self._hits[0][0] < arrive - frame / 2:  # not enough history to judge yet
            return None
        window = [(ts, p) for ts, p in self._hits if ts >= now - arrive]
        if sum(p for _, p in window) < ARRIVE_SHARE * len(window):
            return None
        self.occupied_since = next(ts for ts, p in window if p)
        log.debug("[%s] someone arrived", self.name)
        return self.occupied_since

    def _scan_faces(self, frame: np.ndarray, boxes: np.ndarray, now: float) -> tuple[list[_Sightings], list[int]]:
        """Identify the faces inside the person boxes. Returns the sightings, and ids of faces that are new here."""
        if len(boxes) > MAX_FACE_REGIONS:
            found = self.faces.analyze(frame, None, max_faces=MAX_FACES)
        else:
            found = []
            for b in boxes:
                for obs in self.faces.analyze(frame, tuple(float(v) for v in b), max_faces=2, det_size=CROP_DET_SIZE):
                    same = next((i for i, o in enumerate(found) if _iou(o.box, obs.box) >= SAME_FACE_IOU), None)
                    if same is None:
                        found.append(obs)
                    elif obs.score > found[same].score:
                        found[same] = obs

        self._face_boxes, newly = [], []
        sightings: dict[int, _Sightings] = {}
        for obs in found:
            if not obs.good:  # blurry / turned away: can't tell who it is either way
                self._face_boxes.append((UNCLEAR_FACE_ID, obs.box))
                continue
            f, score = self._identify(obs.embedding)
            if f.id not in sightings:  # (the same person twice in one frame is a duplicate detection)
                stayed = self.occupied_since is not None and f.last_seen >= self.occupied_since
                if now - f.last_seen > self.cfg.events.away_seconds and not stayed:
                    f.visit = now
                    newly.append(f.id)
                f.last_seen = now
            f.embeddings.append(obs.embedding)
            s = sightings.setdefault(f.id, _Sightings(f))
            s.count = 1
            if score is not None:
                s.scores.append(score)
            if s.best is None or obs.quality > s.best.quality:
                s.best = obs
            self._face_boxes.append((f.id, obs.box))
        return list(sightings.values()), newly

    def _identify(self, emb: np.ndarray) -> tuple[SeenFace, float | None]:
        """Known person, else a stranger already seen here, else someone new. Returns the face and match score."""
        m = self.db.match(emb)
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
        keep = 2 * self.cfg.events.away_seconds
        in_event = set(self.event.faces) if self.event else set()
        for fid, f in list(self.seen.items()):
            stayed = self.occupied_since is not None and f.last_seen >= self.occupied_since  # maybe turned away
            if now - f.last_seen > keep and fid not in in_event and not stayed:
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
        for f in list(self.seen.values()):  # called from the UI thread while the detector updates it
            labels[f.id] = f"{f.name}{' (trusted)' if trusted_map.get(f.name) else ''}" if f.name else "Unknown person"
        return labels

    def _finalize(self, ev: _Event, now: float) -> EventResult | None:
        face_cfg = self.cfg.face
        trusted_map = self.db.trusted_map()
        box_labels = {PERSON_BOX_ID: PERSON_LABEL, UNCLEAR_FACE_ID: UNCLEAR_LABEL}
        people: list[PersonInfo] = []
        by_key: dict[str, PersonInfo] = {}
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
            new = f.visit >= ev.start - 1e-6  # this visit began with this event, not before it
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
        if ev.arrival and not people:  # someone came, but no face was clear enough to say who
            people.append(PersonInfo(PERSON_BOX_ID, None, False, f"faceless:{self.name}:{next(self._faceless)}",
                                     None, None, True))
        if not people:
            log.debug("[%s] event dropped: no face seen clearly enough", self.name)
            return None

        # Only people who are new in this event can cause an alert; the others are listed as context.
        new_untrusted = [p for p in people if p.new and not p.trusted]
        if not any(p.new for p in people):
            decision = "unchanged"
        elif not new_untrusted:
            decision = "trusted"
        elif not self.is_armed(self.name):
            decision = "disarmed"
        else:
            decision = "alert"

        result = EventResult(self.name, ev.wall, decision, people, ev.recording.frames, ev.start, ev.end, ev.anchor,
                             box_labels)
        self.last_result = result
        log.info("[%s] %s: %s", self.name, decision, result.summary())
        return result
