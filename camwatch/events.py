"""Per-camera arrivals, face identification, moving vehicles, event lifecycle and alert decisions.

Four rules, all on one time span, `events.away_seconds` (30 s):

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
  * Vehicles (YOLO, cameras with `vehicles: true`). A vehicle is still if its box sits where a vehicle already was
    `STILL_SECONDS` ago (or is mostly inside that spot: someone walking in front of a parked car); any other vehicle
    is moving. A moving vehicle in a camera that saw none moving for `away_seconds` (in most detections for
    `arrive_seconds`) starts an event, and always alerts when armed. Vehicles already there when the camera starts
    watching are parked, and parked spots are remembered for `PARKED_MEMORY`, so a parked car YOLO lost for a
    while isn't new when it's found again.

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
from .detector import PERSON_CLASS, VEHICLE_CLASSES
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
STILL_IOU = 0.8  # a vehicle box this close to a spot (IoU) is in that spot
STILL_INSIDE = 0.9  # ...or one lying this much inside it (partly hidden behind someone)
STILL_SECONDS = 1.0  # a vehicle in a spot it has been in for this long is still
SPOT_MEMORY = 5.0  # spots of moving vehicles are forgotten after this long unseen
PARKED_MEMORY = 3600.0  # ...and spots of parked ones after this long
# overlay ids of person boxes, unidentifiable faces and vehicles (faces use 1, 2, ...)
PERSON_BOX_ID, UNCLEAR_FACE_ID, VEHICLE_BOX_ID, PARKED_BOX_ID = 0, -1, -2, -3
PERSON_LABEL, UNCLEAR_LABEL, FACE_LABEL = "person", "unclear face", "face"
VEHICLE_LABEL, PARKED_LABEL = "moving vehicle", "parked vehicle"
BOX_LABELS = {PERSON_BOX_ID: PERSON_LABEL, UNCLEAR_FACE_ID: UNCLEAR_LABEL,
              VEHICLE_BOX_ID: VEHICLE_LABEL, PARKED_BOX_ID: PARKED_LABEL}


def solid_people(dets: np.ndarray, width: int, height: int, min_height: float) -> list[np.ndarray]:
    """Person boxes that count: tall enough, or twice that if cut off at the frame edge."""
    mx, my = EDGE_MARGIN * width, EDGE_MARGIN * height
    out = []
    for b in dets[:, :4]:
        cut = b[0] <= mx or b[2] >= width - mx or b[3] >= height - my
        if b[3] - b[1] >= min_height * height * (2 if cut else 1):
            out.append(b)
    return out


def solid_vehicles(dets: np.ndarray, width: int, min_width: float) -> list[tuple[np.ndarray, int]]:
    """(box, class) of the vehicles that count: wide enough to be near, not traffic in the distance."""
    return [(d[:4], int(d[5])) for d in dets if int(d[5]) in VEHICLE_CLASSES and d[2] - d[0] >= min_width * width]


def _intersection(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a) -> float:
    return (a[2] - a[0]) * (a[3] - a[1])


def _iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _in_spot(box, spot) -> bool:
    return _iou(box, spot) >= STILL_IOU or _intersection(box, spot) >= STILL_INSIDE * _area(box)


class Presence:
    """Whether something (people, moving vehicles) is in view. The camera is empty once it saw nothing for
    `away_seconds`; something in most detections for `arrive_seconds` in the empty camera has arrived."""

    def __init__(self, cfg: AppConfig, camera: str, what: str):
        self.cfg, self.camera, self.what = cfg, camera, what
        self.since: float | None = None  # when it appeared in the empty camera (None: empty)
        self.last_seen = float("-inf")  # last detection with it in it
        self._hits: deque[tuple[float, bool]] = deque()  # (time, seen) for the arrival window

    def update(self, seen: bool, now: float) -> float | None:
        """Returns when it appeared in the empty camera, else None."""
        away, arrive = self.cfg.events.away_seconds, self.cfg.events.arrive_seconds
        if self.since is not None and now - self.last_seen > away:  # also after a camera outage
            self.since = None
            log.debug("[%s] no %s for %gs", self.camera, self.what, away)
        if seen:
            self.last_seen = now
        self._hits.append((now, seen))
        while self._hits[0][0] < now - arrive - 1.0:
            self._hits.popleft()
        if self.since is not None:
            return None
        frame = 1.0 / max(self.cfg.detection.detect_fps, 0.1)
        if now - self._hits[0][0] < arrive - frame / 2:  # not enough history to judge yet
            return None
        window = [(ts, p) for ts, p in self._hits if ts >= now - arrive]
        if sum(p for _, p in window) < ARRIVE_SHARE * len(window):
            return None
        self.since = next(ts for ts, p in window if p)
        log.debug("[%s] %s arrived", self.camera, self.what)
        return self.since


@dataclass
class _Spot:
    """Where a vehicle was seen: the box of its first sighting there."""

    box: np.ndarray
    first: float
    last: float

    @property
    def parked(self) -> bool:
        return self.last - self.first >= STILL_SECONDS


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
    vehicle: str | None = None  # not a person but a moving vehicle of this kind (car, truck, ...)

    @property
    def label(self) -> str:
        if self.vehicle:
            return f"Moving {self.vehicle}"
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

    @property
    def icon(self) -> str:
        if self.vehicle:
            return "🚗"
        return "🟢" if self.trusted else ("🟠" if self.name else "🔴")


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
    vehicle: str | None = None  # kind of vehicle that started moving in it
    faces: dict[int, _Sightings] = field(default_factory=dict)


class CameraMonitor:
    def __init__(self, cfg: AppConfig, stream: CameraStream, faces: FaceEngine, db: FaceDB,
                 is_armed: Callable[[str], bool]):
        self.cfg = cfg
        self.stream = stream
        self.faces = faces
        self.db = db
        self.is_armed = is_armed
        self.people = Presence(cfg, self.name, "person")
        self.moving = Presence(cfg, self.name, "moving vehicle")
        self._spots: list[_Spot] = []
        self._moving_kind = "vehicle"  # kind of the widest vehicle last seen moving
        self._watching_since = self._last_vehicle_check = float("-inf")
        self._vehicles = itertools.count(1)
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

    @property
    def occupied_since(self) -> float | None:
        """When someone appeared in the empty camera (None: empty)."""
        return self.people.since

    # ---- per inference --------------------------------------------------------------------
    def process(self, frame: np.ndarray, dets: np.ndarray, now: float) -> None:
        """Called by the detection worker for every inference on this camera."""
        if self._last_infer:
            dt = now - self._last_infer
            self.infer_fps = 0.8 * self.infer_fps + 0.2 / dt if self.infer_fps else 1.0 / dt
        self._last_infer = now

        h, w = frame.shape[:2]
        persons = dets[dets[:, 5] == PERSON_CLASS]
        people = solid_people(persons, w, h, self.cfg.detection.person_min_height)
        arrived = self.people.update(bool(people), now)
        moving, parked, vehicle_arrived = [], [], None
        if self.stream.cfg.vehicles:
            moving, parked = self._classify_vehicles(
                solid_vehicles(dets, w, self.cfg.detection.vehicle_min_width), now)
            vehicle_arrived = self.moving.update(bool(moving), now)
            if moving:
                self._moving_kind = VEHICLE_CLASSES[max(moving, key=lambda v: v[0][2] - v[0][0])[1]]

        sightings, new_faces = [], []
        if not len(persons):
            self._face_boxes = []
        elif now - self._last_face_scan >= 1.0 / max(self.cfg.face.check_fps, 0.1) - 1e-6:
            self._last_face_scan = now
            sightings, new_faces = self._scan_faces(frame, persons[:, :4], now)

        arrivals = [t for t in (arrived, vehicle_arrived) if t is not None]
        if self.event is None and (arrivals or new_faces):
            anchor = min(arrivals, default=now)
            start = anchor - self.cfg.events.pre_seconds
            self.event = _Event(start=start, anchor=anchor, end=anchor + self.cfg.events.post_seconds,
                                wall=time.time() - (now - anchor), recording=self.stream.start_recording(start))
            log.debug("[%s] event started (arrival: %s, vehicle: %s, new faces: %s)", self.name,
                      arrived is not None, vehicle_arrived is not None, new_faces)
        if self.event is not None:
            self.event.arrival |= arrived is not None
            if vehicle_arrived is not None and self.event.vehicle is None:
                self.event.vehicle = self._moving_kind
            for s in sightings:
                ev_s = self.event.faces.setdefault(s.face.id, _Sightings(s.face))
                ev_s.count += s.count
                ev_s.scores += s.scores
                if s.best is not None and (ev_s.best is None or s.best.quality > ev_s.best.quality):
                    ev_s.best = s.best
        self._forget(now)

        boxes = [(PERSON_BOX_ID, b) for b in people] + [(PARKED_BOX_ID, b) for b, _ in parked] \
            + [(VEHICLE_BOX_ID, b) for b, _ in moving]
        self.stream.set_overlay(Overlay(now, [(i, tuple(float(v) for v in b)) for i, b in boxes] + self._face_boxes))
        trusted_map = self.db.trusted_map()
        labels = [f"{f.name}{' ✓' if trusted_map.get(f.name) else ''}" if f.name else "unknown"
                  for f in self.seen.values() if now - f.last_seen <= VISIBLE_SECONDS]
        self.visible = ([f"{len(people)} {'person' if len(people) == 1 else 'people'}"] if people else []) + labels
        if moving:
            self.visible.append(f"{len(moving)} moving vehicle{'' if len(moving) == 1 else 's'}")

    def _classify_vehicles(self, vehicles: list[tuple[np.ndarray, int]], now: float) -> tuple[list, list]:
        """Split (box, class) vehicles into moving and parked ones, and remember where each was seen."""
        if now - self._last_vehicle_check > SPOT_MEMORY:  # started, or back after an outage
            self._watching_since = now
        self._last_vehicle_check = now
        warming_up = now - self._watching_since < STILL_SECONDS  # whatever is here already is parked
        moving, parked = [], []
        for v in vehicles:
            spots = [s for s in self._spots if _in_spot(v[0], s.box)]
            if not spots:
                self._spots.append(_Spot(v[0].copy(), now, now))
            still = warming_up or any(now - s.first >= STILL_SECONDS for s in spots)
            (parked if still else moving).append(v)
            for s in spots:
                s.last = now
        self._spots = [s for s in self._spots if now - s.last <= (PARKED_MEMORY if s.parked else SPOT_MEMORY)]
        return moving, parked

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
        labels = dict(BOX_LABELS)
        for f in list(self.seen.values()):  # called from the UI thread while the detector updates it
            labels[f.id] = f"{f.name}{' (trusted)' if trusted_map.get(f.name) else ''}" if f.name else "Unknown person"
        return labels

    def _finalize(self, ev: _Event, now: float) -> EventResult | None:
        face_cfg = self.cfg.face
        trusted_map = self.db.trusted_map()
        box_labels = dict(BOX_LABELS)
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
        if ev.vehicle:  # always a new, untrusted arrival: alerts whoever is in it
            people.append(PersonInfo(VEHICLE_BOX_ID, None, False, f"vehicle:{self.name}:{next(self._vehicles)}",
                                     None, None, True, vehicle=ev.vehicle))
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
