import numpy as np
import pytest

from camwatch.events import CameraMonitor, solid_people
from conftest import FakeFaces, FakeStream, face, unit

FRAME = np.zeros((720, 1280, 3), np.uint8)
LEFT = [100, 100, 400, 700]
RIGHT = [800, 100, 1100, 700]
NESTED = [150, 330, 360, 660]  # torso-only box inside LEFT (YOLO boxing the same person twice)
SMALL = [600, 300, 660, 380]  # far away: too small to count as someone arriving
SMALL_FACE = (615, 305, 645, 335)
LEFT_FACE = (200, 120, 300, 230)
RIGHT_FACE = (900, 120, 1000, 230)
FPS = 5.0


def dets(*boxes):
    return np.array([[*b, 0.9] for b in boxes], np.float32).reshape(-1, 5)


def make_monitor(cfg, db, armed=True):
    faces = FakeFaces()
    mon = CameraMonitor(cfg, FakeStream(), faces, db, lambda _cam: armed)
    return mon, faces


def run(mon, boxes, t0, seconds, faces=None, face_list=None):
    """Feed the same detections (and faces, if given) for `seconds`; returns finished events and the end time."""
    if faces is not None and face_list is not None:
        faces.faces = face_list
    results, t = [], t0
    while t < t0 + seconds - 1e-6:
        mon.process(FRAME, dets(*boxes), t)
        if r := mon.tick(t):
            results.append(r)
        t = round(t + 1 / FPS, 6)
    return results, t


def scripted(mon, faces, t0, seconds, scene):
    """Like run(), but `scene(t)` returns (boxes, faces) for each detection."""
    results, t = [], t0
    while t < t0 + seconds - 1e-6:
        boxes, faces.faces = scene(t)
        mon.process(FRAME, dets(*boxes), t)
        if r := mon.tick(t):
            results.append(r)
        t = round(t + 1 / FPS, 6)
    return results, t


def enroll(db, name, emb, trusted):
    pid = db.add_person(name, trusted)
    db.add_embedding(pid, emb, None)


def near(base: np.ndarray, similarity: float, seed: int) -> np.ndarray:
    """A unit vector with exactly `similarity` cosine to `base` (the same face at a worse angle)."""
    other = unit(seed)
    other = other - (other @ base) * base
    other /= np.linalg.norm(other)
    return similarity * base + np.sqrt(1 - similarity ** 2) * other


# ---- people ---------------------------------------------------------------------------------------
def test_edge_cut_and_small_boxes_do_not_count():
    h, w = FRAME.shape[:2]
    arm_at_edge = [0, 300, 120, 500]  # cut off at the left edge and short
    tall_at_edge = [0, 100, 250, 700]  # someone close to the camera, half in frame
    kept = solid_people(dets(arm_at_edge, SMALL, tall_at_edge, LEFT), w, h, 0.15)
    assert sorted(b.tolist() for b in kept) == sorted([tall_at_edge, LEFT])


def test_arm_at_frame_edge_is_not_an_arrival(cfg, db):
    mon, _ = make_monitor(cfg, db)
    results, _ = run(mon, [[0, 300, 120, 500]], 0.0, 10)
    assert results == [] and mon.occupied_since is None


def test_single_frame_false_positive_is_ignored(cfg, db):
    mon, _ = make_monitor(cfg, db)
    mon.process(FRAME, dets(LEFT), 0.0)
    results, _ = run(mon, [], 0.2, 8)
    assert results == []


# ---- who arrived ----------------------------------------------------------------------------------
def test_trusted_person_alone_is_suppressed(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, alice)])
    assert [r.decision for r in results] == ["trusted"]
    assert [p.name for p in results[0].people] == ["Alice"]  # her face says who arrived


def test_unknown_next_to_trusted_alerts(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT, RIGHT], 0.0, 6, faces, [face(LEFT_FACE, alice), face(RIGHT_FACE, unit(99))])
    assert [r.decision for r in results] == ["alert"]
    labels = sorted(p.label for p in results[0].people)
    assert labels[0] == "Alice (trusted)" and labels[1].startswith("Unknown #")
    assert len(db.list_unknowns()) == 1


def test_masked_person_alerts_as_face_not_visible(cfg, db):
    enroll(db, "Alice", unit(10), trusted=True)
    mon, _ = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 0.0, 6)
    assert results[0].decision == "alert"
    assert [p.label for p in results[0].people] == ["Person (face not visible)"]


def test_unclear_faces_are_ignored(cfg, db):
    mon, faces = make_monitor(cfg, db)
    turned = face(LEFT_FACE, unit(50))
    turned.issue = "turned away"
    results, _ = run(mon, [LEFT], 0.0, 6, faces, [turned])
    assert [p.label for p in results[0].people] == ["Person (face not visible)"]
    assert db.list_unknowns() == []
    assert results[0].labels[-1] == "unclear face"


def test_face_needs_enough_sightings(cfg, db):
    cfg.face.min_sightings = 3
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, alice)]
    run(mon, [LEFT], 0.0, 0.8)  # 2 face scans, then the face turns away
    results, _ = run(mon, [LEFT], 0.8, 6, faces, [])
    assert results[0].decision == "alert"
    assert [p.label for p in results[0].people] == ["Person (face not visible)"]


def test_unknown_face_seen_once_is_dropped(cfg, db):
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(SMALL_FACE, unit(50))]
    mon.process(FRAME, dets(SMALL), 0.0)
    results, _ = run(mon, [], 0.2, 8, faces, [])
    assert results == [] and db.list_unknowns() == []


def test_face_outside_every_person_box_is_not_looked_at(cfg, db):
    mon, faces = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 0.0, 6, faces, [face(RIGHT_FACE, unit(50))])
    assert [p.label for p in results[0].people] == ["Person (face not visible)"]


def test_known_untrusted_person_alerts_with_name(cfg, db):
    bob = unit(20)
    enroll(db, "Bob", bob, trusted=False)
    mon, faces = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, bob)])
    assert results[0].decision == "alert" and [p.label for p in results[0].people] == ["Bob"]


def test_clip_and_alert_labels_show_match_percent(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=False)
    mon, faces = make_monitor(cfg, db)
    seen = near(alice, 0.8, 11)
    results, _ = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, seen)])
    person = results[0].people[0]
    assert person.label == "Alice"  # logs and dashboard stay unchanged
    assert person.scored_label == "Alice 80%" and results[0].labels[person.track_id] == "Alice 80%"
    assert results[0].labels[0] == "person"  # body boxes are drawn neutral


def test_same_face_twice_and_double_boxed_is_one_person(cfg, db):
    mon, faces = make_monitor(cfg, db)
    me = unit(30)
    results, _ = run(mon, [LEFT, NESTED], 0.0, 6, faces, [face(LEFT_FACE, me), face((210, 125, 305, 232), me)])
    assert [r.decision for r in results] == ["alert"]
    assert len(results[0].people) == 1 and results[0].people[0].label.startswith("Unknown #")


def test_two_strangers_arriving_together_alert_once_listing_both(cfg, db):
    mon, faces = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT, RIGHT], 0.0, 6, faces, [face(LEFT_FACE, unit(40)), face(RIGHT_FACE, unit(41))])
    assert [r.decision for r in results] == ["alert"]
    assert len(results[0].people) == 2


def test_lookalike_of_a_known_person_is_a_stranger(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, near(alice, db.threshold - 0.04, 12))])
    assert results[0].decision == "alert" and results[0].people[0].label.startswith("Unknown #")


# ---- the 30 second rule ---------------------------------------------------------------------------
def test_couch_sitter_turning_away_for_ten_minutes_never_realerts(cfg, db):
    mon, faces = make_monitor(cfg, db)
    stranger = unit(60)

    def scene(t):  # sits on the couch the whole time; face visible only now and then
        visible = t < 6 or (t % 20 < 1 and t < 100) or t > 700
        return [LEFT], [face(LEFT_FACE, stranger)] if visible else []

    results, _ = scripted(mon, faces, 0.0, 760, scene)
    assert [r.decision for r in results] == ["alert"]


def test_walking_around_double_boxed_and_occluded_alerts_once(cfg, db):
    mon, faces = make_monitor(cfg, db)

    def scene(t):
        x = int(100 + 25 * (t % 30))  # walks across and back
        box = [x, 100, x + 300, 700]
        if 10 <= t % 15 < 11:  # behind furniture for a second
            return [], []
        if int(t * 5) % 7 == 0:  # YOLO boxes them twice now and then
            return [box, [x + 50, 330, x + 260, 660]], []
        return [box], []

    results, _ = scripted(mon, faces, 0.0, 60, scene)
    assert [r.decision for r in results] == ["alert"]
    assert len(results[0].people) == 1


@pytest.mark.parametrize("gone, alerts", [(20, ["alert"]), (40, ["alert", "alert"])])
def test_masked_person_returning_alerts_only_after_30s_away(cfg, db, gone, alerts):
    mon, _ = make_monitor(cfg, db)
    results, t = run(mon, [LEFT], 0.0, 6)
    more, t = run(mon, [], t, gone)
    back, _ = run(mon, [LEFT], t, 6)
    assert [r.decision for r in results + more + back] == alerts


@pytest.mark.parametrize("gone, alerts", [(20, ["alert"]), (40, ["alert", "alert"])])
def test_stranger_returning_alerts_only_after_30s_away(cfg, db, gone, alerts):
    mon, faces = make_monitor(cfg, db)
    stranger = unit(80)
    first, t = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, stranger)])
    _, t = run(mon, [], t, gone, faces, [])
    back, _ = run(mon, [LEFT], t, 6, faces, [face(LEFT_FACE, stranger)])
    assert [r.decision for r in first + back] == alerts
    assert all(r.people[0].key == first[0].people[0].key for r in back)  # recognised as the same unknown


def test_trusted_person_coming_back_is_trusted(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    results, t = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, alice)])
    _, t = run(mon, [], t, 70, faces, [])  # left the room
    back, _ = run(mon, [LEFT], t, 6, faces, [face(LEFT_FACE, alice)])
    assert [r.decision for r in results + back] == ["trusted", "trusted"]


def test_new_face_joining_someone_present_alerts(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    results, t = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, alice)])
    _, t = run(mon, [LEFT], t, 30, faces, [])  # Alice sits facing away
    later, _ = run(mon, [LEFT, RIGHT], t, 6, faces, [face(RIGHT_FACE, unit(70))])
    assert [r.decision for r in results + later] == ["trusted", "alert"]
    assert [p.label for p in later[0].people if p.new][0].startswith("Unknown #")


def test_masked_person_joining_someone_present_is_not_reported(cfg, db):
    # accepted gap: without a face, only an arrival in an empty camera is noticed
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    results, t = run(mon, [LEFT], 0.0, 6, faces, [face(LEFT_FACE, alice)])
    later, _ = run(mon, [LEFT, RIGHT], t, 10, faces, [])
    assert [r.decision for r in results] == ["trusted"] and later == []


def test_disarmed_camera_does_not_alert(cfg, db):
    mon, _ = make_monitor(cfg, db, armed=False)
    results, _ = run(mon, [LEFT], 0.0, 6)
    assert results[0].decision == "disarmed"


def test_camera_outage_longer_than_30s_counts_as_empty(cfg, db):
    mon, _ = make_monitor(cfg, db)
    results, t = run(mon, [LEFT], 0.0, 6)
    again, _ = run(mon, [LEFT], t + cfg.events.away_seconds + 10, 6)  # stream was down meanwhile
    assert [r.decision for r in results + again] == ["alert", "alert"]


# ---- face scan cost --------------------------------------------------------------------------------
def test_no_face_scans_while_nobody_is_in_view(cfg, db):
    mon, faces = make_monitor(cfg, db)
    run(mon, [], 0.0, 10)
    assert faces.calls == []


def test_faces_are_looked_for_in_each_person_box(cfg, db):
    mon, faces = make_monitor(cfg, db)
    run(mon, [LEFT], 0.0, 2)
    assert len(faces.calls) == 4  # check_fps 2/s
    assert all(region == tuple(map(float, LEFT)) and size == 320 for region, size in faces.calls)


def test_a_crowd_gets_one_whole_frame_scan(cfg, db):
    mon, faces = make_monitor(cfg, db)
    crowd = [[100 + 200 * i, 100, 280 + 200 * i, 700] for i in range(5)]
    run(mon, crowd, 0.0, 0.2)
    assert faces.calls == [(None, None)]


# ---- clip window ---------------------------------------------------------------------------------
def test_default_clip_is_five_seconds_from_first_detection(cfg, db):
    assert (cfg.events.pre_seconds, cfg.events.post_seconds) == (0.0, 5.0)
    mon, _ = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 10.0, 9)  # first detection at t=10, arrival known at t=12
    r = results[0]
    assert (r.start, r.anchor, r.end) == pytest.approx((10.0, 10.0, 15.0))


def test_clip_window_covers_pre_and_post_roll(cfg, db):
    cfg.events.pre_seconds, cfg.events.post_seconds = 2.0, 3.0
    mon, _ = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 10.0, 8)
    r = results[0]
    assert r.end - r.start == pytest.approx(cfg.events.pre_seconds + cfg.events.post_seconds)
    assert r.anchor - r.start == pytest.approx(cfg.events.pre_seconds)


def test_ring_buffer_keeps_frames_from_before_the_arrival_was_confirmed():
    from camwatch.camera import BufferedFrame, CameraStream
    from camwatch.config import CameraConfig, ClipConfig

    clip = ClipConfig()
    stream = CameraStream(CameraConfig(name="c", source="0"), clip, pre_seconds=2.0)  # pre_seconds + arrive_seconds
    for i in range(int(4 * clip.fps)):  # 4 s of buffered frames, 0..4 s
        stream._ring.append(BufferedFrame(i / clip.fps, FRAME, 1.0, None))
    rec = stream.start_recording(since=2.0)  # person first seen 2 s before the arrival was confirmed at t=4
    assert rec.frames and rec.frames[0].ts == pytest.approx(2.0)
