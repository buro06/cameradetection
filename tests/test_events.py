import numpy as np
import pytest

from camwatch.events import CameraMonitor, Tracker, assign_faces
from conftest import FakeFaces, FakeStream, face, unit

FRAME = np.zeros((720, 1280, 3), np.uint8)
LEFT = [100, 100, 400, 700]
RIGHT = [800, 100, 1100, 700]
LEFT_FACE = (200, 120, 300, 230)
RIGHT_FACE = (900, 120, 1000, 230)


def dets(*boxes):
    return np.array([[*b, 0.9] for b in boxes], np.float32).reshape(-1, 5)


def make_monitor(cfg, db, armed=True):
    faces = FakeFaces()
    mon = CameraMonitor(cfg, FakeStream(), faces, db, lambda _cam: armed)
    return mon, faces


def run(mon, boxes, t0, seconds, fps=5.0):
    """Feed the same detections for `seconds`; returns finished event results."""
    results, t = [], t0
    while t < t0 + seconds:
        mon.process(FRAME, dets(*boxes), t)
        if r := mon.tick(t):
            results.append(r)
        t += 1 / fps
    return results, t


def enroll(db, name, emb, trusted):
    pid = db.add_person(name, trusted)
    db.add_embedding(pid, emb, None)


# ---- face-to-person assignment -------------------------------------------------------------
class T:
    def __init__(self, tid, box):
        self.id, self.box = tid, np.array(box, np.float32)


def test_face_goes_to_best_head_position_not_widest_box():
    # Regression: a person's arm-extended box covered another person's face.
    wide, tall = T(1, [120, 202, 1110, 712]), T(2, [737, 40, 1140, 708])
    f_wide, f_tall = face((547, 272, 653, 413), unit(1)), face((922, 154, 1043, 270), unit(2))
    got = {t.id: f for t, f in assign_faces([f_tall, f_wide], [wide, tall])}
    assert got[1] is f_wide and got[2] is f_tall


def test_ambiguous_face_is_dropped():
    a, b = T(1, [100, 100, 400, 700]), T(2, [110, 100, 410, 700])
    assert assign_faces([face((205, 120, 305, 230), unit(1))], [a, b]) == []


def test_face_below_head_region_is_ignored():
    assert assign_faces([face((200, 600, 300, 690), unit(1))], [T(1, LEFT)]) == []


def test_tracker_keeps_identity_while_moving():
    tr = Tracker(ttl=2.0)
    a = tr.update(dets(LEFT), 0.0)[0]
    b = tr.update(dets([130, 100, 430, 700]), 0.2)[0]
    assert a.id == b.id and b.hits == 2
    tr.update(dets(), 3.0)
    assert not tr.tracks


# ---- alert decisions --------------------------------------------------------------------------------
def test_trusted_person_alone_is_suppressed(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, alice)]
    results, _ = run(mon, [LEFT], 0.0, 6)
    assert [r.decision for r in results] == ["trusted"]
    assert results[0].people[0].name == "Alice"


def test_unknown_next_to_trusted_alerts(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, alice), face(RIGHT_FACE, unit(99))]
    results, _ = run(mon, [LEFT, RIGHT], 0.0, 6)
    assert [r.decision for r in results] == ["alert"]
    labels = sorted(p.label for p in results[0].people)
    assert labels[0] == "Alice (trusted)" and labels[1].startswith("Unknown #")
    assert len(db.list_unknowns()) == 1


def test_person_without_visible_face_alerts(cfg, db):
    enroll(db, "Alice", unit(10), trusted=True)
    mon, _ = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 0.0, 6)
    assert results[0].decision == "alert"
    assert results[0].people[0].label == "Unknown person (face not visible)"


def test_single_trusted_match_is_not_enough(cfg, db):
    cfg.face.trusted_min_matches = 3
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, alice)]
    mon.process(FRAME, dets(LEFT), 0.0)
    mon.process(FRAME, dets(LEFT), 0.2)  # 2 matches, then the face turns away
    faces.faces = []
    results, _ = run(mon, [LEFT], 0.4, 6)
    assert results[0].decision == "alert"
    assert results[0].people[0].label == "Alice"  # named, but not confirmed as trusted


def test_mostly_unknown_faces_outvote_a_trusted_match(cfg, db):
    alice = unit(10)
    enroll(db, "Alice", alice, trusted=True)
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, alice)]
    mon.process(FRAME, dets(LEFT), 0.0)
    faces.faces = [face(LEFT_FACE, unit(50))]
    results, _ = run(mon, [LEFT], 0.2, 6)
    assert results[0].decision == "alert"


def test_known_untrusted_person_alerts_with_name(cfg, db):
    bob = unit(20)
    enroll(db, "Bob", bob, trusted=False)
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, bob)]
    results, _ = run(mon, [LEFT], 0.0, 6)
    assert results[0].decision == "alert" and results[0].people[0].label == "Bob"


def test_cooldown_per_person_and_realert_after_expiry(cfg, db):
    cfg.events.cooldown_seconds = 30
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, unit(99))]
    results, t = run(mon, [LEFT], 0.0, 25)  # lingering unknown: one alert, no event spam
    assert [r.decision for r in results] == ["alert"]
    results, t = run(mon, [LEFT], t, 15)  # cooldown expired while still present
    assert [r.decision for r in results] == ["alert"]


def test_new_person_during_cooldown_still_alerts(cfg, db):
    mon, faces = make_monitor(cfg, db)
    faces.faces = [face(LEFT_FACE, unit(99))]
    results, t = run(mon, [LEFT], 0.0, 8)
    assert [r.decision for r in results] == ["alert"]
    faces.faces.append(face(RIGHT_FACE, unit(77)))
    results, _ = run(mon, [LEFT, RIGHT], t, 8)
    assert [r.decision for r in results] == ["alert"]
    assert len(results[0].people) == 2


def test_disarmed_camera_does_not_alert(cfg, db):
    mon, _ = make_monitor(cfg, db, armed=False)
    results, _ = run(mon, [LEFT], 0.0, 6)
    assert results[0].decision == "disarmed"


def test_single_frame_false_positive_is_ignored(cfg, db):
    mon, _ = make_monitor(cfg, db)
    mon.process(FRAME, dets(LEFT), 0.0)
    results, _ = run(mon, [], 0.2, 8)
    assert results == []


def test_default_clip_is_five_seconds_from_first_detection(cfg, db):
    assert (cfg.events.pre_seconds, cfg.events.post_seconds) == (0.0, 5.0)
    mon, _ = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 10.0, 7)  # first detection at t=10.0, confirmed at t=10.2
    r = results[0]
    assert (r.start, r.anchor, r.end) == pytest.approx((10.0, 10.0, 15.0))


def test_ring_buffer_keeps_frames_before_confirmation_with_zero_pre_roll():
    from camwatch.camera import BufferedFrame, CameraStream
    from camwatch.config import CameraConfig, ClipConfig

    clip = ClipConfig()
    stream = CameraStream(CameraConfig(name="c", source="0"), clip, pre_seconds=0.0)
    for i in range(int(3 * clip.fps)):  # 3 s of buffered frames, 0..3 s
        stream._ring.append(BufferedFrame(i / clip.fps, FRAME, 1.0, None))
    rec = stream.start_recording(since=2.0)  # person first seen 1 s before confirmation at t=3
    assert rec.frames and rec.frames[0].ts == pytest.approx(2.0)


def test_clip_window_covers_pre_and_post_roll(cfg, db):
    cfg.events.pre_seconds, cfg.events.post_seconds = 2.0, 3.0
    mon, _ = make_monitor(cfg, db)
    results, _ = run(mon, [LEFT], 10.0, 6)
    r = results[0]
    assert r.end - r.start == pytest.approx(cfg.events.pre_seconds + cfg.events.post_seconds)
    assert r.anchor - r.start == pytest.approx(cfg.events.pre_seconds)
