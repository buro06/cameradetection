import cv2
import numpy as np
import pytest

from conftest import face, unit


def noisy(v, seed, amount=0.3):
    n = v + amount * np.random.default_rng(seed).normal(size=v.shape).astype(np.float32) / np.sqrt(v.size)
    return n / np.linalg.norm(n)


def test_match_known_person(db):
    alice = unit(1)
    pid = db.add_person("Alice", trusted=True)
    db.add_embedding(pid, alice, None)
    m = db.match(noisy(alice, 5))
    assert m and m.name == "Alice" and m.trusted
    assert db.match(unit(2)) is None


def test_unknown_clusters_merge_similar_faces(db):
    crop = np.zeros((160, 160, 3), np.uint8)
    stranger = unit(3)
    a = db.record_unknown(stranger, crop, 0.9, "cam")
    b = db.record_unknown(noisy(stranger, 7), crop, 0.9, "cam")
    c = db.record_unknown(unit(4), crop, 0.9, "cam")
    assert a == b != c
    assert db.list_unknowns()[-1]["sightings"] in (1, 2)
    assert len(db.unknown_images(a)) == 2


def test_assign_unknown_moves_faces_to_person(db):
    crop = np.zeros((160, 160, 3), np.uint8)
    stranger = unit(3)
    uid = db.record_unknown(stranger, crop, 0.9, "cam")
    p = db.assign_unknown(uid, "Carol", trusted=True)
    assert p["trusted"] and p["samples"] == 1
    assert not db.unknown_exists(uid)
    assert db.match(noisy(stranger, 9)).name == "Carol"


def test_assign_to_existing_person_keeps_trust(db):
    pid = db.add_person("Dave", trusted=True)
    uid = db.record_unknown(unit(5), np.zeros((10, 10, 3), np.uint8), 0.9, "cam")
    p = db.assign_unknown(uid, "dave")  # case-insensitive
    assert p["id"] == pid and p["trusted"] and p["samples"] == 1


def test_delete_person_removes_matching(db):
    pid = db.add_person("Eve")
    db.add_embedding(pid, unit(6), None)
    db.delete_person(pid)
    assert db.match(unit(6)) is None


class PixelFaces:
    """Fake FaceEngine: an image's top-left pixel value picks the embedding; 0 = no face, 255 = two faces."""

    class cfg:
        detector_score = 0.85

    def analyze(self, img, region=None, max_faces=1, min_score=None):
        v = int(img[0, 0, 0])
        if v == 0:
            return []
        n = 2 if v == 255 else 1
        return [face((0, 0, 100, 100), unit(v)) for _ in range(n)][:max_faces]


def photo(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((40, 40, 3), value, np.uint8))


def test_rebuild_follows_photos_moved_between_folders(db, tmp_path):
    a, b = db.add_person("Alice"), db.add_person("Bob")
    db.add_embedding(a, unit(10), None)  # Bob's face wrongly stored under Alice, without a photo
    photo(tmp_path / "faces" / str(a) / "alice.png", 20)
    photo(tmp_path / "faces" / str(b) / "moved-from-alice.jpg", 10)  # the user moved Bob's photo here
    photo(tmp_path / "faces" / str(b) / "blank.jpg", 0)
    photo(tmp_path / "faces" / str(b) / "group.jpg", 255)
    photo(tmp_path / "faces" / "99" / "stray.jpg", 30)
    (tmp_path / "faces" / "Thumbs.db").write_bytes(b"x")

    plan = db.plan_rebuild(PixelFaces())
    assert plan.before == {a: 1, b: 0}
    assert [p.name for p in plan.no_face] == ["blank.jpg"]
    assert [p.name for p in plan.multi_face] == ["group.jpg"]
    assert [p.name for p in plan.stray] == ["99"]
    db.apply_rebuild(plan)

    assert db.match(unit(10)).name == "Bob"
    assert db.match(unit(20)).name == "Alice"
    assert {p["name"]: p["samples"] for p in db.list_persons()} == {"Alice": 1, "Bob": 1}
    assert (tmp_path / "faces" / str(b) / "blank.jpg").exists()  # skipped photos are kept


def test_rebuild_refuses_if_samples_added_while_scanning(db, tmp_path):
    a = db.add_person("Alice")
    photo(tmp_path / "faces" / str(a) / "alice.jpg", 20)
    plan = db.plan_rebuild(PixelFaces())
    db.add_embedding(a, unit(21), None)
    with pytest.raises(RuntimeError):
        db.apply_rebuild(plan)
