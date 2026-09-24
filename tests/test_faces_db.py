import numpy as np

from conftest import unit


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
