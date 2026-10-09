"""Real SCRFD + ArcFace models; skipped unless they were downloaded to data/models (run camwatch once)."""

from pathlib import Path

import numpy as np
import pytest

from camwatch.config import FaceConfig

MODELS = Path(__file__).resolve().parent.parent / "data" / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "buffalo_l" / "w600k_r50.onnx").exists(),
                                reason="face models not downloaded")


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("onnxruntime")
    from camwatch.faces import FaceEngine
    return FaceEngine(MODELS, FaceConfig(device="cpu"))


@pytest.fixture(scope="module")
def zidane():
    import cv2
    from ultralytics.utils import ASSETS
    return cv2.imread(str(ASSETS / "zidane.jpg"))


def test_finds_faces_and_flags_the_profile_one(engine, zidane):
    faces = engine.analyze(zidane, None, max_faces=5)
    assert len(faces) == 2
    frontal, profile = sorted(faces, key=lambda f: -f.box[0])
    assert frontal.good and frontal.embedding.shape == (512,)
    assert profile.issue == "turned away"


def test_same_face_matches_across_crops_and_scales(engine, zidane):
    import cv2
    a = engine.analyze(zidane, None, max_faces=2)
    b = engine.analyze(cv2.resize(zidane[50:, 100:], None, fx=0.7, fy=0.7), None, max_faces=2)
    sims = np.array([[x.embedding @ y.embedding for y in b] for x in a])
    assert sims.diagonal().min() > 0.8 and sims[0, 1] < 0.3 and sims[1, 0] < 0.3


def test_region_limits_the_search_to_one_person(engine, zidane):
    faces = engine.analyze(zidane, (850, 40, 1150, 720), max_faces=3)
    assert len(faces) == 1 and faces[0].box[0] >= 850


def test_alignment_puts_landmarks_on_the_template():
    from camwatch.faces import ARCFACE_TEMPLATE, _similarity_transform
    kps = ARCFACE_TEMPLATE * 2.5 + [300, 120]
    m = _similarity_transform(kps.astype(np.float64), ARCFACE_TEMPLATE.astype(np.float64))
    mapped = kps @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, ARCFACE_TEMPLATE, atol=1e-3)
