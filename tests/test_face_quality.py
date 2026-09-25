import cv2
import numpy as np
import pytest

from camwatch.config import FaceConfig
from camwatch.faces import face_issue, head_turn, sharpness


def landmarks(nose_x: float) -> np.ndarray:
    row = np.zeros(15, np.float32)
    row[4:6], row[6:8], row[8:10] = (40, 50), (80, 50), (nose_x, 70)  # right eye, left eye, nose
    return row


@pytest.mark.parametrize("nose_x, turn", [(60, 0.0), (70, 0.25), (80, 0.5), (50, 0.25)])
def test_head_turn_is_nose_offset_in_eye_distances(nose_x, turn):
    assert head_turn(landmarks(nose_x)) == pytest.approx(turn)


def test_blur_lowers_sharpness():
    chip = (np.random.default_rng(0).random((112, 112, 3)) * 255).astype(np.uint8)
    assert sharpness(cv2.GaussianBlur(chip, (9, 9), 0)) < sharpness(chip) / 10


def test_face_issue_checks_blur_and_turn():
    cfg = FaceConfig()
    assert face_issue(cfg, 80, 0.1) == ""
    assert face_issue(cfg, 10, 0.1) == "blurry"
    assert face_issue(cfg, 80, 0.7) == "turned away"
