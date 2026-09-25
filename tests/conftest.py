import numpy as np
import pytest

from camwatch.config import AppConfig, CameraConfig
from camwatch.faces import FaceDB, FaceObs


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


def face(box, emb, quality=0.95) -> FaceObs:
    return FaceObs(box=box, quality=quality, embedding=emb, crop=np.zeros((160, 160, 3), np.uint8))


class FakeStream:
    def __init__(self, name="cam"):
        self.cfg = CameraConfig(name=name, source="0")
        self.overlay = None

    def set_overlay(self, o):
        self.overlay = o

    def start_recording(self, since):
        from camwatch.camera import Recording
        return Recording()

    def stop_recording(self, rec):
        pass


class FakeFaces:
    """Returns scripted faces for any person region that contains them."""

    def __init__(self):
        self.faces: list[FaceObs] = []

    def analyze(self, frame, region=None, max_faces=1):
        out = []
        for f in self.faces:
            cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
            if region is None or (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                out.append(f)
        return out[:max_faces]


@pytest.fixture
def db(tmp_path):
    return FaceDB(tmp_path, match_threshold=0.4)


@pytest.fixture
def cfg(tmp_path):
    c = AppConfig(data_dir=str(tmp_path))
    c.events.cooldown_seconds = 60
    return c
