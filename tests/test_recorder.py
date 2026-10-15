import cv2
import numpy as np

from camwatch.camera import BufferedFrame, Overlay
from camwatch.config import ClipConfig
from camwatch.events import EventResult, PersonInfo
from camwatch.recorder import snapshot_jpeg


def frame(ts, shade, box_ids):
    img = np.full((60, 80, 3), shade, np.uint8)
    return BufferedFrame(ts, img, 1.0, Overlay(ts, [(tid, (5, 5, 30, 50)) for tid in box_ids]))


def event(frames):
    alice = PersonInfo(3, "Alice", False, "alice", None, None)
    return EventResult("door", 1_700_000_000.0, "alert", [alice], frames, 0.0, 5.0, 0.0)


def test_snapshot_is_the_latest_frame_showing_the_most_identified_people():
    frames = [frame(1.0, 10, []), frame(2.0, 120, [3]), frame(3.0, 200, [3]), frame(4.0, 250, [99])]
    jpg = snapshot_jpeg(event(frames), ClipConfig(annotate=False))
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (60, 80, 3) and abs(int(img.mean()) - 200) <= 2


def test_snapshot_is_annotated_and_none_without_frames():
    plain = frame(1.0, 0, [3])
    jpg = snapshot_jpeg(event([plain]), ClipConfig(annotate=True))
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.max() > 100  # box and stamp drawn
    assert not plain.image.any()  # the buffered frame itself is untouched
    assert snapshot_jpeg(event([]), ClipConfig()) is None
