import yaml

from camwatch.config import AppConfig, CameraConfig, load_config


def test_save_marks_changed_settings_with_their_default(tmp_path):
    cfg = AppConfig()
    cfg.face.match_threshold = 0.42
    cfg.clip.annotate = False
    cfg.telegram.chat_ids = [123, -456]
    cfg.cameras = [CameraConfig(name="desk", source="0", width=1920), CameraConfig(name="door", source="rtsp://x/#1")]
    path = tmp_path / "config.yaml"
    cfg.save(path)
    text = path.read_text()

    assert "match_threshold: 0.42  # default: 0.35\n" in text
    assert "annotate: false  # default: true\n" in text
    assert "chat_ids: [123, -456]  # default: []\n" in text
    assert "width: 1920  # default: 0\n" in text
    assert "detector_score: 0.5\n" in text  # unchanged → no comment
    assert "source: '0'\n" in text  # no default → no comment

    loaded = load_config(path)
    assert loaded.to_dict() == cfg.to_dict()
    assert yaml.safe_load(text) == cfg.to_dict()


def test_save_without_cameras_round_trips(tmp_path):
    path = tmp_path / "config.yaml"
    AppConfig().save(path)
    assert load_config(path).to_dict() == AppConfig().to_dict()


def test_legacy_sface_thresholds_are_reset_for_insightface(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("face:\n  detector_score: 0.85\n  match_threshold: 0.4\n  min_face_px: 50\n")
    cfg = load_config(path)
    assert (cfg.face.model, cfg.face.detector_score, cfg.face.match_threshold) == ("buffalo_l", 0.5, 0.35)
    assert cfg.face.min_face_px == 50  # model-independent settings are kept
    cfg.save(path)
    assert load_config(path).face.match_threshold == 0.35
    path.write_text("face:\n  model: buffalo_l\n  match_threshold: 0.4\n")
    assert load_config(path).face.match_threshold == 0.4  # already migrated: user's value stays


def test_settings_of_the_old_tracker_are_dropped(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("detection:\n  min_hits: 3\n  min_box_height: 0.1\n  overlap_confirm_seconds: 1.0\n  confidence: 0.6\n"
                    "events:\n  track_ttl: 2.0\n  lost_memory_seconds: 30\n  handoff_seconds: 1.5\n  cooldown_seconds: 90\n"
                    "face:\n  model: buffalo_l\n  trusted_min_matches: 3\n")
    cfg = load_config(path)
    assert (cfg.detection.confidence, cfg.events.cooldown_seconds, cfg.face.min_sightings) == (0.6, 90, 3)
    assert "events.handoff_seconds" in caplog.text
    cfg.save(path)
    assert "min_hits" not in path.read_text() and load_config(path).to_dict() == cfg.to_dict()


def test_camera_leave_time_round_trips(tmp_path):
    cfg = AppConfig()
    cfg.cameras = [CameraConfig(name="porch", source="rtsp://x", leave_seconds=10.0), CameraConfig(name="den", source="0")]
    path = tmp_path / "config.yaml"
    cfg.save(path)
    loaded = load_config(path)
    assert [c.leave_seconds for c in loaded.cameras] == [10.0, None]
