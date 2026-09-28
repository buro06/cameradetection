import yaml

from camwatch.config import AppConfig, CameraConfig, load_config


def test_save_marks_changed_settings_with_their_default(tmp_path):
    cfg = AppConfig()
    cfg.face.match_threshold = 0.45
    cfg.clip.annotate = False
    cfg.telegram.chat_ids = [123, -456]
    cfg.cameras = [CameraConfig(name="desk", source="0", width=1920), CameraConfig(name="door", source="rtsp://x/#1")]
    path = tmp_path / "config.yaml"
    cfg.save(path)
    text = path.read_text()

    assert "match_threshold: 0.45  # default: 0.4\n" in text
    assert "annotate: false  # default: true\n" in text
    assert "chat_ids: [123, -456]  # default: []\n" in text
    assert "width: 1920  # default: 0\n" in text
    assert "detector_score: 0.85\n" in text  # unchanged → no comment
    assert "source: '0'\n" in text  # no default → no comment

    loaded = load_config(path)
    assert loaded.to_dict() == cfg.to_dict()
    assert yaml.safe_load(text) == cfg.to_dict()


def test_save_without_cameras_round_trips(tmp_path):
    path = tmp_path / "config.yaml"
    AppConfig().save(path)
    assert load_config(path).to_dict() == AppConfig().to_dict()
