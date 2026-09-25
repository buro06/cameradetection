import logging

from camwatch.config import load_config


def test_removed_settings_are_ignored_with_a_warning(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("detection:\n  min_hits: 2\n  confidence: 0.6\nface:\n  quality_min_px: 64\n")
    with caplog.at_level(logging.WARNING, logger="camwatch.config"):
        cfg = load_config(path)
    assert cfg.detection.confidence == 0.6
    assert "min_hits" in caplog.text and "quality_min_px" in caplog.text
    cfg.save()
    assert "min_hits" not in path.read_text()
