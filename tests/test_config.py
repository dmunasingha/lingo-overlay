import json

import pytest

from translator import HERE, Segmenter, load_config

REQUIRED_KEYS = {
    "input_device", "fallback_to_speaker_loopback",
    "skip_languages", "skip_threshold",
    "model", "device", "compute_type", "cpu_threads",
    "vad_silence_ms", "vad_max_segment_s", "vad_min_speech_ms", "vad_threshold_db",
    "voice_enabled", "voice_rate", "voice_volume",
    "overlay_enabled", "overlay_font_size", "overlay_seconds", "overlay_bottom_offset", "overlay_opacity",
    "hotkey_toggle", "hotkey_voice", "hotkey_replay", "hotkey_quit",
}


def test_shipped_config_loads_with_all_keys():
    missing = REQUIRED_KEYS - load_config().keys()
    assert not missing, f"config.json is missing keys: {sorted(missing)}"


def test_shipped_config_values_are_sane():
    cfg = load_config()
    assert all(isinstance(s, str) for s in cfg["skip_languages"])
    assert 0 <= cfg["skip_threshold"] <= 1
    assert 0 < cfg["overlay_opacity"] <= 1
    assert 0 <= cfg["voice_volume"] <= 1
    assert cfg["vad_silence_ms"] > 0 and cfg["vad_min_speech_ms"] > 0
    assert cfg["vad_max_segment_s"] * 1000 > cfg["vad_silence_ms"]


def test_shipped_config_builds_a_segmenter():
    seg = Segmenter(load_config(), out_q=None)
    assert seg.silence_frames > 0 and seg.max_frames > seg.min_speech_frames


def test_default_path_is_next_to_translator():
    assert load_config() == json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def test_custom_path_and_utf8(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"input_device": "Mikrofon (Größe)"}, ensure_ascii=False), encoding="utf-8")
    assert load_config(p) == {"input_device": "Mikrofon (Größe)"}


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_config(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")
