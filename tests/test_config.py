import json

import pytest

from lingo_overlay import ROOT
from lingo_overlay.config import DEFAULT_PATH, Config, ConfigError, load_config
from lingo_overlay.vad import Segmenter

REQUIRED_KEYS = {
    "input_device", "fallback_to_speaker_loopback",
    "skip_languages", "skip_threshold",
    "model", "device", "compute_type", "cpu_threads",
    "vad_silence_ms", "vad_max_segment_s", "vad_min_speech_ms", "vad_threshold_db",
    "voice_enabled", "voice_rate", "voice_volume",
    "overlay_enabled", "overlay_font_size", "overlay_seconds", "overlay_bottom_offset", "overlay_opacity",
    "hotkey_toggle", "hotkey_voice", "hotkey_replay", "hotkey_quit",
}


def write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def shipped():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def test_shipped_config_has_all_keys():
    missing = REQUIRED_KEYS - shipped().keys()
    assert not missing, f"config.json is missing keys: {sorted(missing)}"


def test_shipped_config_sets_only_known_keys():
    assert set(shipped()) <= set(Config())


def test_shipped_config_values_are_sane():
    cfg = load_config()
    assert all(isinstance(s, str) for s in cfg.skip_languages)
    assert cfg.vad_max_segment_s * 1000 > cfg.vad_silence_ms


def test_shipped_config_builds_a_segmenter():
    seg = Segmenter(load_config(), out_q=None)
    assert seg.silence_frames > 0 and seg.max_frames > seg.min_speech_frames


def test_defaults_match_shipped_config():
    assert Config.from_dict(shipped()) == Config()


def test_default_path_is_next_to_translator():
    assert DEFAULT_PATH == ROOT / "config.json" and (ROOT / "translator.py").is_file()
    assert load_config() == Config.from_dict(shipped())


def test_custom_path_and_utf8(tmp_path):
    cfg = load_config(write(tmp_path, {"input_device": "Mikrofon (Größe)"}))
    assert cfg == Config(input_device="Mikrofon (Größe)")


def test_missing_keys_get_defaults(tmp_path):
    assert load_config(write(tmp_path, {})) == Config()


def test_values_are_kept_as_written(tmp_path):
    cfg = load_config(write(tmp_path, {"vad_silence_ms": 500, "skip_languages": ["EN"]}))
    assert cfg.vad_silence_ms == 500 and isinstance(cfg.vad_silence_ms, int)
    assert cfg.skip_languages == ["EN"]  # lower-cased where it is used, as before


def test_config_is_a_read_only_mapping():
    cfg = Config(model="tiny")
    assert cfg["model"] == "tiny" and cfg.get("model") == "tiny"
    assert cfg.get("nope", 1) == 1
    assert dict(cfg, model="base")["model"] == "base"
    assert {**cfg}["vad_backend"] == "silero"
    with pytest.raises(KeyError):
        cfg["nope"]
    with pytest.raises(AttributeError):
        cfg.model = "base"
    assert cfg.replace(model="base").model == "base"


def test_int_accepted_for_float_field():
    assert Config(skip_threshold=1).skip_threshold == 1


@pytest.mark.parametrize("spec", ["off", "last:4", [0, 1], None])
def test_cpu_affinity_shapes_are_accepted(spec):
    assert Config(cpu_affinity=spec).cpu_affinity == spec


@pytest.mark.parametrize("key,value,needle", [
    ("cpu_threads", "6", "'cpu_threads' must be a whole number"),
    ("cpu_threads", 6.5, "'cpu_threads' must be a whole number"),
    ("voice_enabled", "yes", "'voice_enabled' must be true or false"),
    ("vad_silence_ms", True, "'vad_silence_ms' must be a number"),
    ("vad_silence_ms", 0, "'vad_silence_ms' must be > 0"),
    ("skip_languages", "en", "'skip_languages' must be a list of strings"),
    ("skip_threshold", 1.5, "'skip_threshold' must be between 0 and 1"),
    ("min_language_prob", -0.1, "'min_language_prob' must be between 0 and 1"),
    ("overlay_opacity", -0.1, "'overlay_opacity' must be between 0 and 1"),
    ("backend", "whisper", "'backend' must be"),
    ("device", "gpu", "'device' must be"),
    ("vad_threshold_db", 10, "'vad_threshold_db' must be <= 0"),
    ("hotkey_quit", " ", "'hotkey_quit' must not be empty"),
    ("cpu_affinity", 3, "'cpu_affinity' must be"),
])
def test_invalid_values_raise_clear_errors(tmp_path, key, value, needle):
    with pytest.raises(ConfigError, match="config.json") as ex:
        load_config(write(tmp_path, {key: value}))
    assert needle in str(ex.value)


def test_unknown_backends_are_left_to_the_components():
    # These fall back with a log message where they are used, as before.
    assert Config(vad_backend="nope", lid_backend="nope", process_priority="nope").vad_backend == "nope"


def test_unknown_key_warns_with_suggestion(tmp_path, capsys):
    cfg = load_config(write(tmp_path, {"modle": "base"}))
    assert cfg.model == Config().model
    assert "did you mean 'model'" in capsys.readouterr().out


def test_invalid_json_raises(tmp_path):
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(write(tmp_path, "{not json"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")


def test_top_level_must_be_object(tmp_path):
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(write(tmp_path, "[1, 2]"))
