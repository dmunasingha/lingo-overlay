import json

import pytest

from lingo_overlay import ROOT
from lingo_overlay.config import Config, ConfigError, load_config


def write(tmp_path, data):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return p


def test_shipped_config_loads():
    cfg = load_config(ROOT / "config.json")
    assert cfg.model
    assert cfg.skip_languages == ["en"]


def test_defaults_match_shipped_config():
    shipped = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert Config.from_dict(shipped) == Config()


def test_missing_keys_use_defaults(tmp_path):
    cfg = load_config(write(tmp_path, {"model": "small"}))
    assert cfg.model == "small"
    assert cfg.vad_silence_ms == Config().vad_silence_ms


def test_skip_languages_are_lowercased():
    assert Config(skip_languages=["EN", "Pt"]).skip_languages == ["en", "pt"]


def test_int_accepted_for_float_field():
    assert Config(skip_threshold=1).skip_threshold == 1


@pytest.mark.parametrize("key,value,needle", [
    ("cpu_threads", "6", "'cpu_threads' must be a whole number"),
    ("cpu_threads", 6.5, "'cpu_threads' must be a whole number"),
    ("voice_enabled", "yes", "'voice_enabled' must be true or false"),
    ("vad_silence_ms", True, "'vad_silence_ms' must be a whole number"),
    ("skip_languages", "en", "'skip_languages' must be a list of strings"),
    ("skip_threshold", 1.5, "'skip_threshold' must be between 0 and 1"),
    ("overlay_opacity", -0.1, "'overlay_opacity' must be between 0 and 1"),
    ("device", "gpu", "'device' must be"),
    ("vad_threshold_db", 10, "'vad_threshold_db' must be <= 0"),
    ("hotkey_quit", " ", "'hotkey_quit' must not be empty"),
])
def test_invalid_values_raise_clear_errors(tmp_path, key, value, needle):
    with pytest.raises(ConfigError, match="config.json") as ex:
        load_config(write(tmp_path, {key: value}))
    assert needle in str(ex.value)


def test_unknown_key_warns_with_suggestion(tmp_path, capsys):
    cfg = load_config(write(tmp_path, {"modle": "small"}))
    assert cfg.model == Config().model
    assert "did you mean 'model'" in capsys.readouterr().out


def test_bad_json(tmp_path):
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(write(tmp_path, "{ not json"))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")


def test_top_level_must_be_object(tmp_path):
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(write(tmp_path, "[1, 2]"))
