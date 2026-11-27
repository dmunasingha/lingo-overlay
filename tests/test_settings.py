import json

import pytest

from lingo_overlay import ROOT
from lingo_overlay.config import Config, ConfigError, load_config, save_config
from lingo_overlay.ui.settings import FIELDS, form_values, parse_form


def test_fields_are_real_settings():
    keys = [f.key for f in FIELDS]
    assert len(keys) == len(set(keys))
    assert set(keys) <= set(Config())


def test_form_round_trip_is_unchanged():
    cfg = Config()
    changes = parse_form(form_values(cfg))
    assert cfg.replace(**changes) == cfg


def test_parse_form_types():
    values = form_values(Config())
    values.update(skip_languages=" EN, si ;; en ", overlay_opacity="70", overlay_font_size="24",
                  overlay_seconds="4.5", hotkey_quit=" ctrl+alt+x ")
    changes = parse_form(values)
    assert changes["skip_languages"] == ["en", "si"]
    assert changes["overlay_opacity"] == pytest.approx(0.7)
    assert changes["overlay_font_size"] == 24
    assert changes["overlay_seconds"] == 4.5
    assert changes["hotkey_quit"] == "ctrl+alt+x"


@pytest.mark.parametrize("key,value", [("overlay_font_size", "big"), ("cpu_threads", "2.5"),
                                       ("overlay_opacity", "")])
def test_parse_form_rejects_bad_numbers(key, value):
    values = form_values(Config())
    values[key] = value
    with pytest.raises(ConfigError):
        parse_form(values)


def test_parse_form_rejects_duplicate_hotkeys():
    values = form_values(Config())
    values["hotkey_voice"] = values["hotkey_toggle"].upper()
    with pytest.raises(ConfigError, match="hotkey"):
        parse_form(values)


def test_save_config_keeps_order_and_unknown_keys(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"model": "small", "my_note": "keep me", "overlay_font_size": 20}),
                 encoding="utf-8")
    cfg = save_config({"overlay_font_size": 28, "skip_languages": ["en", "si"]}, p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert list(data) == ["model", "my_note", "overlay_font_size", "skip_languages"]
    assert data["my_note"] == "keep me"
    assert cfg.overlay_font_size == 28
    assert load_config(p).skip_languages == ["en", "si"]
    assert not (tmp_path / "config.json.tmp").exists()


def test_save_config_validates_before_writing(tmp_path):
    p = tmp_path / "config.json"
    original = json.dumps({"overlay_opacity": 0.85})
    p.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_config({"overlay_opacity": 1.5}, p)
    with pytest.raises(ConfigError, match="Unknown"):
        save_config({"not_a_setting": 1}, p)
    assert p.read_text(encoding="utf-8") == original


def test_save_shipped_config_round_trip(tmp_path):
    p = tmp_path / "config.json"
    p.write_text((ROOT / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    before = load_config(p)
    save_config(parse_form(form_values(before)), p)
    assert load_config(p) == before
