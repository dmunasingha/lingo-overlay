import json

import pytest

from lingo_overlay.config import Config, ConfigError, load_config, save_settings
from lingo_overlay.ui.placement import Monitor, anchor_from_xy, outline_offsets, pick_monitor, subtitle_xy

PRIMARY = Monitor(0, 0, 1920, 1080)
RIGHT = Monitor(1920, 0, 4480, 1440)
LEFT = Monitor(-1280, 0, 0, 1024)
MONITORS = [PRIMARY, RIGHT, LEFT]


def test_default_is_bottom_centre_like_before():
    assert subtitle_xy(PRIMARY, 400, 60, None, 160) == ((1920 - 400) // 2, 1080 - 60 - 160)


def test_anchor_places_bottom_centre_on_that_monitor():
    x, y = subtitle_xy(RIGHT, 400, 60, [0.5, 0.25], 160)
    assert (x + 200, y + 60) == (1920 + 1280, 360)


def test_subtitle_is_kept_on_screen():
    assert subtitle_xy(PRIMARY, 400, 60, [1.0, 0.0], 160) == (1520, 0)
    assert subtitle_xy(LEFT, 400, 60, [0.0, 1.0], 160) == (-1280, 1024 - 60)


def test_missing_monitor_falls_back_to_primary():
    assert pick_monitor(MONITORS, 1) is RIGHT
    assert pick_monitor(MONITORS, 7) is PRIMARY


@pytest.mark.parametrize("mon_index", [0, 1, 2])
def test_drag_round_trip(mon_index):
    mon = MONITORS[mon_index]
    x, y = subtitle_xy(mon, 500, 80, [0.3, 0.7], 160)
    index, anchor = anchor_from_xy(MONITORS, x, y, 500, 80)
    assert index == mon_index
    assert subtitle_xy(MONITORS[index], 500, 80, anchor, 160) == (x, y)


def test_dragged_off_every_screen_picks_nearest():
    index, anchor = anchor_from_xy(MONITORS, 5000, 100, 400, 60)
    assert index == 1 and anchor[0] == 1.0


def test_outline_offsets():
    assert outline_offsets(0) == []
    ring = outline_offsets(2)
    assert (0, 0) not in ring and (2, 0) in ring and (-2, 0) in ring and (0, -2) in ring
    assert len(set(ring)) == len(ring)


@pytest.mark.parametrize("bad", [
    {"overlay_anchor": [0.5]},
    {"overlay_anchor": [0.5, 1.5]},
    {"overlay_anchor": "bottom"},
    {"overlay_outline": 20},
    {"overlay_monitor": -1},
    {"hotkey_edit": ""},
])
def test_invalid_overlay_settings(bad):
    with pytest.raises(ConfigError):
        Config.from_dict(bad)


def test_save_settings_rewrites_in_place_and_keeps_layout(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{\n  "model": "small",\n\n  "overlay_anchor": null,\n  "overlay_monitor": 0\n}\n',
                 encoding="utf-8")
    save_settings(p, {"overlay_anchor": [0.25, 0.9], "overlay_monitor": 1})
    text = p.read_text(encoding="utf-8")
    assert '  "overlay_anchor": [0.25, 0.9],\n  "overlay_monitor": 1\n}' in text
    assert '"model": "small",\n\n' in text
    cfg = load_config(p)
    assert cfg.overlay_anchor == [0.25, 0.9] and cfg.overlay_monitor == 1


def test_save_settings_adds_missing_keys(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"model": "small"}, indent=2), encoding="utf-8")
    save_settings(p, {"overlay_anchor": [0.5, 0.5]})
    assert load_config(p).overlay_anchor == [0.5, 0.5]


def test_save_settings_refuses_bad_values(tmp_path):
    p = tmp_path / "config.json"
    original = '{\n  "overlay_anchor": null\n}\n'
    p.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_settings(p, {"overlay_anchor": [2, 2]})
    with pytest.raises(ConfigError):
        save_settings(p, {"not_a_setting": 1})
    assert p.read_text(encoding="utf-8") == original
