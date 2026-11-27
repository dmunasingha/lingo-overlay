import queue

import pytest

from lingo_overlay import controls
from lingo_overlay.config import Config
from lingo_overlay.ui import tray


def state(**kw):
    s = {"enabled": True, "voice": False, "quit": False, "last": None, "ready": True, "busy": False}
    s.update(kw)
    return s


@pytest.mark.parametrize("kw, want", [
    ({"enabled": False}, tray.OFF),
    ({"enabled": False, "busy": True}, tray.OFF),
    ({"ready": False}, tray.LOADING),
    ({}, tray.LISTENING),
    ({"busy": True}, tray.TRANSLATING),
])
def test_tray_status(kw, want):
    assert tray.tray_status(state(**kw)) == want


def test_every_status_has_colour_and_label():
    for s in (tray.OFF, tray.LOADING, tray.LISTENING, tray.TRANSLATING):
        assert s in tray.COLORS
        assert tray.tooltip(s).startswith("Lingo Overlay: ")


def test_icon_image():
    pytest.importorskip("PIL")
    img = tray.make_icon_image(tray.LISTENING, size=32)
    assert img.size == (32, 32)
    assert img.getpixel((16, 16))[:3] == tray.COLORS[tray.LISTENING]
    assert img.getpixel((0, 0))[3] == 0  # transparent corner


def test_start_tray_disabled_returns_none():
    assert tray.start_tray(Config(tray_enabled=False), state(), queue.Queue()) is None


def test_controls_toggle_and_quit():
    s, q = state(), queue.Queue()
    controls.toggle_enabled(s, q)
    assert s["enabled"] is False and q.get_nowait() == ("status", "Translator OFF")
    controls.toggle_voice(s, q)
    assert s["voice"] is True and q.get_nowait() == ("status", "English voice ON")
    controls.request_quit(s)
    assert s["quit"] is True


def test_controls_replay():
    s, ui_q, tts_q = state(last=("es", "push left")), queue.Queue(), queue.Queue()
    controls.replay(s, ui_q, tts_q)
    assert ui_q.get_nowait() == ("sub", "es", "push left")
    assert tts_q.get_nowait() == "push left"
    controls.replay(state(), ui_q, tts_q)  # nothing to replay yet
    assert ui_q.empty()
