import threading

import pytest

from lingo_overlay.config import Config, ConfigError
from lingo_overlay.history import Entry, History


def test_keeps_only_the_last_n_oldest_first():
    h = History(size=3)
    for i in range(5):
        h.add("es", f"line {i}", timestamp=1000 + i)
    assert [e.text for e in h.entries()] == ["line 2", "line 3", "line 4"]
    assert len(h) == 3


def test_default_size_is_50():
    h = History()
    for i in range(60):
        h.add("fr", str(i))
    assert len(h) == 50
    assert h.entries()[0].text == "10"


def test_entry_format_has_time_language_and_text(monkeypatch):
    monkeypatch.setattr("time.localtime", lambda _ts: __import__("time").struct_time((2026, 1, 2, 13, 4, 5, 0, 2, 0)))
    assert Entry(0.0, "es", "Push left").format() == "13:04:05  [ES]  Push left"


def test_as_text_is_one_line_per_entry_and_clear_empties():
    h = History()
    h.add("es", "a", timestamp=0)
    h.add("de", "b", timestamp=1)
    lines = h.as_text().splitlines()
    assert len(lines) == 2 and lines[0].endswith("[ES]  a") and lines[1].endswith("[DE]  b")
    h.clear()
    assert h.entries() == [] and h.as_text() == ""


def test_add_is_thread_safe():
    h = History(size=1000)
    threads = [threading.Thread(target=lambda: [h.add("es", "x") for _ in range(100)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(h) == 800


def test_config_history_defaults_and_validation():
    cfg = Config()
    assert cfg.history_size == 50 and cfg.hotkey_history == "ctrl+alt+h"
    with pytest.raises(ConfigError, match="history_size"):
        Config(history_size=0)
    with pytest.raises(ConfigError, match="hotkey_history"):
        Config(hotkey_history="  ")
