"""Tests for the fast language-ID gate logic (no torch / speechbrain needed)."""

import numpy as np

import lingo_overlay.langid.speechbrain_lid as L
from lingo_overlay.langid import lid_gate, load_lid


class FakeLID:
    def __init__(self, probs):
        self._probs = probs

    def probs(self, audio):
        return self._probs


AUDIO = np.zeros(16000, np.float32)


def test_confident_skip_language_is_dropped():
    drop, lang, prob = lid_gate(FakeLID({"en": 0.97, "es": 0.03}), AUDIO, {"en"}, 0.9)
    assert drop and lang == "en" and prob == 0.97


def test_low_confidence_skip_language_goes_to_whisper():
    drop, lang, _ = lid_gate(FakeLID({"en": 0.6, "es": 0.4}), AUDIO, {"en"}, 0.9)
    assert not drop and lang == "en"


def test_foreign_language_goes_to_whisper():
    drop, lang, _ = lid_gate(FakeLID({"en": 0.02, "es": 0.98}), AUDIO, {"en"}, 0.9)
    assert not drop and lang == "es"


def test_skip_probability_is_summed_over_skip_languages():
    probs = {"en": 0.5, "de": 0.45, "fr": 0.05}
    assert lid_gate(FakeLID(probs), AUDIO, {"en", "de"}, 0.9)[0]
    assert not lid_gate(FakeLID(probs), AUDIO, {"en"}, 0.9)[0]


def test_whisper_backend_loads_no_gate():
    assert load_lid({}) is None
    assert load_lid({"lid_backend": "whisper"}) is None


def test_unknown_backend_falls_back_to_whisper():
    assert load_lid({"lid_backend": "nope"}) is None


def test_missing_extras_fall_back_to_whisper(monkeypatch):
    def boom(cfg):
        raise ImportError("No module named 'speechbrain'")

    monkeypatch.setattr(L, "SpeechBrainLID", boom)
    assert load_lid({"lid_backend": "speechbrain"}) is None
