"""ASRBackend skip/translate decision, with fake backends (no Whisper needed)."""

import numpy as np
import pytest

from lingo_overlay.asr import ASRBackend, is_hallucination

AUDIO = np.zeros(16000, np.float32)


class Fake(ASRBackend):
    def __init__(self, probs, text="hello"):
        self.probs, self.text, self.translated = probs, text, []

    def detect_language(self, audio):
        return self.probs

    def translate(self, audio, lang):
        self.translated.append(lang)
        return self.text


def test_foreign_language_is_translated():
    b = Fake({"en": 0.1, "es": 0.9})
    assert b.run(AUDIO, {"en"}, 0.25) == ("es", 0.9, "hello")
    assert b.translated == ["es"]


def test_skip_language_is_not_translated():
    b = Fake({"en": 0.8, "es": 0.2})
    assert b.run(AUDIO, {"en"}, 0.25) == ("en", 0.8, None)
    assert b.translated == []


def test_skip_probability_over_threshold_skips():
    b = Fake({"es": 0.6, "en": 0.3, "de": 0.1})
    assert b.run(AUDIO, {"en"}, 0.25)[2] is None
    assert b.run(AUDIO, {"en"}, 0.35)[2] == "hello"


def test_unsure_language_is_dropped():
    b = Fake({"es": 0.45, "fr": 0.4, "en": 0.15})
    assert b.run(AUDIO, {"en"}, 0.25, min_lang_prob=0.5)[2] is None
    assert b.run(AUDIO, {"en"}, 0.25)[2] == "hello"


def test_float32_probabilities_compare_in_float32():
    # whisper.cpp returns np.float32; the threshold check must not change by converting first.
    p = np.float32(0.7)
    lang, prob, text = Fake({"es": p, "en": np.float32(0.3)}).run(AUDIO, {"en"}, 0.9, min_lang_prob=0.7)
    assert text == ("hello" if not p < 0.7 else None)
    assert type(prob) is float


@pytest.mark.parametrize("text,bad", [("", True), ("Thank you.", True), ("you", True), ("Push B", False)])
def test_hallucinations(text, bad):
    assert is_hallucination(text) is bad
