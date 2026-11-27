"""Tests for the whisper.cpp backend: audio_ctx sizing, skip logic and fallback.

The logic tests use fakes, so neither pywhispercpp nor a model is needed. The smoke test at
the bottom runs only when pywhispercpp and models/ggml-small-q8_0.bin are both present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import translator as T  # noqa: E402

SR = T.TARGET_SR


def test_audio_ctx_covers_clip_plus_margin():
    # 50 encoder positions per second, +16 margin.
    assert T.audio_ctx_for(int(2.0 * SR), minimum=0) == 100 + 16
    assert T.audio_ctx_for(int(2.0 * SR) + 1, minimum=0) == 101 + 16  # rounds up


def test_audio_ctx_respects_minimum_and_maximum():
    assert T.audio_ctx_for(int(2.0 * SR), minimum=384) == 384
    assert T.audio_ctx_for(int(7.0 * SR), minimum=256) == 366
    assert T.audio_ctx_for(int(40 * SR), minimum=384) == 1500


@pytest.mark.parametrize("text, want", [
    ("Tom is 30 years old. Tom is 30 years old. Tom is 30 years old. Tom is 30", "Tom is 30 years old."),
    ("Do not forget to the best to the best to the best to the bes", "Do not forget to the best"),
    ("go go go", "go go go"),                      # real callout, kept
    ("I said no, no, no.", "I said no, no, no."),
    ("a b a b c", "a b a b c"),                    # only two repeats
    ("Pass me the book, please.", "Pass me the book, please."),
    ("", ""),
])
def test_collapse_repeats(text, want):
    assert T.collapse_repeats(text) == want


def test_max_tokens_scales_with_clip():
    assert T.max_tokens_for(2 * SR) == 40
    assert T.max_tokens_for(7 * SR) > T.max_tokens_for(2 * SR)


class FakePW:
    def __init__(self, text=b" Hello there."):
        self.text = text
        self.calls = []

    def whisper_full(self, ctx, params, audio, n):
        self.calls.append((params.language, params.translate, params.audio_ctx, n))
        return 0

    def whisper_full_n_segments(self, ctx):
        return 1

    def whisper_full_get_segment_text(self, ctx, i):
        return self.text


class FakeParams:
    language = ""
    translate = False
    audio_ctx = 0
    max_tokens = 0


def make_backend(probs, text=b" Hello there.", ctx_min=384):
    b = object.__new__(T.WhisperCppBackend)
    b.pw = FakePW(text)
    b.ctx_min = ctx_min
    b.model = type("M", (), {"_params": FakeParams(), "_ctx": object()})()
    b.seen = []

    def detect(audio, actx):
        b.seen.append((len(audio), actx))
        return probs

    b._detect = detect
    return b


def test_skip_language_is_not_translated():
    b = make_backend({"en": 0.9, "es": 0.1})
    lang, prob, text = b.run(np.zeros(2 * SR, np.float32), {"en"}, 0.25)
    assert (lang, text) == ("en", None)
    assert prob == pytest.approx(0.9)
    assert b.pw.calls == []  # no translation pass


def test_skip_threshold_counts_non_top_skip_language():
    b = make_backend({"es": 0.6, "en": 0.3, "pt": 0.1})
    assert b.run(np.zeros(2 * SR, np.float32), {"en"}, 0.25)[2] is None


def test_foreign_is_translated_with_same_audio_ctx():
    b = make_backend({"es": 0.95, "en": 0.05})
    lang, _, text = b.run(np.zeros(3 * SR, np.float32), {"en"}, 0.25)
    assert (lang, text) == ("es", "Hello there.")
    (_, actx), = b.seen
    assert b.pw.calls == [("es", True, actx, 3 * SR)]


def test_hallucination_is_dropped():
    b = make_backend({"es": 0.95}, text=b" Thank you.")
    assert b.run(np.zeros(2 * SR, np.float32), {"en"}, 0.25)[2] is None


def test_short_audio_is_padded_to_one_second_plus():
    b = make_backend({"en": 1.0})
    b.run(np.zeros(int(0.5 * SR), np.float32), {"en"}, 0.25)
    assert b.seen[0][0] == T.MIN_CPP_SAMPLES


def test_explicit_audio_ctx_overrides_sizing():
    b = make_backend({"en": 1.0})
    b.run(np.zeros(2 * SR, np.float32), {"en"}, 0.25, audio_ctx=1500)
    assert b.seen[0][1] == 1500


def test_load_backend_falls_back_when_whispercpp_missing(monkeypatch):
    def boom(cfg):
        raise ImportError("no pywhispercpp")

    class FakeFW:
        name = "faster-whisper"

        def __init__(self, cfg):
            pass

    monkeypatch.setattr(T, "WhisperCppBackend", boom)
    monkeypatch.setattr(T, "FasterWhisperBackend", FakeFW)
    assert T.load_backend({"asr_backend": "whisper.cpp"}).name == "faster-whisper"
    assert T.load_backend({}).name == "faster-whisper"  # default
    assert T.load_backend({"asr_backend": "nope"}).name == "faster-whisper"


def test_shipped_config_defaults_to_faster_whisper():
    cfg = T.load_config()
    assert cfg["asr_backend"] == "faster-whisper"
    assert cfg["whispercpp_audio_ctx_min"] >= 0


MODEL = T.HERE / "models" / "ggml-small-q8_0.bin"


@pytest.mark.skipif(not MODEL.is_file(), reason="models/ggml-small-q8_0.bin not downloaded")
def test_real_whispercpp_smoke():
    pytest.importorskip("pywhispercpp")
    cfg = T.load_config()
    cfg.update(model="small", cpu_threads=4)
    b = T.WhisperCppBackend(cfg)
    probs = b._detect(np.zeros(2 * SR, np.float32), 384)
    assert abs(sum(probs.values()) - 1.0) < 0.05
    lang, prob, text = b.run(np.zeros(2 * SR, np.float32), {"en"}, 0.25)
    assert isinstance(lang, str) and 0.0 <= prob <= 1.0
