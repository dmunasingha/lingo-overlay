"""Routing for the transcribe-then-translate path, with fake models (no downloads needed)."""

import numpy as np
import pytest

pytest.importorskip("pywhispercpp")  # whispercpp_backend imports it lazily, but keep CI lean

import mt  # noqa: E402
import translator as T  # noqa: E402
from whispercpp_backend import WhisperCppModel  # noqa: E402


class FakeMT:
    def __init__(self, langs=("zh", "yue")):
        self.languages = set(langs)
        self.calls = []

    def supports(self, lang):
        return lang in self.languages

    def translate(self, text, lang):
        self.calls.append((text, lang))
        return f"EN({text})"


class FakeASR:
    languages = {"zh", "yue"}

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.text


class FakeWhisper:
    """Stands in for pywhispercpp.Model: fixed language, records transcribe() calls."""

    def __init__(self, lang, prob=0.99):
        self.lang, self.prob = lang, prob
        self.transcribes = []

    def auto_detect_language(self, audio):
        return (self.lang, self.prob), {self.lang: self.prob}

    def transcribe(self, audio, **kw):
        if kw.get("abort_callback") is None:  # the real decode, not the audio_ctx setter
            self.transcribes.append(kw)
        return []


def make(lang, mt_=None, asr=None):
    m = WhisperCppModel.__new__(WhisperCppModel)
    m.min_ctx, m.ctx_pad, m.min_prob = 256, 64, 0.2
    m.hallucinations = set(T.HALLUCINATIONS)
    m.mt, m.source_asr = mt_, asr
    m.model = FakeWhisper(lang)
    m._set_ctx = lambda ctx, audio: None
    return m


AUDIO = np.zeros(32000, np.float32)


def test_chinese_goes_through_sensevoice_and_mt():
    fmt, asr = FakeMT(), FakeASR("往返吗？是单程。")
    m = make("zh", fmt, asr)
    lang, _, text = m.run(AUDIO, {"en"}, 0.25, 0.5)
    assert (lang, text) == ("zh", "EN(往返吗？是单程。)")
    assert asr.calls == 1 and m.model.transcribes == []  # Whisper never decoded
    assert fmt.calls == [("往返吗？是单程。", "zh")]


def test_cantonese_uses_the_same_path():
    m = make("yue", FakeMT(), FakeASR("你好"))
    assert m.run(AUDIO, {"en"}, 0.25, 0.5)[2] == "EN(你好)"


def test_other_languages_keep_whisper_translation():
    asr = FakeASR("unused")
    m = make("es", FakeMT(), asr)
    m.run(AUDIO, {"en"}, 0.25, 0.5)
    assert asr.calls == 0
    assert m.model.transcribes[0]["translate"] is True


def test_without_sensevoice_whisper_transcribes_for_mt():
    m = make("zh", FakeMT(), None)
    m.run(AUDIO, {"en"}, 0.25, 0.5)
    assert m.model.transcribes[0]["translate"] is False


def test_skip_languages_never_reach_mt():
    fmt, asr = FakeMT(), FakeASR("hello")
    m = make("en", fmt, asr)
    assert m.run(AUDIO, {"en"}, 0.25, 0.5)[2] is None
    assert asr.calls == 0 and fmt.calls == []


@pytest.mark.parametrize("heard", ["", "  ", "字幕由Amara.org社区提供", "请不吝点赞 订阅 转发"])
def test_empty_or_hallucinated_transcripts_are_dropped(heard):
    fmt = FakeMT()
    m = make("zh", fmt, FakeASR(heard))
    assert m.run(AUDIO, {"en"}, 0.25, 0.5)[2] is None
    assert fmt.calls == []


def test_translator_aliases_cantonese_to_chinese_model():
    tt = mt.TextTranslator("unused", ["zh", "yue", "xx"])
    assert tt.supports("zh") and tt.supports("yue") and not tt.supports("xx")


def test_config_defaults_and_opt_in():
    assert T.CONFIG_DEFAULTS["text_translation"] == ["zh", "yue"]
    assert T.CONFIG_DEFAULTS["chinese_asr"] == "whisper"  # SenseVoice is opt-in (license)
    assert T.load_source_asr({"chinese_asr": "whisper", "mt_threads": 2}) is None
    assert T.load_source_asr({"chinese_asr": "nope", "mt_threads": 2}) is None
    assert T.load_text_translator({"text_translation": [], "mt_threads": 2}) is None
