"""whisper.cpp backend with a shortened encoder context (audio_ctx).

Whisper's encoder normally processes a fixed 30 s window, even for a 2 s phrase.
whisper.cpp lets us shrink that window (``audio_ctx``, in 20 ms encoder frames), which
cuts encoder time roughly in proportion to the clip length. CTranslate2 (faster-whisper)
rejects inputs shorter than 30 s, which is why this is a separate backend.
"""

import math
import os
import re
from pathlib import Path

import numpy as np

from lingo_overlay.asr.base import ASRBackend

# Phrases Whisper invents for silence/noise when transcribing (not translating), e.g. subtitle
# credits from its training data. Matched as substrings of the original-language transcript.
SOURCE_HALLUCINATIONS = (
    "字幕", "订阅", "訂閱", "点赞", "點贊", "打赏", "明镜", "感谢观看", "感謝觀看", "谢谢观看", "謝謝觀看",
    "Amara.org",
)

FULL_CTX = 1500  # 30 s of 20 ms encoder frames
MIN_SAMPLES = int(1.1 * 16000)  # whisper.cpp skips audio shorter than 1 s; pad short callouts


class WhisperCppModel(ASRBackend):
    """whisper.cpp via pywhispercpp. ``detect_language`` sets the short encoder window first.

    With ``mt`` (``lingo_overlay.translate.mt.TextTranslator``), languages it supports are
    transcribed in the original language and translated as text, which is much more accurate
    for e.g. Chinese. ``source_asr`` (``lingo_overlay.asr.sensevoice.SenseVoiceASR``) then does
    that transcription instead of Whisper for the languages in ``source_asr.languages``.
    """

    name = "whisper.cpp"

    def __init__(self, model_path, threads, min_ctx=256, ctx_pad=64, min_prob=0.2, hallucinations=(),
                 mt=None, source_asr=None):
        from pywhispercpp.model import Model

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")
        self.min_ctx = min_ctx
        self.ctx_pad = ctx_pad
        self.min_prob = min_prob
        self.hallucinations = set(hallucinations)
        self.mt = mt
        self.source_asr = source_asr
        self.last_source = None  # original-language transcript of the last MT phrase (for logs)
        self._skip = set()
        self.model = Model(
            str(model_path),
            redirect_whispercpp_logs_to=None,
            n_threads=threads,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
            no_context=True,
            no_timestamps=True,
            single_segment=True,
            suppress_blank=True,
            suppress_nst=True,
            temperature=0.0,
            temperature_inc=0.0,  # no fallback re-decodes: they cost latency
            max_tokens=120,
        )

    def audio_ctx(self, n_samples):
        frames = math.ceil(n_samples / 320)  # 16 kHz / 50 frames per second
        return int(min(FULL_CTX, max(self.min_ctx, frames + self.ctx_pad)))

    def _set_ctx(self, ctx, audio):
        """whisper.cpp only applies audio_ctx inside whisper_full, so run one that aborts
        immediately; the next encode (language detection) then uses the short window."""
        from pywhispercpp.utils import redirect_stderr

        with redirect_stderr(to=None):  # hide the expected "failed to encode" message
            self.model.transcribe(audio, audio_ctx=ctx, language="en", translate=False,
                                  abort_callback=lambda: True)

    @staticmethod
    def _prepare(audio):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if len(audio) < MIN_SAMPLES:  # whisper.cpp silently ignores input under 1 s
            audio = np.pad(audio, (0, MIN_SAMPLES - len(audio)))
        return audio

    def run(self, audio, skip, skip_thr, min_lang_prob=0.0):
        self.last_source = None
        self._skip = set(skip)  # translate() needs it for the SenseVoice second opinion
        return super().run(audio, skip, skip_thr, min_lang_prob)

    def detect_language(self, audio):
        audio = self._prepare(audio)
        self._set_ctx(self.audio_ctx(len(audio)), audio)
        _, probs = self.model.auto_detect_language(audio)
        return probs

    def translate(self, audio, lang):
        audio = self._prepare(audio)
        use_mt = self.mt is not None and self.mt.supports(lang)
        if use_mt and self.source_asr is not None and lang in self.source_asr.languages:
            source_text, heard_lang = self.source_asr.transcribe(audio)
            # Second opinion: Whisper sometimes labels short English callouts ("Spike down")
            # as Chinese; SenseVoice, which hears them as English, overrules it.
            if heard_lang in self._skip:
                return None
            return self._translate_text(source_text, lang)
        ctx = self.audio_ctx(len(audio))
        # A short audio_ctx makes Whisper prone to repeating itself until max_tokens,
        # which costs seconds. Nobody says more than ~8 English tokens per second;
        # transcripts in languages like Chinese need more tokens per second.
        max_tokens = int(len(audio) / 16000 * (16 if use_mt else 8)) + 12
        segs = self.model.transcribe(audio, audio_ctx=ctx, language=lang, translate=not use_mt,
                                     max_tokens=max_tokens, abort_callback=None,
                                     extract_probability=True)
        text = collapse_repeats(" ".join(s.text.strip() for s in segs).strip())
        if not text or text.lower() in self.hallucinations:
            return None
        # Geometric-mean token probability. Translations score low even when correct
        # (0.25-0.5 is common), so only drop real garbage.
        if segs and np.nanmean([s.probability for s in segs]) < self.min_prob:
            return None
        return self._translate_text(text, lang) if use_mt else text

    def _translate_text(self, source_text, lang):
        """Original-language transcript -> English subtitle, or None if it's not worth showing."""
        source_text = collapse_repeats(collapse_char_loops(source_text.strip()))
        self.last_source = source_text
        if not source_text or any(h in source_text for h in SOURCE_HALLUCINATIONS):
            return None
        text = self.mt.translate(source_text, lang)
        if not text or text.lower() in self.hallucinations:
            return None
        return text


_CHAR_LOOP = re.compile(r"(\S{4,40}?)\1{2,}\S{0,40}$|(\S{4,40}?)\2{2,}")


def collapse_char_loops(text):
    """Loops in languages written without spaces: "王反媽是單臣?王反媽是單臣?王反…" -> "王反媽是單臣?".

    Only units of 4+ characters repeated 3+ times, so real repetition like 哈哈哈 stays.
    """
    return _CHAR_LOOP.sub(lambda m: m.group(1) or m.group(2), text)


def collapse_repeats(text):
    """'Hi. Hi. Hi.' -> 'Hi.' Drops a sentence that repeats the one before it, and a
    trailing fragment that is just the start of that sentence again (a cut-off loop)."""

    # Loops without punctuation: "the one who is the one who is the one ..." -> "the one who is"
    # (a run of 2+ words repeated 3+ times, plus a cut-off partial copy at the end).
    text = re.sub(r"\b((?:\S+\s+){1,11}?\S+?)(?:[\s,]+\1\b){2,}(?:[\s,]+\S+){0,11}?$", r"\1", text)
    text = re.sub(r"\b((?:\S+\s+){1,11}?\S+?)(?:[\s,]+\1\b){2,}", r"\1", text)

    parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p]
    out = []
    for p in parts:
        if out and (p == out[-1] or out[-1].startswith(p)):
            continue
        out.append(p)
    return " ".join(out)


MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
# Optional extra source, tried after Hugging Face: a base URL that serves ggml-<name>.bin files.
MIRROR_ENV = "LINGO_MODEL_MIRROR"


def model_name(model, quant):
    return f"{model}-{quant}" if quant else model


def model_path(folder, model, quant):
    return Path(folder) / f"ggml-{model_name(model, quant)}.bin"


def model_urls(model, quant):
    name = model_name(model, quant)
    urls = [MODEL_URL.format(name=name)]
    mirror = os.environ.get(MIRROR_ENV, "").strip()
    if mirror:
        urls.append(f"{mirror.rstrip('/')}/ggml-{name}.bin")
    return urls


def ensure_model(folder, model, quant, log=print, progress=None, cancel=None):
    """Return the local ggml model path, downloading it once if it's missing.

    See ``lingo_overlay.download``: resumes after stalls and falls back to the mirror
    in ``LINGO_MODEL_MIRROR`` if Hugging Face keeps failing.
    """
    from lingo_overlay.download import download, text_progress

    path = model_path(folder, model, quant)
    if path.is_file():
        return path
    urls = model_urls(model, quant)
    log(f"Model {path.name} not found - downloading once from {urls[0]}")
    return download(urls, path, progress=progress or text_progress(log), log=log, cancel=cancel)
