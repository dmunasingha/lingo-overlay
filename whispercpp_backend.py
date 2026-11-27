"""whisper.cpp backend with a shortened encoder context (audio_ctx).

Whisper's encoder normally processes a fixed 30 s window, even for a 2 s phrase.
whisper.cpp lets us shrink that window (``audio_ctx``, in 20 ms encoder frames), which
cuts encoder time roughly in proportion to the clip length. CTranslate2 (faster-whisper)
rejects inputs shorter than 30 s, which is why this is a separate backend.
"""

import math
import os
from pathlib import Path

import numpy as np

FULL_CTX = 1500  # 30 s of 20 ms encoder frames


class WhisperCppModel:
    """Same contract as translator.run_whisper: ``run(audio, skip, thr) -> (lang, prob, text|None)``."""

    def __init__(self, model_path, threads, min_ctx=256, ctx_pad=64, min_prob=0.2, hallucinations=()):
        from pywhispercpp.model import Model

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")
        self.min_ctx = min_ctx
        self.ctx_pad = ctx_pad
        self.min_prob = min_prob
        self.hallucinations = set(hallucinations)
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

    def run(self, audio, skip, skip_thr, min_lang_prob=0.0):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        ctx = self.audio_ctx(len(audio))
        self._set_ctx(ctx, audio)
        (lang, prob), probs = self.model.auto_detect_language(audio)
        skip_p = float(sum(probs.get(code, 0.0) for code in skip))
        # Unsure about the language usually means it isn't speech (game sounds, music).
        if lang in skip or skip_p >= skip_thr or prob < min_lang_prob:
            return lang, float(prob), None

        # A short audio_ctx makes Whisper prone to repeating itself until max_tokens,
        # which costs seconds. Nobody says more than ~8 tokens per second.
        max_tokens = int(len(audio) / 16000 * 8) + 12
        segs = self.model.transcribe(audio, audio_ctx=ctx, language=lang, translate=True,
                                     max_tokens=max_tokens, abort_callback=None,
                                     extract_probability=True)
        text = collapse_repeats(" ".join(s.text.strip() for s in segs).strip())
        if not text or text.lower() in self.hallucinations:
            return lang, float(prob), None
        # Geometric-mean token probability. Translations score low even when correct
        # (0.25-0.5 is common), so only drop real garbage.
        if segs and np.nanmean([s.probability for s in segs]) < self.min_prob:
            return lang, float(prob), None
        return lang, float(prob), text


def collapse_repeats(text):
    """'Hi. Hi. Hi.' -> 'Hi.' Drops a sentence that repeats the one before it, and a
    trailing fragment that is just the start of that sentence again (a cut-off loop)."""
    import re

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


def ensure_model(folder, model, quant, log=print):
    """Return the local ggml model path, downloading it once if it's missing.

    Plain HTTP download (no huggingface_hub), written to a .part file first so an
    interrupted download is never mistaken for a complete model.
    """
    import urllib.request

    name = f"{model}-{quant}" if quant else model
    path = Path(folder) / f"ggml-{name}.bin"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(name=name)
    log(f"Model {path.name} not found - downloading once from {url}")
    part = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=30) as resp, open(part, "wb") as f:
        total, done, last = int(resp.headers.get("Content-Length", 0)), 0, 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done * 10 // total > last:
                last = done * 10 // total
                log(f"  {last * 10}% of {total >> 20} MB")
    os.replace(part, path)
    return path
