"""faster-whisper (CTranslate2) backend."""

from pathlib import Path

import numpy as np

from lingo_overlay import ROOT
from lingo_overlay.asr.base import ASRBackend, is_hallucination
from lingo_overlay.log import log


def load_model(cfg, models_dir: Path = ROOT / "models"):
    """Use the already-downloaded model offline; only go online if it's missing."""
    from faster_whisper import WhisperModel

    kw = dict(device=cfg.device, compute_type=cfg.compute_type,
              cpu_threads=cfg.cpu_threads, download_root=str(models_dir))
    local_dir = models_dir / cfg.model
    if (local_dir / "model.bin").is_file():
        return WhisperModel(str(local_dir), **kw)
    try:
        return WhisperModel(cfg.model, local_files_only=True, **kw)
    except Exception:
        log("Model not found locally - downloading (one time)...")
        return WhisperModel(cfg.model, **kw)


class FasterWhisperBackend(ASRBackend):
    """Runs the Whisper encoder once per clip, shared by language ID and translation.

    (faster-whisper's transcribe() would run the expensive encoder twice.)
    """

    name = "faster-whisper"

    def __init__(self, model):
        self.model = model
        self._last_audio = None
        self._last_enc = None

    @classmethod
    def from_config(cls, cfg):
        return cls(load_model(cfg))

    def _encode(self, audio: np.ndarray):
        if audio is self._last_audio:
            return self._last_enc
        from faster_whisper.audio import pad_or_trim

        features = pad_or_trim(self.model.feature_extractor(audio)[:, :3000])
        enc = self.model.encode(features)
        self._last_audio, self._last_enc = audio, enc
        return enc

    def detect_language(self, audio):
        enc = self._encode(audio)
        return {tok[2:-2]: p for tok, p in self.model.model.detect_language(enc)[0]}

    def translate(self, audio, lang):
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        enc = self._encode(audio)
        self._last_audio = self._last_enc = None  # free the clip's encoder output
        m = self.model
        tok = Tokenizer(m.hf_tokenizer, True, task="translate", language=lang)
        res = m.model.generate(
            enc, [m.get_prompt(tok, [], without_timestamps=True)],
            beam_size=1, max_length=200, return_scores=True, return_no_speech_prob=True,
            suppress_blank=True, suppress_tokens=get_suppressed_tokens(tok, [-1]),
        )[0]
        ids = [t for t in res.sequences_ids[0] if t < tok.eot]
        avg_logprob = res.scores[0] * len(ids) / (len(ids) + 1)
        if res.no_speech_prob > 0.6 and avg_logprob < -0.8:
            return None
        text = tok.decode(ids).strip()
        if is_hallucination(text):
            return None
        return text
