"""Speech recognition + translation backends behind a common ASRBackend interface.

  * whisper.cpp (default): encoder window sized to the clip (``audio_ctx``)
  * faster-whisper (CTranslate2)
"""

from lingo_overlay import ROOT
from lingo_overlay.asr.base import HALLUCINATIONS, ASRBackend, is_hallucination
from lingo_overlay.log import log

__all__ = ["HALLUCINATIONS", "ASRBackend", "is_hallucination", "load_model"]


def load_model(cfg) -> ASRBackend:
    """Load the backend named by ``cfg.backend``; downloads the model once if it is missing."""
    if cfg.backend == "whisper.cpp":
        from lingo_overlay.asr.whispercpp_backend import WhisperCppModel, ensure_model

        path = ensure_model(ROOT / "models" / "ggml", cfg.model, cfg.whispercpp_quant, log)
        return WhisperCppModel(path, cfg.cpu_threads, min_ctx=cfg.whispercpp_min_ctx,
                               hallucinations=HALLUCINATIONS)

    from lingo_overlay.asr.faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend.from_config(cfg)
