"""Lingo Overlay: live, fully local translation of in-game voice chat (Windows).

Pipeline:
  Discord audio (via VB-Cable "CABLE Output", or speaker loopback as fallback)
    -> speech detector (cuts audio into phrases at pauses)            lingo_overlay.vad
    -> Whisper: detect language; skip if in skip list; else translate  lingo_overlay.asr
    -> on-screen subtitle overlay (click-through, always on top)       lingo_overlay.ui
    -> optional English voice (Windows built-in TTS)                   lingo_overlay.tts

You keep hearing the original Discord audio with zero delay; this only adds
the translation on top.
"""

from pathlib import Path

__version__ = "0.1.0"

# Repo root: config.json and models/ live here (next to translator.py / run.bat).
ROOT = Path(__file__).resolve().parent.parent
