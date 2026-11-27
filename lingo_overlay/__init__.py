"""Lingo Overlay: live, fully local translation of in-game voice chat (Windows).

Pipeline:
  Discord audio (via VB-Cable "CABLE Output", or speaker loopback as fallback)
    -> speech detector (Silero VAD; cuts audio into phrases at pauses)   lingo_overlay.vad
    -> optional fast language-ID gate (SpeechBrain)                      lingo_overlay.langid
    -> Whisper: detect language; skip if in skip list; else translate    lingo_overlay.asr
    -> on-screen subtitle overlay (click-through, always on top)         lingo_overlay.ui
    -> optional English voice (Windows built-in TTS)                     lingo_overlay.tts

You keep hearing the original Discord audio with zero delay; this only adds
the translation on top.
"""

import os
import sys
from pathlib import Path

__version__ = "0.1.0"

# Repo root: config.json and models/ live here (next to translator.py / run.bat).
# In the installed app (PyInstaller) that is the folder holding LingoOverlay.exe instead.
FROZEN = bool(getattr(sys, "frozen", False))
ROOT = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent


def _add_cuda_dlls():
    """Let CTranslate2 find the pip-installed NVIDIA cuBLAS / cuDNN DLLs."""
    if not hasattr(os, "add_dll_directory"):  # non-Windows
        return
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn", "cuda_nvrtc"):
        d = nv / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


# Done at import time, before anything loads CTranslate2 (as the single-file app did).
_add_cuda_dlls()
