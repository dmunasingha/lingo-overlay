"""SenseVoice-Small speech recognition for Chinese (and Cantonese) phrases.

Whisper ``small`` often mishears Mandarin in noisy voice chat, and that error carries into
the translation. SenseVoice-Small (FunAudioLLM / Alibaba) is trained heavily on Chinese,
is non-autoregressive (one pass, ~0.1 s per phrase on CPU) and transcribes far more
accurately. It runs through sherpa-onnx (Apache-2.0) on ONNX Runtime.

License note: the weights are under the "FunASR Model Open Source License Agreement 1.1"
(free use, modification and sharing with attribution; not an OSI license), so this is
opt-in via ``"chinese_asr": "sensevoice"``. Model: SenseVoiceSmall by FunAudioLLM,
https://github.com/FunAudioLLM/SenseVoice, converted to ONNX by the sherpa-onnx project.
"""

import os
import shutil
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

import numpy as np

from lingo_overlay import ROOT
from lingo_overlay.log import log as _log

NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{NAME}.tar.bz2"
LANGUAGES = {"zh", "yue"}  # languages we route to SenseVoice (it also knows en/ja/ko)


def ensure_model(root, log=print):
    """Return the local model folder, downloading it once (~155 MB) if missing."""
    out = Path(root) / NAME
    if (out / "model.int8.onnx").is_file() and (out / "tokens.txt").is_file():
        return out
    log(f"SenseVoice model not found - downloading once from {URL}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp = Path(tmp)
        archive = tmp / "model.tar.bz2"
        with urllib.request.urlopen(URL, timeout=30) as resp, open(archive, "wb") as f:
            shutil.copyfileobj(resp, f, 1 << 20)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(tmp, filter="data")
        os.replace(tmp / NAME, out)  # only a complete model ever appears at `out`
    return out


def load_source_asr(cfg):
    """SenseVoice for Chinese/Cantonese transcripts when ``cfg.chinese_asr`` is "sensevoice"."""
    choice = str(cfg.chinese_asr).lower()
    if choice == "whisper":
        return None
    if choice != "sensevoice":
        _log(f"Unknown chinese_asr '{choice}' - using Whisper.")
        return None
    try:
        asr = SenseVoiceASR(ROOT / "models" / "asr", threads=cfg.mt_threads, log=_log)
        _log("Chinese speech recognition: SenseVoice-Small (FunAudioLLM, FunASR model license)")
        return asr
    except Exception as e:  # sherpa-onnx missing, download failed, ...: keep working
        _log(f"SenseVoice unavailable ({e}); Whisper transcribes Chinese.")
        return None


class SenseVoiceASR:
    languages = LANGUAGES

    def __init__(self, root, threads=2, log=print):
        import sherpa_onnx

        d = ensure_model(root, log)
        self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=threads, language="auto", use_itn=True)
        self._lock = threading.Lock()

    def transcribe(self, audio, sr=16000):
        """Return (transcript, language) for a 16 kHz mono float32 phrase.

        language is SenseVoice's own tag ("zh", "en", "yue", "ja", "ko", or "" if unknown).
        """
        with self._lock:
            s = self._rec.create_stream()
            s.accept_waveform(sr, np.ascontiguousarray(audio, dtype=np.float32))
            self._rec.decode_stream(s)
            return s.result.text.strip(), s.result.lang.strip("<|>").lower()
