"""Common interface for speech-recognition / translation backends.

faster-whisper today; whisper.cpp (issue #5) and others plug in by implementing ASRBackend.
"""

from abc import ABC, abstractmethod

import numpy as np

HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.",
    "thanks for watching.", "thank you for watching!", "you", "bye.",
    "please subscribe.", "subtitles by the amara.org community",
}


class ASRBackend(ABC):
    """A model that can identify the spoken language and translate speech to English.

    `audio` is always 16 kHz mono float32. Backends may cache work done in
    `detect_language` (e.g. the encoder pass) and reuse it when `translate` is then
    called with the same array object.
    """

    name: str = "base"

    @abstractmethod
    def detect_language(self, audio: np.ndarray) -> dict[str, float]:
        """Return language-code -> probability for the clip (e.g. {"en": 0.91, ...})."""

    @abstractmethod
    def translate(self, audio: np.ndarray, lang: str) -> str | None:
        """Translate speech in `lang` to English. Returns None if it isn't real speech."""


def is_hallucination(text: str) -> bool:
    return not text or text.lower() in HALLUCINATIONS


def detect_and_translate(backend: ASRBackend, audio: np.ndarray, skip, skip_thr: float):
    """Detect the language, skip it if it is in `skip`, otherwise translate.

    Returns (lang, lang_prob, text). text is None when skipped / not speech.
    """
    probs = backend.detect_language(audio)
    lang = max(probs, key=probs.get)
    skip_p = sum(probs.get(code, 0.0) for code in skip)
    if lang in skip or skip_p >= skip_thr:
        return lang, probs[lang], None
    return lang, probs[lang], backend.translate(audio, lang)
