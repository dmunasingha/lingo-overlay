"""Common interface for speech-recognition / translation backends."""

from abc import ABC, abstractmethod

import numpy as np

HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.",
    "thanks for watching.", "thank you for watching!", "you", "bye.",
    "please subscribe.", "subtitles by the amara.org community",
}


def is_hallucination(text: str) -> bool:
    return not text or text.lower() in HALLUCINATIONS


class ASRBackend(ABC):
    """A model that identifies the spoken language and translates speech to English.

    ``audio`` is always 16 kHz mono float32. Backends may cache work done in
    ``detect_language`` (e.g. the encoder pass) and reuse it when ``translate`` is then
    called with the same array object.
    """

    name: str = "base"

    @abstractmethod
    def detect_language(self, audio: np.ndarray) -> dict:
        """Return language code -> probability for the clip (e.g. {"en": 0.91, ...})."""

    @abstractmethod
    def translate(self, audio: np.ndarray, lang: str) -> str | None:
        """Translate speech in ``lang`` to English. Returns None if it isn't real speech."""

    def run(self, audio, skip, skip_thr, min_lang_prob=0.0):
        """Detect the language; skip it if it is in ``skip``, otherwise translate.

        Returns (lang, lang_prob, text). text is None when skipped / not speech.
        """
        probs = self.detect_language(audio)
        lang = max(probs, key=probs.get)
        prob = probs[lang]  # compared in the backend's own dtype, converted only for the caller
        skip_p = float(sum(probs.get(code, 0.0) for code in skip))
        # Unsure about the language usually means it isn't speech (game sounds, music).
        if lang in skip or skip_p >= skip_thr or prob < min_lang_prob:
            return lang, float(prob), None
        return lang, float(prob), self.translate(audio, lang)
