"""Text translation (machine translation) for languages where Whisper's own translation is weak.

Whisper's built-in "translate" task is noticeably worse for some languages (Chinese most of
all) than transcribing in the original language and translating the text with a dedicated
model. This module runs Opus-MT models (University of Helsinki, CC-BY-4.0) with CTranslate2,
the same inference engine faster-whisper uses.

Models live in ``models/mt/opus-mt-<lang>-en/``. A missing model is downloaded once from the
Tatoeba-MT release server (the original Marian weights) and converted locally to int8, so no
PyTorch is needed. Only the languages in ``MODELS`` are supported; everything else keeps
using Whisper's translation.
"""

import os
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path

# Whisper language code -> Tatoeba-MT release (original Marian weights, CC-BY-4.0).
# Tatoeba-test scores are from each release's model card.
MODELS = {
    "zh": "zho-eng/opus-2020-07-17.zip",  # BLEU 36.1, chrF 0.548
}
# Languages served by another language's model (the zho-eng model covers Cantonese too).
ALIASES = {"yue": "zh"}
RELEASE_URL = "https://object.pouta.csc.fi/Tatoeba-MT-models/{}"


def model_dir(root, lang):
    return Path(root) / f"opus-mt-{lang}-en"


def ensure_model(root, lang, log=print):
    """Return the local CTranslate2 model folder for ``lang``, downloading and converting once."""
    out = model_dir(root, lang)
    if (out / "model.bin").is_file():
        return out
    from ctranslate2.converters import OpusMTConverter

    url = RELEASE_URL.format(MODELS[lang])
    log(f"Translation model for '{lang}' not found - downloading once from {url}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp = Path(tmp)
        zpath = tmp / "model.zip"
        with urllib.request.urlopen(url, timeout=30) as resp, open(zpath, "wb") as f:
            shutil.copyfileobj(resp, f, 1 << 20)
        raw = tmp / "raw"
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
        staging = tmp / "ct2"
        OpusMTConverter(str(raw)).convert(str(staging), quantization="int8")
        for name in ("source.spm", "target.spm", "LICENSE", "README.md"):
            if (raw / name).is_file():
                shutil.copy2(raw / name, staging / name)
        os.replace(staging, out)  # only a complete model ever appears at `out`
    return out


class TextTranslator:
    """Translate short phrases to English with per-language Opus-MT models (loaded lazily)."""

    def __init__(self, root, languages=None, threads=2, log=print, glossary=None):
        self.root = Path(root)
        known = set(MODELS) | set(ALIASES)
        self.languages = known if languages is None else set(languages) & known
        self.threads = threads
        self.glossary = glossary  # glossary.Glossary: game terms -> English names
        self.log = log
        self._models = {}
        self._lock = threading.Lock()

    def supports(self, lang):
        return lang in self.languages

    def _load(self, lang):
        lang = ALIASES.get(lang, lang)
        with self._lock:
            if lang not in self._models:
                import ctranslate2
                import sentencepiece as spm

                d = ensure_model(self.root, lang, self.log)
                self._models[lang] = (
                    ctranslate2.Translator(str(d), device="cpu", compute_type="int8",
                                           inter_threads=1, intra_threads=self.threads),
                    spm.SentencePieceProcessor(model_file=str(d / "source.spm")),
                    spm.SentencePieceProcessor(model_file=str(d / "target.spm")),
                )
            return self._models[lang]

    def preload(self):
        for lang in sorted({ALIASES.get(code, code) for code in self.languages}):
            self._load(lang)

    def translate(self, text, lang):
        model, src, tgt = self._load(lang)
        if self.glossary is not None:
            text = self.glossary.apply(text, lang)
        tokens = src.encode(text.strip(), out_type=str)
        if not tokens:
            return ""
        res = model.translate_batch([tokens], beam_size=2, max_decoding_length=128,
                                    repetition_penalty=1.2)
        out = tgt.decode(res[0].hypotheses[0]).strip()
        return self.glossary.repair(text, out) if self.glossary is not None else out
