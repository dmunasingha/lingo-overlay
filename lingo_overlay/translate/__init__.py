"""Text translation: an original-language transcript -> English subtitle.

  * ``mt``: Opus-MT (CC-BY-4.0) on CTranslate2, per language pair
  * ``glossary``: VALORANT names and gamer slang kept intact around the translator
"""

from lingo_overlay import ROOT
from lingo_overlay.log import log

__all__ = ["load_text_translator"]


def load_text_translator(cfg):
    """Opus-MT for the languages in ``cfg.text_translation`` (None when off or unavailable)."""
    langs = [code.lower() for code in cfg.text_translation]
    if not langs:
        return None
    try:
        from lingo_overlay.translate.mt import ALIASES, MODELS, TextTranslator

        unsupported = sorted(set(langs) - set(MODELS) - set(ALIASES))
        if unsupported:
            log(f"No text-translation model for {', '.join(unsupported)}; Whisper translates those.")
        glossary = None
        if cfg.game_glossary:
            from lingo_overlay.translate.glossary import load_default

            glossary = load_default(log)
        mt = TextTranslator(ROOT / "models" / "mt", langs, threads=cfg.mt_threads, log=log,
                            glossary=glossary)
        mt.preload()
        if mt.languages:
            log(f"Text translation (Opus-MT) for: {', '.join(sorted(mt.languages))}")
        return mt
    except Exception as e:  # missing sentencepiece, download failed, ...: keep working
        log(f"Text translation unavailable ({e}); Whisper translates everything.")
        return None
