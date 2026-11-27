"""Game word list applied to a transcript before text translation.

General translation models don't know game vocabulary: Opus-MT turns "这个clo" (a cut-off
"Clove") into "this cl", and the official Chinese weapon and agent names (幻影 = Phantom)
come out as literal words. This replaces them with the English game names first, which
the translator then copies through.

The word list is data (``glossary/valorant.json``) so players can fix and extend it.
"""

import json
import re
from pathlib import Path

from lingo_overlay import ROOT

LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z/']*")


class Glossary:
    def __init__(self, english=(), zh=None, zh_patterns=None):
        self.english = {w.lower(): w for w in english}
        # Regex -> English for slang that isn't a fixed string ("6666666" -> "nice!").
        self._zh_patterns = [(re.compile(p), dst) for p, dst in (zh_patterns or {}).items()]
        zh = zh or {}
        # Every English word the list can insert: candidates for repair() besides the names.
        self._vocab = {w.lower(): w for dst in [*zh.values(), *(zh_patterns or {}).values()]
                       for w in LATIN_WORD.findall(dst) if len(w) >= 3}
        # Longest first, so 经济局 wins over shorter overlapping terms.
        self._zh = sorted(zh.items(), key=lambda kv: -len(kv[0]))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("english", []), data.get("zh", {}), data.get("zh_patterns", {}))

    def _english_word(self, m):
        word = m.group(0)
        low = word.lower()
        if low in self.english:
            return self.english[low]
        if len(low) >= 3:  # a cut-off name: complete it only if exactly one name fits
            hits = [canon for key, canon in self.english.items() if key.startswith(low)]
            if len(hits) == 1:
                return hits[0]
        return word

    def apply(self, text, lang):
        """Return ``text`` with game terms normalised to their English names."""
        if lang in ("zh", "yue"):
            for pattern, dst in self._zh_patterns:
                text = pattern.sub(f" {dst} ", text)
            for src, dst in self._zh:
                text = text.replace(src, f" {dst} ")
        text = LATIN_WORD.sub(self._english_word, text)
        return re.sub(r"\s{2,}", " ", text).strip()

    def repair(self, source, translation):
        """Fix game names the translator misspelled ("Phandom" -> "Phantom").

        Only names that are in ``source`` (after ``apply``) are candidates, and only output
        words within a small edit distance of one are changed, so ordinary words stay put.
        """
        known = {**self._vocab, **self.english}
        names = {known[w.lower()] for w in LATIN_WORD.findall(source) if w.lower() in known}
        if not names:
            return translation
        present = {w.lower() for w in LATIN_WORD.findall(translation)}
        missing = [n for n in names if n.lower() not in present]

        def fix(m):
            word = m.group(0)
            if word.lower() in present and word.lower() in known:
                return word
            best = min(missing, key=lambda n: _distance(word.lower(), n.lower()), default=None)
            if best and len(word) >= 3 and _distance(word.lower(), best.lower()) <= max(1, len(best) // 4):
                missing.remove(best)
                return best
            return word

        return LATIN_WORD.sub(fix, translation)


def _distance(a, b):
    """Levenshtein edit distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_default(log=print):
    path = ROOT / "glossary" / "valorant.json"
    try:
        return Glossary.load(path)
    except (OSError, ValueError) as e:
        log(f"Game word list not loaded ({e}); translating without it.")
        return None
