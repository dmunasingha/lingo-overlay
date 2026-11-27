"""Per-game vocabulary hints, passed to Whisper as its initial prompt.

Whisper spells names it has seen in the prompt much more reliably, so agent, weapon and map
names stop coming out as "Jet", "Kill Joy" or "Vandel". Profiles are plain text files in
``profiles/<game>.txt`` (one term per line, ``#`` comments) so players can add their game.
"""

from __future__ import annotations

import re
from pathlib import Path

from lingo_overlay import ROOT
from lingo_overlay.log import log

PROFILES_DIR = ROOT / "profiles"

# Whisper accepts up to 223 prompt tokens, but every prompt token costs decode time and a long
# prompt makes it more likely to parrot the list on noise. ~450 characters of names is ~150 tokens.
MAX_PROMPT_CHARS = 450

_NAME = re.compile(r"^[\w.-]+$")
_WORDS = re.compile(r"[^\w/]+")


def read_terms(text: str) -> list[str]:
    """Terms from a profile file: one per line, ``#`` comments and blank lines ignored,
    duplicates (any case) dropped, order kept."""
    seen, terms = set(), []
    for line in text.splitlines():
        term = line.split("#", 1)[0].strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)
    return terms


def build_prompt(terms, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """"Astra, Breach, ..." cut at a whole term so it fits in ``max_chars``."""
    prompt = ""
    for term in terms:
        candidate = f"{prompt}, {term}" if prompt else term
        if len(candidate) + 1 > max_chars:  # + the final "."
            break
        prompt = candidate
    return f"{prompt}." if prompt else ""


def load_prompt(profile: str, folder: Path | str = PROFILES_DIR) -> str:
    """The initial prompt for ``profile`` ("" turns it off). A missing or invalid profile is
    logged and ignored, so a typo in config.json never stops the translator."""
    profile = profile.strip()
    if not profile:
        return ""
    if not _NAME.match(profile):
        log(f"Game profile '{profile}': use a file name from profiles/ such as \"valorant\"; ignoring it")
        return ""
    path = Path(folder) / f"{profile}.txt"
    try:
        terms = read_terms(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        found = sorted(p.stem for p in Path(folder).glob("*.txt"))
        log(f"Game profile '{profile}' not found ({path}); available: {', '.join(found) or 'none'}")
        return ""
    prompt = build_prompt(terms)
    used = prompt.count(", ") + 1 if prompt else 0
    log(f"Game profile '{profile}': {used} vocabulary hints"
        + (f" (first {used} of {len(terms)}; the rest don't fit in the prompt)" if used < len(terms) else ""))
    return prompt


def _words(text: str) -> list[str]:
    return [w for w in _WORDS.split(text.lower()) if w]


def is_prompt_echo(text: str, prompt: str) -> bool:
    """True when Whisper just repeated part of the vocabulary list ("Astra, Breach, Brimstone"),
    which it tends to do on noise. A short real callout that names agents ("Jett, Sage") is kept:
    only three or more consecutive words of the list count as an echo."""
    if not prompt or not text:
        return False
    words, listed = _words(text), _words(prompt)
    if len(words) < 3:
        return False
    n = len(words)
    return any(listed[i:i + n] == words for i in range(len(listed) - n + 1))
