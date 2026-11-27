"""Per-game vocabulary hints (profiles/<game>.txt -> Whisper initial prompt)."""

import json

import pytest

from lingo_overlay import ROOT
from lingo_overlay.asr.vocab import PROFILES_DIR, build_prompt, is_prompt_echo, load_prompt, read_terms
from lingo_overlay.config import Config


def test_read_terms_skips_comments_blanks_and_duplicates():
    text = "# Agents\nJett\n\n  Sage  # healer\nJETT\nKAY/O\n#Sova\n"
    assert read_terms(text) == ["Jett", "Sage", "KAY/O"]


def test_build_prompt_joins_terms():
    assert build_prompt(["Jett", "Sage", "Sova"]) == "Jett, Sage, Sova."
    assert build_prompt([]) == ""


def test_build_prompt_cuts_at_a_whole_term():
    assert build_prompt(["Jett", "Sage", "Brimstone"], max_chars=11) == "Jett, Sage."
    assert build_prompt(["Jett", "Sage", "Brimstone"], max_chars=10) == "Jett."


def test_load_prompt_off_by_default():
    assert Config().game_profile == ""
    assert load_prompt("") == ""


def test_load_prompt_from_folder(tmp_path):
    (tmp_path / "mygame.txt").write_text("# hi\nAlpha\nBravo\n", encoding="utf-8")
    assert load_prompt("mygame", tmp_path) == "Alpha, Bravo."


@pytest.mark.parametrize("name", ["nope", "../config", "a b"])
def test_bad_profile_is_ignored(tmp_path, name):
    assert load_prompt(name, tmp_path) == ""


def test_shipped_valorant_profile():
    prompt = load_prompt("valorant")
    for name in ("Jett", "KAY/O", "Killjoy", "Vandal", "Ascent"):
        assert name in prompt
    terms = read_terms((PROFILES_DIR / "valorant.txt").read_text(encoding="utf-8"))
    assert len(prompt) <= 450
    assert prompt == build_prompt(terms), "every VALORANT term should fit in the prompt"


def test_valorant_profile_has_every_glossary_name():
    glossary = json.loads((ROOT / "glossary" / "valorant.json").read_text(encoding="utf-8"))
    names = {w for w in glossary["english"] if w[0].isupper()}
    terms = set(read_terms((PROFILES_DIR / "valorant.txt").read_text(encoding="utf-8")))
    assert names <= terms, f"missing from profiles/valorant.txt: {sorted(names - terms)}"


PROMPT = "Astra, Breach, Brimstone, Chamber, Jett, KAY/O, Killjoy."


@pytest.mark.parametrize("text,echo", [
    ("Astra, Breach, Brimstone.", True),
    ("Breach Brimstone Chamber", True),
    ("Jett, KAY/O, Killjoy", True),
    ("Jett, Sage.", False),            # two names: could be a real callout
    ("Breach is on B.", False),
    ("Chamber, Brimstone, Breach", False),  # not in list order: someone saying names
    ("", False),
])
def test_prompt_echo(text, echo):
    assert is_prompt_echo(text, PROMPT) is echo


def test_no_echo_without_prompt():
    assert is_prompt_echo("Astra, Breach, Brimstone", "") is False
