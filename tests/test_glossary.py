import pytest

from lingo_overlay.translate.glossary import Glossary, load_default


@pytest.fixture(scope="module")
def g():
    return load_default()


def test_shipped_word_list_loads(g):
    assert g is not None and "phantom" in g.english


@pytest.mark.parametrize("src, expected", [
    ("我要买幻影，你买狂徒吧。", "我要买 Phantom ，你买 Vandal 吧。"),
    ("捷风大招好了。", "Jett ult is ready 。"),
    ("这把经济局。", "这把 eco round 。"),
    ("我特别讨厌这个clo。", "我特别讨厌这个Clove。"),
    ("他有 VANDAL", "他有 Vandal"),
    ("对面有冥驹", "the enemy 有 Operator"),
    ("这什么6666666。", "这什么 nice! 。"),
    ("他是演员，一直送人头", "他是 thrower ，一直 feeding"),
    ("臭小孩儿。", "brat 。"),
])
def test_apply_chinese(g, src, expected):
    assert g.apply(src, "zh") == expected


def test_chinese_terms_only_for_chinese(g):
    assert g.apply("幻影", "ja") == "幻影"


def test_ambiguous_or_short_prefixes_are_left_alone():
    g = Glossary(["Sage", "Sava", "Clove"])
    assert g.apply("sa", "zh") == "sa"     # too short
    assert g.apply("sav", "zh") == "Sava"  # unique
    g2 = Glossary(["Sage", "Sagebrush"])
    assert g2.apply("sag", "zh") == "sag"  # ambiguous


def test_longest_chinese_term_wins():
    g = Glossary(zh={"大招": "ult", "大招好了": "ult is ready"})
    assert g.apply("大招好了", "zh") == "ult is ready"


@pytest.mark.parametrize("source, translation, expected", [
    ("我要买 Phantom", "I'll buy Phandom.", "I'll buy Phantom."),
    ("对面有 Operator", "There's Opperator over there.", "There's Operator over there."),
    ("plant the spike", "plant the spine", "plant the spike"),
    ("对面有 Operator", "There's a spine.", "There's a spine."),  # not a near-miss of Operator
    ("没有游戏词", "Phandom", "Phandom"),  # nothing inserted, nothing repaired
])
def test_repair_misspelled_names(g, source, translation, expected):
    assert g.repair(source, translation) == expected


def test_repair_covers_slang_words(g):
    assert g.repair("牛逼 awesome", "Aawesome, carry me.") == "awesome, carry me."
    assert g.repair("the enemy 有 Operator", "The eremy has Operator") == "The enemy has Operator"
