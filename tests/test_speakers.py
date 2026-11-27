"""Tests for speaker labels and splitting at voice changes (no torch / speechbrain needed)."""

import numpy as np

from lingo_overlay.config import Config
from lingo_overlay.speakers import SpeakerTracker, change_points, load_speakers, split_by_speaker
from lingo_overlay.ui.overlay import subtitle_text

SR = 16000
# Orthogonal "voices", so similarity between different speakers is 0.
A, B, C = np.eye(3, dtype=np.float32)


def near(v, seed):
    """A slightly noisy copy of a voice embedding (similarity stays well above 0.5)."""
    rng = np.random.default_rng(seed)
    return v + rng.normal(0, 0.1, v.shape).astype(np.float32)


def test_same_voice_keeps_its_number_and_new_voice_gets_next():
    t = SpeakerTracker(threshold=0.5)
    assert t.identify(near(A, 0)) == 1
    assert t.identify(near(B, 1)) == 2
    assert t.identify(near(A, 2)) == 1
    assert t.identify(near(B, 3)) == 2
    assert len(t) == 2


def test_least_recently_heard_voice_is_reused_when_full():
    t = SpeakerTracker(threshold=0.5, max_speakers=2)
    assert t.identify(A) == 1
    assert t.identify(B) == 2
    assert t.identify(A) == 1  # B is now the oldest
    assert t.identify(C) == 2
    assert t.identify(A) == 1
    assert len(t) == 2


def test_reset_forgets_voices():
    t = SpeakerTracker()
    t.identify(A)
    t.reset()
    assert len(t) == 0 and t.best_match(A) == (None, -1.0)


def test_change_points_groups_consecutive_windows():
    assert change_points([A, A, B, B, B], 0.5) == [(0, 2), (2, 5)]
    assert change_points([A, near(A, 1), near(A, 2)], 0.5) == [(0, 3)]
    assert change_points([A, B, A], 0.5) == [(0, 1), (1, 2), (2, 3)]
    assert change_points([], 0.5) == []


def voices(*spans):
    """Fake audio where each sample's value is the voice index; the embedder reads the majority."""
    return np.concatenate([np.full(int(s * SR), v, np.float32) for v, s in spans])


def fake_embed(audio):
    vals, counts = np.unique(audio.astype(int), return_counts=True)
    return np.eye(3, dtype=np.float32)[vals[np.argmax(counts)]]


def test_two_callouts_in_one_phrase_are_split():
    audio = voices((0, 3.0), (1, 3.0))
    parts = split_by_speaker(audio, fake_embed, SpeakerTracker(), window_s=1.5)
    assert [p for p, _ in parts] == [1, 2]
    assert [len(a) for _, a in parts] == [3 * SR, 3 * SR]
    assert np.array_equal(np.concatenate([a for _, a in parts]), audio)


def test_one_voice_is_one_part():
    audio = voices((0, 5.2))
    parts = split_by_speaker(audio, fake_embed, SpeakerTracker(), window_s=1.5)
    assert len(parts) == 1 and parts[0][0] == 1 and len(parts[0][1]) == len(audio)


def test_short_phrase_or_split_off_is_labelled_whole():
    t = SpeakerTracker()
    short = voices((0, 1.0), (1, 1.0))
    assert [p for p, _ in split_by_speaker(short, fake_embed, t, window_s=1.5)] == [1]
    long = voices((0, 3.0), (1, 3.0))
    parts = split_by_speaker(long, fake_embed, t, split=False, window_s=1.5)
    assert len(parts) == 1 and len(parts[0][1]) == len(long)


def test_labels_stay_stable_across_phrases():
    t = SpeakerTracker()
    split_by_speaker(voices((0, 3.0), (1, 3.0)), fake_embed, t)
    parts = split_by_speaker(voices((1, 3.0), (0, 3.0)), fake_embed, t)
    assert [p for p, _ in parts] == [2, 1]


def test_subtitle_text_with_and_without_speaker():
    assert subtitle_text("es", "Push left") == "[ES]  Push left"
    assert subtitle_text("es", "Push left", "P2") == "P2 [ES]  Push left"


def test_speaker_labels_off_by_default_loads_nothing():
    assert load_speakers(Config()) is None


def test_speaker_labels_fall_back_when_model_cannot_load(monkeypatch):
    import lingo_overlay.speakers.speechbrain_spk as S

    def boom(cfg):
        raise ImportError("no torch")

    monkeypatch.setattr(S, "SpeechBrainEmbedder", boom)
    assert S.load_speakers(Config(speaker_labels=True)) is None
