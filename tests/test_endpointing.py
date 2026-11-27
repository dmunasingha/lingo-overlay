"""Segmenter phrase-ending tests on synthetic audio (a 220 Hz tone with an amplitude envelope)."""

import numpy as np
import pytest

from lingo_overlay.audio import TARGET_SR
from lingo_overlay.vad import EnergyDetector, Segmenter

BASE = {"vad_max_segment_s": 7, "vad_min_speech_ms": 400, "vad_threshold_db": -45, "vad_backend": "energy"}
FRAME = EnergyDetector.frame_len  # these tests use the deterministic energy detector


class Q:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def tone(ms, db_start, db_end=None):
    """A tone whose level moves linearly (in dB) from db_start to db_end, frame by frame."""
    n = int(ms / 30)
    levels = np.linspace(db_start, db_start if db_end is None else db_end, n)
    t = np.arange(FRAME) / TARGET_SR
    frames = [np.sqrt(2) * 10 ** (lv / 20) * np.sin(2 * np.pi * 220 * t) for lv in levels]
    return np.concatenate(frames).astype(np.float32)


def silence(ms):
    return np.zeros(int(ms / 30) * FRAME, np.float32)


def emit_frame(cfg, audio):
    """Feed audio frame by frame; return (frame index at first emit, segmenter, queue)."""
    q = Q()
    seg = Segmenter({**BASE, **cfg}, q)
    for i in range(0, len(audio), FRAME):
        seg.feed(audio[i:i + FRAME])
        if q.items:
            return seg.frame_idx, seg, q
    return None, seg, q


TRAILING = np.concatenate([tone(900, -20), tone(240, -20, -38)])  # sentence that trails off
ABRUPT = tone(1100, -20)  # loud right up to the cut, like a mid-word stop


def test_fixed_mode_waits_vad_silence_ms():
    audio = np.concatenate([ABRUPT, silence(1000)])
    idx, seg, _ = emit_frame({"vad_adaptive": False, "vad_silence_ms": 450}, audio)
    speech_frames = len(ABRUPT) // FRAME
    assert idx - speech_frames == 15
    assert seg.last_emit == ("fixed", False)


def test_trailing_off_ends_fast():
    audio = np.concatenate([TRAILING, silence(1000)])
    idx, seg, _ = emit_frame({}, audio)
    assert idx - len(TRAILING) // FRAME == round(250 / 30)
    assert seg.last_emit == ("fast", False)


def test_abrupt_stop_waits_longer():
    audio = np.concatenate([ABRUPT, silence(1500)])
    idx, seg, _ = emit_frame({}, audio)
    assert idx - len(ABRUPT) // FRAME == round(700 / 30)
    assert seg.last_emit == ("slow", False)


def test_abrupt_stop_does_not_cut_a_word_with_a_500ms_gap():
    """Fixed 450 ms splits this in two; adaptive keeps one phrase."""
    audio = np.concatenate([ABRUPT, silence(500), TRAILING, silence(1500)])
    fixed = Q()
    seg = Segmenter({**BASE, "vad_adaptive": False, "vad_silence_ms": 450}, fixed)
    seg.feed(audio)
    adaptive = Q()
    seg = Segmenter(BASE, adaptive)
    seg.feed(audio)
    assert len(fixed.items) == 2
    assert len(adaptive.items) == 1


def test_short_fragment_merges_with_speech_that_follows_quickly():
    frag = np.concatenate([tone(420, -20), tone(150, -20, -40)])  # under vad_fragment_ms
    audio = np.concatenate([frag, silence(450), TRAILING, silence(1500)])
    q = Q()
    seg = Segmenter(BASE, q)
    seg.feed(audio)
    assert len(q.items) == 1
    assert seg.last_emit[1] is True  # merged
    assert len(q.items[0][0]) >= len(frag) + len(TRAILING)


def test_short_fragment_alone_is_emitted_after_merge_window():
    frag = np.concatenate([tone(480, -20), tone(150, -20, -40)])
    audio = np.concatenate([frag, silence(2000)])
    q = Q()
    seg = Segmenter(BASE, q)
    seg.feed(audio)
    assert len(q.items) == 1
    assert seg.last_emit == ("fragment", False)


def test_flush_emits_held_fragment():
    frag = np.concatenate([tone(480, -20), tone(150, -20, -40)])
    audio = np.concatenate([frag, silence(400)])
    q = Q()
    seg = Segmenter(BASE, q)
    seg.feed(audio)
    assert seg.pending is not None and not q.items
    seg.flush()
    assert len(q.items) == 1


def test_too_short_noise_is_still_dropped():
    audio = np.concatenate([tone(150, -20), silence(2000)])
    q = Q()
    Segmenter(BASE, q).feed(audio)
    assert q.items == []


def test_max_segment_length_still_applies():
    audio = np.concatenate([tone(8000, -20), silence(1000)])
    q = Q()
    seg = Segmenter(BASE, q)
    seg.feed(audio)
    assert len(q.items[0][0]) == int(7 * 1000 / 30) * FRAME


@pytest.mark.parametrize("adaptive", [True, False])
def test_old_config_without_new_keys_works(adaptive):
    cfg = {**BASE, "vad_silence_ms": 450}
    if not adaptive:
        cfg["vad_adaptive"] = False
    q = Q()
    Segmenter(cfg, q).feed(np.concatenate([TRAILING, silence(1500)]))
    assert len(q.items) == 1
