import numpy as np
from helpers import SR, drain, silence, tone

from lingo_overlay.vad import EnergyDetector, Segmenter

FRAME = EnergyDetector.frame_len  # these tests cover the energy detector's timing


def segments(cfg, q, audio, chunk=None):
    seg = Segmenter(cfg, q)
    if chunk is None:
        seg.feed(audio)
    else:
        for i in range(0, len(audio), chunk):
            seg.feed(audio[i:i + chunk])
    return [a for a, _t in drain(q)]


def test_silence_emits_nothing(vad_cfg, out_q):
    assert segments(vad_cfg, out_q, silence(3)) == []


def test_quiet_noise_below_threshold_emits_nothing(vad_cfg, out_q):
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(3 * SR) * 0.001).astype(np.float32)  # about -60 dB
    assert segments(vad_cfg, out_q, noise) == []


def test_single_phrase_is_emitted_once(vad_cfg, out_q):
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0)])
    segs = segments(vad_cfg, out_q, audio)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.dtype == np.float32
    assert len(seg) % FRAME == 0
    # The whole phrase, plus at most ~300 ms pre-roll and the 450 ms closing silence.
    assert len(tone(1.0)) <= len(seg) <= len(tone(1.0)) + int(0.3 * SR) + int(0.45 * SR) + FRAME
    assert np.max(np.abs(seg)) > 0.25


def test_emits_end_timestamp(vad_cfg, out_q):
    seg = Segmenter(vad_cfg, out_q)
    seg.feed(np.concatenate([tone(1.0), silence(1.0)]))
    _audio, t_end = out_q.get_nowait()
    assert isinstance(t_end, float)


def test_short_blip_is_dropped(vad_cfg, out_q):
    # 200 ms is under vad_min_speech_ms (400 ms): a click, not speech.
    audio = np.concatenate([silence(0.5), tone(0.2), silence(1.0)])
    assert segments(vad_cfg, out_q, audio) == []


def test_two_phrases_split_at_pause(vad_cfg, out_q):
    audio = np.concatenate([tone(1.0), silence(1.0), tone(1.0, freq=660), silence(1.0)])
    assert len(segments(vad_cfg, out_q, audio)) == 2


def test_short_pause_does_not_split(vad_cfg, out_q):
    # A 200 ms gap is shorter than vad_silence_ms (450 ms), so it stays one phrase.
    audio = np.concatenate([tone(0.8), silence(0.2), tone(0.8), silence(1.0)])
    assert len(segments(vad_cfg, out_q, audio)) == 1


def test_long_speech_is_cut_at_max_segment(vad_cfg, out_q):
    max_frames = int(vad_cfg["vad_max_segment_s"] * 1000 / 30)
    segs = segments(vad_cfg, out_q, np.concatenate([tone(10.0), silence(1.0)]))
    assert len(segs) == 2
    assert len(segs[0]) == max_frames * FRAME


def test_chunk_size_does_not_matter(vad_cfg, out_q):
    audio = np.concatenate([silence(0.3), tone(1.0), silence(0.8), tone(0.7), silence(1.0)])
    whole = segments(vad_cfg, out_q, audio)
    chunked = segments(vad_cfg, out_q, audio, chunk=1601)  # not frame aligned
    assert len(whole) == len(chunked) == 2
    for a, b in zip(whole, chunked, strict=True):
        np.testing.assert_array_equal(a, b)


def test_partial_frame_is_buffered(vad_cfg, out_q):
    seg = Segmenter(vad_cfg, out_q)
    seg.feed(silence(0.01))  # 160 samples, less than one frame
    assert len(seg.buf) == 160
    seg.feed(silence(0.03))  # +480 samples -> one frame consumed
    assert len(seg.buf) == 160 + 480 - FRAME
