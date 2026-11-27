import json
import queue
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

translator = pytest.importorskip("translator")
SR = translator.TARGET_SR


def cfg(**kw):
    c = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    c.update(kw)
    return c


def segments(c, audio, block=SR // 10):
    q = queue.Queue()
    seg = translator.Segmenter(c, q)
    for i in range(0, len(audio), block):
        seg.feed(audio[i:i + block])
    return seg, [q.get_nowait()[0] for _ in range(q.qsize())]


def tone_between_silence(tone_s=1.0):
    t = np.arange(int(tone_s * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    gap = np.zeros(SR, dtype=np.float32)
    return np.concatenate([gap, tone, gap]).astype(np.float32)


def test_energy_detects_tone_phrase():
    seg, out = segments(cfg(vad_backend="energy"), tone_between_silence())
    assert seg.detector.name == "energy"
    assert len(out) == 1
    # tone + pre-roll + closing silence (up to 700 ms: an abrupt stop looks like a mid-word cut)
    assert 1.0 <= len(out[0]) / SR <= 2.1


def test_energy_ignores_silence():
    _, out = segments(cfg(vad_backend="energy"), np.zeros(3 * SR, dtype=np.float32))
    assert out == []


def test_frame_counts_follow_detector_frame_length():
    c = cfg(vad_backend="energy", vad_silence_ms=450)
    seg = translator.Segmenter(c, queue.Queue())
    assert seg.frame_len == 480 and seg.silence_frames == 15


def test_unknown_backend_still_gives_a_detector():
    seg = translator.Segmenter(cfg(vad_backend="nope"), queue.Queue())
    assert seg.detector.name in {"silero", "energy"}


def _silero_or_skip():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("faster_whisper")


def test_silero_ignores_noise():
    _silero_or_skip()
    rng = np.random.default_rng(0)
    noise = (0.05 * rng.standard_normal(10 * SR)).astype(np.float32)  # about -26 dBFS
    seg, out = segments(cfg(vad_backend="silero"), noise)
    assert seg.detector.name == "silero"
    assert seg.frame_len == 512 and seg.silence_frames == 14
    assert out == []


def test_silero_ignores_pure_tone():
    _silero_or_skip()
    _, out = segments(cfg(vad_backend="silero"), tone_between_silence(2.0))
    assert out == []
