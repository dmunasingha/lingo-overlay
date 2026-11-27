import queue

import pytest


@pytest.fixture
def vad_cfg():
    """Segmenter settings matching the shipped config.json, with the deterministic energy detector."""
    return {
        "vad_silence_ms": 450,
        "vad_max_segment_s": 7,
        "vad_min_speech_ms": 400,
        "vad_threshold_db": -45,
        "vad_backend": "energy",
        "vad_adaptive": False,  # fixed 450 ms ending; adaptive is covered in test_endpointing.py
    }


@pytest.fixture
def out_q():
    return queue.Queue()
