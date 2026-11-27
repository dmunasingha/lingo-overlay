"""Synthetic audio helpers shared by the tests."""

import numpy as np

SR = 16000


def tone(seconds, freq=440.0, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items
