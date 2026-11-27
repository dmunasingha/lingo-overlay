"""Downmix and resample captured audio to 16 kHz mono float32 (what VAD and Whisper expect)."""

import numpy as np

TARGET_SR = 16000


def to_16k_mono(block, sr):
    mono = block.mean(axis=1) if block.ndim == 2 else block
    if sr == TARGET_SR:
        return mono.astype(np.float32)
    ratio = sr / TARGET_SR
    if ratio == int(ratio):  # e.g. 48k -> 16k: average groups (cheap anti-alias)
        r = int(ratio)
        n = len(mono) // r * r
        return mono[:n].reshape(-1, r).mean(axis=1).astype(np.float32)
    n_out = int(len(mono) / ratio)
    x = np.linspace(0, len(mono) - 1, n_out)
    return np.interp(x, np.arange(len(mono)), mono).astype(np.float32)
