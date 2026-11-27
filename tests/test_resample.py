import numpy as np
import pytest

from lingo_overlay.audio import TARGET_SR, to_16k_mono


def sine(freq, sr, seconds=1.0):
    t = np.arange(int(sr * seconds)) / sr
    return np.sin(2 * np.pi * freq * t)


def peak_hz(x, sr):
    spec = np.abs(np.fft.rfft(x))
    return np.argmax(spec) * sr / len(x)


def test_16k_mono_passthrough_casts_to_float32():
    x = sine(440, TARGET_SR)  # float64
    y = to_16k_mono(x, TARGET_SR)
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_stereo_is_averaged_to_mono():
    block = np.stack([np.ones(1000), np.zeros(1000)], axis=1)
    y = to_16k_mono(block, TARGET_SR)
    assert y.shape == (1000,)
    np.testing.assert_allclose(y, 0.5)


def test_single_channel_2d_block():
    # soundcard returns shape (n, 1) when recording with channels=1.
    x = sine(440, TARGET_SR)[:, None]
    y = to_16k_mono(x, TARGET_SR)
    assert y.ndim == 1 and len(y) == len(x)


def test_integer_ratio_averages_groups():
    x = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)  # 48k -> 16k is ratio 3
    np.testing.assert_allclose(to_16k_mono(x, 48000), [2.0, 5.0])  # trailing partial group dropped


@pytest.mark.parametrize("sr", [48000, 32000, 44100, 22050])
def test_length_dtype_and_pitch(sr):
    y = to_16k_mono(sine(440, sr), sr)
    assert y.dtype == np.float32
    assert abs(len(y) - TARGET_SR) <= 1
    assert peak_hz(y, TARGET_SR) == pytest.approx(440, abs=2)


@pytest.mark.parametrize("sr", [48000, 44100])
def test_dc_level_preserved(sr):
    np.testing.assert_allclose(to_16k_mono(np.full(sr, 0.25), sr), 0.25, atol=1e-6)


def test_empty_block():
    assert len(to_16k_mono(np.zeros((0, 1)), 48000)) == 0
