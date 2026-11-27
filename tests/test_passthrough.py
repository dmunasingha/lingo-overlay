from types import SimpleNamespace

import numpy as np
import pytest

from lingo_overlay.audio.passthrough import BLOCK, SAMPLE_RATE, Ducker, find_devices
from lingo_overlay.config import Config


def ducker(volume=0.3, attack_ms=30, release_ms=400):
    return Ducker(SAMPLE_RATE, volume, attack_ms, release_ms)


def run(d, ms, ducking):
    """Feed `ms` of 10 ms blocks of ones and return the output gains."""
    out = [d.process(np.ones((BLOCK, 2), dtype=np.float32), ducking) for _ in range(int(ms // 10))]
    return np.concatenate(out)[:, 0]


def test_block_is_10ms():
    assert BLOCK * 1000 / SAMPLE_RATE == 10


def test_unity_gain_when_not_ducking():
    x = np.random.default_rng(0).standard_normal((BLOCK, 2)).astype(np.float32)
    np.testing.assert_array_equal(ducker().process(x, False), x)


def test_ducks_within_attack_time_and_holds():
    g = run(ducker(attack_ms=30), 100, True)
    assert g[0] < 1.0  # starts ramping immediately
    n_attack = 30 * SAMPLE_RATE // 1000
    assert g[n_attack - 1] == pytest.approx(0.3, abs=1e-3)
    np.testing.assert_allclose(g[n_attack:], 0.3, atol=1e-6)


def test_release_is_smooth_and_returns_to_unity():
    d = ducker(release_ms=400)
    run(d, 100, True)
    g = run(d, 500, False)
    assert np.all(np.diff(g) >= -1e-7)  # never jumps down while releasing
    assert np.max(np.abs(np.diff(g))) < 1e-3  # no clicks
    n_release = 400 * SAMPLE_RATE // 1000
    assert g[n_release - 2] < 1.0
    np.testing.assert_allclose(g[n_release:], 1.0, atol=1e-6)


def test_ramp_is_continuous_across_blocks():
    d = ducker()
    g = run(d, 60, True)
    steps = np.abs(np.diff(g))
    assert steps.max() < 1e-3


def test_changing_mind_mid_ramp_continues_from_current_gain():
    d = ducker(attack_ms=30, release_ms=30)
    run(d, 10, True)
    mid = d.gain
    assert 0.3 < mid < 1.0
    g = run(d, 10, False)
    assert g[0] == pytest.approx(mid, abs=1e-3)


def test_mono_block():
    y = ducker(attack_ms=0).process(np.ones(BLOCK, dtype=np.float32), True)
    assert y.shape == (BLOCK,)
    np.testing.assert_allclose(y, 0.3)


def test_volume_one_never_changes_signal():
    d = ducker(volume=1.0)
    np.testing.assert_array_equal(run(d, 50, True), 1.0)


# ---- device selection (fake soundcard module)

def fake_sc(mics, speakers, default="Speakers (Realtek)"):
    dev = lambda n: SimpleNamespace(name=n)  # noqa: E731
    return SimpleNamespace(
        all_microphones=lambda include_loopback=False: [dev(n) for n in mics],
        all_speakers=lambda: [dev(n) for n in speakers],
        default_speaker=lambda: dev(default),
    )


def test_finds_cable_and_default_speaker():
    sc = fake_sc(["Microphone (Realtek)", "CABLE Output (VB-Audio)"], ["Speakers (Realtek)"])
    mic, spk = find_devices(Config(), sc)
    assert mic.name == "CABLE Output (VB-Audio)"
    assert spk.name == "Speakers (Realtek)"


def test_named_output_device():
    sc = fake_sc(["CABLE Output (VB-Audio)"], ["Speakers (Realtek)", "Headset (Arctis 7)"])
    _, spk = find_devices(Config(passthrough_output_device="arctis"), sc)
    assert spk.name == "Headset (Arctis 7)"


def test_missing_cable_never_falls_back_to_loopback():
    sc = fake_sc(["Microphone (Realtek)"], ["Speakers (Realtek)"])
    assert find_devices(Config(), sc) is None


def test_refuses_to_play_into_the_cable():
    sc = fake_sc(["CABLE Output (VB-Audio)"], ["CABLE Input (VB-Audio)"], default="CABLE Input (VB-Audio)")
    assert find_devices(Config(), sc) is None


def test_missing_output_device():
    sc = fake_sc(["CABLE Output (VB-Audio)"], ["Speakers (Realtek)"])
    assert find_devices(Config(passthrough_output_device="nope"), sc) is None


@pytest.mark.parametrize("bad", [{"duck_volume": 1.5}, {"duck_volume": -0.1}, {"duck_attack_ms": -1},
                                 {"duck_release_ms": -5}])
def test_duck_config_validation(bad):
    from lingo_overlay.config import ConfigError
    with pytest.raises(ConfigError):
        Config(**bad)
