"""In-app voice-chat passthrough (replaces Windows "Listen to this device") with ducking.

We play CABLE Output to the headset ourselves, in 10 ms blocks, so we can lower the original
voice chat while the translation is being spoken. No game hooks are needed.
"""

import numpy as np

from lingo_overlay.log import log

SAMPLE_RATE = 48000
BLOCK = 480  # 10 ms at 48 kHz; keeps the added latency well under 30 ms


class Ducker:
    """Smoothly moves the gain between 1.0 and ``duck_volume`` with linear ramps.

    ``attack_ms`` is how fast the volume drops when ducking starts; ``release_ms`` is how fast
    it comes back afterwards. Ramps run per sample, so block boundaries never click.
    """

    def __init__(self, sr, duck_volume, attack_ms, release_ms):
        self.duck_volume = float(duck_volume)
        span = 1.0 - self.duck_volume
        self.down_step = span / max(1.0, attack_ms * sr / 1000) if span > 0 else 0.0
        self.up_step = span / max(1.0, release_ms * sr / 1000) if span > 0 else 0.0
        self.gain = 1.0

    def gains(self, n, ducking):
        target = self.duck_volume if ducking else 1.0
        if self.gain == target or n == 0:
            return np.full(n, self.gain, dtype=np.float32)
        step = -self.down_step if target < self.gain else self.up_step
        g = self.gain + step * np.arange(1, n + 1, dtype=np.float64)
        g = np.maximum(g, target) if step < 0 else np.minimum(g, target)
        self.gain = float(g[-1])
        return g.astype(np.float32)

    def process(self, block, ducking):
        g = self.gains(len(block), ducking)
        return block * (g[:, None] if block.ndim == 2 else g)


def find_devices(cfg, sc):
    """Return (microphone, speaker) for the passthrough, or None with a logged reason."""
    want = (cfg.input_device or "").lower()
    mic = next((m for m in sc.all_microphones(include_loopback=False) if want and want in m.name.lower()),
               None)
    if mic is None:
        # Never pass through a speaker loopback: it would play the PC's audio back into itself.
        log(f"Passthrough off: input device '{cfg.input_device}' not found.")
        return None
    out = (cfg.passthrough_output_device or "").lower()
    if out:
        spk = next((s for s in sc.all_speakers() if out in s.name.lower()), None)
        if spk is None:
            log(f"Passthrough off: output device '{cfg.passthrough_output_device}' not found.")
            return None
    else:
        spk = sc.default_speaker()
    if "cable" in spk.name.lower():
        log(f"Passthrough off: output '{spk.name}' is the virtual cable (would loop back).")
        return None
    return mic, spk


def passthrough_loop(cfg, state):
    import warnings

    import soundcard as sc

    devices = find_devices(cfg, sc)
    if devices is None:
        return
    mic, spk = devices
    ducker = Ducker(SAMPLE_RATE, cfg.duck_volume if cfg.duck_enabled else 1.0,
                    cfg.duck_attack_ms, cfg.duck_release_ms)
    log(f"Passthrough: {mic.name} -> {spk.name}"
        + (f" (ducks to {cfg.duck_volume:.0%} while speaking)" if cfg.duck_enabled else ""))
    log("  Turn off Windows 'Listen to this device' on the cable, or you will hear voice chat twice.")
    warnings.filterwarnings("ignore", message="data discontinuity")
    try:
        with mic.recorder(samplerate=SAMPLE_RATE, channels=2, blocksize=BLOCK) as rec, \
                spk.player(samplerate=SAMPLE_RATE, channels=2, blocksize=BLOCK) as player:
            while not state["quit"]:
                data = rec.record(numframes=BLOCK)
                player.play(ducker.process(data, state.get("tts_speaking", False)))
    except Exception as ex:  # a device unplugged mid-game should not take down the translator
        log(f"Passthrough stopped: {ex}")
