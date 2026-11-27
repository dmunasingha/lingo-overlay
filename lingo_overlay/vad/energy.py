"""Classic energy-threshold speech detector."""

import numpy as np

FRAME = 480  # 30 ms analysis frame (energy detector, speech-end measurement in scripts)


class EnergyDetector:
    """Classic energy threshold with a slowly tracked noise floor. 30 ms frames."""

    name = "energy"
    frame_len = FRAME  # 30 ms @ 16 kHz

    def __init__(self, cfg):
        self.abs_threshold = cfg.get("vad_threshold_db", -45)
        self.noise_db = -60.0

    def is_speech(self, f, active):
        db = 20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-9)
        speech = db > max(self.abs_threshold, self.noise_db + 10)
        if not speech:  # slowly track background level
            self.noise_db = 0.98 * self.noise_db + 0.02 * db
        return speech
