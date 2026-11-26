import time

import numpy as np

FRAME = 480  # 30 ms @ 16 kHz


class Segmenter:
    """Energy-based speech detector. Emits one numpy array per spoken phrase."""

    def __init__(self, cfg, out_q):
        self.out_q = out_q
        self.silence_frames = int(cfg.vad_silence_ms / 30)
        self.max_frames = int(cfg.vad_max_segment_s * 1000 / 30)
        self.min_speech_frames = int(cfg.vad_min_speech_ms / 30)
        self.abs_threshold = cfg.vad_threshold_db
        self.noise_db = -60.0
        self.buf = np.zeros(0, dtype=np.float32)
        self.preroll = []
        self.active = False
        self.frames = []
        self.speech_count = 0
        self.silence_count = 0
        self.start_streak = 0

    def feed(self, samples):
        self.buf = np.concatenate([self.buf, samples])
        while len(self.buf) >= FRAME:
            f, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            self._frame(f)

    def _frame(self, f):
        db = 20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-9)
        is_speech = db > max(self.abs_threshold, self.noise_db + 10)
        if not is_speech:  # slowly track background level
            self.noise_db = 0.98 * self.noise_db + 0.02 * db

        if not self.active:
            self.preroll.append(f)
            self.preroll = self.preroll[-10:]  # ~300 ms before speech starts
            self.start_streak = self.start_streak + 1 if is_speech else 0
            if self.start_streak >= 3:
                self.active = True
                self.frames = list(self.preroll)
                self.speech_count = 3
                self.silence_count = 0
            return

        self.frames.append(f)
        if is_speech:
            self.speech_count += 1
            self.silence_count = 0
        else:
            self.silence_count += 1

        if self.silence_count >= self.silence_frames or len(self.frames) >= self.max_frames:
            if self.speech_count >= self.min_speech_frames:
                self.out_q.put((np.concatenate(self.frames), time.time()))
            self.active = False
            self.frames = []
            self.preroll = []
            self.start_streak = 0
