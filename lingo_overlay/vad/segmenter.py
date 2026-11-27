"""Phrase segmentation: turns a stream of 16 kHz audio into one array per spoken phrase."""

import time

import numpy as np

from lingo_overlay.audio.resample import TARGET_SR
from lingo_overlay.log import log
from lingo_overlay.vad.energy import EnergyDetector
from lingo_overlay.vad.silero import SileroDetector


def make_detector(cfg):
    backend = str(cfg.get("vad_backend", "silero")).lower()
    if backend == "energy":
        return EnergyDetector(cfg)
    if backend != "silero":
        log(f"Unknown vad_backend '{backend}', using silero.")
    try:
        return SileroDetector(cfg)
    except Exception as e:  # missing onnxruntime / model file: still usable
        log(f"Silero VAD unavailable ({e}); falling back to the energy detector.")
        return EnergyDetector(cfg)


class Segmenter:
    """Speech detector front end. Emits one numpy array per spoken phrase.

    The per-frame speech/non-speech decision comes from a detector (Silero by
    default, or the energy threshold); pre-roll, min-speech and max-segment
    logic is shared.

    Phrase ending ("endpointing"):
      * fixed (``vad_adaptive: false``): a phrase ends after ``vad_silence_ms`` of silence.
      * adaptive (default): when silence starts, the energy of the last few speech frames
        decides how long to wait:
          - energy fell off gradually (a sentence trailing off)  -> ``vad_fast_silence_ms``
          - energy was still near its peak (cut off mid-word)   -> ``vad_slow_silence_ms``
          - anything else                                       -> ``vad_silence_ms``
        Short fragments (under ``vad_fragment_ms`` of speech) are held for up to
        ``vad_merge_gap_ms`` after they end; if speech resumes in that window, the fragment
        and the new speech become one phrase.
    """

    TAIL = 3  # frames (~90 ms) that make up the "end of speech" energy window

    def __init__(self, cfg, out_q, detector=None):
        self.out_q = out_q
        self.detector = detector or make_detector(cfg)
        self.frame_len = self.detector.frame_len
        frame_ms = self.frame_len * 1000 / TARGET_SR

        def frames(ms):
            return max(1, round(ms / frame_ms))

        self.silence_frames = frames(cfg.get("vad_silence_ms", 450))
        self.max_frames = frames(cfg["vad_max_segment_s"] * 1000)
        self.min_speech_frames = frames(cfg["vad_min_speech_ms"])
        self.preroll_frames = frames(300)  # audio kept from before speech starts
        self.start_frames = 3  # consecutive speech frames needed to open a phrase

        self.adaptive = cfg.get("vad_adaptive", True)
        self.fast_frames = frames(cfg.get("vad_fast_silence_ms", 250))
        self.slow_frames = frames(cfg.get("vad_slow_silence_ms", 700))
        self.merge_frames = round(cfg.get("vad_merge_gap_ms", 300) / frame_ms)
        self.fragment_frames = round(cfg.get("vad_fragment_ms", 800) / frame_ms)
        self.fall_db = cfg.get("vad_fall_db", 3.0)  # tail this much below the frames before it
        self.peak_db = cfg.get("vad_peak_db", 3.0)  # tail within this much of the peak = mid-word

        self.buf = np.zeros(0, dtype=np.float32)
        self.preroll = []
        self.active = False
        self.frames = []
        self.speech_db = []  # dB of each speech frame in the current phrase
        self.speech_count = 0
        self.silence_count = 0
        self.start_streak = 0
        self.need_silence = self.silence_frames
        self.end_kind = "fixed"
        self.pending = None  # (frames, speech_count) of a held fragment
        self.pending_age = 0
        # Bookkeeping for logs and scripts/endpoint_benchmark.py
        self.frame_idx = 0
        self.last_emit = None  # (end_kind, merged) of the latest emitted phrase
        self._merged = False

    def feed(self, samples):
        self.buf = np.concatenate([self.buf, samples])
        n = self.frame_len
        while len(self.buf) >= n:
            f, self.buf = self.buf[:n], self.buf[n:]
            self._frame(f)
            self.frame_idx += 1

    def flush(self):
        """Emit a held fragment right away (e.g. at end of input)."""
        if self.pending is not None:
            self._emit_pending()

    # ------------------------------------------------------------------ internals
    def _emit(self, frames, speech_count, kind):
        if speech_count >= self.min_speech_frames:
            self.last_emit = (kind, self._merged)
            self.out_q.put((np.concatenate(frames), time.time()))
        self._merged = False

    def _emit_pending(self):
        frames, speech = self.pending
        self.pending = None
        self._emit(frames, speech, "fragment")

    def _pick_silence(self):
        """How many silent frames end this phrase, judged from the energy contour."""
        if not self.adaptive or len(self.speech_db) < 2 * self.TAIL:
            return self.silence_frames, "fixed"
        db = self.speech_db
        tail = float(np.mean(db[-self.TAIL:]))
        before = float(np.mean(db[-2 * self.TAIL:-self.TAIL]))
        peak = float(np.percentile(db, 90))
        if tail >= peak - self.peak_db:
            return self.slow_frames, "slow"  # still loud when it stopped: cut mid-word
        if (tail <= before - self.fall_db and tail <= peak - 2 * self.peak_db
                and self.speech_count >= self.min_speech_frames):
            return self.fast_frames, "fast"  # trailed off: the sentence is done
        return self.silence_frames, "normal"

    def _frame(self, f):
        is_speech = self.detector.is_speech(f, self.active)

        if not self.active:
            self.preroll.append(f)
            keep = self.preroll_frames if self.pending is None else self.merge_frames + self.start_frames
            self.preroll = self.preroll[-max(self.preroll_frames, keep):]
            self.start_streak = self.start_streak + 1 if is_speech else 0
            if self.start_streak >= self.start_frames:
                self.active = True
                self.frames = list(self.preroll)
                self.speech_db = []
                self.speech_count = self.start_frames
                if self.pending is not None:  # speech resumed soon enough: merge
                    frames, speech = self.pending
                    self.pending = None
                    self.frames = frames + self.frames
                    self.speech_count += speech
                    self._merged = True
                self.silence_count = 0
                return
            if self.pending is not None:
                self.pending_age += 1
                # the start streak means speech began a few frames before we know it
                if self.pending_age > self.merge_frames + self.start_frames:
                    self._emit_pending()
            return

        self.frames.append(f)
        if is_speech:
            self.speech_count += 1
            self.speech_db.append(20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-9))
            self.silence_count = 0
        else:
            if self.silence_count == 0:
                self.need_silence, self.end_kind = self._pick_silence()
            self.silence_count += 1

        hit_max = len(self.frames) >= self.max_frames
        if self.silence_count >= self.need_silence or hit_max:
            if (self.adaptive and not hit_max and self.merge_frames > 0
                    and self.speech_count < self.fragment_frames):
                self.pending = (self.frames, self.speech_count)  # hold: maybe more is coming
                self.pending_age = 0
            else:
                self._emit(self.frames, self.speech_count, "max" if hit_max else self.end_kind)
            self.active = False
            self.frames = []
            self.preroll = []
            self.start_streak = 0
