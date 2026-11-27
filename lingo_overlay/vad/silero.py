"""Silero VAD speech detector."""

from pathlib import Path

import numpy as np


class SileroDetector:
    """Silero VAD v6 (MIT), streamed frame by frame through ONNX Runtime on one CPU thread.

    Uses the model file bundled with faster-whisper, so nothing extra is downloaded.
    The model takes 512-sample (32 ms) frames at 16 kHz plus 64 samples of context.
    """

    name = "silero"
    frame_len = 512  # 32 ms @ 16 kHz (fixed by the model)
    context_len = 64

    def __init__(self, cfg):
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        path = Path(get_assets_path()) / "silero_vad_v6.onnx"
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=opts)
        self.threshold = float(cfg.get("vad_speech_threshold", 0.5))
        # Hysteresis (as in Silero's own VADIterator): once speech has started,
        # keep it going until the probability drops clearly below the threshold.
        self.neg_threshold = max(self.threshold - 0.15, 0.01)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.context_len, dtype=np.float32)

    def prob(self, f):
        x = np.concatenate([self.context, f]).astype(np.float32, copy=False)[None, :]
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = f[-self.context_len:]
        return float(out.ravel()[0])

    def is_speech(self, f, active):
        return self.prob(f) >= (self.neg_threshold if active else self.threshold)
