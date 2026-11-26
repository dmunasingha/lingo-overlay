"""Speech recognition backends behind a common ASRBackend interface."""

import queue
import time

from lingo_overlay.asr.base import ASRBackend, detect_and_translate
from lingo_overlay.log import log

__all__ = ["ASRBackend", "detect_and_translate", "make_backend", "translate_loop"]


def make_backend(cfg) -> ASRBackend:
    from lingo_overlay.asr.faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend.from_config(cfg)


def translate_loop(cfg, seg_q, ui_q, tts_q, state):
    log(f"Loading Whisper '{cfg.model}' on {cfg.device} ({cfg.compute_type})...")
    backend = make_backend(cfg)
    log("Model ready. Listening for speech...")
    ui_q.put(("status", "Translator ready"))
    skip = set(cfg.skip_languages)
    skip_thr = cfg.skip_threshold

    while not state["quit"]:
        try:
            audio, t_end = seg_q.get(timeout=0.5)
        except queue.Empty:
            continue
        # Keep latency low: if phrases piled up, drop the oldest ones.
        while seg_q.qsize() > 2:
            audio, t_end = seg_q.get_nowait()
        if not state["enabled"]:
            continue

        t0 = time.time()
        lang, prob, text = detect_and_translate(backend, audio, skip, skip_thr)
        if text is None:
            log(f"skip  ({lang} {prob:.0%}, {time.time() - t0:.2f}s)")
            continue

        lag = time.time() - t_end
        log(f"[{lang.upper()} {prob:.0%}] {text}   "
            f"(took {time.time() - t0:.2f}s, {lag:.2f}s after speech ended)")
        state["last"] = (lang, text)
        ui_q.put(("sub", lang, text))
        if state["voice"]:
            tts_q.put(text)
