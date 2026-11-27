"""Wiring: the capture, translate and TTS threads, hotkeys, and the overlay."""

import queue
import sys
import threading
import time

from lingo_overlay import perf_limits
from lingo_overlay.asr import load_model
from lingo_overlay.audio.capture import capture_loop, print_devices
from lingo_overlay.config import ConfigError, load_config
from lingo_overlay.hotkeys import setup_hotkeys
from lingo_overlay.langid import lid_gate, load_lid
from lingo_overlay.log import log
from lingo_overlay.speakers import load_speakers, split_by_speaker
from lingo_overlay.tts import tts_loop
from lingo_overlay.ui import Overlay


def translate_loop(cfg, seg_q, ui_q, tts_q, state):
    log(f"Loading Whisper '{cfg.model}' ({cfg.backend}, {cfg.cpu_threads} CPU threads)...")
    model = load_model(cfg)
    lid = load_lid(cfg)
    speakers = load_speakers(cfg)
    spk_split, spk_window = cfg.speaker_split, cfg.speaker_window_s
    translate_one = _translate_one(cfg, model, lid, ui_q, tts_q, state)
    log("Model ready. Listening for speech...")
    ui_q.put(("status", "Translator ready"))

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

        if speakers is None:
            parts = [(None, audio)]
        else:
            t0 = time.time()
            embed, tracker = speakers
            parts = [(f"P{n}", a) for n, a in
                     split_by_speaker(audio, embed, tracker, spk_split, spk_window)]
            log(f"speakers: {', '.join(p for p, _ in parts)}  ({time.time() - t0:.2f}s)")

        for speaker, part in parts:
            translate_one(part, t_end, speaker)


def _translate_one(cfg, model, lid, ui_q, tts_q, state):
    """Return a function that translates one phrase and sends it to the overlay and voice."""
    lid_conf = float(cfg.lid_confidence)
    skip = {s.lower() for s in cfg.skip_languages}
    skip_thr = cfg.skip_threshold
    min_lang_prob = cfg.min_language_prob

    def translate_one(audio, t_end, speaker=None):
        t0 = time.time()
        who = f"{speaker} " if speaker else ""
        if lid is not None and skip:
            drop, lang, prob = lid_gate(lid, audio, skip, lid_conf)
            if drop:
                log(f"skip  {who}({lang} {prob:.0%}, fast LID, {time.time() - t0:.2f}s)")
                return
        lang, prob, text = model.run(audio, skip, skip_thr, min_lang_prob)
        if text is None:
            log(f"skip  {who}({lang} {prob:.0%}, {time.time() - t0:.2f}s)")
            return

        lag = time.time() - t_end
        heard = getattr(model, "last_source", None)
        log(f"{who}[{lang.upper()} {prob:.0%}] {text}   "
            f"(took {time.time() - t0:.2f}s, {lag:.2f}s after speech ended)"
            + (f"   heard: {heard}" if heard else ""))
        state["last"] = (lang, text, speaker)
        ui_q.put(("sub", lang, text, speaker))
        if state["voice"]:
            tts_q.put(text)

    return translate_one


def main():
    try:
        cfg = load_config()
    except ConfigError as ex:
        raise SystemExit(str(ex)) from None
    perf_limits.apply(cfg, log)
    state = {"enabled": True, "voice": cfg.voice_enabled, "quit": False, "last": None}
    seg_q, ui_q, tts_q = queue.Queue(), queue.Queue(), queue.Queue()

    log(f"Hotkeys: toggle={cfg.hotkey_toggle}  voice={cfg.hotkey_voice}  "
        f"replay={cfg.hotkey_replay}  quit={cfg.hotkey_quit}")
    log(f"Skipping languages: {', '.join(cfg.skip_languages)}")

    for target, args in (
        (capture_loop, (cfg, seg_q, state)),
        (translate_loop, (cfg, seg_q, ui_q, tts_q, state)),
        (tts_loop, (cfg, tts_q, state)),
    ):
        threading.Thread(target=target, args=args, daemon=True).start()
    setup_hotkeys(cfg, state, ui_q, tts_q)

    overlay = Overlay(cfg, ui_q, state)
    try:
        overlay.root.mainloop()
    except KeyboardInterrupt:
        pass
    state["quit"] = True


def cli(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--devices" in argv:
        print_devices()
        sys.exit(0)
    main()
