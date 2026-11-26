"""Wiring: start the capture, translate and TTS threads, hotkeys, and the overlay."""

import queue
import sys
import threading

from lingo_overlay.asr import translate_loop
from lingo_overlay.audio.capture import capture_loop, print_devices
from lingo_overlay.config import ConfigError, load_config
from lingo_overlay.hotkeys import setup_hotkeys
from lingo_overlay.log import log
from lingo_overlay.tts import tts_loop
from lingo_overlay.ui import Overlay


def main():
    try:
        cfg = load_config()
    except ConfigError as ex:
        raise SystemExit(str(ex)) from None
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
