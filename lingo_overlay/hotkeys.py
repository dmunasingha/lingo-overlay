"""Global hotkeys."""

from lingo_overlay.log import log


def setup_hotkeys(cfg, state, ui_q, tts_q):
    import keyboard

    def toggle():
        state["enabled"] = not state["enabled"]
        msg = "Translator ON" if state["enabled"] else "Translator OFF"
        log(msg)
        ui_q.put(("status", msg))

    def voice():
        state["voice"] = not state["voice"]
        msg = "English voice ON" if state["voice"] else "English voice OFF"
        log(msg)
        ui_q.put(("status", msg))

    def replay():
        if state["last"]:
            lang, text, speaker = state["last"]
            ui_q.put(("sub", lang, text, speaker))
            tts_q.put(text)

    def quit_():
        log("Quitting...")
        state["quit"] = True

    keyboard.add_hotkey(cfg.hotkey_toggle, toggle)
    keyboard.add_hotkey(cfg.hotkey_voice, voice)
    keyboard.add_hotkey(cfg.hotkey_replay, replay)
    keyboard.add_hotkey(cfg.hotkey_quit, quit_)
