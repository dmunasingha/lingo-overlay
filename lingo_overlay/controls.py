"""User actions shared by the global hotkeys and the tray menu."""

from lingo_overlay.log import log


def toggle_enabled(state, ui_q):
    state["enabled"] = not state["enabled"]
    msg = "Translator ON" if state["enabled"] else "Translator OFF"
    log(msg)
    ui_q.put(("status", msg))


def toggle_voice(state, ui_q):
    state["voice"] = not state["voice"]
    msg = "English voice ON" if state["voice"] else "English voice OFF"
    log(msg)
    ui_q.put(("status", msg))


def replay(state, ui_q, tts_q):
    if state["last"]:
        lang, text = state["last"]
        ui_q.put(("sub", lang, text))
        tts_q.put(text)


def request_quit(state):
    log("Quitting...")
    state["quit"] = True
