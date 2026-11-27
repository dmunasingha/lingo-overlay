"""Global hotkeys."""

from lingo_overlay import controls


def setup_hotkeys(cfg, state, ui_q, tts_q):
    import keyboard

    keyboard.add_hotkey(cfg.hotkey_toggle, lambda: controls.toggle_enabled(state, ui_q))
    keyboard.add_hotkey(cfg.hotkey_voice, lambda: controls.toggle_voice(state, ui_q))
    keyboard.add_hotkey(cfg.hotkey_replay, lambda: controls.replay(state, ui_q, tts_q))
    keyboard.add_hotkey(cfg.hotkey_quit, lambda: controls.request_quit(state))
