"""System tray icon: shows off / loading / listening / translating, with a small menu.

pystray and Pillow are imported lazily, so the rest of the app (and the tests) work without
them. If they are missing the tray is skipped with a log line.
"""

import os
import threading
import time

from lingo_overlay import controls
from lingo_overlay.config import DEFAULT_PATH
from lingo_overlay.log import log

OFF, LOADING, LISTENING, TRANSLATING = "off", "loading", "listening", "translating"

# Fill colour of the status dot for each state.
COLORS = {
    OFF: (128, 128, 128),
    LOADING: (66, 135, 245),
    LISTENING: (46, 204, 113),
    TRANSLATING: (243, 156, 18),
}

LABELS = {
    OFF: "Off",
    LOADING: "Loading model...",
    LISTENING: "Listening",
    TRANSLATING: "Translating",
}


def tray_status(state):
    """Map the shared app state dict to one of the tray states."""
    if not state.get("enabled", True):
        return OFF
    if not state.get("ready", False):
        return LOADING
    if state.get("busy", False):
        return TRANSLATING
    return LISTENING


def tooltip(status):
    return f"Lingo Overlay: {LABELS[status]}"


def make_icon_image(status, size=64):
    """A round status dot on a transparent background."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = size // 10
    draw.ellipse((pad, pad, size - pad, size - pad), fill=COLORS[status] + (255,),
                 outline=(20, 20, 20, 255), width=max(1, size // 16))
    return img


def open_settings(path=DEFAULT_PATH):
    """Open config.json in the default editor (until there is a settings window)."""
    try:
        os.startfile(str(path))
    except (AttributeError, OSError) as ex:
        log(f"Tray: could not open {path}: {ex}")


class Tray:
    def __init__(self, cfg, state, ui_q):
        import pystray

        self.state, self.ui_q = state, ui_q
        self._status = tray_status(state)
        self._images = {s: make_icon_image(s) for s in COLORS}
        item = pystray.MenuItem
        menu = pystray.Menu(
            item(lambda _: tooltip(tray_status(state)), None, enabled=False),
            pystray.Menu.SEPARATOR,
            item("Translator", self._toggle, checked=lambda _: state["enabled"], default=True),
            item("Spoken translation", self._voice, checked=lambda _: state["voice"]),
            item("Settings...", lambda: open_settings()),
            pystray.Menu.SEPARATOR,
            item("Quit", self._quit),
        )
        self.icon = pystray.Icon("lingo_overlay", self._images[self._status],
                                 tooltip(self._status), menu)

    def _toggle(self):
        controls.toggle_enabled(self.state, self.ui_q)
        self.refresh()

    def _voice(self):
        controls.toggle_voice(self.state, self.ui_q)
        self.icon.update_menu()

    def _quit(self):
        controls.request_quit(self.state)

    def refresh(self):
        status = tray_status(self.state)
        if status != self._status:
            self._status = status
            self.icon.icon = self._images[status]
            self.icon.title = tooltip(status)
            self.icon.update_menu()

    def _watch(self):
        while not self.state["quit"]:
            self.refresh()
            time.sleep(0.2)
        self.icon.stop()

    def start(self):
        threading.Thread(target=self.icon.run, daemon=True).start()
        threading.Thread(target=self._watch, daemon=True).start()

    def stop(self):
        try:
            self.icon.stop()
        except Exception:
            pass


def start_tray(cfg, state, ui_q):
    """Start the tray icon, or return None if it is turned off or unavailable."""
    if not cfg.tray_enabled:
        return None
    try:
        tray = Tray(cfg, state, ui_q)
    except ImportError:
        log("Tray icon unavailable: pip install pystray pillow")
        return None
    except Exception as ex:  # the tray is optional: never stop the translator
        log(f"Tray icon failed to start: {ex}")
        return None
    tray.start()
    return tray
