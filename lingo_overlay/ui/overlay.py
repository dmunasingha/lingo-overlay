"""Subtitle overlay (click-through, always on top).

Text is drawn on a canvas with an outline so it stays readable on any background, with or
without the dark box behind it. Pressing the edit hotkey makes the overlay clickable for a
moment so it can be dragged to any spot on any monitor; pressing it again saves the spot.
"""

import ctypes
import queue
import time
import tkinter as tk

from lingo_overlay.config import DEFAULT_PATH, save_settings
from lingo_overlay.log import log
from lingo_overlay.ui.placement import (
    anchor_from_xy,
    list_monitors,
    outline_offsets,
    pick_monitor,
    subtitle_xy,
)

BOX_BG = "#111111"
KEY_BG = "#010101"  # made fully transparent when the box is turned off
TEXT_FG = "#FFFFFF"
OUTLINE_FG = "#000000"
EDIT_BORDER = "#3FA9F5"
PAD_X, PAD_Y = 18, 8
EDIT_SAMPLE = "[ES]  Subtitles will appear here"

GWL_EXSTYLE = -20
WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x80000, 0x20
WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80, 0x08000000


class Overlay:
    def __init__(self, cfg, ui_q, state, config_path=DEFAULT_PATH):
        self.cfg, self.ui_q, self.state = cfg, ui_q, state
        self.config_path = config_path
        self.monitor_index = cfg.overlay_monitor
        self.anchor = list(cfg.overlay_anchor) if cfg.overlay_anchor is not None else None
        self.editing = False
        self.text = ""
        self.size = (0, 0)
        self.pos = (0, 0)
        self._drag_from = None

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", cfg.overlay_opacity)
        self.bg = BOX_BG if cfg.overlay_background else KEY_BG
        self.root.configure(bg=self.bg)
        if not cfg.overlay_background:
            self.root.attributes("-transparentcolor", KEY_BG)
        self.font = ("Segoe UI Semibold", cfg.overlay_font_size)
        self.canvas = tk.Canvas(self.root, bg=self.bg, highlightthickness=0, bd=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.hide_at = 0
        self.root.withdraw()
        self.root.update_idletasks()
        self._set_click_through(True)
        self.root.after(50, self.poll)

    # ---- window style
    def _hwnd(self):
        return ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()

    def _set_click_through(self, on):
        hwnd = self._hwnd()
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        style = style | WS_EX_TRANSPARENT if on else style & ~WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

    # ---- drawing
    def _monitors(self):
        return list_monitors((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))

    def _draw(self, text, mon):
        c = self.canvas
        c.delete("all")
        wrap = int(mon.width * 0.6)
        ow = self.cfg.overlay_outline
        x0, y0 = PAD_X + ow, PAD_Y + ow
        for dx, dy in outline_offsets(ow):
            c.create_text(x0 + dx, y0 + dy, text=text, fill=OUTLINE_FG, font=self.font,
                          width=wrap, justify="center", anchor="nw")
        main = c.create_text(x0, y0, text=text, fill=TEXT_FG, font=self.font,
                             width=wrap, justify="center", anchor="nw")
        x1, y1, x2, y2 = c.bbox(main)
        w = (x2 - x1) + 2 * (PAD_X + ow)
        h = (y2 - y1) + 2 * (PAD_Y + ow)
        if self.editing:
            c.create_rectangle(1, 1, w - 2, h - 2, outline=EDIT_BORDER, width=2, dash=(6, 4))
        c.config(width=w, height=h)
        return w, h

    def show(self, text, seconds=None):
        if not self.cfg.overlay_enabled and not self.editing:
            return
        self.text = text
        monitors = self._monitors()
        mon = pick_monitor(monitors, self.monitor_index)
        w, h = self._draw(text, mon)
        if self.editing and self.size != (0, 0):
            # keep the dragged spot; grow upward from the same bottom-centre
            x = round(self.pos[0] + self.size[0] / 2 - w / 2)
            y = self.pos[1] + self.size[1] - h
        else:
            x, y = subtitle_xy(mon, w, h, self.anchor, self.cfg.overlay_bottom_offset)
        self._place(x, y, w, h)
        self.root.deiconify()
        self.root.attributes("-topmost", True)
        self.hide_at = 0 if self.editing else time.time() + (seconds or self.cfg.overlay_seconds)

    def _place(self, x, y, w, h):
        self.pos, self.size = (x, y), (w, h)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ---- edit mode (drag to position)
    def toggle_edit(self):
        if not self.editing:
            self.editing = True
            self.size = (0, 0)
            self._set_click_through(False)
            self.show(self.text or EDIT_SAMPLE)
            log(f"Overlay edit mode: drag the subtitle, then press {self.cfg.hotkey_edit} to save")
            return
        self.editing = False
        self._set_click_through(True)
        w, h = self.size
        self.monitor_index, self.anchor = anchor_from_xy(self._monitors(), *self.pos, w, h)
        try:
            save_settings(self.config_path, {"overlay_monitor": self.monitor_index,
                                             "overlay_anchor": self.anchor})
            msg = "Subtitle position saved"
        except Exception as ex:  # keep the position for this session even if the file can't be written
            log(f"Could not save the subtitle position to {self.config_path}: {ex}")
            msg = "Subtitle position set (not saved)"
        log(f"{msg}: monitor {self.monitor_index}, anchor {self.anchor}")
        self.show(msg, seconds=2)

    def _drag_start(self, event):
        if self.editing:
            self._drag_from = (event.x_root - self.pos[0], event.y_root - self.pos[1])

    def _drag_move(self, event):
        if self.editing and self._drag_from:
            self._place(event.x_root - self._drag_from[0], event.y_root - self._drag_from[1], *self.size)

    # ---- main loop
    def poll(self):
        if self.state["quit"]:
            self.root.destroy()
            return
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "sub":
                    self.show(f"[{msg[1].upper()}]  {msg[2]}")
                elif msg[0] == "status":
                    self.show(msg[1], seconds=2)
                elif msg[0] == "edit":
                    self.toggle_edit()
        except queue.Empty:
            pass
        if self.hide_at and time.time() > self.hide_at:
            self.root.withdraw()
            self.hide_at = 0
        self.root.after(50, self.poll)
