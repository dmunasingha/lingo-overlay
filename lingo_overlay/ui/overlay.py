"""Subtitle overlay (click-through, always on top)."""

import ctypes
import queue
import time
import tkinter as tk


def subtitle_text(lang, text, speaker=None):
    """Overlay line, e.g. "P2 [ES]  Push left" (the speaker label only when speaker labels are on)."""
    tag = f"[{lang.upper()}]"
    return f"{speaker} {tag}  {text}" if speaker else f"{tag}  {text}"


class Overlay:
    def __init__(self, cfg, ui_q, state):
        self.cfg, self.ui_q, self.state = cfg, ui_q, state
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", cfg.overlay_opacity)
        self.root.configure(bg="#111111")
        self.label = tk.Label(
            self.root, text="", fg="#FFFFFF", bg="#111111",
            font=("Segoe UI Semibold", cfg.overlay_font_size),
            wraplength=int(self.root.winfo_screenwidth() * 0.6), justify="center",
            padx=18, pady=8,
        )
        self.label.pack()
        self.hide_at = 0
        self.root.withdraw()
        self.root.update_idletasks()
        self._click_through()
        self.root.after(50, self.poll)

    def _click_through(self):
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        GWL_EXSTYLE = -20
        WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x80000, 0x20
        WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80, 0x08000000
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE,
            style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        )

    def show(self, text, seconds=None):
        if not self.cfg.overlay_enabled:
            return
        self.label.config(text=text)
        self.root.update_idletasks()
        w, h = self.label.winfo_reqwidth(), self.label.winfo_reqheight()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = sh - h - self.cfg.overlay_bottom_offset
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.deiconify()
        self.root.attributes("-topmost", True)
        self.hide_at = time.time() + (seconds or self.cfg.overlay_seconds)

    def poll(self):
        if self.state["quit"]:
            self.root.destroy()
            return
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "sub":
                    self.show(subtitle_text(*msg[1:]))
                elif msg[0] == "status":
                    self.show(msg[1], seconds=2)
        except queue.Empty:
            pass
        if self.hide_at and time.time() > self.hide_at:
            self.root.withdraw()
            self.hide_at = 0
        self.root.after(50, self.poll)
