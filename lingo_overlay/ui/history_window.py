"""Translation history window: the last N translations with time and language, copyable."""

import tkinter as tk


class HistoryWindow:
    """A normal (focusable, not click-through) window listing recent translations.

    Toggled by the history hotkey. Select one or more lines and press Ctrl+C or
    "Copy selected", or use "Copy all".
    """

    def __init__(self, root, history):
        self.root, self.history = root, history
        self.win = None
        self.listbox = None
        self._shown = 0  # entries currently in the listbox, to skip needless redraws

    def is_open(self) -> bool:
        return self.win is not None and bool(self.win.winfo_exists())

    def toggle(self):
        if self.is_open():
            self.close()
        else:
            self.open()

    def open(self):
        if self.is_open():
            self.win.deiconify()
            self.win.lift()
            return
        self.win = tk.Toplevel(self.root)
        self.win.title("Lingo Overlay: translation history")
        self.win.geometry("720x420")
        self.win.attributes("-topmost", True)
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        frame = tk.Frame(self.win)
        frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        scroll = tk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        self.listbox = tk.Listbox(
            frame, selectmode="extended", font=("Segoe UI", 11),
            activestyle="none", yscrollcommand=scroll.set,
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.listbox.yview)

        buttons = tk.Frame(self.win)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(buttons, text="Copy selected", command=self.copy_selected).pack(side="left")
        tk.Button(buttons, text="Copy all", command=self.copy_all).pack(side="left", padx=6)
        tk.Button(buttons, text="Clear", command=self.clear).pack(side="left")
        tk.Button(buttons, text="Close", command=self.close).pack(side="right")

        self.listbox.bind("<Control-c>", lambda _e: self.copy_selected())
        self.listbox.bind("<Control-a>", lambda _e: self.listbox.select_set(0, "end"))
        self.win.bind("<Escape>", lambda _e: self.close())

        self._shown = -1
        self.refresh()
        self.win.focus_force()

    def close(self):
        if self.win is not None:
            try:
                self.win.destroy()
            except tk.TclError:
                pass
        self.win = self.listbox = None

    def refresh(self):
        """Redraw if the history changed. Called from the overlay's poll loop."""
        if not self.is_open():
            return
        entries = self.history.entries()
        # History is a ring buffer, so compare on the newest entry as well as the count.
        key = (len(entries), entries[-1].timestamp if entries else 0)
        if key == self._shown:
            return
        at_bottom = self.listbox.yview()[1] >= 0.999
        self.listbox.delete(0, "end")
        for e in entries:
            self.listbox.insert("end", e.format())
        if at_bottom or self._shown == -1:
            self.listbox.see("end")
        self._shown = key

    def _copy(self, text):
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def copy_selected(self):
        if self.listbox is None:
            return
        lines = [self.listbox.get(i) for i in self.listbox.curselection()]
        self._copy("\n".join(lines))

    def copy_all(self):
        self._copy(self.history.as_text())

    def clear(self):
        self.history.clear()
        self.refresh()
