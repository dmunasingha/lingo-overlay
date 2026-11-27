"""Settings window: edit the common config.json settings without touching JSON.

Run with ``python translator.py --settings`` (or ``settings.bat``). Changes are saved to
config.json and take effect the next time the translator starts.

The form logic (``FIELDS``, ``form_values`` and ``parse_form``) is plain Python so it can be
tested without a display; only ``SettingsWindow`` needs Tk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lingo_overlay.config import Config, ConfigError, load_config, save_config

MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
BACKENDS = ("whisper.cpp", "faster-whisper")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str  # "text" | "choice" | "device" | "langs" | "int" | "float" | "percent" | "bool"
    group: str
    choices: tuple[str, ...] = ()
    hint: str = ""


FIELDS: tuple[Field, ...] = (
    Field("input_device", "Input device", "device", "Audio",
          hint="Part of the recording device name, e.g. CABLE Output"),
    Field("fallback_to_speaker_loopback", "Use speaker audio if the device is missing", "bool", "Audio"),
    Field("skip_languages", "Skip languages", "langs", "Languages",
          hint="Comma-separated codes left untranslated, e.g. en, si"),
    Field("model", "Whisper model", "choice", "Speech model", MODELS,
          hint="small is fast and light; medium translates better but uses ~2x CPU"),
    Field("backend", "Backend", "choice", "Speech model", BACKENDS),
    Field("cpu_threads", "CPU threads", "int", "Speech model",
          hint="Lower to 2-3 if game FPS drops"),
    Field("hotkey_toggle", "Translator on/off", "text", "Hotkeys"),
    Field("hotkey_voice", "Spoken translation on/off", "text", "Hotkeys"),
    Field("hotkey_replay", "Replay last translation", "text", "Hotkeys"),
    Field("hotkey_quit", "Quit", "text", "Hotkeys"),
    Field("overlay_enabled", "Show subtitles", "bool", "Overlay"),
    Field("overlay_font_size", "Font size", "int", "Overlay"),
    Field("overlay_bottom_offset", "Distance from bottom (px)", "int", "Overlay"),
    Field("overlay_seconds", "Seconds on screen", "float", "Overlay"),
    Field("overlay_opacity", "Opacity (%)", "percent", "Overlay"),
    Field("voice_enabled", "Speak translations at start", "bool", "Voice"),
)


def form_values(cfg: Config) -> dict[str, Any]:
    """Config -> the values the form widgets show (strings, or bools for checkboxes)."""
    out: dict[str, Any] = {}
    for f in FIELDS:
        v = cfg[f.key]
        if f.kind == "bool":
            out[f.key] = bool(v)
        elif f.kind == "langs":
            out[f.key] = ", ".join(v)
        elif f.kind == "percent":
            out[f.key] = str(round(v * 100))
        elif f.kind == "float":
            out[f.key] = f"{v:g}"
        else:
            out[f.key] = str(v)
    return out


def _number(text: str, label: str, cast):
    try:
        return cast(str(text).strip())
    except ValueError:
        want = "a whole number" if cast is int else "a number"
        raise ConfigError(f"'{label}' must be {want} (got {text!r})") from None


def parse_form(values: dict[str, Any]) -> dict[str, Any]:
    """Form widget values -> config changes. Raises ConfigError on bad input.

    Range checks (opacity 0-100 %, font size >= 1, ...) are left to ``Config`` so the rules live in
    one place; this only turns text into the right types.
    """
    changes: dict[str, Any] = {}
    for f in FIELDS:
        if f.key not in values:
            continue
        raw = values[f.key]
        if f.kind == "bool":
            changes[f.key] = bool(raw)
        elif f.kind == "langs":
            langs = [s.strip().lower() for s in str(raw).replace(";", ",").split(",")]
            changes[f.key] = list(dict.fromkeys(s for s in langs if s))
        elif f.kind == "int":
            changes[f.key] = _number(raw, f.label, int)
        elif f.kind == "float":
            changes[f.key] = _number(raw, f.label, float)
        elif f.kind == "percent":
            changes[f.key] = _number(raw, f.label, float) / 100
        else:  # text, device, choice (Config rejects an unknown backend)
            changes[f.key] = str(raw).strip()
    hotkeys = [changes[k].lower() for k in ("hotkey_toggle", "hotkey_voice", "hotkey_replay", "hotkey_quit")
               if changes.get(k)]
    if len(hotkeys) != len(set(hotkeys)):
        raise ConfigError("Each hotkey must be different")
    return changes


def list_input_devices() -> list[str]:
    """Names of the recording devices, or [] if soundcard is unavailable."""
    try:
        import soundcard as sc
        return [m.name for m in sc.all_microphones(include_loopback=False)]
    except Exception:  # no audio stack / not Windows: the field stays free text
        return []


class SettingsWindow:  # pragma: no cover - needs a display
    def __init__(self, path: Path | str | None = None):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.path = tk, path
        try:
            cfg = load_config(path)
        except ConfigError as ex:
            from tkinter import messagebox
            messagebox.showwarning("Lingo Overlay settings", f"{ex}\n\nShowing defaults instead.")
            cfg = Config()
        self.root = tk.Tk()
        self.root.title("Lingo Overlay settings")
        self.root.resizable(False, False)
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")

        devices = list_input_devices()
        values = form_values(cfg)
        self.vars: dict[str, Any] = {}
        row, group = 0, None
        for f in FIELDS:
            if f.group != group:
                group = f.group
                ttk.Label(frame, text=group, font=("Segoe UI Semibold", 10)).grid(
                    row=row, column=0, columnspan=2, sticky="w", pady=(10 if row else 0, 2))
                row += 1
            if f.kind == "bool":
                var = tk.BooleanVar(value=values[f.key])
                ttk.Checkbutton(frame, text=f.label, variable=var).grid(
                    row=row, column=0, columnspan=2, sticky="w", padx=(12, 0))
            else:
                var = tk.StringVar(value=values[f.key])
                ttk.Label(frame, text=f.label).grid(row=row, column=0, sticky="w", padx=(12, 8))
                if f.kind == "choice":
                    widget = ttk.Combobox(frame, textvariable=var, values=f.choices, width=34,
                                          state="readonly" if f.key == "backend" else "normal")
                elif f.kind == "device":
                    widget = ttk.Combobox(frame, textvariable=var, values=devices, width=34)
                else:
                    widget = ttk.Entry(frame, textvariable=var, width=37)
                widget.grid(row=row, column=1, sticky="we", pady=1)
            self.vars[f.key] = var
            if f.hint:
                row += 1
                ttk.Label(frame, text=f.hint, foreground="#666666").grid(
                    row=row, column=1, sticky="w")
            row += 1

        self.status = tk.StringVar(value="Changes apply the next time the translator starts.")
        ttk.Label(frame, textvariable=self.status, wraplength=420, justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 4))
        buttons = ttk.Frame(frame)
        buttons.grid(row=row + 1, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Save", command=self.save).pack(side="left", padx=4)
        ttk.Button(buttons, text="Close", command=self.root.destroy).pack(side="left")
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def save(self):
        try:
            changes = parse_form({k: v.get() for k, v in self.vars.items()})
            save_config(changes, self.path)
        except ConfigError as ex:
            self.status.set(str(ex))
            return
        self.status.set("Saved. Restart the translator to apply.")

    def run(self):
        self.root.mainloop()


def open_settings(path: Path | str | None = None) -> None:  # pragma: no cover - needs a display
    SettingsWindow(path).run()
