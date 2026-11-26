"""Typed configuration loaded from config.json, with validation and clear errors."""

from __future__ import annotations

import dataclasses
import difflib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from lingo_overlay import ROOT
from lingo_overlay.log import log

DEFAULT_PATH = ROOT / "config.json"


class ConfigError(ValueError):
    """Raised when config.json is missing, malformed, or has invalid values."""


@dataclass(frozen=True)
class Config:
    # Audio input
    input_device: str = "CABLE Output"
    fallback_to_speaker_loopback: bool = True

    # Language skipping
    skip_languages: list[str] = field(default_factory=lambda: ["en"])
    skip_threshold: float = 0.25

    # Whisper
    model: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 6

    # Speech detector
    vad_silence_ms: int = 450
    vad_max_segment_s: float = 7
    vad_min_speech_ms: int = 400
    vad_threshold_db: float = -45

    # Text-to-speech
    voice_enabled: bool = False
    voice_rate: int = 200
    voice_volume: float = 0.8

    # Overlay
    overlay_enabled: bool = True
    overlay_font_size: int = 20
    overlay_seconds: float = 6
    overlay_bottom_offset: int = 160
    overlay_opacity: float = 0.85

    # Hotkeys
    hotkey_toggle: str = "ctrl+alt+t"
    hotkey_voice: str = "ctrl+alt+v"
    hotkey_replay: str = "ctrl+alt+r"
    hotkey_quit: str = "ctrl+alt+q"

    def __post_init__(self):
        errors = []
        for f in fields(self):
            msg = _check_type(f.name, getattr(self, f.name), f.type)
            if msg:
                errors.append(msg)
        if not errors:
            errors += self._check_values()
        if errors:
            raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errors))
        # Normalise language codes once, here, rather than at every use site.
        object.__setattr__(self, "skip_languages", [s.lower() for s in self.skip_languages])

    def _check_values(self) -> list[str]:
        e = []

        def rng(name, lo=None, hi=None):
            v = getattr(self, name)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                bounds = f"between {lo} and {hi}" if lo is not None and hi is not None else (
                    f">= {lo}" if lo is not None else f"<= {hi}")
                e.append(f"'{name}' must be {bounds} (got {v!r})")

        rng("skip_threshold", 0, 1)
        rng("cpu_threads", 0)
        rng("vad_silence_ms", 30)
        rng("vad_max_segment_s", 0.1)
        rng("vad_min_speech_ms", 0)
        rng("vad_threshold_db", hi=0)
        rng("voice_volume", 0, 1)
        rng("overlay_font_size", 1)
        rng("overlay_seconds", 0.1)
        rng("overlay_opacity", 0, 1)
        if not self.model.strip():
            e.append("'model' must not be empty (e.g. \"small\" or \"medium\")")
        if self.device not in ("cpu", "cuda", "auto"):
            e.append(f"'device' must be \"cpu\", \"cuda\" or \"auto\" (got {self.device!r})")
        for name in ("hotkey_toggle", "hotkey_voice", "hotkey_replay", "hotkey_quit"):
            if not getattr(self, name).strip():
                e.append(f"'{name}' must not be empty")
        return e

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        if not isinstance(data, dict):
            raise ConfigError("Invalid config: top level must be a JSON object")
        known = {f.name for f in fields(cls)}
        for key in data.keys() - known:
            hint = difflib.get_close_matches(key, known, n=1)
            log(f"Config: ignoring unknown setting '{key}'"
                + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        return cls(**{k: v for k, v in data.items() if k in known})

    def replace(self, **changes) -> Config:
        return dataclasses.replace(self, **changes)


def _check_type(name: str, value: Any, annotation: str) -> str | None:
    # Annotations are strings because of `from __future__ import annotations`.
    if annotation == "list[str]":
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        want = "a list of strings"
    elif annotation == "bool":
        ok, want = isinstance(value, bool), "true or false"
    elif annotation == "int":
        ok, want = isinstance(value, int) and not isinstance(value, bool), "a whole number"
    elif annotation == "float":
        ok, want = isinstance(value, int | float) and not isinstance(value, bool), "a number"
    elif annotation == "str":
        ok, want = isinstance(value, str), "a string"
    else:  # pragma: no cover - only if a new field type is added without a check
        raise TypeError(f"no validator for {name}: {annotation}")
    if ok:
        return None
    return f"'{name}' must be {want} (got {json.dumps(value)})"


def load_config(path: Path | str = DEFAULT_PATH) -> Config:
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}") from None
    except json.JSONDecodeError as ex:
        raise ConfigError(f"{path.name} is not valid JSON (line {ex.lineno}, column {ex.colno}): "
                          f"{ex.msg}") from None
    try:
        return Config.from_dict(data)
    except ConfigError as ex:
        raise ConfigError(f"{path.name}: {ex}") from None
