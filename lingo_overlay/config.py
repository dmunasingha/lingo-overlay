"""Typed configuration loaded from config.json, with validation and clear errors.

``Config`` is a frozen dataclass. It is also a read-only mapping (``cfg["model"]``,
``cfg.get("vad_backend")``, ``dict(cfg)``), so the speech detector, language-ID gate and
scripts accept either a ``Config`` or a plain dict of settings.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from lingo_overlay import ROOT
from lingo_overlay.log import log

DEFAULT_PATH = ROOT / "config.json"


class ConfigError(ValueError):
    """Raised when config.json is missing, malformed, or has invalid values."""


@dataclass(frozen=True)
class Config(Mapping):
    # Audio input
    input_device: str = "CABLE Output"
    fallback_to_speaker_loopback: bool = True

    # Language skipping
    skip_languages: list[str] = field(default_factory=lambda: ["en"])
    skip_threshold: float = 0.25
    min_language_prob: float = 0.5  # below this Whisper is unsure: usually not speech
    lid_backend: str = "whisper"  # "whisper" | "speechbrain" (fast gate before Whisper)
    lid_confidence: float = 0.5
    lid_threads: int = 2

    # Text translation (transcript in the original language -> English text)
    text_translation: list[str] = field(default_factory=lambda: ["zh", "yue"])  # Opus-MT languages
    chinese_asr: str = "whisper"  # "whisper" | "sensevoice" (opt-in: FunASR model licence)
    mt_threads: int = 2
    game_glossary: bool = True  # keep VALORANT names/slang intact (glossary/valorant.json)

    # Whisper
    model: str = "small"
    backend: str = "whisper.cpp"  # "whisper.cpp" | "faster-whisper"
    whispercpp_quant: str = "q8_0"
    whispercpp_min_ctx: int = 256
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    process_priority: str = "below_normal"
    cpu_affinity: str | list[int] | None = "off"

    # Speech detector
    vad_backend: str = "silero"  # "silero" | "energy"
    vad_speech_threshold: float = 0.5
    vad_silence_ms: float = 450
    vad_adaptive: bool = True
    vad_fast_silence_ms: float = 250
    vad_slow_silence_ms: float = 700
    vad_merge_gap_ms: float = 300
    vad_fragment_ms: float = 800
    vad_fall_db: float = 3.0
    vad_peak_db: float = 3.0
    vad_max_segment_s: float = 7
    vad_min_speech_ms: float = 400
    vad_threshold_db: float = -45

    # Text-to-speech
    voice_enabled: bool = False
    voice_rate: float = 200
    voice_volume: float = 0.8

    # Overlay
    overlay_enabled: bool = True
    overlay_font_size: int = 20
    overlay_seconds: float = 6
    overlay_bottom_offset: int = 160
    overlay_opacity: float = 0.85

    # History window
    history_size: int = 50

    # Hotkeys
    hotkey_toggle: str = "ctrl+alt+t"
    hotkey_voice: str = "ctrl+alt+v"
    hotkey_replay: str = "ctrl+alt+r"
    hotkey_quit: str = "ctrl+alt+q"
    hotkey_history: str = "ctrl+alt+h"

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

    def _check_values(self) -> list[str]:
        e = []

        def rng(name, lo=None, hi=None, lo_open=False):
            v = getattr(self, name)
            bad_lo = lo is not None and (v <= lo if lo_open else v < lo)
            if bad_lo or (hi is not None and v > hi):
                if lo is not None and hi is not None:
                    bounds = f"between {lo} and {hi}"
                elif lo is not None:
                    bounds = f"> {lo}" if lo_open else f">= {lo}"
                else:
                    bounds = f"<= {hi}"
                e.append(f"'{name}' must be {bounds} (got {v!r})")

        for name in ("skip_threshold", "min_language_prob", "lid_confidence", "vad_speech_threshold",
                     "voice_volume", "overlay_opacity"):
            rng(name, 0, 1)
        for name in ("cpu_threads", "lid_threads", "mt_threads", "whispercpp_min_ctx", "vad_merge_gap_ms",
                     "vad_fragment_ms", "vad_fall_db", "vad_peak_db", "vad_min_speech_ms"):
            rng(name, 0)
        for name in ("vad_silence_ms", "vad_fast_silence_ms", "vad_slow_silence_ms",
                     "vad_max_segment_s", "overlay_seconds"):
            rng(name, 0, lo_open=True)
        rng("vad_threshold_db", hi=0)
        rng("overlay_font_size", 1)
        rng("history_size", 1)
        if not self.model.strip():
            e.append("'model' must not be empty (e.g. \"small\" or \"medium\")")
        if self.backend not in ("whisper.cpp", "faster-whisper"):
            e.append(f"'backend' must be \"whisper.cpp\" or \"faster-whisper\" (got {self.backend!r})")
        if self.device not in ("cpu", "cuda", "auto"):
            e.append(f"'device' must be \"cpu\", \"cuda\" or \"auto\" (got {self.device!r})")
        for name in ("hotkey_toggle", "hotkey_voice", "hotkey_replay", "hotkey_quit", "hotkey_history"):
            if not getattr(self, name).strip():
                e.append(f"'{name}' must not be empty")
        # vad_backend, lid_backend, chinese_asr, process_priority and cpu_affinity are checked where
        # they are used: an unknown value is logged and a safe default is used instead.
        return e

    # ---- read-only mapping, so code written for a dict of settings keeps working
    def __getitem__(self, key: str) -> Any:
        if key not in _FIELD_NAMES_SET:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(_FIELD_NAMES)

    def __len__(self) -> int:
        return len(_FIELD_NAMES)

    __hash__ = None  # holds a list

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        if not isinstance(data, dict):
            raise ConfigError("Invalid config: top level must be a JSON object")
        for key in sorted(data.keys() - _FIELD_NAMES_SET):
            hint = difflib.get_close_matches(key, _FIELD_NAMES, n=1)
            log(f"Config: ignoring unknown setting '{key}'"
                + (f" (did you mean '{hint[0]}'?)" if hint else ""))
        return cls(**{k: v for k, v in data.items() if k in _FIELD_NAMES_SET})

    def replace(self, **changes) -> Config:
        return dataclasses.replace(self, **changes)


_FIELD_NAMES = tuple(f.name for f in fields(Config))
_FIELD_NAMES_SET = frozenset(_FIELD_NAMES)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


_TYPES = {
    # Annotations are strings because of `from __future__ import annotations`.
    "str": (lambda v: isinstance(v, str), "a string"),
    "bool": (lambda v: isinstance(v, bool), "true or false"),
    "int": (_is_int, "a whole number"),
    "float": (lambda v: _is_int(v) or isinstance(v, float), "a number"),
    "list[str]": (lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
                  "a list of strings"),
    "str | list[int] | None": (lambda v: v is None or isinstance(v, str) or isinstance(v, list),
                               "\"off\", \"last:N\" or a list of CPU numbers"),
}


def _check_type(name: str, value: Any, annotation: str) -> str | None:
    if annotation not in _TYPES:  # pragma: no cover - only if a new field type is added without a check
        raise TypeError(f"no validator for {name}: {annotation}")
    ok, want = _TYPES[annotation]
    if ok(value):
        return None
    return f"'{name}' must be {want} (got {json.dumps(value)})"


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path is not None else DEFAULT_PATH
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
