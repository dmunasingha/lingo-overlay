"""Speech detection: per-frame detectors (Silero, energy) and the phrase Segmenter."""

from lingo_overlay.vad.energy import FRAME, EnergyDetector
from lingo_overlay.vad.segmenter import Segmenter, make_detector
from lingo_overlay.vad.silero import SileroDetector

__all__ = ["FRAME", "EnergyDetector", "Segmenter", "SileroDetector", "make_detector"]
