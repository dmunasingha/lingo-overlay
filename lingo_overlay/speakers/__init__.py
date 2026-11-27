"""Optional speaker labels (P1, P2, ...) so back-to-back callouts by different people stay apart."""

from lingo_overlay.speakers.speechbrain_spk import SPK_MODEL, SpeechBrainEmbedder, load_speakers
from lingo_overlay.speakers.tracker import SpeakerTracker, change_points, split_by_speaker

__all__ = ["SPK_MODEL", "SpeakerTracker", "SpeechBrainEmbedder", "change_points", "load_speakers",
           "split_by_speaker"]
