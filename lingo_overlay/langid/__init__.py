"""Optional fast language-ID gate in front of Whisper."""

from lingo_overlay.langid.speechbrain_lid import LID_MODEL, SpeechBrainLID, lid_gate, load_lid

__all__ = ["LID_MODEL", "SpeechBrainLID", "lid_gate", "load_lid"]
