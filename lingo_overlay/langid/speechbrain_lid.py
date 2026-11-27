"""Fast language-ID gate (optional): drop skip-language speech before Whisper runs."""

import time

import numpy as np

from lingo_overlay import ROOT
from lingo_overlay.log import log

LID_MODEL = "speechbrain/lang-id-voxlingua107-ecapa"
# VoxLingua107 codes that differ from Whisper's.
LID_ALIASES = {"iw": "he"}


class SpeechBrainLID:
    """SpeechBrain VoxLingua107 ECAPA language ID (Apache-2.0, 107 languages).

    Needs the optional extras in requirements-lid.txt (torch, torchaudio, speechbrain).
    The model (~85 MB) is downloaded once into models/ and then used offline.
    """

    def __init__(self, cfg):
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.torch = torch
        torch.set_num_threads(max(1, int(cfg.get("lid_threads", 2))))
        savedir = ROOT / "models" / LID_MODEL.split("/")[-1]
        # COPY instead of symlinks: symlinks need admin / developer mode on Windows.
        kw = dict(source=LID_MODEL, savedir=str(savedir), run_opts={"device": "cpu"},
                  local_strategy=LocalStrategy.COPY)
        if (savedir / "hyperparams.yaml").is_file():
            kw["source"] = str(savedir)  # already downloaded: stay offline
        self.model = EncoderClassifier.from_hparams(**kw)
        self.model.eval()
        labels = self.model.hparams.label_encoder.ind2lab
        self.codes = []
        for i in range(len(labels)):
            code = str(labels[i]).split(":")[0].strip().lower()
            self.codes.append(LID_ALIASES.get(code, code))

    def probs(self, audio):
        """Return {lang_code: probability} for a 16 kHz mono float32 clip."""
        with self.torch.inference_mode():
            wav = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
            log_probs = self.model.classify_batch(wav)[0][0]
            p = log_probs.exp().tolist()
        return dict(zip(self.codes, p, strict=True))


def lid_gate(lid, audio, skip, confidence):
    """Decide whether a clip can be dropped without running Whisper.

    Returns (drop, lang, prob). drop is True only when the fast model is confident
    (combined probability of the skip languages >= confidence). Anything else,
    including low-confidence English, goes on to Whisper, whose own language ID
    then decides as before.
    """
    probs = lid.probs(audio)
    lang = max(probs, key=probs.get)
    skip_p = sum(probs.get(code, 0.0) for code in skip)
    return skip_p >= confidence, lang, probs[lang]


def load_lid(cfg):
    """Load the configured fast language-ID gate, or None to use Whisper's LID only."""
    backend = str(cfg.get("lid_backend", "whisper")).lower()
    if backend == "whisper":
        return None
    if backend != "speechbrain":
        log(f"Unknown lid_backend '{backend}' - using Whisper's language detection.")
        return None
    try:
        t0 = time.time()
        lid = SpeechBrainLID(cfg)
        log(f"Fast language-ID gate ready ({LID_MODEL}, {time.time() - t0:.1f}s).")
        return lid
    except Exception as e:  # missing extras, no network on first run, ...
        log(f"Fast language-ID gate unavailable ({type(e).__name__}: {e}).")
        log("  Install it with: pip install -r requirements-lid.txt. Using Whisper's detection.")
        return None
