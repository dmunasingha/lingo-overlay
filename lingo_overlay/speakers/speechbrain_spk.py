"""Voice embeddings for speaker labels (optional): SpeechBrain ECAPA trained on VoxCeleb."""

import time

import numpy as np

from lingo_overlay import ROOT
from lingo_overlay.log import log
from lingo_overlay.speakers.tracker import SpeakerTracker

SPK_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


class SpeechBrainEmbedder:
    """192-dim ECAPA-TDNN speaker embeddings (Apache-2.0, ~80 MB, ~20 ms per phrase on CPU).

    Needs the same optional extras as the language-ID gate (requirements-lid.txt).
    The model is downloaded once into models/ and then used offline.
    """

    def __init__(self, cfg):
        import torch
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.torch = torch
        torch.set_num_threads(max(1, int(cfg.get("lid_threads", 2))))
        savedir = ROOT / "models" / SPK_MODEL.split("/")[-1]
        # COPY instead of symlinks: symlinks need admin / developer mode on Windows.
        kw = dict(source=SPK_MODEL, savedir=str(savedir), run_opts={"device": "cpu"},
                  local_strategy=LocalStrategy.COPY)
        if (savedir / "hyperparams.yaml").is_file():
            kw["source"] = str(savedir)  # already downloaded: stay offline
        self.model = EncoderClassifier.from_hparams(**kw)
        self.model.eval()

    def __call__(self, audio):
        """Return the embedding of a 16 kHz mono float32 clip as a 1-D numpy array."""
        with self.torch.inference_mode():
            wav = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
            return self.model.encode_batch(wav).reshape(-1).numpy()


def load_speakers(cfg):
    """Return (embed, tracker) when speaker labels are on and the model loads, else None."""
    if not cfg.get("speaker_labels", False):
        return None
    try:
        t0 = time.time()
        embed = SpeechBrainEmbedder(cfg)
        log(f"Speaker labels ready ({SPK_MODEL}, {time.time() - t0:.1f}s).")
    except Exception as e:  # missing extras, no network on first run, ...
        log(f"Speaker labels unavailable ({type(e).__name__}: {e}).")
        log("  Install them with: pip install -r requirements-lid.txt. Continuing without labels.")
        return None
    tracker = SpeakerTracker(cfg.get("speaker_similarity", 0.5), cfg.get("speaker_max", 9))
    return embed, tracker
