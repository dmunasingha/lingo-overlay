"""Measure phrase ending: latency gain vs sentences cut in half.

Takes the local Tatoeba clips (``benchmarks/clips``, from scripts/fetch_clips.py) or any folder
of audio files, joins them into one long "voice chat" stream with realistic gaps between
speakers, and runs the real ``lingo_overlay.vad.Segmenter`` over it with different settings.

For every sentence (one clip) it reports:
  end wait     time from the speaker's last loud frame to the phrase being emitted
               (only for sentences that ended a phrase cleanly)
  cut          the sentence was split across 2+ phrases (a cut in the middle of it)
  merged       the sentence came out in the same phrase as the previous sentence

No Whisper involved, so it runs in seconds.

Usage:
    python scripts/endpoint_benchmark.py                   # benchmarks/clips
    python scripts/endpoint_benchmark.py --clips my_wavs --gap 0.5 1.5 --seed 2
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingo_overlay.audio import TARGET_SR  # noqa: E402
from lingo_overlay.vad import FRAME, Segmenter  # noqa: E402

# Energy detector: frame indices below assume its 30 ms frames.
BASE = {"vad_max_segment_s": 7, "vad_min_speech_ms": 400, "vad_threshold_db": -45, "vad_backend": "energy"}

SETTINGS = {
    "fixed 450 ms (v0.1)": {"vad_adaptive": False, "vad_silence_ms": 450},
    "fixed 250 ms": {"vad_adaptive": False, "vad_silence_ms": 250},
    "adaptive (default)": {"vad_adaptive": True},
    "adaptive, no merge": {"vad_adaptive": True, "vad_merge_gap_ms": 0},
    "adaptive, fast 200": {"vad_adaptive": True, "vad_fast_silence_ms": 200},
}


def decode_16k_mono(path):
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            chunks += [o.to_ndarray().reshape(-1) for o in resampler.resample(frame)]
        chunks += [o.to_ndarray().reshape(-1) for o in resampler.resample(None)]
    if not chunks:
        return np.zeros(0, np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def loud_span(audio, threshold_db):
    """(first, last) 30 ms frame louder than the threshold, or None."""
    n = len(audio) // FRAME
    frames = audio[: n * FRAME].reshape(n, FRAME)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    loud = np.flatnonzero(db > threshold_db)
    return (int(loud[0]), int(loud[-1])) if len(loud) else None


def build_stream(clips, gap_range, seed, threshold_db):
    """Concatenate clips with random silent gaps. Returns audio and per-sentence frame spans."""
    rng = random.Random(seed)
    noise = np.random.default_rng(seed)
    parts, spans, pos = [], [], 0
    for audio in clips:
        span = loud_span(audio, threshold_db)
        if span is None:
            continue
        audio = audio[: (len(audio) // FRAME) * FRAME]
        gap = int(rng.uniform(*gap_range) * 1000 / 30) * FRAME
        parts += [np.zeros(gap, np.float32), audio]
        pos += gap // FRAME
        spans.append((pos + span[0], pos + span[1]))
        pos += len(audio) // FRAME
    parts.append(np.zeros(TARGET_SR * 2, np.float32))
    stream = np.concatenate(parts)
    stream += noise.normal(0, 10 ** (-70 / 20), len(stream)).astype(np.float32)  # faint hiss
    return stream, spans


class Recorder:
    """Stands in for the phrase queue; records which frames each phrase covered."""

    def __init__(self):
        self.seg = None
        self.phrases = []  # (first_frame, emit_frame)

    def put(self, item):
        audio, _ = item
        emit = self.seg.frame_idx + 1
        n = len(audio) // FRAME
        # A held fragment is emitted some frames after its audio ends.
        end = emit if self.seg.last_emit[0] != "fragment" else emit - self.seg.pending_age
        self.phrases.append((end - n, end, emit))


def run(stream, spans, overrides):
    rec = Recorder()
    seg = Segmenter({**BASE, **overrides}, rec)
    rec.seg = seg
    for i in range(0, len(stream), 4800):  # 100 ms blocks, as in capture_loop
        seg.feed(stream[i:i + 4800])
    seg.flush()

    waits, cuts, merged, missed = [], 0, 0, 0
    for k, (s0, s1) in enumerate(spans):
        covering = [p for p in rec.phrases if p[0] <= s1 and p[1] > s0]
        if not covering:
            missed += 1
            continue
        if len(covering) > 1:
            cuts += 1
            continue
        first, _, emit = covering[0]
        if k > 0 and first <= spans[k - 1][1]:
            merged += 1
            continue
        if k + 1 < len(spans) and covering[0][1] > spans[k + 1][0]:
            continue  # counted as "merged" for the next sentence
        waits.append((emit - s1 - 1) * 0.030)
    return {
        "n": len(spans), "waits": waits, "cut": cuts, "merged": merged, "missed": missed,
        "phrases": len(rec.phrases),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=str(ROOT / "benchmarks" / "clips"))
    ap.add_argument("--gap", nargs=2, type=float, default=(0.4, 1.5), metavar=("MIN_S", "MAX_S"),
                    help="random silence between sentences (default 0.4 1.5)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    files = sorted(p for p in Path(args.clips).iterdir() if p.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac"})
    if not files:
        raise SystemExit(f"No audio files in {args.clips}. Run scripts/fetch_clips.py first.")
    random.Random(args.seed).shuffle(files)
    clips = [decode_16k_mono(p) for p in files]
    stream, spans = build_stream(clips, args.gap, args.seed, BASE["vad_threshold_db"])
    print(f"{len(spans)} sentences, {len(stream) / TARGET_SR:.0f} s of audio, "
          f"gaps {args.gap[0]}-{args.gap[1]} s\n")

    print("| Setting | End wait p50 | End wait p95 | Mean | Cut in half | Merged with previous | Missed |")
    print("|---|---|---|---|---|---|---|")
    for name, overrides in SETTINGS.items():
        r = run(stream, spans, overrides)
        w = r["waits"]
        p50, p95 = (np.percentile(w, 50), np.percentile(w, 95)) if w else (float("nan"),) * 2
        mean = statistics.mean(w) if w else float("nan")
        print(f"| {name} | {p50:.2f} s | {p95:.2f} s | {mean:.2f} s | "
              f"{r['cut'] / r['n']:.1%} | {r['merged'] / r['n']:.1%} | {r['missed']} |")


if __name__ == "__main__":
    main()
