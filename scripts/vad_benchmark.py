"""Compare the speech detectors (energy vs Silero) without running Whisper.

    python scripts/vad_benchmark.py                    # synthetic noise only
    python scripts/vad_benchmark.py --clips benchmarks/clips

Reports, per backend:
  * false triggers per minute on a synthetic, noise-only clip set (keyboard clicks,
    breathing, gunfire, engine hum, music-like tones, broadband noise);
  * on real speech clips (optional): miss rate, extra segments, and the delay from
    the end of the clip audio to the moment the phrase is emitted, both clean and
    with game-like noise mixed in;
  * CPU cost per second of audio.

The noise set is generated from a fixed seed, so it is redistributable and
results are reproducible. Output is a markdown table you can paste into a PR.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingo_overlay.audio import TARGET_SR  # noqa: E402
from lingo_overlay.config import load_config  # noqa: E402
from lingo_overlay.vad import Segmenter  # noqa: E402

SR = TARGET_SR
BLOCK = SR // 10  # capture delivers 100 ms blocks


# --------------------------------------------------------------------------
# Synthetic noise (no speech in any of these)
# --------------------------------------------------------------------------
def _db(x, level_db):
    rms = np.sqrt(np.mean(x * x)) + 1e-12
    return (x / rms * 10 ** (level_db / 20)).astype(np.float32)


def _lowpass(x, alpha):
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def noise_set(seconds=20, seed=0):
    rng = np.random.default_rng(seed)
    n = seconds * SR
    t = np.arange(n) / SR
    out = {}

    out["white -40 dB"] = _db(rng.standard_normal(n), -40)
    out["white -25 dB"] = _db(rng.standard_normal(n), -25)
    out["rumble -25 dB"] = _db(_lowpass(rng.standard_normal(n), 0.02), -25)

    clicks = np.zeros(n, np.float32)
    k = np.exp(-np.arange(int(0.006 * SR)) / (0.0015 * SR)) * rng.standard_normal(int(0.006 * SR))
    for pos in np.cumsum(rng.exponential(SR / 9, size=400)).astype(int):  # ~9 keys/s
        if pos + len(k) < n:
            clicks[pos:pos + len(k)] += k * rng.uniform(0.3, 1.0)
    out["keyboard -30 dB"] = _db(clicks, -30)

    breath = _lowpass(rng.standard_normal(n), 0.15) - _lowpass(rng.standard_normal(n), 0.01)
    env = np.clip(np.sin(2 * np.pi * t / 2.6), 0, None) ** 2  # ~1 s in-breath every 2.6 s
    out["breathing -32 dB"] = _db(breath * env, -32)

    shots = np.zeros(n, np.float32)
    shot = rng.standard_normal(int(0.25 * SR)) * np.exp(-np.arange(int(0.25 * SR)) / (0.04 * SR))
    pos = 0
    while True:
        pos += int(rng.uniform(0.08, 1.2) * SR)  # bursts and single shots
        if pos + len(shot) >= n:
            break
        shots[pos:pos + len(shot)] += shot
    out["gunfire -22 dB"] = _db(shots, -22)

    hum = sum(np.sin(2 * np.pi * 85 * h * t) / h for h in range(1, 8))
    hum *= 1 + 0.3 * np.sin(2 * np.pi * 0.7 * t)
    out["engine -25 dB"] = _db(hum, -25)

    notes = 220 * 2 ** (rng.integers(0, 24, size=(seconds * 4, 3)) / 12)
    music = np.zeros(n)
    step = SR // 4
    for i, chord in enumerate(notes):
        s = slice(i * step, min((i + 1) * step, n))
        music[s] = sum(np.sin(2 * np.pi * f * t[s]) for f in chord)
    out["music -28 dB"] = _db(music, -28)
    return out


def game_noise(n, seed):
    """Keyboard + gunfire + rumble, mixed in under speech clips."""
    ns = noise_set(seconds=int(np.ceil(n / SR)) + 1, seed=seed)
    return (ns["keyboard -30 dB"] + ns["gunfire -22 dB"] + ns["rumble -25 dB"])[:n]


# --------------------------------------------------------------------------
# Running a detector the way capture_loop does
# --------------------------------------------------------------------------
class _Recorder:
    def __init__(self):
        self.pos = 0
        self.emitted = []

    def put(self, item):
        self.emitted.append((self.pos, len(item[0])))


def run(backend, cfg, audio):
    """Feed audio in 100 ms blocks. Returns ([emit sample positions], cpu seconds)."""
    c = dict(cfg, vad_backend=backend)
    rec = _Recorder()
    seg = Segmenter(c, rec)
    if seg.detector.name != backend:
        raise SystemExit(f"{backend} detector could not be loaded")
    t0 = time.process_time()
    for i in range(0, len(audio), BLOCK):
        rec.pos = min(i + BLOCK, len(audio))
        seg.feed(audio[i:i + BLOCK])
    return [p for p, _ in rec.emitted], time.process_time() - t0


def load_clip(path):
    import av

    with av.open(str(path)) as f:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        parts = [fr.to_ndarray().ravel() for frame in f.decode(audio=0) for fr in rs.resample(frame)]
        parts += [fr.to_ndarray().ravel() for fr in rs.resample(None)]
    return np.concatenate(parts).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, help="folder of speech clips (e.g. benchmarks/clips)")
    ap.add_argument("--backends", default="energy,silero")
    args = ap.parse_args()
    cfg = load_config()
    backends = args.backends.split(",")

    print("## Noise-only false triggers (segments per minute)\n")
    noises = noise_set()
    print("| Noise | " + " | ".join(backends) + " |")
    print("|---|" + "---|" * len(backends))
    totals = dict.fromkeys(backends, 0)
    cpu = dict.fromkeys(backends, 0.0)
    cpu_audio = dict.fromkeys(backends, 0.0)
    for name, audio in noises.items():
        row = []
        for b in backends:
            emits, c = run(b, cfg, audio)
            totals[b] += len(emits)
            cpu[b] += c
            cpu_audio[b] += len(audio) / SR
            row.append(f"{len(emits) * 60 / (len(audio) / SR):.1f}")
        print(f"| {name} | " + " | ".join(row) + " |")
    minutes = sum(len(a) for a in noises.values()) / SR / 60
    print("| **all** | " + " | ".join(f"**{totals[b] / minutes:.1f}**" for b in backends) + " |")

    if args.clips:
        files = sorted(p for p in args.clips.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"})
        if not files:
            raise SystemExit(f"No clips in {args.clips}")
        print(f"\n## Speech clips ({len(files)})\n")
        print("| Backend | Condition | Miss | Extra segments | Emit delay p50 | p95 |")
        print("|---|---|---|---|---|---|")
        clips = [load_clip(p) for p in files]
        for b in backends:
            for cond in ("clean", "game noise"):
                misses, extra, delays = 0, 0, []
                for i, speech in enumerate(clips):
                    lead, tail = np.zeros(SR, np.float32), np.zeros(int(1.5 * SR), np.float32)
                    audio = np.concatenate([lead, speech, tail])
                    if cond == "game noise":
                        audio = audio + game_noise(len(audio), seed=1000 + i)
                    emits, c = run(b, cfg, audio)
                    cpu[b] += c
                    cpu_audio[b] += len(audio) / SR
                    if not emits:
                        misses += 1
                        continue
                    extra += len(emits) - 1
                    delays.append((emits[-1] - len(lead) - len(speech)) / SR)
                d = np.array(delays) if delays else np.array([np.nan])
                print(f"| {b} | {cond} | {misses / len(clips):.1%} | {extra} | "
                      f"{np.percentile(d, 50):.2f} s | {np.percentile(d, 95):.2f} s |")
        print("\nEmit delay = clip audio end to phrase emitted. Clips have their own trailing")
        print("silence, so compare backends with each other rather than reading it as absolute.")

    print("\n## CPU\n")
    print("| Backend | CPU ms per second of audio |")
    print("|---|---|")
    for b in backends:
        print(f"| {b} | {cpu[b] / cpu_audio[b] * 1000:.2f} |")


if __name__ == "__main__":
    main()
