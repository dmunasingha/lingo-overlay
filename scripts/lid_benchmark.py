"""Compare the fast SpeechBrain language-ID gate with Whisper's own language detection.

For every clip in the clip set this measures:
  * SpeechBrain VoxLingua107: top-1 language, skip-language probability, time and CPU time;
  * Whisper (ASRBackend.run from lingo_overlay.asr): the full skip/translate decision, time and CPU time.

It then reports, for several values of lid_confidence:
  gate drop (skip lang)   share of skip-language clips the gate drops before Whisper (higher = more saving)
  gate wrong-drop         share of foreign clips the gate drops (each one is a lost subtitle)
  wrong-skip (pipeline)   foreign clips with no subtitle: gate drop OR Whisper skip
  wrong-translate         skip-language clips that still produced a subtitle
  skip-lang time vs today processing time per skip-language clip, gated pipeline vs Whisper only
                          (wall time: process CPU time is inflated by idle thread-pool spinning)

Clips: benchmarks/clips (see scripts/fetch_clips.py) with manifest.json, or any folder of
audio files named <iso639-3>_<id>.<ext>.

Usage:
    pip install -r requirements-lid.txt
    python scripts/lid_benchmark.py --model small
    python scripts/lid_benchmark.py --model small --lid-threads 1 --no-whisper
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingo_overlay.asr import load_model  # noqa: E402
from lingo_overlay.audio import TARGET_SR  # noqa: E402
from lingo_overlay.config import load_config  # noqa: E402
from lingo_overlay.langid import LID_MODEL, SpeechBrainLID  # noqa: E402

ISO3_TO_1 = {"ara": "ar", "cmn": "zh", "deu": "de", "eng": "en", "fra": "fr", "jpn": "ja",
             "por": "pt", "rus": "ru", "spa": "es", "tur": "tr", "hin": "hi", "ita": "it",
             "kor": "ko", "pol": "pl", "nld": "nl", "ukr": "uk", "vie": "vi"}
CONFIDENCES = (0.5, 0.7, 0.8, 0.9, 0.95)


def decode_16k_mono(path):
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def list_clips(folder):
    manifest = folder / "manifest.json"
    if manifest.is_file():
        items = json.loads(manifest.read_text(encoding="utf-8"))
        return [(folder / it["file"], it["lang"]) for it in items if (folder / it["file"]).is_file()]
    clips = []
    for p in sorted(folder.iterdir()):
        code = p.stem.split("_")[0]
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"} and code in ISO3_TO_1:
            clips.append((p, ISO3_TO_1[code]))
    return clips


def timed(fn, *a):
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn(*a)
    return out, time.perf_counter() - w0, time.process_time() - c0


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=ROOT / "benchmarks" / "clips")
    ap.add_argument("--model", help="Whisper model (default: config.json)")
    ap.add_argument("--lid-threads", type=int, help="torch threads for the gate (default: config / 2)")
    ap.add_argument("--no-whisper", action="store_true", help="only time and score the gate")
    ap.add_argument("--json", type=Path, help="write per-clip results here")
    args = ap.parse_args()

    cfg = load_config()
    if args.model:
        cfg = cfg.replace(model=args.model)
    if args.lid_threads:
        cfg = cfg.replace(lid_threads=args.lid_threads)
    skip = {s.lower() for s in cfg["skip_languages"]}

    clips = list_clips(args.clips)
    if not clips:
        raise SystemExit(f"No clips in {args.clips}. Run scripts/fetch_clips.py first.")
    print(f"{len(clips)} clips, skip languages: {', '.join(sorted(skip))}", flush=True)

    lid = SpeechBrainLID(cfg)
    model = None if args.no_whisper else load_model(cfg)
    warm = decode_16k_mono(clips[0][0])
    lid.probs(warm)
    if model is not None:
        model.run(warm, skip, cfg["skip_threshold"])

    rows = []
    for i, (path, lang) in enumerate(clips, 1):
        audio = decode_16k_mono(path)
        probs, lid_wall, lid_cpu = timed(lid.probs, audio)
        top = max(probs, key=probs.get)
        row = dict(file=path.name, lang=lang, dur=len(audio) / TARGET_SR, lid_lang=top,
                   lid_prob=probs[top], lid_skip_p=sum(probs.get(c, 0.0) for c in skip),
                   lid_wall=lid_wall, lid_cpu=lid_cpu)
        if model is not None:
            (w_lang, _, text), w_wall, w_cpu = timed(
                model.run, audio, skip, cfg["skip_threshold"])
            row.update(w_lang=w_lang, w_text=text, w_wall=w_wall, w_cpu=w_cpu)
        rows.append(row)
        print(f"  [{i:3}/{len(clips)}] {path.name:22} {lang} -> lid {top} {probs[top]:.2f}"
              f" ({lid_wall * 1000:.0f} ms)" + (f", whisper {row['w_lang']} ({row['w_wall']:.2f}s)"
                                               if model is not None else ""), flush=True)

    skip_rows = [r for r in rows if r["lang"] in skip]
    foreign = [r for r in rows if r["lang"] not in skip]
    lid_ms = [r["lid_wall"] * 1000 for r in rows]
    durs = [r["dur"] for r in rows]

    out = [f"## Fast language-ID gate: {LID_MODEL}", "",
           f"{len(rows)} clips ({len(foreign)} foreign, {len(skip_rows)} in skip languages), "
           f"clip length p50 {statistics.median(durs):.1f} s, lid_threads {cfg.get('lid_threads', 2)}, "
           f"Whisper `{cfg['model']}` {cfg['device']}/{cfg['compute_type']} {cfg['cpu_threads']} threads", "",
           f"- Gate time per clip: p50 {pct(lid_ms, 50):.0f} ms, p95 {pct(lid_ms, 95):.0f} ms, "
           f"CPU {statistics.mean(r['lid_cpu'] for r in rows) * 1000:.0f} ms",
           f"- Gate top-1 accuracy: {sum(r['lid_lang'] == r['lang'] for r in rows) / len(rows):.1%}"]
    if model is not None:
        w_ms = [r["w_wall"] * 1000 for r in rows]
        out.append(f"- Whisper decision per clip: p50 {pct(w_ms, 50):.0f} ms, "
                   f"CPU {statistics.mean(r['w_cpu'] for r in rows) * 1000:.0f} ms")
        w_wrong_skip = sum(r["w_text"] is None for r in foreign) / max(1, len(foreign))
        w_wrong_tr = sum(r["w_text"] is not None for r in skip_rows) / max(1, len(skip_rows))
        out.append(f"- Today (Whisper only): wrong-skip {w_wrong_skip:.1%}, wrong-translate {w_wrong_tr:.1%}")
    out += ["", "| lid_confidence | Gate drop (skip lang) | Gate wrong-drop | Wrong-skip (pipeline) "
                "| Wrong-translate | Skip-lang time vs today |",
            "|---|---|---|---|---|---|"]
    for conf in CONFIDENCES:
        drop = [r["lid_skip_p"] >= conf for r in rows]
        by = dict(zip([r["file"] for r in rows], drop, strict=True))
        g_drop = sum(by[r["file"]] for r in skip_rows) / max(1, len(skip_rows))
        g_wrong = sum(by[r["file"]] for r in foreign) / max(1, len(foreign))
        if model is not None:
            ws = sum(by[r["file"]] or r["w_text"] is None for r in foreign) / max(1, len(foreign))
            wt = sum((not by[r["file"]]) and r["w_text"] is not None for r in skip_rows) / max(1, len(skip_rows))
            today = sum(r["w_wall"] for r in skip_rows)
            gated = sum(r["lid_wall"] + (0 if by[r["file"]] else r["w_wall"]) for r in skip_rows)
            ratio = f"{gated / today:.0%}" if today else "n/a"
            out.append(f"| {conf} | {g_drop:.1%} | {g_wrong:.1%} | {ws:.1%} | {wt:.1%} | {ratio} |")
        else:
            out.append(f"| {conf} | {g_drop:.1%} | {g_wrong:.1%} | n/a | n/a | n/a |")

    wrong = [r for r in rows if r["lid_lang"] != r["lang"]]
    if wrong:
        out += ["", "Gate top-1 misses: " + ", ".join(
            f"{r['file']} ({r['lang']}→{r['lid_lang']} {r['lid_prob']:.2f})" for r in wrong)]
    print("\n" + "\n".join(out))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
