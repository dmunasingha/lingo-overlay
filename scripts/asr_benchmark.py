"""
Compare ASR backends (faster-whisper vs whisper.cpp with shortened audio_ctx).

Each clip goes through the real Segmenter, then the backend's skip/translate decision.
Reports inference time per phrase (p50/p95, end of segment -> text), wrong-skip,
wrong-translate and chrF against the human English translation.

End-to-end latency is roughly this inference time plus the end-of-phrase pause
(vad_silence_ms, 450 ms by default).

Clips come from scripts/fetch_clips.py (benchmarks/clips/manifest.json, local only).

    python scripts/asr_benchmark.py --backend faster-whisper --model small
    python scripts/asr_benchmark.py --backend whisper.cpp --model small
    python scripts/asr_benchmark.py --backend whisper.cpp --model small --audio-ctx-min 0 --full-ctx
"""

import argparse
import json
import queue
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import translator as T  # noqa: E402


def chrf(hyp, ref, n=6, beta=2.0):
    """Character n-gram F-score (chrF), 0-100, whitespace removed."""
    hyp, ref = "".join(hyp.split()), "".join(ref.split())
    precs, recs = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = statistics.mean(precs), statistics.mean(recs)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def decode_16k_mono(path):
    import av

    with av.open(str(path)) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=T.TARGET_SR)
        chunks = [f.to_ndarray().ravel() for fr in c.decode(audio=0) for f in rs.resample(fr)]
        chunks += [f.to_ndarray().ravel() for f in rs.resample(None)]
    return np.concatenate(chunks).astype(np.float32)


def segment(cfg, audio):
    """Run the real Segmenter over clip + trailing silence; return the phrases."""
    q = queue.Queue()
    seg = T.Segmenter(cfg, q)
    pad = np.zeros(int(0.3 * T.TARGET_SR), np.float32)
    tail = np.zeros(int((cfg["vad_silence_ms"] / 1000 + 0.5) * T.TARGET_SR), np.float32)
    seg.feed(np.concatenate([pad, audio, tail]))
    phrases = []
    while not q.empty():
        phrases.append(q.get()[0])
    return phrases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["faster-whisper", "whisper.cpp"], default="whisper.cpp")
    ap.add_argument("--model", default="small", help="small / medium (whisper.cpp uses <model>-q8_0)")
    ap.add_argument("--whispercpp-model", default=None, help="e.g. small-q5_1 (default <model>-q8_0)")
    ap.add_argument("--audio-ctx-min", type=int, default=None, help="minimum audio_ctx (whisper.cpp)")
    ap.add_argument("--full-ctx", action="store_true", help="whisper.cpp with the full 1500 context")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--clips", default=str(ROOT / "benchmarks" / "clips"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips per language")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()

    cfg = T.load_config()
    cfg.update(model=args.model, asr_backend=args.backend, device="cpu", compute_type="int8")
    if args.threads:
        cfg["cpu_threads"] = args.threads
    if args.whispercpp_model:
        cfg["whispercpp_model"] = args.whispercpp_model
    if args.audio_ctx_min is not None:
        cfg["whispercpp_audio_ctx_min"] = args.audio_ctx_min
    skip = {s.lower() for s in cfg["skip_languages"]}
    skip_thr = cfg["skip_threshold"]

    clips_dir = Path(args.clips)
    manifest = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.limit:
        seen = Counter()
        manifest = [c for c in manifest if (seen.update([c["lang"]]) or seen[c["lang"]] <= args.limit)]

    backend = T.load_backend(cfg)
    if backend.name != args.backend:
        sys.exit(f"Requested {args.backend} but got {backend.name}")
    run = backend.run
    if args.full_ctx:
        def run(a, s, t):
            return backend.run(a, s, t, audio_ctx=1500)
    run(np.zeros(T.TARGET_SR * 2, np.float32), skip, skip_thr)  # warm-up

    results = []
    for i, clip in enumerate(manifest, 1):
        audio = decode_16k_mono(clips_dir / clip["file"])
        phrases = segment(cfg, audio)
        texts, langs, times = [], [], []
        for ph in phrases:
            t0 = time.perf_counter()
            lang, prob, text = run(ph, skip, skip_thr)
            times.append(time.perf_counter() - t0)
            langs.append(lang)
            if text:
                texts.append(text)
        out = " ".join(texts)
        is_skip_lang = clip["lang"] in skip
        results.append({
            "file": clip["file"], "lang": clip["lang"], "clip_s": round(len(audio) / T.TARGET_SR, 2),
            "detected": langs, "phrases": len(phrases), "infer_s": [round(t, 3) for t in times],
            "output": out, "reference": clip["english"],
            "chrf": round(chrf(out, clip["english"]), 1) if (out and not is_skip_lang) else None,
        })
        print(f"[{i}/{len(manifest)}] {clip['lang']} {sum(times):.2f}s  {out[:70]}", flush=True)

    foreign = [r for r in results if r["lang"] not in skip]
    native = [r for r in results if r["lang"] in skip]
    # Latency of the phrase that produced (or would produce) the text: last phrase per clip.
    lat = [r["infer_s"][-1] for r in results if r["infer_s"]]
    lat_f = [r["infer_s"][-1] for r in foreign if r["infer_s"]]
    lat_n = [r["infer_s"][-1] for r in native if r["infer_s"]]
    chrfs = [r["chrf"] for r in foreign if r["chrf"] is not None]
    wrong_skip = sum(1 for r in foreign if r["phrases"] and not r["output"]) / max(1, len(foreign))
    wrong_tr = sum(1 for r in native if r["output"]) / max(1, len(native))

    label = args.backend if args.backend == "faster-whisper" else (
        f"whisper.cpp {cfg.get('whispercpp_model') or args.model + '-q8_0'}, "
        + ("audio_ctx 1500" if args.full_ctx else f"audio_ctx min {getattr(backend, 'ctx_min', '?')}"))
    row = (f"| {args.model} | {label} | {pct(lat_f, 50):.2f} s | {pct(lat_f, 95):.2f} s "
           f"| {pct(lat_n, 50):.2f} s | {100 * wrong_skip:.1f} % | {100 * wrong_tr:.1f} % "
           f"| {statistics.mean(chrfs) if chrfs else float('nan'):.1f} |")
    header = ("| Model | Backend | Foreign p50 | Foreign p95 | Skip-lang p50 | Wrong-skip "
              "| Wrong-translate | chrF |\n|---|---|---|---|---|---|---|---|")
    print()
    print(f"{len(results)} clips ({len(foreign)} foreign, {len(native)} skip-language), "
          f"{cfg['cpu_threads']} threads. Inference time per phrase, all clips p50 {pct(lat, 50):.2f} s.")
    print(header)
    print(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = label.replace(" ", "_").replace(",", "").replace(".", "")
    (out_dir / f"asr_{args.model}_{slug}_{stamp}.json").write_text(
        json.dumps({"row": row, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
