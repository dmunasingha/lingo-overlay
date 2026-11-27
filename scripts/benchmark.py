"""Benchmark the translation pipeline: latency, skip accuracy, translation quality, CPU use.

Every clip goes through the REAL pipeline code from lingo_overlay (Segmenter + ASRBackend.run),
with real-time arrival simulated:

  * the clip is padded with silence and fed to the Segmenter frame by frame;
  * a phrase becomes available when the Segmenter emits it (audio time);
  * phrases are processed one at a time, as in the app, so a slow phrase delays the next one;
  * latency = time the last phrase's result is ready - time the speaker stopped talking.

Metrics (see docs/GOALS.md#success-metrics):
  latency p50/p95   end of speech -> subtitle ready
  wrong-skip        foreign clip produced no subtitle
  wrong-translate   clip in a skip language (English) produced a subtitle
  VAD miss          the Segmenter never detected speech
  chrF              character n-gram F-score of the output vs the human English
                    translation (0-100, rough automatic quality signal)
  CPU cores         average cores busy during inference (process CPU time / wall time)

Usage:
    python scripts/fetch_clips.py                     # once, downloads clips
    python scripts/benchmark.py                       # uses config.json settings
    python scripts/benchmark.py --model small --label small-baseline
    python scripts/benchmark.py --with-load 6         # 6 busy CPU processes, roughly a game
"""

import argparse
import json
import multiprocessing as mp
import queue
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lingo_overlay.asr import load_model  # noqa: E402
from lingo_overlay.audio import TARGET_SR  # noqa: E402
from lingo_overlay.config import load_config  # noqa: E402
from lingo_overlay.vad import FRAME, Segmenter  # noqa: E402

PAD_BEFORE_S = 0.5
PAD_AFTER_S = 1.5


# --------------------------------------------------------------------------- scoring
def chrf(hyp, ref, n=6, beta=2.0):
    """Character n-gram F-score (chrF, Popovic 2015), whitespace removed, 0-100."""
    hyp, ref = "".join(hyp.lower().split()), "".join(ref.lower().split())
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def pct(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


def decode_16k_mono(path):
    """Decode any audio file to 16 kHz mono float32 with PyAV.

    (faster_whisper.decode_audio breaks on PyAV >= 17: it passes a removed keyword.)
    """
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- CPU load
def _burn(stop):
    x = 0
    while not stop.is_set():
        for _ in range(100_000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def start_load(n):
    stop = mp.Event()
    procs = [mp.Process(target=_burn, args=(stop,), daemon=True) for _ in range(n)]
    for p in procs:
        p.start()
    return stop, procs


# --------------------------------------------------------------------------- pipeline sim
def speech_end_s(audio, threshold_db):
    """Time (s) of the last 30 ms frame louder than the VAD threshold."""
    n = len(audio) // FRAME
    if n == 0:
        return 0.0
    frames = audio[: n * FRAME].reshape(n, FRAME)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    loud = np.nonzero(db > threshold_db)[0]
    return float((loud[-1] + 1) * FRAME / TARGET_SR) if len(loud) else 0.0


def segment(cfg, audio):
    """Feed audio through the real Segmenter; return [(phrase_audio, emit_time_s)]."""
    q = queue.Queue()
    seg = Segmenter(cfg, q)
    out = []
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        seg.feed(audio[i:i + FRAME])
        while not q.empty():
            phrase, _ = q.get()
            out.append((phrase, (i + FRAME) / TARGET_SR))
    return out


def run_clip(model, cfg, clip, audio16k, skip, skip_thr):
    sil_a = np.zeros(int(PAD_BEFORE_S * TARGET_SR), np.float32)
    sil_b = np.zeros(int(PAD_AFTER_S * TARGET_SR), np.float32)
    stream = np.concatenate([sil_a, audio16k, sil_b])
    end_s = PAD_BEFORE_S + speech_end_s(audio16k, cfg["vad_threshold_db"])

    phrases = segment(cfg, stream)
    busy_until, texts, langs, infer_total, cpu_total = 0.0, [], [], 0.0, 0.0
    for phrase, emit_s in phrases:
        start = max(emit_s, busy_until)
        c0, w0 = time.process_time(), time.perf_counter()
        lang, prob, text = model.run(phrase, skip, skip_thr, cfg["min_language_prob"])
        wall = time.perf_counter() - w0
        cpu_total += time.process_time() - c0
        infer_total += wall
        busy_until = start + wall
        langs.append((lang, round(prob, 3)))
        if text:
            texts.append(text)

    output = " ".join(texts)
    is_skip_lang = clip["lang"] in skip
    return {
        "file": clip["file"], "lang": clip["lang"], "source_text": clip["text"],
        "reference": clip["english"], "output": output,
        "detected": langs, "phrases": len(phrases),
        "vad_miss": len(phrases) == 0,
        "subtitled": bool(output),
        "wrong_skip": (not is_skip_lang) and len(phrases) > 0 and not output,
        "wrong_translate": is_skip_lang and bool(output),
        "latency_s": round(busy_until - end_s, 3) if phrases else None,
        "infer_s": round(infer_total, 3),
        "cpu_cores": round(cpu_total / infer_total, 2) if infer_total else None,
        "chrf": round(chrf(output, clip["english"]), 1) if (output and not is_skip_lang) else None,
        "clip_s": round(len(audio16k) / TARGET_SR, 2),
    }


# --------------------------------------------------------------------------- report
def summarize(results, skip):
    def rate(flag, subset):
        return 100 * sum(r[flag] for r in subset) / len(subset) if subset else float("nan")

    foreign = [r for r in results if r["lang"] not in skip]
    native = [r for r in results if r["lang"] in skip]
    lat = [r["latency_s"] for r in results if r["latency_s"] is not None]
    lat_foreign = [r["latency_s"] for r in foreign if r["latency_s"] is not None and r["subtitled"]]
    lat_native = [r["latency_s"] for r in native if r["latency_s"] is not None]
    chrfs = [r["chrf"] for r in foreign if r["chrf"] is not None]
    cores = [r["cpu_cores"] for r in results if r["cpu_cores"]]
    return {
        "clips": len(results), "foreign_clips": len(foreign), "skip_lang_clips": len(native),
        "latency_p50": pct(lat, 50), "latency_p95": pct(lat, 95),
        "latency_translated_p50": pct(lat_foreign, 50), "latency_translated_p95": pct(lat_foreign, 95),
        "latency_skipped_p50": pct(lat_native, 50),
        "wrong_skip_pct": rate("wrong_skip", foreign),
        "wrong_translate_pct": rate("wrong_translate", native),
        "vad_miss_pct": rate("vad_miss", results),
        "chrf_mean": statistics.mean(chrfs) if chrfs else float("nan"),
        "cpu_cores_mean": statistics.mean(cores) if cores else float("nan"),
    }


def per_language(results, skip):
    rows = []
    by = defaultdict(list)
    for r in results:
        by[r["lang"]].append(r)
    for lang, rs in sorted(by.items()):
        lat = [r["latency_s"] for r in rs if r["latency_s"] is not None]
        ch = [r["chrf"] for r in rs if r["chrf"] is not None]
        bad = sum(r["wrong_translate"] if lang in skip else r["wrong_skip"] for r in rs)
        rows.append((lang, len(rs), pct(lat, 50), statistics.mean(ch) if ch else float("nan"),
                     bad, sum(r["vad_miss"] for r in rs)))
    return rows


def markdown(meta, s, rows, results):
    f = lambda v, d=2: "n/a" if v != v else f"{v:.{d}f}"  # noqa: E731  (nan-safe)
    L = [
        f"## Benchmark: {meta['label']}",
        "",
        f"`{meta['model']}` via {meta['backend']} "
        f"({meta['quant'] if meta['backend'] == 'whisper.cpp' else meta['device'] + '/' + meta['compute_type']}), "
        f"{meta['threads']} threads, "
        f"CPU load processes: {meta['with_load']}, clips: {s['clips']} "
        f"({s['foreign_clips']} foreign, {s['skip_lang_clips']} in skip languages), "
        f"skip: {', '.join(meta['skip'])}, {meta['date']}",
        "",
        "| Metric | Value | Phase 1 target |",
        "|---|---|---|",
        f"| Latency p50 (all) | {f(s['latency_p50'])} s | ≤ 1.0 s |",
        f"| Latency p95 (all) | {f(s['latency_p95'])} s | |",
        f"| Latency p50, translated clips | {f(s['latency_translated_p50'])} s | ≤ 1.0 s |",
        f"| Latency p50, skipped clips | {f(s['latency_skipped_p50'])} s | |",
        f"| Wrong-skip (foreign → no subtitle) | {f(s['wrong_skip_pct'], 1)} % | ≤ 5 % |",
        f"| Wrong-translate (skip lang → subtitle) | {f(s['wrong_translate_pct'], 1)} % | ≤ 5 % |",
        f"| VAD miss (no speech detected) | {f(s['vad_miss_pct'], 1)} % | 0 % |",
        f"| chrF vs human translation | {f(s['chrf_mean'], 1)} | higher is better |",
        f"| CPU cores busy during inference | {f(s['cpu_cores_mean'])} | |",
        "",
        "| Lang | Clips | Latency p50 (s) | chrF | Wrong skip/translate | VAD miss |",
        "|---|---|---|---|---|---|",
    ]
    for lang, n, lat, ch, bad, miss in rows:
        L.append(f"| {lang} | {n} | {f(lat)} | {f(ch, 1)} | {bad} | {miss} |")
    L += ["", "<details><summary>Sample outputs</summary>", "", "| Lang | Output | Reference |", "|---|---|---|"]
    shown = Counter()
    for r in results:
        if shown[r["lang"]] < 2 and r["lang"] not in meta["skip"]:
            shown[r["lang"]] += 1
            out = (r["output"] or "*(no subtitle)*").replace("|", "/")
            L.append(f"| {r['lang']} | {out} | {r['reference'].replace('|', '/')} |")
    L += ["", "</details>"]
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=str(ROOT / "benchmarks" / "clips"))
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--compute-type")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--backend", choices=["faster-whisper", "whisper.cpp"])
    ap.add_argument("--min-ctx", type=int, help="whisper.cpp minimum audio_ctx (20 ms frames)")
    ap.add_argument("--skip", help="comma-separated skip languages (default: config.json)")
    ap.add_argument("--langs", help="only these Whisper language codes, e.g. es,ru,en")
    ap.add_argument("--limit", type=int, help="max clips per language")
    ap.add_argument("--with-load", type=int, default=0, help="busy CPU processes to simulate a game")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()

    cfg = load_config()
    overrides = (("model", args.model), ("device", args.device),
                 ("compute_type", args.compute_type), ("cpu_threads", args.threads),
                 ("backend", args.backend), ("whispercpp_min_ctx", args.min_ctx))
    cfg = cfg.replace(**{key: val for key, val in overrides if val is not None})
    skip = {s.strip() for s in (args.skip.split(",") if args.skip else cfg["skip_languages"])}

    manifest_path = Path(args.clips) / "manifest.json"
    if not manifest_path.exists():
        sys.exit("No clips found. Run:  python scripts/fetch_clips.py")
    clips = json.loads(manifest_path.read_text("utf-8"))
    if args.langs:
        wanted = set(args.langs.split(","))
        clips = [c for c in clips if c["lang"] in wanted]
    if args.limit:
        seen = Counter()
        clips = [c for c in clips if (seen.update([c["lang"]]) or seen[c["lang"]] <= args.limit)]

    engine = "cpp" if cfg["backend"] == "whisper.cpp" else cfg["device"]
    label = args.label or f"{cfg['model']}-{engine}-t{cfg['cpu_threads']}" + (
        f"-load{args.with_load}" if args.with_load else "")
    print(f"Loading {cfg['model']} ({cfg['backend']}, {cfg['cpu_threads']} threads)...", flush=True)
    model = load_model(cfg)
    model.run(np.zeros(TARGET_SR, np.float32), skip, cfg["skip_threshold"])  # warm-up

    stop, procs = start_load(args.with_load) if args.with_load else (None, [])
    results = []
    try:
        for i, clip in enumerate(clips, 1):
            path = Path(args.clips) / clip["file"]
            if not path.exists():
                continue
            audio = decode_16k_mono(path)
            r = run_clip(model, cfg, clip, audio, skip, cfg["skip_threshold"])
            results.append(r)
            mark = ("MISS" if r["vad_miss"] else "BAD " if (r["wrong_skip"] or r["wrong_translate"])
                    else "ok  ")
            lat = f"{r['latency_s']:.2f}s" if r["latency_s"] is not None else "  -  "
            print(f"[{i:3}/{len(clips)}] {mark} {clip['lang']} {lat}  {r['output'][:70]}", flush=True)
    finally:
        if stop:
            stop.set()
            for p in procs:
                p.join(timeout=2)

    s = summarize(results, skip)
    meta = {"label": label, "backend": cfg["backend"], "quant": cfg["whispercpp_quant"],
            "min_ctx": cfg["whispercpp_min_ctx"], "model": cfg["model"], "device": cfg["device"],
            "compute_type": cfg["compute_type"], "threads": cfg["cpu_threads"],
            "with_load": args.with_load, "skip": sorted(skip),
            "date": time.strftime("%Y-%m-%d %H:%M")}
    md = markdown(meta, s, per_language(results, skip), results)
    print("\n" + md)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S") + "_" + label
    (out / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": s, "results": results},
                                                 ensure_ascii=False, indent=1), "utf-8")
    (out / f"{stem}.md").write_text(md, "utf-8")
    print(f"\nSaved {out / stem}.md / .json")


if __name__ == "__main__":
    mp.freeze_support()
    main()
