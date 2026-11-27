"""
Summarise PresentMon / CapFrameX frame-time CSVs and report the translator's FPS impact.

    python tools/fps_summary.py bench/off.csv bench/idle.csv bench/translating.csv

The first file is the baseline (translator off). Each later file is compared to
it. Exit code is 1 if any scenario's average FPS drops by more than --max-impact
percent (default 5, the v0.2 target), so it can gate a manual test run.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

# PresentMon 1.x/2.x and CapFrameX all write MsBetweenPresents; some PresentMon 2
# metric sets call it FrameTime.
FRAME_COLUMNS = ("MsBetweenPresents", "FrameTime", "msBetweenPresents")


def read_frametimes(path, process=None):
    """Return the list of frame times (ms) in a PresentMon-style CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("//"))
        col = next((c for c in FRAME_COLUMNS if c in (rows.fieldnames or [])), None)
        if col is None:
            raise ValueError(f"{path}: no frame-time column (looked for {', '.join(FRAME_COLUMNS)})")
        out = []
        for r in rows:
            if process and r.get("Application", "").lower() != process.lower():
                continue
            try:
                ms = float(r[col])
            except (TypeError, ValueError):
                continue
            if ms > 0:
                out.append(ms)
    if not out:
        raise ValueError(f"{path}: no frames found")
    return out


def summarise(frametimes):
    """Average FPS, 1 % low FPS (from the 99th-percentile frame time) and frame count."""
    ft = sorted(frametimes)
    p99 = ft[min(len(ft) - 1, int(round(0.99 * (len(ft) - 1))))]
    return {
        "frames": len(ft),
        "avg_fps": 1000.0 / statistics.fmean(ft),
        "low1_fps": 1000.0 / p99,
    }


def impact_pct(baseline_fps, fps):
    """How much lower fps is than baseline_fps, in percent (negative = faster)."""
    return (baseline_fps - fps) / baseline_fps * 100.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path, help="baseline (translator off) first, then other runs")
    ap.add_argument("--process", help="only count frames from this executable, e.g. VALORANT-Win64-Shipping.exe")
    ap.add_argument("--max-impact", type=float, default=5.0, help="fail above this avg-FPS drop in %%")
    a = ap.parse_args(argv)

    results = [(p.stem, summarise(read_frametimes(p, a.process))) for p in a.csv]
    base = results[0][1]
    print(f"{'run':<16}{'frames':>8}{'avg FPS':>10}{'1% low':>10}{'avg Δ':>9}{'1% Δ':>9}")
    worst = 0.0
    for name, r in results:
        d_avg = impact_pct(base["avg_fps"], r["avg_fps"])
        d_low = impact_pct(base["low1_fps"], r["low1_fps"])
        worst = max(worst, d_avg)
        print(f"{name:<16}{r['frames']:>8}{r['avg_fps']:>10.1f}{r['low1_fps']:>10.1f}"
              f"{d_avg:>8.1f}%{d_low:>8.1f}%")
    ok = worst <= a.max_impact
    print(f"\nWorst avg-FPS impact {worst:.1f}% (target <= {a.max_impact:g}%): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
