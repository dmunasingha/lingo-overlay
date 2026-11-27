# Benchmarking

Every change to the pipeline (VAD, language ID, ASR backend, model, threads) is judged by
`scripts/benchmark.py`. Include its summary table in your PR.

## 1. Get the clip set (once)

```powershell
.\.venv\Scripts\python scripts\fetch_clips.py
```

This downloads short spoken sentences with human English translations from
[Tatoeba](https://tatoeba.org), 12 per language by default (es, pt, ru, zh, de, fr, ja, ar, tr, en),
into `benchmarks/clips/`. Short everyday sentences are close to voice-chat callouts.

> **Do not commit or share the clips.** Audio licenses on Tatoeba are set per contributor and
> aren't reported by the API, so each developer downloads them locally. The folder is git-ignored.

Options: `--langs spa,rus,eng`, `--per-lang 20`, `--min-words 3 --max-words 12`.
Re-running only tops up missing clips. Delete the folder for a fresh random set.

## 2. Run

```powershell
.\.venv\Scripts\python scripts\benchmark.py                          # settings from config.json
.\.venv\Scripts\python scripts\benchmark.py --model small            # override model
.\.venv\Scripts\python scripts\benchmark.py --with-load 6            # 6 busy CPU processes, roughly a game
.\.venv\Scripts\python scripts\benchmark.py --langs es,ru,en --limit 5   # quick smoke run
```

Results are printed as markdown and saved to `benchmarks/results/<time>_<label>.md` and `.json`
(the JSON has per-clip outputs, detected languages and timings).

## 3. How it works

Each clip runs through the **real app code** (`lingo_overlay.vad.Segmenter` and `ASRBackend.run` from `lingo_overlay.asr`),
simulating real-time arrival:

1. The clip is padded with 0.5 s of silence before and 1.5 s after, then fed frame by frame to the Segmenter.
2. A phrase becomes available when the Segmenter emits it, in audio time. This includes the
   `vad_silence_ms` wait.
3. Phrases are processed one at a time, like the app, so a slow phrase delays the next one.
4. **Latency = when the last phrase's result is ready − when the speaker stopped talking.**

## 4. Metrics

| Metric | Meaning | Phase 1 target |
|---|---|---|
| Latency p50 / p95 | End of speech → subtitle ready | p50 ≤ 1.0 s |
| Wrong-skip | Foreign clip produced no subtitle | ≤ 5 % |
| Wrong-translate | Clip in a skip language (English) produced a subtitle | ≤ 5 % |
| VAD miss | Segmenter never detected speech | 0 % |
| chrF | Character n-gram F-score vs the human translation (0–100). A rough automatic quality signal: compare runs with each other, not with other projects | higher |
| CPU cores | Average cores busy during inference | lower |

chrF punishes valid paraphrases ("Give me the book" vs "Pass me the book" scores about 60), so
judge real quality from the sample-outputs table too.

## 5. Baseline (v0.1)

Ryzen 7 7700X, 16 GB, Windows 11, CPU int8, 6 threads, 120 clips (10 languages), skip = en.

| Run | Latency p50 | p95 | Wrong-skip | Wrong-translate | VAD miss | chrF |
|---|---|---|---|---|---|---|
| `small` | **1.66 s** | 1.78 s | 0.9 % | 0.0 % | 0.0 % | 50.6 |
| `medium` | 3.52 s | 4.07 s | 0.0 % | 0.0 % | 0.0 % | **59.9** |
| `small --with-load 6` | 1.67 s | 1.85 s | 0.9 % | 0.0 % | 0.0 % | 50.6 |

What this tells us:

- **Accuracy is already fine; latency is the problem.** Language gating is near-perfect on this set. Both models
  miss the 1.0 s target, small by 0.66 s.
- Of small's 1.66 s, about **0.45 s is waiting for the pause** (`vad_silence_ms`) and **about 1.2 s is inference**,
  most of it the fixed 30 s Whisper encoder window. That's why the whisper.cpp `audio_ctx` work (#5) and
  adaptive phrase ending (#6) are the big Phase 1 levers.
- English clips are skipped about 0.25 s faster than translated ones, but they still pay the full encoder cost,
  which is what the language-ID gate (#4) addresses.
- medium gains +9 chrF (Spanish 79, German 72) at 2.1× the latency.
- Synthetic load barely moved small (16 logical cores, so 6 + 6 threads still fit). Real games also compete for
  cache and memory bandwidth, so confirm with an in-game FPS test (#7).

## 6. whisper.cpp backend (v0.2, #5)

Same machine and clips. whisper.cpp q8_0 models, encoder window sized to the clip
(`audio_ctx` = clip frames + 64, minimum 256, i.e. ≥ 5.12 s instead of a fixed 30 s).

| Run | Latency p50 | p95 | Wrong-skip | Wrong-translate | VAD miss | chrF |
|---|---|---|---|---|---|---|
| `--backend whisper.cpp --model small` | **1.00 s** | 1.12 s | 2.8 % | 0.0 % | 0.0 % | 48.1 |
| `--backend whisper.cpp --model medium` | 2.15 s | 2.63 s | 0.9 % | 0.0 % | 0.0 % | **54.8** |
| `--backend whisper.cpp --model small --with-load 6` | 1.09 s | 1.21 s | 2.8 % | 0.0 % | 0.0 % | 48.1 |
| `--backend whisper.cpp --model small --min-ctx 128` | 0.83 s | 0.94 s | 3.7 % | 0.0 % | 0.0 % | 47.5 |

What changed vs the faster-whisper baseline:

- **Latency: small 1.66 → 1.00 s, medium 3.52 → 2.15 s.** Inference for small went from about 1.2 s to
  about 0.5 s; the remaining ~0.45 s is the end-of-phrase pause (#6).
- **Quality dropped a little** (chrF −2.5 for small, −5 for medium). Two reasons: a shorter encoder window
  sees less context, and the backend lets low-confidence translations through (a rough subtitle is more
  useful than none). Language gating is unchanged: 0 % English wrongly translated.
- **Short windows make Whisper loop** ("The next room is noisy." ×20). The backend caps output at
  8 tokens per second of audio and collapses repeated sentences; without that, medium's p95 was 4.1 s.
- `--min-ctx 128` is another 0.17 s faster but misses more subtitles, so 256 stays the default.
- Language detection and translation each run the (short) encoder, so the encoder runs twice per
  translated phrase. A single-pass version or the language-ID gate (#4) would save another ~0.2 s.

## Tips for fair comparisons

- Close heavy apps, including the translator itself, and compare runs made on the same machine.
- Use the same clip set for before and after (don't re-fetch between runs).
- For gaming-realistic numbers use `--with-load` with about as many processes as your game's busy threads.
