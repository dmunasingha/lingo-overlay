# Roadmap

Each phase is a GitHub milestone, and each bullet is (or will become) an issue.
Targets refer to the metrics in [GOALS.md](GOALS.md#success-metrics).

---

## ✅ Phase 0: Working prototype (v0.1), done

- [x] Capture voice chat from VB-Cable (fallback: all speaker audio)
- [x] Energy-based phrase detection
- [x] Whisper language detection with a skip list (English skipped by default)
- [x] Translate to English with one shared encoder pass (2× faster than `transcribe()`)
- [x] Click-through, always-on-top subtitle overlay
- [x] Optional Windows TTS voice
- [x] Global hotkeys (toggle, voice, replay, quit)
- [x] CPU-first design. Measured: GPU inference took 8+ s while a game was running

---

## Phase 1: Fast and measurable (v0.2)

**Goal:** median subtitle latency **≤ 1.0 s** on CPU while gaming, with numbers to prove it.

1. ✅ **Benchmark harness** (`scripts/benchmark.py`, [BENCHMARKING.md](BENCHMARKING.md)). Baseline:
   small 1.66 s / medium 3.52 s median latency, ≤ 0.9 % skip errors, chrF 50.6 / 59.9.
2. **Refactor into a package** (`lingo_overlay/`, see [ARCHITECTURE.md](ARCHITECTURE.md#planned-code-layout-phase-1-refactor))
   with a common `ASRBackend` interface.
3. **Silero VAD** replaces the energy detector, giving fewer false triggers from game noise.
4. **Fast language gate**: SpeechBrain VoxLingua107 decides "skip / translate" in ~20 ms, so
   English speech never reaches Whisper.
5. ✅ **whisper.cpp backend with `audio_ctx`**, so the encoder is sized to the clip instead of 30 s.
   Result: small 1.66 → 1.00 s, medium 3.52 → 2.15 s median latency (inference about 2.4× faster).
6. **Adaptive phrase ending**: end a phrase sooner (250 ms) when the sentence is clearly finished,
   and later when it's mid-word.
7. **FPS impact test**: measure game FPS with and without the translator (PresentMon), and
   tune `cpu_threads` and process priority (below-normal).
8. Unit tests for segmenter, resampler, config; CI on Windows.

**Exit criteria:** benchmark shows p50 ≤ 1.0 s, FPS impact ≤ 5 %, wrong-skip/wrong-translate ≤ 5 %.

---

## Phase 2: Real-time subtitles and voice (v0.3)

**Goal:** subtitles appear *while* people talk, and a natural translated voice is available.

1. ✅ **Streaming partial subtitles** (opt-in, `stream_partials`): re-decode the growing buffer every ~0.5 s and show the stable
   prefix (LocalAgreement). Grey text = live, white = final.
2. **Neural TTS**: Piper (multi-voice, CPU-fast) and Kokoro (English quality), streamed
   sentence by sentence.
3. **Audio ducking**: lower the original voice-chat volume while the translation speaks
   (we control the Listen path, so this needs no game hooks).
4. **Speaker separation (basic)**: label lines by speaker (`P1`, `P2`) using voice embeddings, so
   overlapping callouts don't merge.
5. **Gaming vocabulary**: per-game hint prompts (agent, map and weapon names) passed to Whisper as
   `initial_prompt` to fix things like "hama" → the actual agent name.
6. **Optional 2-stage translation**: Whisper transcribes in the source language, then Opus-MT
   translates. Better quality for some language pairs.

**Exit criteria:** first partial text ≤ 0.6 s after speech starts; spoken translation starts
≤ 1.5 s after speech ends.

---

## Phase 3: Easy for everyone (v0.4)

**Goal:** a non-technical gamer can install it in under 5 minutes.

1. **Per-process audio capture** (Windows process loopback), so Discord works with no VB-Cable.
2. **Settings window**: device pickers, skip languages, model size, hotkeys, overlay position,
   font and opacity, all saved to config.
3. **System-tray app** with status (listening / translating / off).
4. **Installer** (PyInstaller + Inno Setup): bundled Python, first-run model download with
   progress, VB-Cable setup guide or auto-install, `audio_setup` built in.
5. **Overlay positioning and themes**: drag to place, per-monitor, outline text for readability.
6. **Translation history window** (last 50 lines, copyable).

---

## Phase 4: Community and reach (v1.0)

1. Target languages other than English (translate *into* Sinhala, Spanish and others).
2. Per-game profiles (auto-switch when a known game is focused).
3. Optional plugin interface for cloud back-ends (user brings their own key; never default).
4. Localization of the UI.
5. Linux support (PipeWire capture) if contributors step up.
6. Docs site, demo video, and published benchmark results.

---

## Ideas parking lot (not scheduled)

- Translating *your* voice to teammates (opt-in, virtual mic output)
- Profanity filter / toxicity blur on subtitles
- Lip-sync-free "dub" mode that fully replaces the original voice
- Mobile companion app showing subtitles on a phone instead of an overlay
