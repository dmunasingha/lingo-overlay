# Architecture

## Current pipeline (v0.1)

```
Game voice chat ──► VB-Cable "CABLE Input" ──┬──► Windows "Listen" ──► your headset  (live, 0 delay)
                                             │
                                             └──► CABLE Output (recording device)
                                                     │
                                         capture thread (soundcard, 48 kHz → 16 kHz mono)
                                                     │
                                         Segmenter: Silero VAD, cuts phrases at 250-700 ms pauses (adaptive)
                                                     │  queue (oldest phrases dropped if > 2 waiting)
                                         translate thread: Whisper (whisper.cpp q8_0, CPU)
                                           1. encoder window sized to the phrase (audio_ctx), not 30 s
                                           2. detect language → skip if in skip_languages
                                           3. decode with task="translate" → English text
                                           (alternative backend: faster-whisper / CTranslate2 int8)
                                                     │
                                     ┌───────────────┴───────────────┐
                              Overlay (tkinter,                TTS thread (Windows SAPI,
                              click-through, topmost)          off by default)
```

The code lives in the `lingo_overlay` package (`translator.py` and `run.bat` remain as entry points;
`python -m lingo_overlay` also works), with settings in `config.json`:

| Module | What it does |
|---|---|
| `config.py` | Typed, frozen `Config` loaded from `config.json`: defaults for missing keys, type/range validation with clear errors, a warning for unknown keys. Also a read-only mapping, so components and scripts accept a `Config` or a plain dict |
| `audio/` | Capture (`capture.py`: VB-Cable or speaker loopback) and `to_16k_mono` resampling |
| `vad/` | Per-frame detectors (`silero.py`, `energy.py`) and the `Segmenter` with adaptive phrase ending (`segmenter.py`) |
| `langid/` | Optional SpeechBrain fast language-ID gate in front of Whisper |
| `asr/` | `ASRBackend` interface (`detect_language`, `translate`, shared skip decision in `run`) with the whisper.cpp and faster-whisper backends |
| `tts/sapi.py` | Windows SAPI voice |
| `ui/overlay.py` | Click-through subtitle overlay |
| `perf_limits.py` | Process priority and CPU affinity (keeps FPS up) |
| `hotkeys.py`, `app.py` | Global hotkeys; wiring of threads and queues |

### Where the time goes (small model, CPU, game running)

| Step | Time | Why |
|---|---|---|
| Waiting for the pause that ends the phrase | 0.24 s typical, 0.45 s when unsure | `vad_adaptive`, `vad_*silence_ms` (see `scripts/endpoint_benchmark.py`) |
| Whisper encoder | ~1.1 s | **Always processes a fixed 30 s window**, even for a 2 s phrase |
| Language detection | ~0.05 s | reuses the encoder output |
| Decoder (translation) | ~0.3 s | proportional to sentence length |
| Overlay | < 0.05 s | |

**The single biggest cost is the 30-second encoder window.** Most of the Phase 1 latency work is
about not paying for that.

## Target pipeline (Phase 1–2)

```
audio ─► Silero VAD (neural, ~1 ms/frame) ─► streaming chunks every ~0.5 s while speech is ongoing
            │
            ├─► fast language ID (SpeechBrain VoxLingua107, ~20 ms)  ── skip-language? ──► drop
            │
            └─► whisper.cpp with shortened audio_ctx (encoder sized to the clip, not 30 s)
                   ├─ partial hypothesis every 0.5 s ──► overlay shows grey "live" text
                   └─ final text when the phrase ends ──► overlay shows white final text
                                                       └─► TTS (Piper / Kokoro) ──► ducked voice in headset
```

Key ideas:

1. ✅ **Shorter encoder context** (done in v0.2, see [BENCHMARKING.md](BENCHMARKING.md#6-whispercpp-backend-v02-5)).
   whisper.cpp exposes `audio_ctx`, which sizes the encoder to the
   actual clip length. For 2–3 s phrases that's a 3–5× speed-up. CTranslate2 rejects inputs
   shorter than 3000 frames, so this means moving from faster-whisper to whisper.cpp (via
   `pywhispercpp`), or patching CTranslate2.
2. **Cheap language gate.** A dedicated language-ID model (~20 MB, ~20 ms) decides whether to
   bother running Whisper at all. English teammates then cost almost no CPU.
3. **Streaming partials.** Re-run on the growing buffer every ~0.5 s and keep the prefix that
   stays stable across runs (the "LocalAgreement" approach from `whisper_streaming`). Subtitles appear
   while the person is still talking.
4. **Neural VAD.** ✅ Done: Silero VAD replaced the energy threshold (`vad_backend`). It ignores game
   noise, breathing and keyboard clicks; compare backends with `scripts/vad_benchmark.py`.

## Audio capture options

| Method | Pros | Cons | Status |
|---|---|---|---|
| VB-Cable + game's voice output setting | Only voice chat, clean signal, works with any game that has a voice output device setting | Extra driver install, one game setting | **Used now** |
| Speaker loopback (all PC audio) | Zero setup | Hears game sounds too, so more false triggers | Fallback |
| **Per-process loopback** (Windows 10 2004+ `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`) | No driver; capture only Discord.exe | Captures the whole process, so in-game voice comes mixed with game sounds | Planned, ideal for Discord |

## Models and licenses

The project is GPL-3.0. A model may be **shipped or recommended by default** only if its license
allows free redistribution, including commercial use. Non-commercial (NC) models may be supported as
opt-in downloads only, clearly labelled.

| Purpose | Model | License | Default OK? | Notes |
|---|---|---|---|---|
| ASR + language ID + translate | OpenAI Whisper (tiny … large-v3) | MIT | ✅ | `large-v3-turbo` is poor at the translate task. Use small/medium |
| Runtime | faster-whisper / CTranslate2 | MIT | ✅ | alternative backend (`"backend": "faster-whisper"`) |
| Runtime | whisper.cpp (via pywhispercpp, MIT) | MIT | ✅ | default backend, shortened `audio_ctx` |
| VAD | Silero VAD | MIT | ✅ | current (v6 ONNX, bundled with faster-whisper) |
| Language ID | SpeechBrain `lang-id-voxlingua107-ecapa` | Apache-2.0 | ✅ | optional gate (`lid_backend`), 107 languages |
| Text MT (2-stage) | Opus-MT `zho-eng` (Helsinki-NLP, Tatoeba-MT) | CC-BY-4.0 | ✅ | **on for Chinese/Cantonese** (`text_translation`); attribution required; converted locally to CTranslate2 int8 |
| ASR (Chinese) | SenseVoice-Small (FunAudioLLM) via sherpa-onnx (Apache-2.0) | FunASR Model License 1.1 | ❌ opt-in only | `"chinese_asr": "sensevoice"`; custom licence (attribution, conduct clause, self-updating terms) |
| Text MT (optional) | MADLAD-400 | Apache-2.0 | ✅ | large; GPU or slow |
| Text MT (optional) | M2M100 | MIT | ✅ | |
| Text MT | NLLB-200 | CC-BY-NC-4.0 | ❌ opt-in only | non-commercial |
| Speech-to-speech | SeamlessM4T v2 / SeamlessStreaming | CC-BY-NC-4.0 | ❌ opt-in only | non-commercial |
| TTS | Piper (`piper1-gpl`) | GPL-3.0 | ✅ | fast on CPU; **each voice has its own license, so check it** |
| TTS | Kokoro-82M | Apache-2.0 | ✅ | high-quality English |
| TTS | Windows SAPI voices | built into Windows | ✅ | current fallback, robotic |

## Planned code layout (later phases)

```
lingo_overlay/
  audio/        capture backends (vbcable, loopback, process_loopback), resampling
  vad/          energy, silero
  langid/       whisper_lid, speechbrain_lid
  asr/          faster_whisper_backend, whispercpp_backend  (common interface)
  translate/    whisper_translate, opus_mt (optional 2-stage)
  tts/          sapi, piper, kokoro
  ui/           overlay, tray, settings window
  hotkeys.py
  config.py     typed config + validation
  app.py        wiring / threads
scripts/
  benchmark.py  latency + accuracy on a clip set
tests/
```
