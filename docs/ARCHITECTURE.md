# Architecture

## Current pipeline (v0.1)

```
Game voice chat ──► VB-Cable "CABLE Input" ──┬──► Windows "Listen" ──► your headset  (live, 0 delay)
                                             │
                                             └──► CABLE Output (recording device)
                                                     │
                                         capture thread (soundcard, 48 kHz → 16 kHz mono)
                                                     │
                                         Segmenter: energy VAD, cuts phrases at 250-700 ms pauses (adaptive)
                                                     │  queue (oldest phrases dropped if > 2 waiting)
                                         translate thread: Whisper (faster-whisper / CTranslate2, CPU int8)
                                           1. encode audio once
                                           2. detect language → skip if in skip_languages
                                           3. decode with task="translate" → English text
                                                     │
                                     ┌───────────────┴───────────────┐
                              Overlay (tkinter,                TTS thread (Windows SAPI,
                              click-through, topmost)          off by default)
```

Everything lives in `translator.py` today, with settings in `config.json`.

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

1. **Shorter encoder context.** whisper.cpp exposes `audio_ctx`, which sizes the encoder to the
   actual clip length. For 2–3 s phrases that's a 3–5× speed-up. CTranslate2 rejects inputs
   shorter than 3000 frames, so this means moving from faster-whisper to whisper.cpp (via
   `pywhispercpp`), or patching CTranslate2.
2. **Cheap language gate.** A dedicated language-ID model (~20 MB, ~20 ms) decides whether to
   bother running Whisper at all. English teammates then cost almost no CPU.
3. **Streaming partials.** Re-run on the growing buffer every ~0.5 s and keep the prefix that
   stays stable across runs (the "LocalAgreement" approach from `whisper_streaming`). Subtitles appear
   while the person is still talking.
4. **Neural VAD.** Silero VAD instead of the energy threshold. It works better with game noise,
   breathing and keyboard clicks.

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
| Runtime | faster-whisper / CTranslate2 | MIT | ✅ | current |
| Runtime | whisper.cpp | MIT | ✅ | planned, for `audio_ctx` |
| VAD | Silero VAD | MIT | ✅ | planned |
| Language ID | SpeechBrain `lang-id-voxlingua107-ecapa` | Apache-2.0 | ✅ | planned, 107 languages |
| Text MT (optional 2-stage) | Opus-MT (Helsinki-NLP) | CC-BY-4.0 | ✅ | attribution required; per-language-pair models |
| Text MT (optional) | MADLAD-400 | Apache-2.0 | ✅ | large; GPU or slow |
| Text MT (optional) | M2M100 | MIT | ✅ | |
| Text MT | NLLB-200 | CC-BY-NC-4.0 | ❌ opt-in only | non-commercial |
| Speech-to-speech | SeamlessM4T v2 / SeamlessStreaming | CC-BY-NC-4.0 | ❌ opt-in only | non-commercial |
| TTS | Piper (`piper1-gpl`) | GPL-3.0 | ✅ | fast on CPU; **each voice has its own license, so check it** |
| TTS | Kokoro-82M | Apache-2.0 | ✅ | high-quality English |
| TTS | Windows SAPI voices | built into Windows | ✅ | current fallback, robotic |

## Planned code layout (Phase 1 refactor)

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
