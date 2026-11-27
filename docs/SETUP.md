# Setup guide

## 1. Install

```powershell
git clone https://github.com/dmunasingha/lingo-overlay.git
cd lingo-overlay
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The first run downloads the Whisper model (small ≈ 265 MB, medium ≈ 820 MB) into `models/ggml/`.
If the automatic download stalls, download `ggml-<model>-q8_0.bin` from
`https://huggingface.co/ggerganov/whisper.cpp/tree/main` into `models/ggml/`.

With `"backend": "faster-whisper"`, download `config.json`, `model.bin`, `tokenizer.json` and
`vocabulary.txt` from `https://huggingface.co/Systran/faster-whisper-<model>/tree/main` into
`models/<model>/` instead.

## 2. Route voice chat into the translator (recommended)

Without this step the translator listens to **all** PC audio, so gunshots and music can
trigger it. With it, it hears only voice chat.

### 2a. Install VB-Cable (free virtual audio cable)

1. Download from <https://vb-audio.com/Cable/>, unzip, right-click `VBCABLE_Setup_x64.exe`,
   choose *Run as administrator*, then *Install Driver*.
2. **The installer often makes the cable your default speaker and microphone.** Fix that, and
   make the cable audible in your headset, in one step:

   ```powershell
   powershell -ExecutionPolicy Bypass -File audio_setup.ps1
   ```

   This sets your Realtek speakers and mic back as the defaults and turns on *Listen to this device*
   for **CABLE Output**, so anything sent into the cable plays through your headset live.
   If your headset isn't a Realtek device, pass a different name:
   `audio_setup.ps1 -SpeakerMatch "HyperX" -MicMatch "HyperX"`.

   <details><summary>Doing it manually instead</summary>

   Sound settings → *More sound settings* →
   **Playback**: set your headset as default. **Recording**: set your mic as default; then
   right-click **CABLE Output** → Properties → **Listen** → tick *Listen to this device* →
   *Playback through* = your headset.
   </details>

### 2b. Point the game's voice chat at the cable

| App | Setting |
|---|---|
| **Valorant** | Settings → Audio → Voice Chat → **Output Device** = `CABLE Input (VB-Audio Virtual Cable)` |
| **Discord** | User Settings → Voice & Video → **Output Device** = `CABLE Input (VB-Audio Virtual Cable)` |
| **CS2** | Not possible in-game (voice follows the game's output). Use Discord, or wait for per-process capture (Phase 3) |
| Other games | Look for a separate *voice chat output device* setting |

Leave the **input** (microphone) device unchanged.

## 3. Run

Double-click `run.bat`, or run `.\.venv\Scripts\python translator.py`.

Run the game in **Borderless Windowed** mode so the subtitle overlay shows on top.

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+T` | Translator on/off |
| `Ctrl+Alt+V` | Spoken translation on/off |
| `Ctrl+Alt+R` | Replay last translation |
| `Ctrl+Alt+Q` | Quit |

## 4. Configure (`config.json`)

| Key | Default | Meaning |
|---|---|---|
| `skip_languages` | `["en"]` | Languages left alone (Whisper codes: `en`, `si`, `ta`, `hi`, `es`, `ru`, `pt`, `fr`, `de`, `ja`, `ko`, `zh`, `ar` …) |
| `min_language_prob` | `0.5` | Ignore a phrase when the language detector is less sure than this. Filters game sounds and music; 0 disables |
| `model` | `small` | `small` ≈ 1.0 s, good, light · `medium` ≈ 2.2 s, better, about 2× the CPU and RAM (7700X) |
| `backend` | `whisper.cpp` | `whisper.cpp` (fast, short encoder window) or `faster-whisper` (slower, slightly higher quality) |
| `whispercpp_min_ctx` | `256` | Smallest encoder window in 20 ms frames. 128 is ~0.15 s faster but misses more phrases |
| `device`, `compute_type` | `cpu`, `int8` | faster-whisper only. `cuda` only when **not** gaming. Games keep the GPU at ~100 % |
| `cpu_threads` | `4` | Threads used while translating. Lower to 2–3 if game FPS drops. See [PERFORMANCE.md](PERFORMANCE.md) for a per-CPU table |
| `process_priority` | `below_normal` | `normal`, `below_normal` or `idle`. Below-normal lets the game win every contested core |
| `cpu_affinity` | `off` | `off`, `"last:N"` (highest N logical CPUs) or a list like `[8, 9, 10, 11]`. Only turn on if a measurement shows it helps |
| `vad_backend` | `silero` | Speech detector. `silero` (neural, ignores game noise, breathing and keyboard clicks) or `energy` (old volume threshold, tuned with `vad_threshold_db`) |
| `vad_speech_threshold` | `0.5` | Silero only. Raise (e.g. `0.6`) if noise still triggers, lower if quiet speakers are missed |
| `vad_silence_ms` | `450` | Pause that ends a phrase (with `vad_adaptive`: the pause when the end is unclear). Lower is faster, but more mid-sentence cuts |
| `vad_adaptive` | `true` | Pick the pause from the speaker's energy: end sooner when a sentence trails off, wait longer when it stopped mid-word. `false` = always `vad_silence_ms` |
| `vad_fast_silence_ms`, `vad_slow_silence_ms` | `250`, `700` | The adaptive "trailed off" and "cut mid-word" pauses |
| `vad_merge_gap_ms` | `300` | A short fragment (under 0.8 s of speech) is held this long and joined to speech that starts again in that time. `0` = off |
| `lid_backend` | `"whisper"` | `"speechbrain"` turns on the fast language-ID gate: speech it is confident is in a skip language is dropped before Whisper runs. Needs `pip install -r requirements-lid.txt` (see that file for the small CPU-only torch). Falls back to `"whisper"` if it can't load |
| `lid_confidence` | `0.5` | How sure the gate must be (0–1) before it drops a phrase. Anything less goes to Whisper as usual |
| `lid_threads` | `2` | CPU threads for the gate |
| `voice_enabled` | `false` | Speak translations |
| `voice_engine` | `"sapi"` | `"sapi"` (built-in Windows voice, robotic), `"piper"` or `"kokoro"` (natural, see below). Falls back to `"sapi"` if the engine can't load |
| `voice_output_device` | `""` | Part of the speaker name the voice plays on (e.g. `"HyperX"`). `""` = default speaker. Piper/Kokoro only |
| `piper_voice` | `models/piper/en_US-lessac-medium.onnx` | Piper voice file. The matching `.onnx.json` must be next to it |
| `kokoro_model`, `kokoro_voices`, `kokoro_voice` | `models/kokoro/...`, `af_heart` | Kokoro model files and voice name |
| `voice_rate`, `voice_volume` | `200`, `0.8` | Speed (175 = normal) and loudness (0–1) |
| `overlay_font_size`, `overlay_seconds`, `overlay_bottom_offset`, `overlay_opacity` | | Subtitle look |

### Natural voice (Piper / Kokoro)

The built-in Windows voice is robotic. For a natural voice, install one neural engine
(both run on the CPU and are streamed sentence by sentence; a new subtitle cuts off the old one):

```powershell
pip install -r requirements-tts.txt
```

- **Piper** (fast, many voices and languages). Download a voice into `models/piper` (it saves to the current folder):

  ```powershell
  mkdir models/piper; cd models/piper
  python -m piper.download_voices en_US-lessac-medium
  cd ../..
  ```

  Each Piper voice has its own license. Read its `MODEL_CARD` on
  [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) before using it.
  Then set `"voice_engine": "piper"`.

- **Kokoro-82M** (best English quality, a bit slower). Download `kokoro-v1.0.onnx` and
  `voices-v1.0.bin` from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
  into `models/kokoro`, then set `"voice_engine": "kokoro"`.

Turn the voice on with `"voice_enabled": true` or the voice hotkey (`ctrl+alt+v`).

## Troubleshooting

| Problem | Fix |
|---|---|
| No sound at all after installing VB-Cable | Run `audio_setup.ps1`. The cable became your default speaker |
| Teammates can't hear you | Run `audio_setup.ps1`. The cable became your default mic |
| Console says `Input device 'CABLE Output' not found` | VB-Cable isn't installed, so the app is using all speaker audio |
| Subtitles don't appear over the game | Switch the game to Borderless Windowed |
| Hotkeys don't work in game | If the game runs as administrator, run `run.bat` as administrator too |
| Game FPS drops while translating | Lower `cpu_threads`, try `"cpu_affinity": "last:4"`, and measure with [PERFORMANCE.md](PERFORMANCE.md) |
| Translations come in seconds late | Use `"model": "small"`, check `cpu_threads`, close other heavy apps |
| High CPU use | Route voice chat through VB-Cable (step 2). On speaker audio every game sound is checked for speech |
