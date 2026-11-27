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
| `model` | `medium` | `small` ≈ 1.0 s, good · `medium` ≈ 2.2 s, better (CPU, 7700X) |
| `backend` | `whisper.cpp` | `whisper.cpp` (fast, short encoder window) or `faster-whisper` (slower, slightly higher quality) |
| `whispercpp_min_ctx` | `256` | Smallest encoder window in 20 ms frames. 128 is ~0.15 s faster but misses more phrases |
| `device`, `compute_type` | `cpu`, `int8` | faster-whisper only. `cuda` only when **not** gaming. Games keep the GPU at ~100 % |
| `cpu_threads` | `6` | Lower to 4 if game FPS drops |
| `vad_silence_ms` | `450` | Pause that ends a phrase. Lower is faster, but more mid-sentence cuts |
| `voice_enabled` | `false` | Speak translations |
| `overlay_font_size`, `overlay_seconds`, `overlay_bottom_offset`, `overlay_opacity` | | Subtitle look |

## Troubleshooting

| Problem | Fix |
|---|---|
| No sound at all after installing VB-Cable | Run `audio_setup.ps1`. The cable became your default speaker |
| Teammates can't hear you | Run `audio_setup.ps1`. The cable became your default mic |
| Console says `Input device 'CABLE Output' not found` | VB-Cable isn't installed, so the app is using all speaker audio |
| Subtitles don't appear over the game | Switch the game to Borderless Windowed |
| Hotkeys don't work in game | If the game runs as administrator, run `run.bat` as administrator too |
| Translations come in seconds late | Use `"model": "small"`, check `cpu_threads`, close other heavy apps |
