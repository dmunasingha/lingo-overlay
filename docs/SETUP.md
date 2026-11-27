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

> **Using Discord? Skip VB-Cable.** On Windows 10 2004 or later, set
> `"capture_process": "Discord.exe"` in `config.json`. The translator then records only what Discord
> plays (all of its processes), with no driver and no change to Discord's settings. Start Discord
> before the translator. If Discord isn't running or capture fails, the console says why and the
> translator uses `input_device` instead. Don't use this for a game's own voice chat: you'd get
> every game sound too.

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
| **CS2** | Not possible in-game (voice follows the game's output). Use Discord (with `capture_process`, below, no cable is needed) |
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
| `capture_process` | `""` | App to record on its own, e.g. `"Discord.exe"` (Windows 10 2004+, no VB-Cable needed). Empty = use `input_device` |
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
| `text_translation` | `["zh", "yue"]` | Languages transcribed in the original language and translated as text with Opus-MT (better than Whisper's own translation for Chinese). `[]` = Whisper translates everything. The model (~85 MB) is downloaded and converted once |
| `chinese_asr` | `whisper` | `sensevoice` = recognise Chinese/Cantonese speech with SenseVoice-Small instead of Whisper: far more accurate and faster (~0.15 s). Opt-in because of its licence; needs `pip install -r requirements-sensevoice.txt`, then downloads ~155 MB once |
| `mt_threads` | `2` | CPU threads for Opus-MT and SenseVoice |
| `game_glossary` | `true` | Keep VALORANT agent, weapon and map names intact in translations ("这个clo" → "this Clove", 幻影 → Phantom). The word list is `glossary/valorant.json`; fixes welcome |
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
| Game FPS drops while translating | Lower `cpu_threads`, try `"cpu_affinity": "last:4"`, and measure with [PERFORMANCE.md](PERFORMANCE.md) |
| Translations come in seconds late | Use `"model": "small"`, check `cpu_threads`, close other heavy apps |
| High CPU use | Route voice chat through VB-Cable (step 2). On speaker audio every game sound is checked for speech |
