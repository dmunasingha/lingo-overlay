# Lingo Overlay

**Free, open-source, fully local real-time translation of in-game voice chat.**
Teammates speak any language, and you see English subtitles on screen (and can optionally hear a translated voice),
without the game's audio ever being delayed.

> Status: **v0.1 prototype**, working and usable. See the [roadmap](docs/ROADMAP.md) for where it's going.

```
Teammate (Spanish) ──► you hear them live, as normal
                  └──► Lingo Overlay: detect language → not English → translate
                                                         └──► subtitle: [ES] Push left, enemy behind the wall
```

## Features

- 🎧 **Listens only to voice chat** (via VB-Cable), never delays or touches game sounds
- 🌍 **Auto language detection** for ~99 languages, with a skip list (English skipped by default)
- 💬 **Click-through subtitle overlay**, always on top (borderless windowed games)
- 🔊 **Optional spoken translation** (off by default)
- ⌨️ **Global hotkeys**: toggle, voice, replay, quit
- 🖥️ **CPU-first**, so the GPU stays with your game
- 🔒 **100 % offline**: no accounts, no API keys, no cloud, no telemetry
- 🛡️ **Anti-cheat safe**: no injection, no hooks, no memory reading, just Windows audio APIs and a normal window

## Quick start (Windows 10/11)

```powershell
git clone https://github.com/dmunasingha/lingo-overlay.git
cd lingo-overlay
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\run.bat
```

For voice-chat-only listening (recommended), install VB-Cable and point your game's voice output
to it. See the **[setup guide](docs/SETUP.md)** (Valorant, Discord and others).

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+T` | Translator on/off |
| `Ctrl+Alt+V` | Spoken translation on/off |
| `Ctrl+Alt+R` | Replay last translation |
| `Ctrl+Alt+Q` | Quit |

## Performance today

From `scripts/benchmark.py`: 120 short clips in 10 languages, Ryzen 7 7700X, CPU only.
Latency is measured from the end of speech to the subtitle being ready.

| Model | Backend | Latency (median) | Foreign wrongly skipped | English wrongly translated | Quality (chrF) |
|---|---|---|---|---|---|
| `small` (default) | whisper.cpp (default) | **1.00 s** | 2.8 % | 0 % | 48.1 |
| `medium` | whisper.cpp | 2.15 s | 0.9 % | 0 % | 54.8 |
| `small` | faster-whisper | 1.66 s | 0.9 % | 0 % | 50.6 |
| `medium` | faster-whisper | 3.52 s | 0 % | 0 % | 59.9 |

`medium` translates better but uses about twice the CPU and RAM. The goal for v0.2 is **≤ 1.0 s**. See [GOALS.md](docs/GOALS.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[BENCHMARKING.md](docs/BENCHMARKING.md).

## Documentation

- [Setup guide](docs/SETUP.md): install, audio routing, config, troubleshooting
- [Goals](docs/GOALS.md): vision, principles, success metrics, non-goals
- [Roadmap](docs/ROADMAP.md): phased build plan
- [Architecture](docs/ARCHITECTURE.md): pipeline, latency breakdown, model licenses
- [Performance](docs/PERFORMANCE.md): measuring and limiting game FPS impact
- [Contributing](CONTRIBUTING.md): guidelines, anti-cheat rules, workflow

## License

[GPL-3.0](LICENSE). Whisper models are MIT-licensed by OpenAI. See the
[model license table](docs/ARCHITECTURE.md#models-and-licenses) for all models.

Not affiliated with Riot Games, Valve, Discord, or any game publisher.
