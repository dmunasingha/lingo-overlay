# Contributing to Lingo Overlay

Thanks for helping! This project aims to give gamers free, local, real-time translation of voice chat.
Please read [docs/GOALS.md](docs/GOALS.md) and [docs/ROADMAP.md](docs/ROADMAP.md) first, so your work lines up with where the project is going.

## Ground rules

### Anti-cheat rules

These protect our users' game accounts. A PR that breaks them will not be merged.

- ❌ **Never** inject DLLs, hook DirectX/Vulkan, or attach to game processes.
- ❌ **Never** read or write game memory or game files.
- ❌ **Never** send simulated input to games.
- ✅ Only use standard OS audio APIs (WASAPI, loopback, virtual audio devices).
- ✅ The overlay is an ordinary always-on-top, click-through window. If that doesn't show over
  exclusive fullscreen, the answer is "use borderless windowed", not hooking.
- ✅ Global hotkeys use standard OS keyboard hooks, the same thing Discord and OBS use.

### Privacy rules

- Audio and transcripts stay on the user's PC. No telemetry, no analytics, no "phone home".
- Any network access (model downloads, update checks) must be explicit and documented.
- Optional cloud back-ends may only ever be opt-in plugins, never the default.

### Licensing rules

- Code is **GPL-3.0**. By contributing you agree your contribution is licensed under it.
- New dependencies must be GPL-3.0-compatible (MIT, BSD, Apache-2.0, LGPL, GPL-3.0 …).
- New **models** must allow free redistribution, including commercial use, to be on by default.
  Non-commercial (e.g. CC-BY-NC) models may only be opt-in and clearly labelled. Add every model
  to the table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#models-and-licenses).

### Performance rules

- Never add work to the game's audio path. We only process a *copy* of voice chat.
- Default to CPU. GPU features must be opt-in, because games keep the GPU at ~100 %.
- Latency-sensitive changes must include before/after numbers from `scripts/benchmark.py`
  (once it exists, see Phase 1) or the translator's console timings.

## Development setup

Requirements: Windows 10/11, Python 3.11–3.13, git.

```powershell
git clone https://github.com/dmunasingha/lingo-overlay.git
cd lingo-overlay
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python translator.py --devices   # list audio devices
.\.venv\Scripts\python translator.py             # run (downloads the model on first start)
```

Audio routing for testing is described in [docs/SETUP.md](docs/SETUP.md).
Without VB-Cable the app listens to all speaker audio, which is fine for development:
play a foreign-language YouTube video and watch the subtitles.

## Code style

- Format and lint with **ruff**: `.\.venv\Scripts\ruff check .` and `.\.venv\Scripts\ruff format .`
- Keep threads communicating through queues. No shared mutable state except the small `state` dict.
- No blocking work on the Tk (overlay) thread.
- Prefer small, focused functions with a docstring that says *why*, not *what*.

## Workflow

1. **Pick or open an issue.** Comment that you're working on it. For anything bigger than a bug fix,
   describe your approach in the issue first.
2. **Branch** from `main`: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
3. **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):
   `feat: add Silero VAD`, `fix: overlay hidden on second monitor`, `perf: shorter encoder context`.
4. **Open a PR** using the template. Link the issue (`Closes #12`), include latency numbers for
   pipeline changes, and screenshots or GIFs for UI changes.
5. CI (ruff and a byte-compile check on Windows) must pass. One maintainer approval is required.

## Reporting bugs

Use the **Bug report** issue template. Always include:

- the console output (it shows device names, detected languages and timings),
- your `config.json` (no secrets in it, but check first),
- the game, Windows version, CPU and GPU.

## Good first issues

Look for the `good first issue` label. Typical starters: overlay styling options, config
validation, hotkey conflicts, docs and translations of the README.

## Code of conduct

Be respectful. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
