# Goals

## Vision

**Understand every teammate, in any language, without taking your eyes off the game.**

Lingo Overlay is a free, open-source, fully local tool that listens to in-game voice chat,
detects which language each person is speaking, and, if it isn't one you already understand,
shows an English subtitle on screen and can optionally speak the translation into your headset.
You always hear the original voice live. The translation is added on top.

## Principles

1. **Free and local.** No accounts, no API keys, no per-minute costs. Audio never leaves the PC.
2. **Never delay the game.** Game sounds and original voices are not delayed, muted or processed.
   Only a copy of the voice chat audio is analysed.
3. **Anti-cheat safe.** No injecting into game processes, no reading game memory, no hooking
   DirectX. Only standard Windows audio APIs and a normal always-on-top window. See
   [CONTRIBUTING.md](../CONTRIBUTING.md#anti-cheat-rules).
4. **Don't steal FPS.** Inference runs on the CPU by default, because during a match the GPU belongs to the game.
5. **Open all the way down.** Code is GPL-3.0. Every model we ship or recommend must allow free
   redistribution (see the model license table in [ARCHITECTURE.md](ARCHITECTURE.md#models-and-licenses)).

## Success metrics

These are measured by `scripts/benchmark.py` (planned in Phase 1) on a fixed set of voice clips,
**with a game running**, on a reference PC (Ryzen 7 7700X, GTX 1660, 16 GB RAM).

| Metric | Today (v0.1) | Phase 1 target | Long-term target |
|---|---|---|---|
| Subtitle latency (end of speech → subtitle, median) | 1.5 s (small) / 3.8 s (medium) | **≤ 1.0 s** | ≤ 0.5 s, with partial subtitles while the person is still talking |
| Spoken-translation latency (end of speech → voice starts) | n/a | ≤ 1.5 s | ≤ 1.0 s |
| Game FPS impact while translating | not measured | ≤ 5 % | ≤ 3 % |
| English speech wrongly translated (should be skipped) | not measured | ≤ 5 % of phrases | ≤ 2 % |
| Foreign speech wrongly skipped | not measured | ≤ 5 % of phrases | ≤ 2 % |
| Translation quality (human "meaning correct" rating) | ~75–80 % (estimate) | ≥ 80 % | ≥ 90 % |
| Setup time for a new user | ~15 min, manual | ≤ 5 min with installer | 1-click installer, no VB-Cable needed |

## Target users and platforms

- **Primary:** PC gamers on Windows 10/11 playing team games with international voice chat
  (Valorant, CS2, Overwatch, Apex, Rainbow Six and others) or using Discord.
- **Hardware floor:** 6-core CPU from 2019 or newer, 8 GB RAM. No GPU required.
- **Languages:** detect any of Whisper's ~99 languages; translate into English first. Other
  target languages come later.

## Non-goals (for now)

- **Translating your own voice to teammates.** It's a different problem (latency is
  much more visible, and it means injecting into the mic). It may come later as an
  opt-in feature.
- **Overlays inside exclusive-fullscreen games.** That needs DirectX hooking, which anti-cheat
  systems ban. We support borderless windowed only.
- **Cloud services.** Optional cloud back-ends may be accepted as plugins, but the default
  and the core must work 100 % offline.
- **macOS and Linux.** The architecture should not block them, but Windows comes first because that's
  where the games are.
