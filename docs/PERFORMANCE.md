# Game FPS impact

Target for v0.2: the translator costs **≤ 5 % average FPS**, measured in the game, while it is
actively translating.

## What the translator does to protect FPS

| Setting | Default | Effect |
|---|---|---|
| `process_priority` | `below_normal` | Set at startup. When the game and Whisper want the same core, Windows runs the game first. `idle` is even more polite but can make translations late under heavy load. `normal` restores the v0.1 behaviour. |
| `cpu_affinity` | `off` | `"last:N"` pins the translator to the highest N logical CPUs; a list like `[8, 9, 10, 11]` pins it to exactly those. Most games put their main and render threads on low-numbered cores, so this keeps Whisper out of their way. Off by default: turn it on only if a measurement shows it helps. |
| `cpu_threads` | `6` | How many threads Whisper uses. The biggest single knob: fewer threads means less contention but slower translations. |
| `device` | `cpu` | Keep it on `cpu` while gaming. The game usually has the GPU at ~100 %, and `cuda` competes with it directly. |

## Recommended `cpu_threads`

These are starting points, not measurements. Run the benchmark below on your own machine and
adjust. If FPS drops more than 5 %, lower `cpu_threads` by one or two, or switch to
`"model": "small"`.

| CPU class | Example | `cpu_threads` | `cpu_affinity` to try | Model |
|---|---|---|---|---|
| 4 cores / 8 threads | i3-12100, Ryzen 3 | 2 | `off` | `small` |
| 6 cores / 12 threads | i5-10600K, Ryzen 5 5600X | 3–4 | `"last:4"` | `small` |
| 8 cores / 16 threads | Ryzen 7 7700X, i7-10700K | 4–6 | `"last:6"` | `medium` or `small` |
| 12+ cores / 24+ threads | Ryzen 9 7900X, 7950X | 6–8 | `"last:8"` | `medium` |
| Intel hybrid (P + E cores) | i5-13600K, i7-14700K | 4 | `"last:4"` (E-cores) | `small` |

Notes:

- `cpu_threads` larger than the number of CPUs in `cpu_affinity` only adds contention. Keep it
  equal or lower.
- On Intel 12th gen and later, the highest logical CPUs are E-cores, so `"last:N"` keeps
  Whisper off the P-cores the game wants. Translations get slower; check the latency is still OK.
- On Ryzen 9 parts with two CCDs, `"last:N"` lands on the second CCD, which is usually what
  you want: games prefer the first (and the X3D cache die).
- Affinity covers one processor group (up to 64 logical CPUs).

## Measuring it

You need [PresentMon 2.x](https://github.com/GameTechDev/PresentMon/releases) (the console
build). CapFrameX works too; export its CSVs and use `tools/fps_summary.py` directly.

1. Start the game in Borderless Windowed and go somewhere static (practice range, menu scene
   with 3D background). Use the same spot for every run.
2. In an **administrator** PowerShell in the repo folder:

   ```powershell
   .\tools\fps_bench.ps1 -Game VALORANT-Win64-Shipping.exe -PresentMon C:\Tools\PresentMon.exe
   ```

3. The script records 60 s for each scenario and waits for Enter before each one:
   - `off`: translator not running
   - `idle`: translator running, nobody talking
   - `translating`: foreign speech playing through Discord (loop a clip so Whisper is busy the
     whole run)
4. It prints a table and PASS/FAIL against the 5 % target. CSVs are kept in `bench\fps\<time>\`.

To compare settings, re-run with only the scenarios you need and summarise against the old
baseline:

```powershell
python tools\fps_summary.py bench\fps\<time>\off.csv bench\fps\<new>\translating.csv
```

Example output:

```
run               frames   avg FPS    1% low    avg Δ    1% Δ
off                14310     238.5     171.2     0.0%     0.0%
idle               14280     238.0     170.4     0.2%     0.5%
translating        13950     232.5     158.9     2.5%     7.2%

Worst avg-FPS impact 2.5% (target <= 5%): PASS
```

(Illustrative numbers. Please post real results, with your CPU and settings, on the issue
tracker so the table above can be filled with measurements.)
