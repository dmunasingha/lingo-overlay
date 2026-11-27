"""
Keep the translator from stealing frames from the game.

Two knobs, both read from config.json:

  "process_priority": "below_normal"   # "normal" | "below_normal" | "idle"
  "cpu_affinity": "off"                 # "off" | "last:N" | [list of logical CPU indices]

Below-normal priority means Windows always schedules the game's threads first
when both want the same core; Whisper only gets the leftovers. Affinity pins the
translator to a subset of logical CPUs (e.g. "last:4") so it stays away from the
low-numbered cores most games put their main and render threads on. Affinity is
off by default: measure with tools/fps_bench.ps1 before turning it on.

Everything here is a no-op off Windows, and failures are logged, never fatal.
"""

import ctypes
import os
import sys

PRIORITY_CLASSES = {
    "idle": 0x00000040,           # IDLE_PRIORITY_CLASS
    "below_normal": 0x00004000,   # BELOW_NORMAL_PRIORITY_CLASS
    "normal": 0x00000020,         # NORMAL_PRIORITY_CLASS
}


def affinity_mask(spec, cpu_count):
    """Turn a cpu_affinity config value into a bitmask, or None for "don't pin".

    spec: "off" / None / "" / []  -> None
          "last:N"                -> the highest N logical CPUs
          [i, j, ...]             -> exactly those logical CPUs
    Raises ValueError for anything malformed or out of range.
    """
    if spec in (None, "", "off", []):
        return None
    if cpu_count < 1:
        raise ValueError("cpu_count must be >= 1")
    if isinstance(spec, str):
        kind, _, n = spec.partition(":")
        if kind != "last" or not n.strip().isdigit():
            raise ValueError(f"cpu_affinity must be 'off', 'last:N' or a list, got {spec!r}")
        n = int(n)
        if not 1 <= n <= cpu_count:
            raise ValueError(f"cpu_affinity 'last:{n}' needs 1..{cpu_count} CPUs")
        cpus = range(cpu_count - n, cpu_count)
    elif isinstance(spec, (list, tuple)):
        cpus = list(spec)
        for c in cpus:
            if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c < cpu_count:
                raise ValueError(f"cpu_affinity entry {c!r} is not a CPU index in 0..{cpu_count - 1}")
    else:
        raise ValueError(f"cpu_affinity must be 'off', 'last:N' or a list, got {spec!r}")
    mask = 0
    for c in cpus:
        mask |= 1 << c
    return mask


def apply(cfg, log=print):
    """Apply process priority and CPU affinity from cfg to the current process."""
    if sys.platform != "win32":
        return
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    proc = ctypes.c_void_p(k32.GetCurrentProcess())

    prio = cfg.get("process_priority", "below_normal")
    if prio not in PRIORITY_CLASSES:
        log(f"Unknown process_priority {prio!r}; using below_normal")
        prio = "below_normal"
    if k32.SetPriorityClass(proc, PRIORITY_CLASSES[prio]):
        log(f"Process priority: {prio}")
    else:
        log(f"Could not set process priority (error {ctypes.get_last_error()})")

    try:
        # Affinity masks cover one processor group (max 64 logical CPUs).
        mask = affinity_mask(cfg.get("cpu_affinity", "off"), min(os.cpu_count() or 1, 64))
    except ValueError as e:
        log(f"Ignoring cpu_affinity: {e}")
        return
    if mask is None:
        return
    k32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    if k32.SetProcessAffinityMask(proc, mask):
        cpus = [i for i in range(64) if mask >> i & 1]
        log(f"Pinned to logical CPUs {cpus}")
    else:
        log(f"Could not set CPU affinity (error {ctypes.get_last_error()})")
