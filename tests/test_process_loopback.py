import ctypes
import os

import numpy as np
import pytest

from lingo_overlay.audio import process_loopback as pl
from lingo_overlay.config import Config


def test_pick_root_pid_prefers_topmost_process_of_the_tree():
    procs = [
        (4, 0, "System"),
        (100, 50, "Discord.exe"),   # root (parent is explorer)
        (120, 100, "Discord.exe"),  # renderer / voice child
        (130, 100, "Discord.exe"),
        (50, 1, "explorer.exe"),
    ]
    assert pl.pick_root_pid(procs, "Discord.exe") == 100
    assert pl.pick_root_pid(procs, "discord") == 100  # case-insensitive, ".exe" optional
    assert pl.pick_root_pid(procs, "Telegram.exe") is None


def test_pick_root_pid_with_two_separate_instances_takes_lowest_root():
    procs = [(300, 9, "app.exe"), (200, 8, "app.exe"), (210, 200, "app.exe")]
    assert pl.pick_root_pid(procs, "app.exe") == 200


def test_structure_layouts_match_windows_headers():
    assert ctypes.sizeof(pl.WAVEFORMATEX) == 18
    assert ctypes.sizeof(pl.ACTIVATION_PARAMS) == 12
    assert ctypes.sizeof(pl.GUID) == 16
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(pl.PROPVARIANT) == 24


def test_guid_parsing_is_little_endian():
    g = pl.GUID.of("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2")
    assert g.Data1 == 0x1CB9AD4C and g.Data2 == 0xDBFA and g.Data3 == 0x4C32
    assert bytes(g.Data4) == bytes.fromhex("B178C2F568A703B2")


def test_pcm_format_and_conversion():
    f = pl.pcm_format(48000)
    assert (f.nChannels, f.nSamplesPerSec, f.nBlockAlign, f.nAvgBytesPerSec) == (2, 48000, 4, 192000)
    raw = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes()
    out = pl.pcm16_to_float(raw)
    assert out.shape == (2, 2) and out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.0, 0.5], [-1.0, 32767 / 32768]])


def test_capture_process_defaults_off():
    assert Config().capture_process == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_missing_process_raises_clear_error():
    with pytest.raises(pl.ProcessCaptureError, match="not running|Windows 10 2004"):
        pl.open_process_recorder("no-such-app-lingo-test.exe")
