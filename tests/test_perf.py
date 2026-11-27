import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fps_summary  # noqa: E402

import perf_limits  # noqa: E402


@pytest.mark.parametrize("spec", [None, "", "off", []])
def test_affinity_off(spec):
    assert perf_limits.affinity_mask(spec, 16) is None


def test_affinity_last():
    assert perf_limits.affinity_mask("last:4", 16) == 0b1111 << 12
    assert perf_limits.affinity_mask("last:16", 16) == (1 << 16) - 1


def test_affinity_list():
    assert perf_limits.affinity_mask([0, 2, 5], 8) == 0b100101


@pytest.mark.parametrize("spec", ["last:0", "last:17", "first:2", "last:x", [16], [-1], [True], 3])
def test_affinity_invalid(spec):
    with pytest.raises(ValueError):
        perf_limits.affinity_mask(spec, 16)


def test_apply_bad_config_is_not_fatal():
    logs = []
    perf_limits.apply({"process_priority": "normal", "cpu_affinity": "nonsense"}, logs.append)
    if sys.platform == "win32":
        assert any("Ignoring cpu_affinity" in m for m in logs)


def _csv(tmp_path, name, frametimes, col="MsBetweenPresents", app="game.exe"):
    p = tmp_path / f"{name}.csv"
    lines = [f"Application,ProcessID,{col}"]
    lines += [f"{app},1,{ft}" for ft in frametimes]
    lines.append("other.exe,2,1.0")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_summary_numbers(tmp_path):
    p = _csv(tmp_path, "off", [10.0] * 98 + [20.0] * 2)
    r = fps_summary.summarise(fps_summary.read_frametimes(p, "game.exe"))
    assert r["frames"] == 100
    assert r["avg_fps"] == pytest.approx(1000 / 10.2)
    assert r["low1_fps"] == pytest.approx(50.0)


def test_frametime_column_fallback(tmp_path):
    p = _csv(tmp_path, "x", [5.0, 5.0], col="FrameTime")
    assert fps_summary.read_frametimes(p, "game.exe") == [5.0, 5.0]


def test_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Application,Foo\ngame.exe,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        fps_summary.read_frametimes(p)


def test_main_pass_fail(tmp_path, capsys):
    off = _csv(tmp_path, "off", [10.0] * 50)
    ok = _csv(tmp_path, "ok", [10.3] * 50)      # ~2.9 % slower
    bad = _csv(tmp_path, "bad", [11.0] * 50)    # ~9.1 % slower
    assert fps_summary.main([str(off), str(ok), "--process", "game.exe"]) == 0
    assert fps_summary.main([str(off), str(bad), "--process", "game.exe"]) == 1
    assert "FAIL" in capsys.readouterr().out
