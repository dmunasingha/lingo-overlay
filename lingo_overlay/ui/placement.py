"""Where the subtitle goes: monitors, saved anchor, and text-outline offsets.

Everything here except ``list_monitors`` is pure, so it can be tested without a screen.

A saved position is an *anchor*: ``[fx, fy]`` fractions of a monitor, where ``fx`` is the
subtitle's horizontal centre and ``fy`` its bottom edge. Anchoring the bottom keeps a
two-line subtitle growing upward from where it was placed, like the default layout.
"""

from __future__ import annotations

from typing import NamedTuple


class Monitor(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom


def list_monitors(fallback_size: tuple[int, int] | None = None) -> list[Monitor]:
    """All monitors, primary first, then left to right. Falls back to one screen."""
    monitors: list[tuple[bool, Monitor]] = []
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        user32 = ctypes.windll.user32
        proc_type = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                                       ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

        def cb(hmon, _hdc, _rect, _data):
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                r = info.rcMonitor
                monitors.append((bool(info.dwFlags & 1), Monitor(r.left, r.top, r.right, r.bottom)))
            return 1

        user32.EnumDisplayMonitors(None, None, proc_type(cb), 0)
    except Exception:  # not Windows, or the call failed: use the fallback below
        monitors = []
    if not monitors:
        w, h = fallback_size or (1920, 1080)
        return [Monitor(0, 0, w, h)]
    monitors.sort(key=lambda pm: (not pm[0], pm[1].left, pm[1].top))
    return [m for _, m in monitors]


def pick_monitor(monitors: list[Monitor], index: int) -> Monitor:
    """Monitor ``index`` (0 = primary); the primary if that monitor is not connected."""
    return monitors[index] if 0 <= index < len(monitors) else monitors[0]


def subtitle_xy(mon: Monitor, w: int, h: int, anchor, bottom_offset: int) -> tuple[int, int]:
    """Top-left corner for a ``w`` x ``h`` subtitle, kept fully on the monitor."""
    if anchor is None:
        cx, bottom = mon.left + mon.width / 2, mon.bottom - bottom_offset
    else:
        cx, bottom = mon.left + anchor[0] * mon.width, mon.top + anchor[1] * mon.height
    x = round(cx - w / 2)
    y = round(bottom - h)
    x = max(mon.left, min(x, mon.right - w))
    y = max(mon.top, min(y, mon.bottom - h))
    return x, y


def anchor_from_xy(monitors: list[Monitor], x: int, y: int, w: int, h: int) -> tuple[int, list[float]]:
    """(monitor index, anchor) for a subtitle whose top-left is at (x, y).

    The monitor is the one under the subtitle's centre, or the nearest one if the centre
    is off every screen.
    """
    cx, cy = x + w / 2, y + h / 2

    def dist(m: Monitor) -> float:
        dx = max(m.left - cx, 0, cx - m.right)
        dy = max(m.top - cy, 0, cy - m.bottom)
        return dx * dx + dy * dy

    index = next((i for i, m in enumerate(monitors) if m.contains(cx, cy)), None)
    if index is None:
        index = min(range(len(monitors)), key=lambda i: dist(monitors[i]))
    mon = monitors[index]
    fx = (cx - mon.left) / mon.width
    fy = (y + h - mon.top) / mon.height
    clamp = lambda v: round(min(max(v, 0.0), 1.0), 4)  # noqa: E731
    return index, [clamp(fx), clamp(fy)]


def outline_offsets(width: int) -> list[tuple[int, int]]:
    """Offsets to draw the outline copies at: a ring of radius ``width`` around the text."""
    if width <= 0:
        return []
    return [(dx, dy) for dx in range(-width, width + 1) for dy in range(-width, width + 1)
            if (dx or dy) and dx * dx + dy * dy <= width * width + width]
