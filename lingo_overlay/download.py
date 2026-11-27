"""Model download that survives stalls: resume, retry, then fall back to the next URL.

Plain ``urllib`` (no huggingface_hub, whose downloads can hang forever with no progress).
Data goes to ``<file>.part`` first and is renamed only when complete, so an interrupted
download is never mistaken for a finished model. A read that gets no bytes for
``stall_timeout`` seconds counts as a stall: the connection is dropped and the download
resumes from where it stopped (HTTP Range), up to ``retries`` times per URL.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

CHUNK = 1 << 20
USER_AGENT = "lingo-overlay-model-download"

# progress(done_bytes, total_bytes_or_0)
Progress = Callable[[int, int], None]


class DownloadError(RuntimeError):
    """Every URL failed."""


def download(urls: Iterable[str], dest: Path | str, progress: Progress | None = None,
             stall_timeout: float = 30.0, retries: int = 3, log: Callable[..., None] = print,
             cancel: Callable[[], bool] | None = None) -> Path:
    """Download the first URL that works to ``dest``. Returns ``dest``.

    ``cancel`` is polled between chunks; when it returns True the download stops with
    ``DownloadError`` and the ``.part`` file is kept, so the next attempt resumes.
    """
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                _fetch(url, part, progress, stall_timeout, cancel)
                os.replace(part, dest)
                return dest
            except _Cancelled:
                raise DownloadError("Download cancelled") from None
            except urllib.error.HTTPError as ex:
                if ex.code == 416:  # Range not satisfiable: the .part is bad, start over
                    part.unlink(missing_ok=True)
                errors.append(f"{url}: HTTP {ex.code}")
                log(f"  download failed ({url}): HTTP {ex.code}")
                if 400 <= ex.code < 500 and ex.code not in (408, 416, 429):
                    break  # 404 etc. won't fix itself: try the next URL
            except (OSError, urllib.error.URLError, _Incomplete) as ex:
                errors.append(f"{url}: {ex}")
                log(f"  download interrupted ({url}, try {attempt}/{retries}): {ex}")
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 5))
    raise DownloadError("Could not download " + dest.name + ":\n  " + "\n  ".join(errors))


class _Cancelled(Exception):
    pass


class _Incomplete(Exception):
    pass


def _fetch(url, part: Path, progress, stall_timeout, cancel):
    have = part.stat().st_size if part.is_file() else 0
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if have:
        req.add_header("Range", f"bytes={have}-")
    # The timeout applies to each socket read, so it doubles as the stall detector.
    with urllib.request.urlopen(req, timeout=stall_timeout) as resp:
        if have and resp.status != 206:  # server ignored Range: start from scratch
            have = 0
        length = int(resp.headers.get("Content-Length") or 0)
        total = have + length if length else 0
        with open(part, "ab" if have else "wb") as f:
            done = have
            if progress:
                progress(done, total)
            while True:
                if cancel and cancel():
                    raise _Cancelled
                try:
                    chunk = resp.read(CHUNK)
                except TimeoutError as ex:
                    raise _Incomplete(f"stalled: no data for {stall_timeout:g}s") from ex
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    if total and done < total:
        raise _Incomplete(f"connection closed at {done} of {total} bytes")


def text_progress(log: Callable[..., None] = print, steps: int = 10) -> Progress:
    """Progress callback that logs every 1/steps of the file (for the console)."""
    last = [-1]

    def cb(done, total):
        if not total:
            return
        step = done * steps // total
        if step > last[0]:
            last[0] = step
            log(f"  {step * 100 // steps}% of {total >> 20} MB")

    return cb
