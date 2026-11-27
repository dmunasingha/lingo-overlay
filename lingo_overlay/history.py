"""In-memory history of recent translations (shown in the history window)."""

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    timestamp: float
    lang: str
    text: str

    def format(self) -> str:
        clock = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"{clock}  [{self.lang.upper()}]  {self.text}"


class History:
    """Thread-safe ring buffer of the last ``size`` translations, oldest first."""

    def __init__(self, size: int = 50):
        self._items: deque[Entry] = deque(maxlen=max(1, int(size)))
        self._lock = threading.Lock()

    def add(self, lang: str, text: str, timestamp: float | None = None) -> Entry:
        entry = Entry(time.time() if timestamp is None else timestamp, lang, text)
        with self._lock:
            self._items.append(entry)
        return entry

    def entries(self) -> list[Entry]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def as_text(self) -> str:
        """All entries, one per line, ready to paste."""
        return "\n".join(e.format() for e in self.entries())

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
