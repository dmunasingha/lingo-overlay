"""Speaker labels (P1, P2, ...) from voice embeddings, and splitting a phrase at speaker changes.

Pure numpy: the embedding model is passed in as a function, so this is testable without torch.
"""

import numpy as np

SR = 16000


def _unit(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class SpeakerTracker:
    """Remembers the voices heard so far and gives each one a stable number.

    A new embedding joins the most similar known voice when the cosine similarity is at least
    ``threshold``; otherwise it becomes a new speaker. When ``max_speakers`` voices are known, the
    one heard least recently is forgotten and its number reused (players change between matches).
    """

    def __init__(self, threshold=0.5, max_speakers=9, max_weight=20):
        self.threshold = float(threshold)
        self.max_speakers = max(1, int(max_speakers))
        self.max_weight = max_weight  # cap, so a voice's profile keeps adapting slowly
        self.reset()

    def reset(self):
        self._centroids = {}  # speaker number -> unit vector
        self._counts = {}
        self._last_seen = {}
        self._tick = 0

    def __len__(self):
        return len(self._centroids)

    def best_match(self, emb):
        """Return (speaker number, similarity) of the closest known voice, or (None, -1.0)."""
        e = _unit(emb)
        best, best_sim = None, -1.0
        for spk, c in self._centroids.items():
            sim = float(np.dot(c, e))
            if sim > best_sim:
                best, best_sim = spk, sim
        return best, best_sim

    def identify(self, emb):
        """Return the speaker number (1-based) for this embedding and learn from it."""
        e = _unit(emb)
        self._tick += 1
        spk, sim = self.best_match(e)
        if spk is None or sim < self.threshold:
            spk = self._new_slot()
            self._centroids[spk] = e
            self._counts[spk] = 1
        else:
            n = min(self._counts[spk], self.max_weight)
            self._centroids[spk] = _unit(self._centroids[spk] * n + e)
            self._counts[spk] += 1
        self._last_seen[spk] = self._tick
        return spk

    def _new_slot(self):
        for spk in range(1, self.max_speakers + 1):
            if spk not in self._centroids:
                return spk
        oldest = min(self._last_seen, key=self._last_seen.get)
        del self._centroids[oldest], self._counts[oldest], self._last_seen[oldest]
        return oldest


def change_points(window_embs, threshold):
    """Split a run of window embeddings into groups of consecutive windows by the same voice.

    Each window is compared to the mean of the current group; below ``threshold`` cosine
    similarity a new group starts. Returns a list of (first_window, end_window) index pairs.
    """
    if not window_embs:
        return []
    groups, start = [], 0
    mean = _unit(window_embs[0])
    for i in range(1, len(window_embs)):
        e = _unit(window_embs[i])
        if float(np.dot(mean, e)) < threshold:
            groups.append((start, i))
            start, mean = i, e
        else:
            mean = _unit(mean * (i - start) + e)
    groups.append((start, len(window_embs)))
    return groups


def split_by_speaker(audio, embed, tracker, split=True, window_s=1.5, sr=SR):
    """Label a phrase by speaker, splitting it where the voice changes.

    ``embed(audio) -> vector``. Returns a list of (speaker number, audio) in time order. Without
    ``split``, or when the phrase is shorter than two windows, the whole phrase is one part.
    Neighbouring parts that turn out to be the same known speaker are joined again.
    """
    win = int(window_s * sr)
    if not split or win <= 0 or len(audio) < 2 * win:
        return [(tracker.identify(embed(audio)), audio)]

    n = len(audio) // win  # the leftover (< 1 window) is added to the last window
    bounds = [(i * win, (i + 1) * win if i < n - 1 else len(audio)) for i in range(n)]
    embs = [embed(audio[a:b]) for a, b in bounds]
    groups = change_points(embs, tracker.threshold)

    parts = []
    for g0, g1 in groups:
        a, b = bounds[g0][0], bounds[g1 - 1][1]
        chunk = audio[a:b]
        emb = embs[g0] if g1 - g0 == 1 else embed(chunk)
        spk = tracker.identify(emb)
        if parts and parts[-1][0] == spk:
            parts[-1] = (spk, np.concatenate([parts[-1][1], chunk]))
        else:
            parts.append((spk, chunk))
    return parts
