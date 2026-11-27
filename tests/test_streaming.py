"""Streaming partial subtitles: segmenter snapshots and the LocalAgreement policy."""

import queue

import numpy as np
from helpers import SR, drain, silence, tone

from translator import LocalAgreement, Segmenter, common_prefix, latest_partial


def run(cfg, audio):
    out_q, partial_q = queue.Queue(), queue.Queue()
    Segmenter({**cfg, "stream_first_ms": 400, "stream_interval_ms": 500}, out_q,
              partial_q=partial_q).feed(audio)
    return drain(out_q), drain(partial_q)


# ---------------------------------------------------------------- segmenter
def test_no_partial_queue_means_no_partials(vad_cfg, out_q):
    seg = Segmenter(vad_cfg, out_q)
    seg.feed(np.concatenate([tone(2.0), silence(1.0)]))
    assert seg.partial_q is None and len(drain(out_q)) == 1


def test_partials_grow_while_speaking_then_end_marker(vad_cfg):
    finals, partials = run(vad_cfg, np.concatenate([silence(0.5), tone(2.0), silence(1.0)]))
    assert len(finals) == 1
    snaps = [a for _pid, a in partials if a is not None]
    assert len(snaps) >= 3  # ~2 s of speech at 0.5 s intervals
    assert all(len(b) > len(a) for a, b in zip(snaps, snaps[1:], strict=False))
    assert partials[-1] == (1, None)  # phrase closed after its last snapshot
    assert {pid for pid, _a in partials} == {1}
    # Every snapshot is a prefix of the final phrase.
    final = finals[0][0]
    for s in snaps:
        np.testing.assert_array_equal(s, final[:len(s)])


def test_first_partial_within_600ms_of_speech_start(vad_cfg):
    lead = 0.5
    out_q, partial_q = queue.Queue(), queue.Queue()
    seg = Segmenter({**vad_cfg, "stream_first_ms": 400}, out_q, partial_q=partial_q)
    audio = np.concatenate([silence(lead), tone(2.0), silence(1.0)])
    step = seg.frame_len
    first_at = None
    for i in range(0, len(audio), step):
        seg.feed(audio[i:i + step])
        if not partial_q.empty():
            first_at = (i + step) / SR
            break
    assert first_at is not None and first_at - lead <= 0.6


def test_phrase_ids_increase_and_close(vad_cfg):
    audio = np.concatenate([tone(1.5), silence(1.0), tone(1.5, freq=660), silence(1.0)])
    finals, partials = run(vad_cfg, audio)
    assert len(finals) == 2
    ends = [pid for pid, a in partials if a is None]
    assert ends == [1, 2]
    # The end marker of phrase 1 comes before any snapshot of phrase 2.
    end_1 = next(i for i, (pid, a) in enumerate(partials) if pid == 1 and a is None)
    first_2 = next(i for i, (pid, _a) in enumerate(partials) if pid == 2)
    assert end_1 < first_2


def test_dropped_blip_still_closes_its_phrase(vad_cfg):
    finals, partials = run(vad_cfg, np.concatenate([tone(0.2), silence(1.0)]))
    assert finals == []
    assert [(pid, a is None) for pid, a in partials] == [(1, True)]  # no snapshot of a click


# ---------------------------------------------------------------- LocalAgreement
def test_common_prefix_ignores_case_and_punctuation():
    assert common_prefix("Go B, now".split(), "go b now please".split()) == 3
    assert common_prefix(["push"], ["rotate"]) == 0
    assert common_prefix([], ["a"]) == 0


def test_local_agreement_commits_stable_prefix():
    la = LocalAgreement()
    assert la.update("enemy on") == ("", "enemy on")  # nothing to agree with yet
    assert la.update("enemy on B site") == ("enemy on", "B site")
    assert la.update("enemy on B main") == ("enemy on B", "main")
    assert la.update("enemy on B main two") == ("enemy on B main", "two")


def test_local_agreement_never_retracts_committed_words():
    la = LocalAgreement()
    la.update("rotate to A")
    la.update("rotate to A now")
    committed, tentative = la.update("go to A now")  # decoder changed its mind
    assert committed == "rotate to A" and tentative == ""
    la.update("rotate to A now")  # back on track; needs two agreeing runs again
    committed, tentative = la.update("rotate to A now quickly")
    assert committed == "rotate to A now"
    assert tentative == "quickly"


# ---------------------------------------------------------------- scheduling
def test_latest_partial_keeps_newest_open_snapshot():
    q, finished = queue.Queue(), set()
    a, b = np.zeros(10, np.float32), np.ones(20, np.float32)
    q.put((1, a))
    q.put((1, b))
    pid, audio = latest_partial(q, finished)
    assert pid == 1 and audio is b
    assert latest_partial(q, finished) is None  # drained


def test_latest_partial_drops_finished_phrases():
    q, finished = queue.Queue(), set()
    q.put((1, np.zeros(10, np.float32)))
    q.put((1, None))
    assert latest_partial(q, finished) is None
    assert finished == {1}
    q.put((2, np.zeros(10, np.float32)))
    assert latest_partial(q, finished)[0] == 2
