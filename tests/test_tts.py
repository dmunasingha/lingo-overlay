import threading

import numpy as np

import tts
from translator import load_config


class FakeEngine:
    name = "fake"

    def __init__(self, sr=1000, on_synth=None):
        self.sr, self.said, self.on_synth = sr, [], on_synth

    def synth(self, text):
        self.said.append(text)
        if self.on_synth:
            self.on_synth(text)
        return np.ones(len(text) * 10, np.float32), self.sr


class FakePlayer:
    def __init__(self, on_play=None):
        self.blocks, self.on_play = [], on_play

    def play(self, block):
        self.blocks.append(block.copy())
        if self.on_play:
            self.on_play(len(self.blocks))


def speaker(engine, player, volume=1.0):
    return tts.StreamingSpeaker(engine, lambda sr: player, volume=volume, block_s=0.01)


def drain(sp):
    while sp.play_next(timeout=0):
        pass


# --- sentence splitting ---------------------------------------------------
def test_split_sentences_keeps_punctuation():
    assert tts.split_sentences("Push B. They're rotating! Where?") == ["Push B.", "They're rotating!", "Where?"]


def test_split_sentences_single_and_empty():
    assert tts.split_sentences("  one tapped  ") == ["one tapped"]
    assert tts.split_sentences("") == []


def test_split_sentences_does_not_split_mid_number():
    assert tts.split_sentences("He has 1.5 hp left.") == ["He has 1.5 hp left."]


def test_split_sentences_cuts_long_sentences():
    long = ", ".join(["enemy on site"] * 40)
    parts = tts.split_sentences(long, max_chars=50)
    assert all(len(p) <= 51 for p in parts)
    assert " ".join(parts).replace("  ", " ") == long


def test_speed_from_rate_is_clamped():
    assert tts.speed_from_rate(175) == 1.0
    assert tts.speed_from_rate(10) == 0.5
    assert tts.speed_from_rate(1000) == 2.0


# --- streaming playback ---------------------------------------------------
def test_streams_each_sentence_and_applies_volume():
    eng, pl = FakeEngine(), FakePlayer()
    sp = speaker(eng, pl, volume=0.5)
    sp.say("Push B. Now!")
    assert eng.said == ["Push B.", "Now!"]
    drain(sp)
    audio = np.concatenate(pl.blocks)
    assert len(audio) == (len("Push B.") + len("Now!")) * 10
    assert np.allclose(audio, 0.5)
    assert all(len(b) <= 10 for b in pl.blocks)  # 0.01 s blocks at 1 kHz


def test_new_line_cuts_off_playback():
    eng = FakeEngine()
    sp = None

    def interrupt(n):
        if n == 2:  # a new subtitle arrives while the first one is playing
            sp.gen += 1

    pl = FakePlayer(on_play=interrupt)
    sp = speaker(eng, pl)
    sp.say("A long first line that would take a while to say.")
    drain(sp)
    assert len(pl.blocks) == 2


def test_pending_line_stops_synthesis_of_remaining_sentences():
    calls = {"n": 0}

    def pending():
        return calls["n"] >= 1

    eng = FakeEngine(on_synth=lambda t: calls.__setitem__("n", calls["n"] + 1))
    sp = speaker(eng, FakePlayer())
    sp.say("One. Two. Three.", pending=pending)
    assert eng.said == ["One."]
    assert sp.play_q.empty()  # synthesized but already stale, so not queued


def test_stale_queued_audio_is_skipped():
    eng, pl = FakeEngine(), FakePlayer()
    sp = speaker(eng, pl)
    sp.say("Old line.")
    sp.say("New line.")
    drain(sp)
    assert len(np.concatenate(pl.blocks)) == len("New line.") * 10


def test_player_opened_once_per_sample_rate():
    opened = []
    pl = FakePlayer()
    sp = tts.StreamingSpeaker(FakeEngine(sr=800), lambda sr: opened.append(sr) or pl, block_s=0.01)
    sp.say("One. Two.")
    drain(sp)
    assert opened == [800]


def test_play_next_times_out_quietly():
    assert tts.StreamingSpeaker(FakeEngine(), lambda sr: FakePlayer()).play_next(timeout=0) is False


# --- engine selection -----------------------------------------------------
def test_sapi_selected_by_default():
    assert tts.load_engine({**tts.TTS_DEFAULTS}, log=lambda *a: None) is None


def test_unknown_engine_falls_back_to_sapi():
    logs = []
    assert tts.load_engine({**tts.TTS_DEFAULTS, "voice_engine": "nope"}, log=logs.append) is None
    assert "Unknown" in logs[0]


def test_failed_engine_falls_back_to_sapi(monkeypatch):
    class Broken:
        def __init__(self, cfg):
            raise FileNotFoundError("no model")

    monkeypatch.setitem(tts.ENGINES, "piper", Broken)
    logs = []
    assert tts.load_engine({**tts.TTS_DEFAULTS, "voice_engine": "piper"}, log=logs.append) is None
    assert "unavailable" in logs[0]


def test_engine_loads_when_available(monkeypatch):
    monkeypatch.setitem(tts.ENGINES, "kokoro", lambda cfg: FakeEngine())
    eng = tts.load_engine({**tts.TTS_DEFAULTS, "voice_engine": "kokoro"}, log=lambda *a: None)
    assert isinstance(eng, FakeEngine)


def test_shipped_config_has_tts_keys():
    cfg = load_config()
    assert set(tts.TTS_DEFAULTS) <= cfg.keys()
    assert cfg["voice_engine"] in ("sapi", *tts.ENGINES)


def test_tts_loop_speaks_newest_line_with_neural_engine(monkeypatch):
    import queue

    eng, pl = FakeEngine(), FakePlayer()
    monkeypatch.setattr(tts, "load_engine", lambda cfg, log: eng)
    monkeypatch.setattr(tts, "open_soundcard_player", lambda cfg, log: (lambda sr: pl))
    state = {"quit": False}
    q = queue.Queue()
    q.put("Stale line.")
    q.put("Fresh line.")
    t = threading.Thread(target=tts.tts_loop, args=({"voice_volume": 1.0}, q, state, lambda *a: None))
    t.start()
    for _ in range(200):
        if pl.blocks and sum(len(b) for b in pl.blocks) >= len("Fresh line.") * 10:
            break
        threading.Event().wait(0.01)
    state["quit"] = True
    t.join(timeout=2)
    assert eng.said == ["Fresh line."]
