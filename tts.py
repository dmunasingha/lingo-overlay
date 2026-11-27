"""
Spoken translation: Windows SAPI (built in), Piper and Kokoro (neural, offline).

Neural engines are streamed sentence by sentence: one thread synthesizes the next sentence
while another plays the current one, so the voice starts as soon as the first sentence is
ready. A new subtitle line cuts off whatever is still being said (stale speech is useless in a
fast game), both mid-synthesis and mid-playback.

Config (config.json):
  voice_engine          "sapi" (default), "piper" or "kokoro"; falls back to "sapi" if it can't load
  voice_output_device   part of the speaker name to play on; "" = default speaker
  piper_voice           path to a Piper .onnx voice (its .onnx.json must sit next to it)
  kokoro_model          path to kokoro-v1.0.onnx
  kokoro_voices         path to voices-v1.0.bin
  kokoro_voice          voice name, e.g. "af_heart"
  voice_rate            words per minute; 175 = normal speed
  voice_volume          0..1
"""

import queue
import re
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NORMAL_RATE = 175  # words per minute that maps to 1.0x speed
BLOCK_S = 0.04  # playback block; also how fast a new line can cut off old speech

TTS_DEFAULTS = {
    "voice_engine": "sapi",
    "voice_output_device": "",
    "piper_voice": "models/piper/en_US-lessac-medium.onnx",
    "kokoro_model": "models/kokoro/kokoro-v1.0.onnx",
    "kokoro_voices": "models/kokoro/voices-v1.0.bin",
    "kokoro_voice": "af_heart",
}

# Split after . ! ? (and CJK ones) followed by space or end, keeping the punctuation.
_SENTENCE_END = re.compile(r"(?<=[.!?…。！？])\s+")


def split_sentences(text, max_chars=200):
    """Split text into sentences for streaming. Very long sentences are cut at commas/spaces."""
    out = []
    for s in _SENTENCE_END.split(text.strip()):
        s = s.strip()
        while len(s) > max_chars:
            cut = max(s.rfind(", ", 0, max_chars), s.rfind(" ", 0, max_chars))
            if cut <= 0:
                cut = max_chars
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


def speed_from_rate(voice_rate):
    return max(0.5, min(2.0, float(voice_rate) / NORMAL_RATE))


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


# --------------------------------------------------------------------------
# Engines: synth(text) -> (float32 mono array, sample_rate)
# --------------------------------------------------------------------------
class PiperEngine:
    name = "piper"

    def __init__(self, cfg):
        from piper import PiperVoice

        model = _path(cfg["piper_voice"])
        if not model.is_file():
            raise FileNotFoundError(f"Piper voice not found: {model}")
        self.voice = PiperVoice.load(str(model))
        self.syn_config = None
        try:
            from piper import SynthesisConfig

            self.syn_config = SynthesisConfig(length_scale=1.0 / speed_from_rate(cfg["voice_rate"]))
        except ImportError:  # older piper-tts without SynthesisConfig: default speed
            pass

    def synth(self, text):
        chunks = list(self.voice.synthesize(text, syn_config=self.syn_config))
        if not chunks:
            return np.zeros(0, np.float32), 22050
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, chunks[0].sample_rate


class KokoroEngine:
    name = "kokoro"

    def __init__(self, cfg):
        from kokoro_onnx import Kokoro

        model, voices = _path(cfg["kokoro_model"]), _path(cfg["kokoro_voices"])
        for p in (model, voices):
            if not p.is_file():
                raise FileNotFoundError(f"Kokoro file not found: {p}")
        self.kokoro = Kokoro(str(model), str(voices))
        self.voice = cfg["kokoro_voice"]
        self.speed = speed_from_rate(cfg["voice_rate"])

    def synth(self, text):
        samples, sr = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return np.asarray(samples, np.float32), sr


ENGINES = {"piper": PiperEngine, "kokoro": KokoroEngine}


def load_engine(cfg, log):
    """Return a neural engine, or None to use SAPI."""
    name = (cfg.get("voice_engine") or "sapi").lower()
    if name == "sapi":
        return None
    if name not in ENGINES:
        log(f"Unknown voice_engine '{name}'. Using Windows SAPI voice.")
        return None
    try:
        engine = ENGINES[name](cfg)
        log(f"Voice: {name} ready.")
        return engine
    except Exception as e:  # missing package, missing model file, ...
        log(f"Voice: {name} unavailable ({type(e).__name__}: {e}). Using Windows SAPI voice.")
        log("  See docs/SETUP.md, 'Natural voice (Piper / Kokoro)'.")
        return None


# --------------------------------------------------------------------------
# Streaming speaker: synth thread -> play thread, newest line wins
# --------------------------------------------------------------------------
class StreamingSpeaker:
    def __init__(self, engine, open_player, volume=1.0, block_s=BLOCK_S):
        """open_player(sample_rate) -> object with .play(float32 array); reused per rate."""
        self.engine, self.open_player = engine, open_player
        self.volume, self.block_s = volume, block_s
        self.gen = 0  # bumped by every new line; older audio is dropped
        self.play_q = queue.Queue()
        self.players = {}

    def stale(self, gen):
        return gen != self.gen

    def say(self, text, pending=lambda: False):
        """Synthesize text sentence by sentence (call from the synth thread).

        Stops early when `pending()` says a newer line is waiting."""
        self.gen += 1
        gen = self.gen
        for sentence in split_sentences(text):
            if pending() or self.stale(gen):
                return
            audio, sr = self.engine.synth(sentence)
            if pending() or self.stale(gen):
                return
            if len(audio):
                self.play_q.put((gen, audio * self.volume, sr))

    def play_next(self, timeout=0.3):
        """Play one queued sentence (call from the play thread). Returns False if nothing came."""
        try:
            gen, audio, sr = self.play_q.get(timeout=timeout)
        except queue.Empty:
            return False
        if self.stale(gen):
            return True
        player = self.players.get(sr)
        if player is None:
            player = self.players[sr] = self.open_player(sr)
        block = max(1, int(sr * self.block_s))
        for i in range(0, len(audio), block):
            if self.stale(gen):
                break
            player.play(audio[i:i + block])
        return True


def open_soundcard_player(cfg, log):
    import soundcard as sc

    want = (cfg.get("voice_output_device") or "").lower()
    spk = None
    if want:
        spk = next((s for s in sc.all_speakers() if want in s.name.lower()), None)
        if spk is None:
            log(f"Voice output '{cfg['voice_output_device']}' not found. Using default speaker.")
    spk = spk or sc.default_speaker()
    log(f"Voice plays on: {spk.name}")

    def open_player(sr):
        p = spk.player(samplerate=sr, channels=1, blocksize=int(sr * BLOCK_S))
        p.__enter__()  # kept open for the life of the app
        return p

    return open_player


def _sapi_loop(cfg, tts_q, state):
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.Rate = max(-10, min(10, int((cfg["voice_rate"] - NORMAL_RATE) / 15)))
    voice.Volume = int(cfg["voice_volume"] * 100)
    SVSF_ASYNC, SVSF_PURGE = 1, 2
    while not state["quit"]:
        try:
            text = tts_q.get(timeout=0.3)
        except queue.Empty:
            continue
        while not tts_q.empty():  # only speak the newest line
            text = tts_q.get_nowait()
        # Purge = cut off anything still being spoken; new speech wins.
        voice.Speak(text, SVSF_ASYNC | SVSF_PURGE)


def tts_loop(cfg, tts_q, state, log):
    cfg = {**TTS_DEFAULTS, **cfg}
    engine = load_engine(cfg, log)
    if engine is None:
        _sapi_loop(cfg, tts_q, state)
        return

    speaker = StreamingSpeaker(engine, open_soundcard_player(cfg, log), float(cfg["voice_volume"]))

    def play_loop():
        while not state["quit"]:
            speaker.play_next()

    threading.Thread(target=play_loop, daemon=True).start()
    while not state["quit"]:
        try:
            text = tts_q.get(timeout=0.3)
        except queue.Empty:
            continue
        while not tts_q.empty():  # only speak the newest line
            text = tts_q.get_nowait()
        try:
            speaker.say(text, pending=lambda: not tts_q.empty())
        except Exception as e:  # one bad line shouldn't kill the voice
            log(f"Voice error ({type(e).__name__}: {e})")
