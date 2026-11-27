"""
Live voice-chat translator for gaming (Windows, fully local).

Pipeline:
  Discord audio (via VB-Cable "CABLE Output", or speaker loopback as fallback)
    -> speech detector (cuts audio into phrases at pauses)
    -> Whisper: detect language; skip if in skip list; else translate to English
    -> on-screen subtitle overlay (click-through, always on top)
    -> optional English voice (Windows built-in TTS)

You keep hearing the original Discord audio with zero delay; this only adds
the translation on top.
"""

import ctypes
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _add_cuda_dlls():
    """Let CTranslate2 find the pip-installed NVIDIA cuBLAS / cuDNN DLLs."""
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn", "cuda_nvrtc"):
        d = nv / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


_add_cuda_dlls()
TARGET_SR = 16000
FRAME = 480  # 30 ms @ 16 kHz

HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.",
    "thanks for watching.", "thank you for watching!", "you", "bye.",
    "please subscribe.", "subtitles by the amara.org community",
}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


CONFIG_DEFAULTS = {
    "backend": "whisper.cpp",
    "whispercpp_quant": "q8_0",
    "whispercpp_min_ctx": 256,
    "min_language_prob": 0.5,
}


def load_config():
    with open(HERE / "config.json", encoding="utf-8") as f:
        return {**CONFIG_DEFAULTS, **json.load(f)}


# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------
def open_recorder(cfg):
    import soundcard as sc

    want = (cfg.get("input_device") or "").lower()
    if want:
        for mic in sc.all_microphones(include_loopback=False):
            if want in mic.name.lower():
                log(f"Listening to: {mic.name}")
                return mic
        log(f"Input device '{cfg['input_device']}' not found.")

    if cfg.get("fallback_to_speaker_loopback", True):
        spk = sc.default_speaker()
        mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        log(f"Falling back to ALL audio from speakers: {spk.name}")
        log("  (game sounds will be listened to as well - set up VB-Cable for best results)")
        return mic
    raise SystemExit("No input device available.")


def to_16k_mono(block, sr):
    mono = block.mean(axis=1) if block.ndim == 2 else block
    if sr == TARGET_SR:
        return mono.astype(np.float32)
    ratio = sr / TARGET_SR
    if ratio == int(ratio):  # e.g. 48k -> 16k: average groups (cheap anti-alias)
        r = int(ratio)
        n = len(mono) // r * r
        return mono[:n].reshape(-1, r).mean(axis=1).astype(np.float32)
    n_out = int(len(mono) / ratio)
    x = np.linspace(0, len(mono) - 1, n_out)
    return np.interp(x, np.arange(len(mono)), mono).astype(np.float32)


class Segmenter:
    """Energy-based speech detector. Emits one numpy array per spoken phrase."""

    def __init__(self, cfg, out_q):
        self.out_q = out_q
        self.silence_frames = int(cfg["vad_silence_ms"] / 30)
        self.max_frames = int(cfg["vad_max_segment_s"] * 1000 / 30)
        self.min_speech_frames = int(cfg["vad_min_speech_ms"] / 30)
        self.abs_threshold = cfg["vad_threshold_db"]
        self.noise_db = -60.0
        self.buf = np.zeros(0, dtype=np.float32)
        self.preroll = []
        self.active = False
        self.frames = []
        self.speech_count = 0
        self.silence_count = 0
        self.start_streak = 0

    def feed(self, samples):
        self.buf = np.concatenate([self.buf, samples])
        while len(self.buf) >= FRAME:
            f, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            self._frame(f)

    def _frame(self, f):
        db = 20 * np.log10(np.sqrt(np.mean(f * f)) + 1e-9)
        is_speech = db > max(self.abs_threshold, self.noise_db + 10)
        if not is_speech:  # slowly track background level
            self.noise_db = 0.98 * self.noise_db + 0.02 * db

        if not self.active:
            self.preroll.append(f)
            self.preroll = self.preroll[-10:]  # ~300 ms before speech starts
            self.start_streak = self.start_streak + 1 if is_speech else 0
            if self.start_streak >= 3:
                self.active = True
                self.frames = list(self.preroll)
                self.speech_count = 3
                self.silence_count = 0
            return

        self.frames.append(f)
        if is_speech:
            self.speech_count += 1
            self.silence_count = 0
        else:
            self.silence_count += 1

        if self.silence_count >= self.silence_frames or len(self.frames) >= self.max_frames:
            if self.speech_count >= self.min_speech_frames:
                self.out_q.put((np.concatenate(self.frames), time.time()))
            self.active = False
            self.frames = []
            self.preroll = []
            self.start_streak = 0


def capture_loop(cfg, seg_q, state):
    mic = open_recorder(cfg)
    sr = 48000
    seg = Segmenter(cfg, seg_q)
    import warnings
    warnings.filterwarnings("ignore", message="data discontinuity")
    with mic.recorder(samplerate=sr, channels=1, blocksize=4800) as rec:
        while not state["quit"]:
            data = rec.record(numframes=4800)  # 100 ms
            if state["enabled"]:
                seg.feed(to_16k_mono(data, sr))


# --------------------------------------------------------------------------
# Whisper: detect language + translate
# --------------------------------------------------------------------------
def load_model(cfg):
    """Use the already-downloaded model offline; only go online if it's missing."""
    if cfg["backend"] == "whisper.cpp":
        from whispercpp_backend import WhisperCppModel, ensure_model

        path = ensure_model(HERE / "models" / "ggml", cfg["model"], cfg["whispercpp_quant"], log)
        return WhisperCppModel(path, cfg["cpu_threads"], min_ctx=cfg["whispercpp_min_ctx"],
                               hallucinations=HALLUCINATIONS)

    from faster_whisper import WhisperModel

    kw = dict(device=cfg["device"], compute_type=cfg["compute_type"],
              cpu_threads=cfg["cpu_threads"], download_root=str(HERE / "models"))
    local_dir = HERE / "models" / cfg["model"]
    if (local_dir / "model.bin").is_file():
        return WhisperModel(str(local_dir), **kw)
    try:
        return WhisperModel(cfg["model"], local_files_only=True, **kw)
    except Exception:
        log("Model not found locally - downloading (one time)...")
        return WhisperModel(cfg["model"], **kw)


def run_whisper(model, audio, skip, skip_thr, min_lang_prob=0.0):
    """One encoder pass, shared by language detection and translation.

    Returns (lang, lang_prob, text). text is None when skipped / not speech.
    (faster-whisper's transcribe() would run the expensive encoder twice.)
    """
    if hasattr(model, "run"):  # whisper.cpp backend
        return model.run(audio, skip, skip_thr, min_lang_prob)

    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import get_suppressed_tokens

    features = pad_or_trim(model.feature_extractor(audio)[:, :3000])
    enc = model.encode(features)

    probs = {tok[2:-2]: p for tok, p in model.model.detect_language(enc)[0]}
    lang = max(probs, key=probs.get)
    skip_p = sum(probs.get(code, 0.0) for code in skip)
    # Unsure about the language usually means it isn't speech (game sounds, music).
    if lang in skip or skip_p >= skip_thr or probs[lang] < min_lang_prob:
        return lang, probs[lang], None

    tok = Tokenizer(model.hf_tokenizer, True, task="translate", language=lang)
    res = model.model.generate(
        enc, [model.get_prompt(tok, [], without_timestamps=True)],
        beam_size=1, max_length=200, return_scores=True, return_no_speech_prob=True,
        suppress_blank=True, suppress_tokens=get_suppressed_tokens(tok, [-1]),
    )[0]
    ids = [t for t in res.sequences_ids[0] if t < tok.eot]
    avg_logprob = res.scores[0] * len(ids) / (len(ids) + 1)
    if res.no_speech_prob > 0.6 and avg_logprob < -0.8:
        return lang, probs[lang], None
    text = tok.decode(ids).strip()
    if not text or text.lower() in HALLUCINATIONS:
        return lang, probs[lang], None
    return lang, probs[lang], text


def translate_loop(cfg, seg_q, ui_q, tts_q, state):
    log(f"Loading Whisper '{cfg['model']}' ({cfg['backend']}, {cfg['cpu_threads']} CPU threads)...")
    model = load_model(cfg)
    log("Model ready. Listening for speech...")
    ui_q.put(("status", "Translator ready"))
    skip = {s.lower() for s in cfg["skip_languages"]}
    skip_thr = cfg["skip_threshold"]
    min_lang_prob = cfg["min_language_prob"]

    while not state["quit"]:
        try:
            audio, t_end = seg_q.get(timeout=0.5)
        except queue.Empty:
            continue
        # Keep latency low: if phrases piled up, drop the oldest ones.
        while seg_q.qsize() > 2:
            audio, t_end = seg_q.get_nowait()
        if not state["enabled"]:
            continue

        t0 = time.time()
        lang, prob, text = run_whisper(model, audio, skip, skip_thr, min_lang_prob)
        if text is None:
            log(f"skip  ({lang} {prob:.0%}, {time.time() - t0:.2f}s)")
            continue

        lag = time.time() - t_end
        log(f"[{lang.upper()} {prob:.0%}] {text}   "
            f"(took {time.time() - t0:.2f}s, {lag:.2f}s after speech ended)")
        state["last"] = (lang, text)
        ui_q.put(("sub", lang, text))
        if state["voice"]:
            tts_q.put(text)


# --------------------------------------------------------------------------
# Text-to-speech (Windows built-in voices, offline)
# --------------------------------------------------------------------------
def tts_loop(cfg, tts_q, state):
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.Rate = max(-10, min(10, int((cfg["voice_rate"] - 175) / 15)))
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


# --------------------------------------------------------------------------
# Subtitle overlay (click-through, always on top)
# --------------------------------------------------------------------------
class Overlay:
    def __init__(self, cfg, ui_q, state):
        self.cfg, self.ui_q, self.state = cfg, ui_q, state
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", cfg["overlay_opacity"])
        self.root.configure(bg="#111111")
        self.label = tk.Label(
            self.root, text="", fg="#FFFFFF", bg="#111111",
            font=("Segoe UI Semibold", cfg["overlay_font_size"]),
            wraplength=int(self.root.winfo_screenwidth() * 0.6), justify="center",
            padx=18, pady=8,
        )
        self.label.pack()
        self.hide_at = 0
        self.root.withdraw()
        self.root.update_idletasks()
        self._click_through()
        self.root.after(50, self.poll)

    def _click_through(self):
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        GWL_EXSTYLE = -20
        WS_EX_LAYERED, WS_EX_TRANSPARENT = 0x80000, 0x20
        WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80, 0x08000000
        style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE,
            style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        )

    def show(self, text, seconds=None):
        if not self.cfg["overlay_enabled"]:
            return
        self.label.config(text=text)
        self.root.update_idletasks()
        w, h = self.label.winfo_reqwidth(), self.label.winfo_reqheight()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = sh - h - self.cfg["overlay_bottom_offset"]
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.deiconify()
        self.root.attributes("-topmost", True)
        self.hide_at = time.time() + (seconds or self.cfg["overlay_seconds"])

    def poll(self):
        if self.state["quit"]:
            self.root.destroy()
            return
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if msg[0] == "sub":
                    self.show(f"[{msg[1].upper()}]  {msg[2]}")
                elif msg[0] == "status":
                    self.show(msg[1], seconds=2)
        except queue.Empty:
            pass
        if self.hide_at and time.time() > self.hide_at:
            self.root.withdraw()
            self.hide_at = 0
        self.root.after(50, self.poll)


# --------------------------------------------------------------------------
# Hotkeys
# --------------------------------------------------------------------------
def setup_hotkeys(cfg, state, ui_q, tts_q):
    import keyboard

    def toggle():
        state["enabled"] = not state["enabled"]
        msg = "Translator ON" if state["enabled"] else "Translator OFF"
        log(msg)
        ui_q.put(("status", msg))

    def voice():
        state["voice"] = not state["voice"]
        msg = "English voice ON" if state["voice"] else "English voice OFF"
        log(msg)
        ui_q.put(("status", msg))

    def replay():
        if state["last"]:
            lang, text = state["last"]
            ui_q.put(("sub", lang, text))
            tts_q.put(text)

    def quit_():
        log("Quitting...")
        state["quit"] = True

    keyboard.add_hotkey(cfg["hotkey_toggle"], toggle)
    keyboard.add_hotkey(cfg["hotkey_voice"], voice)
    keyboard.add_hotkey(cfg["hotkey_replay"], replay)
    keyboard.add_hotkey(cfg["hotkey_quit"], quit_)


def main():
    cfg = load_config()
    state = {"enabled": True, "voice": cfg["voice_enabled"], "quit": False, "last": None}
    seg_q, ui_q, tts_q = queue.Queue(), queue.Queue(), queue.Queue()

    log(f"Hotkeys: toggle={cfg['hotkey_toggle']}  voice={cfg['hotkey_voice']}  "
        f"replay={cfg['hotkey_replay']}  quit={cfg['hotkey_quit']}")
    log(f"Skipping languages: {', '.join(cfg['skip_languages'])}")

    for target, args in (
        (capture_loop, (cfg, seg_q, state)),
        (translate_loop, (cfg, seg_q, ui_q, tts_q, state)),
        (tts_loop, (cfg, tts_q, state)),
    ):
        threading.Thread(target=target, args=args, daemon=True).start()
    setup_hotkeys(cfg, state, ui_q, tts_q)

    overlay = Overlay(cfg, ui_q, state)
    try:
        overlay.root.mainloop()
    except KeyboardInterrupt:
        pass
    state["quit"] = True


if __name__ == "__main__":
    if "--devices" in sys.argv:
        import soundcard as sc
        print("Recording devices:")
        for m in sc.all_microphones(include_loopback=False):
            print("  ", m.name)
        print("Speakers:")
        for s in sc.all_speakers():
            print("  ", s.name)
        sys.exit(0)
    main()
