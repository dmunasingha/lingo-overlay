"""Audio capture from VB-Cable (or speaker loopback as a fallback)."""

from lingo_overlay.audio.resample import to_16k_mono
from lingo_overlay.log import log
from lingo_overlay.vad import Segmenter


def open_recorder(cfg):
    import soundcard as sc

    want = (cfg.input_device or "").lower()
    if want:
        for mic in sc.all_microphones(include_loopback=False):
            if want in mic.name.lower():
                log(f"Listening to: {mic.name}")
                return mic
        log(f"Input device '{cfg.input_device}' not found.")

    if cfg.fallback_to_speaker_loopback:
        spk = sc.default_speaker()
        mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        log(f"Falling back to ALL audio from speakers: {spk.name}")
        log("  (game sounds will be listened to as well - set up VB-Cable for best results)")
        return mic
    raise SystemExit("No input device available.")


def capture_loop(cfg, seg_q, state):
    mic = open_recorder(cfg)
    sr = 48000
    seg = Segmenter(cfg, seg_q)
    log(f"Speech detector: {seg.detector.name}")
    import warnings
    warnings.filterwarnings("ignore", message="data discontinuity")
    with mic.recorder(samplerate=sr, channels=1, blocksize=4800) as rec:
        while not state["quit"]:
            data = rec.record(numframes=4800)  # 100 ms
            if state["enabled"]:
                seg.feed(to_16k_mono(data, sr))


def print_devices():
    import soundcard as sc
    print("Recording devices:")
    for m in sc.all_microphones(include_loopback=False):
        print("  ", m.name)
    print("Speakers:")
    for s in sc.all_speakers():
        print("  ", s.name)
