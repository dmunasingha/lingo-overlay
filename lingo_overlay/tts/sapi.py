"""Text-to-speech with Windows built-in SAPI voices (offline)."""

import queue


def tts_loop(cfg, tts_q, state):
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    voice.Rate = max(-10, min(10, int((cfg.voice_rate - 175) / 15)))
    voice.Volume = int(cfg.voice_volume * 100)
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
