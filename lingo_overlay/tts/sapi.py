"""Text-to-speech with Windows built-in SAPI voices (offline)."""

import queue

SRSE_IS_SPEAKING = 2  # SpeechRunState flag in voice.Status.RunningState


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
            # Short timeout so state["tts_speaking"] (used to duck voice chat) stays current.
            text = tts_q.get(timeout=0.05)
        except queue.Empty:
            state["tts_speaking"] = bool(voice.Status.RunningState & SRSE_IS_SPEAKING)
            continue
        while not tts_q.empty():  # only speak the newest line
            text = tts_q.get_nowait()
        # Duck before speech starts so the first syllable is not masked.
        state["tts_speaking"] = True
        # Purge = cut off anything still being spoken; new speech wins.
        voice.Speak(text, SVSF_ASYNC | SVSF_PURGE)
