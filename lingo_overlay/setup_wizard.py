"""First-run setup: download the speech model, guide the VB-Cable install, fix audio routing.

Opens automatically on the first launch of the installed app (and with ``--setup`` from a
source checkout). Three steps in one small window:

  1. Speech model   download with a progress bar (resumes after stalls, see download.py)
  2. VB-Cable       checks for "CABLE Output"; links to the installer and re-checks
  3. Audio routing  runs the bundled audio_setup.ps1 (no admin needed)

Only step 1 is required. Finishing writes ``.setup_done`` next to config.json so the
wizard does not come back; ``--setup`` always opens it again.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from lingo_overlay import FROZEN, ROOT

VB_CABLE_URL = "https://vb-audio.com/Cable/"
MARKER = ".setup_done"


def marker_path(root: Path = ROOT) -> Path:
    return root / MARKER


def needs_setup(argv, root: Path = ROOT, frozen: bool = FROZEN) -> bool:
    """``--setup`` always; otherwise only the installed app's first launch."""
    if "--setup" in argv:
        return True
    return frozen and not marker_path(root).is_file()


def mark_done(root: Path = ROOT) -> None:
    marker_path(root).write_text("1\n", encoding="utf-8")


def has_vb_cable(device_names) -> bool:
    return any("cable output" in n.lower() for n in device_names)


def recording_devices() -> list[str]:
    try:
        import soundcard as sc

        return [m.name for m in sc.all_microphones(include_loopback=False)]
    except Exception:  # no audio stack (CI) or COM error: treat as "not found"
        return []


def audio_setup_command(script: Path, speaker: str, mic: str) -> list[str]:
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
            "-SpeakerMatch", speaker, "-MicMatch", mic]


def run(cfg) -> str:
    """Show the wizard. Returns "start" (finished), "quit" (closed, model ready) or
    "incomplete" (closed before the model was ready)."""
    import tkinter as tk
    from tkinter import ttk

    from lingo_overlay.asr.whispercpp_backend import ensure_model, model_path

    models_dir = ROOT / "models" / "ggml"
    whispercpp = cfg.backend == "whisper.cpp"
    events: queue.Queue = queue.Queue()
    cancel = threading.Event()
    result = {"start": False, "ready": not whispercpp or model_path(models_dir, cfg.model, cfg.whispercpp_quant).is_file()}

    root = tk.Tk()
    root.title("Lingo Overlay setup")
    root.resizable(False, False)
    pad = {"padx": 14, "pady": 4}
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    def heading(text):
        ttk.Label(frm, text=text, font=("Segoe UI Semibold", 11)).pack(anchor="w", **pad)

    # ---- 1. model
    heading("1. Speech model")
    model_lbl = ttk.Label(frm, width=60)
    model_lbl.pack(anchor="w", **pad)
    bar = ttk.Progressbar(frm, length=420, maximum=1000)
    bar.pack(anchor="w", **pad)
    dl_btn = ttk.Button(frm, text="Download")
    dl_btn.pack(anchor="w", **pad)

    def start_download():
        dl_btn.state(["disabled"])
        model_lbl.config(text="Downloading... (you can carry on with steps 2 and 3)")

        def work():
            try:
                ensure_model(models_dir, cfg.model, cfg.whispercpp_quant,
                             log=lambda *a: events.put(("log", " ".join(map(str, a)))),
                             progress=lambda d, t: events.put(("progress", d, t)),
                             cancel=cancel.is_set)
                events.put(("model_done",))
            except Exception as ex:
                events.put(("model_failed", str(ex)))

        threading.Thread(target=work, daemon=True).start()

    dl_btn.config(command=start_download)
    if result["ready"]:
        model_lbl.config(text=f"Whisper '{cfg.model}' is installed." if whispercpp
                         else "faster-whisper downloads its model on first start.")
        bar.config(value=1000)
        dl_btn.state(["disabled"])

    # ---- 2. VB-Cable
    ttk.Separator(frm).pack(fill="x", pady=8)
    heading("2. VB-Cable (recommended: listen to voice chat only)")
    cable_lbl = ttk.Label(frm, width=60, wraplength=430)
    cable_lbl.pack(anchor="w", **pad)
    row = ttk.Frame(frm)
    row.pack(anchor="w", **pad)
    ttk.Button(row, text="Get VB-Cable", command=lambda: webbrowser.open(VB_CABLE_URL)).pack(side="left")
    ttk.Button(row, text="Check again", command=lambda: check_cable()).pack(side="left", padx=6)

    def check_cable():
        if has_vb_cable(recording_devices()):
            cable_lbl.config(text="VB-Cable found. Set your game/Discord voice output to "
                                  "'CABLE Input (VB-Audio Virtual Cable)'.")
            return True
        cable_lbl.config(text="Not found. Download it, unzip, right-click VBCABLE_Setup_x64.exe > "
                              "Run as administrator > Install, then reboot and press Check again. "
                              "Without it the app listens to all speaker audio.")
        return False

    # ---- 3. routing
    ttk.Separator(frm).pack(fill="x", pady=8)
    heading("3. Audio routing (after installing VB-Cable)")
    ttk.Label(frm, text="Part of the name of your headset/speakers and microphone:").pack(anchor="w", **pad)
    row3 = ttk.Frame(frm)
    row3.pack(anchor="w", **pad)
    spk, mic = tk.StringVar(value="Realtek"), tk.StringVar(value="Realtek")
    ttk.Label(row3, text="Speakers").pack(side="left")
    ttk.Entry(row3, textvariable=spk, width=14).pack(side="left", padx=4)
    ttk.Label(row3, text="Mic").pack(side="left")
    ttk.Entry(row3, textvariable=mic, width=14).pack(side="left", padx=4)
    route_btn = ttk.Button(row3, text="Fix audio routing")
    route_btn.pack(side="left", padx=6)
    route_lbl = ttk.Label(frm, width=60, wraplength=430)
    route_lbl.pack(anchor="w", **pad)

    def fix_routing():
        script = ROOT / "audio_setup.ps1"
        if not script.is_file():
            route_lbl.config(text=f"audio_setup.ps1 not found in {ROOT}")
            return
        route_btn.state(["disabled"])
        route_lbl.config(text="Running...")

        def work():
            try:
                p = subprocess.run(audio_setup_command(script, spk.get(), mic.get()),
                                   capture_output=True, text=True, timeout=60,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                out = (p.stdout + p.stderr).strip().splitlines()
                events.put(("routing", p.returncode == 0, out[-1] if out else ""))
            except Exception as ex:
                events.put(("routing", False, str(ex)))

        threading.Thread(target=work, daemon=True).start()

    route_btn.config(command=fix_routing)

    # ---- finish
    ttk.Separator(frm).pack(fill="x", pady=8)
    finish = ttk.Button(frm, text="Start Lingo Overlay")
    finish.pack(anchor="e", **pad)

    def update_finish():
        finish.state(["!disabled"] if result["ready"] else ["disabled"])

    def on_finish():
        mark_done()
        result["start"] = True
        root.destroy()

    def on_close():
        cancel.set()
        root.destroy()

    finish.config(command=on_finish)
    root.protocol("WM_DELETE_WINDOW", on_close)

    def poll():
        try:
            while True:
                ev = events.get_nowait()
                if ev[0] == "progress":
                    _, done, total = ev
                    if total:
                        bar.config(value=done * 1000 // total)
                        model_lbl.config(text=f"Downloading... {done >> 20} of {total >> 20} MB")
                elif ev[0] == "log":
                    if "interrupted" in ev[1] or "failed" in ev[1]:
                        model_lbl.config(text="Connection trouble, retrying...")
                elif ev[0] == "model_done":
                    result["ready"] = True
                    bar.config(value=1000)
                    model_lbl.config(text=f"Whisper '{cfg.model}' is installed.")
                elif ev[0] == "model_failed":
                    model_lbl.config(text="Download failed. Check your connection and press Download "
                                          "to resume.\n" + ev[1].splitlines()[-1])
                    dl_btn.state(["!disabled"])
                elif ev[0] == "routing":
                    _, ok, last = ev
                    route_btn.state(["!disabled"])
                    route_lbl.config(text=("Done. " if ok else "Failed: ") + last)
        except queue.Empty:
            pass
        update_finish()
        root.after(100, poll)

    check_cable()
    update_finish()
    if not result["ready"]:
        start_download()  # don't make people press a button for the one required step
    root.after(100, poll)
    root.mainloop()
    if result["start"]:
        return "start"
    return "quit" if result["ready"] else "incomplete"


def main_from_cli(cfg) -> None:
    """Run the wizard; exit unless it was finished with the Start button."""
    outcome = run(cfg)
    if outcome == "incomplete":
        sys.exit("Setup was closed before the speech model finished downloading. "
                 "Start Lingo Overlay again to resume.")
    if outcome == "quit":
        sys.exit(0)
