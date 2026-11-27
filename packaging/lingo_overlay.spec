# PyInstaller spec for the Windows build. Run via packaging/build.ps1 (from the repo root):
#   pyinstaller packaging/lingo_overlay.spec --noconfirm
# Produces dist/LingoOverlay/LingoOverlay.exe (one-folder build: starts faster than one-file
# and doesn't unpack hundreds of MB to %TEMP% on every launch).
#
# config.json, glossary/ and audio_setup.ps1 are NOT bundled here: the installer copies them next
# to the exe, where the app looks for them (lingo_overlay.ROOT) and where users can edit them.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# Native libraries and data files these packages load at runtime
# (whisper.cpp DLLs, CTranslate2 DLLs, faster-whisper's Silero VAD .onnx, sentencepiece).
for pkg in ("pywhispercpp", "ctranslate2", "faster_whisper", "sentencepiece", "soundcard"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
hiddenimports += ["win32com.client", "pythoncom", "pywintypes", "_cffi_backend"]

a = Analysis(
    ["../translator.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # Optional extras users install separately (requirements-lid.txt / -sensevoice.txt).
    excludes=["torch", "torchaudio", "speechbrain", "funasr", "matplotlib", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LingoOverlay",
    console=True,  # keep the log window, like run.bat
    upx=False,  # UPX-packed exes trip antivirus and some anti-cheat scanners
)
coll = COLLECT(exe, a.binaries, a.datas, name="LingoOverlay", upx=False)
