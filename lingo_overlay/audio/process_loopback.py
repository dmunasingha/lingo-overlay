"""Capture the audio of one process (e.g. Discord.exe) with Windows process loopback.

Windows 10 2004 (build 19041) and later can record what a single process tree plays, through
``ActivateAudioInterfaceAsync`` with ``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK``. No virtual
cable or driver is needed, and the user keeps hearing the app normally.

This is plain ctypes (no comtypes): a tiny hand-made COM object for the completion handler, and
vtable calls into IAudioClient / IAudioCaptureClient. Only Windows audio APIs are used, so it stays
anti-cheat safe (no injection, no hooks).

``open_process_recorder(name, samplerate)`` returns a recorder with the same
``with ...: rec.record(numframes)`` shape as a soundcard recorder, so capture_loop can use either.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import uuid
from ctypes import POINTER, byref, c_int, c_long, c_longlong, c_uint, c_ulong, c_ulonglong, c_void_p, c_wchar
from ctypes import wintypes as wt

import numpy as np

MIN_BUILD = 19041  # Windows 10 2004
_FUNCTYPE = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)  # importable (for tests) off Windows

_VIRTUAL_DEVICE = "VAD\\Process_Loopback"
_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
_LOOPBACK_MODE_INCLUDE_TREE = 0
_VT_BLOB = 65

_AUDCLNT_SHAREMODE_SHARED = 0
_STREAMFLAGS = (0x00020000  # LOOPBACK
                | 0x00040000  # EVENTCALLBACK
                | 0x80000000  # AUTOCONVERTPCM
                | 0x08000000)  # SRC_DEFAULT_QUALITY
_BUFFERFLAGS_SILENT = 0x2
_WAVE_FORMAT_PCM = 1
_BITS = 16
_CHANNELS = 2

_S_OK = 0
_E_NOINTERFACE = -2147467262  # 0x80004002
_RPC_E_CHANGED_MODE = -2147417850  # 0x80010106
_WAIT_OBJECT_0 = 0


class ProcessCaptureError(RuntimeError):
    """Process loopback is unavailable (old Windows, process not running, activation failed)."""


# ---------------------------------------------------------------- process lookup (pure, testable)

def _norm_exe(name: str) -> str:
    n = name.strip().lower()
    return n if n.endswith(".exe") else n + ".exe"


def pick_root_pid(procs, name: str) -> int | None:
    """From ``(pid, parent_pid, exe_name)`` tuples, the root process of the named app.

    Apps like Discord run many processes with the same name and play audio from a child, so we
    target the topmost one and capture its whole process tree.
    """
    want = _norm_exe(name)
    matches = [(pid, ppid) for pid, ppid, exe in procs if exe.lower() == want]
    if not matches:
        return None
    pids = {pid for pid, _ in matches}
    roots = [pid for pid, ppid in matches if ppid not in pids]
    return min(roots or pids)


# ---------------------------------------------------------------- ctypes structures

class GUID(ctypes.Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def of(cls, text: str) -> GUID:
        b = uuid.UUID(text).bytes_le
        g = cls()
        ctypes.memmove(byref(g), b, 16)
        return g

    def __eq__(self, other):
        return bytes(self) == bytes(other)

    __hash__ = None


IID_IUnknown = GUID.of("00000000-0000-0000-C000-000000000046")
IID_IAgileObject = GUID.of("94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90")
IID_IActivateCompletionHandler = GUID.of("41D949AB-9862-444A-80F6-C261334DA5EB")
IID_IAudioClient = GUID.of("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2")
IID_IAudioCaptureClient = GUID.of("C8ADBD64-E71E-48a0-A4DE-185C395CD317")


class LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [("TargetProcessId", wt.DWORD), ("ProcessLoopbackMode", c_int)]


class ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [("ActivationType", c_int), ("ProcessLoopbackParams", LOOPBACK_PARAMS)]


class BLOB(ctypes.Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", wt.USHORT), ("r1", wt.WORD), ("r2", wt.WORD), ("r3", wt.WORD), ("blob", BLOB)]


class WAVEFORMATEX(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("wFormatTag", wt.WORD), ("nChannels", wt.WORD), ("nSamplesPerSec", wt.DWORD),
                ("nAvgBytesPerSec", wt.DWORD), ("nBlockAlign", wt.WORD), ("wBitsPerSample", wt.WORD),
                ("cbSize", wt.WORD)]


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD),
                ("cntThreads", wt.DWORD), ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", c_long),
                ("dwFlags", wt.DWORD), ("szExeFile", c_wchar * 260)]


def pcm_format(samplerate: int) -> WAVEFORMATEX:
    block = _CHANNELS * _BITS // 8
    return WAVEFORMATEX(_WAVE_FORMAT_PCM, _CHANNELS, samplerate, samplerate * block, block, _BITS, 0)


def pcm16_to_float(raw: bytes, channels: int = _CHANNELS) -> np.ndarray:
    """Interleaved int16 bytes -> float32 array shaped (frames, channels) in [-1, 1)."""
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return a.reshape(-1, channels)


# ---------------------------------------------------------------- Win32 helpers

def list_processes():
    """``(pid, parent_pid, exe_name)`` for every running process."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wt.HANDLE(-1).value:
        raise ProcessCaptureError("could not list processes")
    out = []
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = k32.Process32FirstW(snap, byref(e))
        while ok:
            out.append((e.th32ProcessID, e.th32ParentProcessID, e.szExeFile))
            ok = k32.Process32NextW(snap, byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def _vcall(ptr, index, argtypes, *args):
    """Call method ``index`` of the COM interface at ``ptr``; returns the HRESULT."""
    vtbl = ctypes.cast(ptr, POINTER(POINTER(c_void_p)))[0]
    fn = _FUNCTYPE(c_long, c_void_p, *argtypes)(vtbl[index])
    return fn(ptr, *args)


def _check(hr, what):
    if hr < 0:
        raise ProcessCaptureError(f"{what} failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")


def _release(ptr):
    if ptr:
        _vcall(ptr, 2, [])


# ---------------------------------------------------------------- completion handler COM object

_QI = _FUNCTYPE(c_long, c_void_p, POINTER(GUID), POINTER(c_void_p))
_ADDREF = _FUNCTYPE(c_ulong, c_void_p)
_COMPLETED = _FUNCTYPE(c_long, c_void_p, c_void_p)


class _HandlerVtbl(ctypes.Structure):
    _fields_ = [("QueryInterface", _QI), ("AddRef", _ADDREF), ("Release", _ADDREF),
                ("ActivateCompleted", _COMPLETED)]


class _HandlerObj(ctypes.Structure):
    _fields_ = [("lpVtbl", POINTER(_HandlerVtbl))]


class _CompletionHandler:
    """IActivateAudioInterfaceCompletionHandler that just signals a threading.Event.

    It answers QueryInterface for IAgileObject, which Windows requires of this handler. Python
    keeps the object alive until activation is done, so the ref count is only nominal.
    """

    def __init__(self):
        self.done = threading.Event()
        self._refs = 1

        def qi(this, riid, ppv):
            iid = riid.contents
            if iid in (IID_IUnknown, IID_IAgileObject, IID_IActivateCompletionHandler):
                ppv[0] = this
                self._refs += 1
                return _S_OK
            ppv[0] = None
            return _E_NOINTERFACE

        def addref(this):
            self._refs += 1
            return self._refs

        def release(this):
            self._refs -= 1
            return max(self._refs, 0)

        def completed(this, op):
            self.done.set()
            return _S_OK

        self._vtbl = _HandlerVtbl(_QI(qi), _ADDREF(addref), _ADDREF(release), _COMPLETED(completed))
        self._obj = _HandlerObj(ctypes.pointer(self._vtbl))

    @property
    def ptr(self):
        return ctypes.cast(ctypes.pointer(self._obj), c_void_p)


# ---------------------------------------------------------------- recorder

class ProcessLoopbackRecorder:
    """Records one process tree. Use as a context manager; ``record(n)`` returns (n, 2) float32."""

    def __init__(self, pid: int, samplerate: int = 48000, activate_timeout: float = 5.0):
        self.pid = pid
        self.samplerate = samplerate
        self.activate_timeout = activate_timeout
        self._client = None
        self._capture = None
        self._event = None
        self._com = False
        self._pending = np.zeros((0, _CHANNELS), dtype=np.float32)
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateEventW.restype = wt.HANDLE
        self._k32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]

    # -- lifetime
    def open(self):
        try:
            self._open()
        except BaseException:
            self.close()
            raise
        return self

    def _open(self):
        ole32 = ctypes.WinDLL("ole32")
        ole32.CoInitializeEx.restype = c_long
        hr = ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        self._com = hr >= 0  # S_OK / S_FALSE need a matching CoUninitialize
        if hr < 0 and hr != _RPC_E_CHANGED_MODE:
            _check(hr, "CoInitializeEx")

        params = ACTIVATION_PARAMS(_ACTIVATION_TYPE_PROCESS_LOOPBACK,
                                   LOOPBACK_PARAMS(self.pid, _LOOPBACK_MODE_INCLUDE_TREE))
        pv = PROPVARIANT()
        pv.vt = _VT_BLOB
        pv.blob.cbSize = ctypes.sizeof(params)
        pv.blob.pBlobData = ctypes.cast(byref(params), c_void_p)

        mmdev = ctypes.WinDLL("Mmdevapi")
        activate = mmdev.ActivateAudioInterfaceAsync
        activate.restype = c_long
        activate.argtypes = [wt.LPCWSTR, POINTER(GUID), POINTER(PROPVARIANT), c_void_p, POINTER(c_void_p)]
        handler = _CompletionHandler()
        op = c_void_p()
        _check(activate(_VIRTUAL_DEVICE, byref(IID_IAudioClient), byref(pv), handler.ptr, byref(op)),
               "ActivateAudioInterfaceAsync")
        try:
            if not handler.done.wait(self.activate_timeout):
                raise ProcessCaptureError("audio activation timed out")
            act_hr = c_long()
            client = c_void_p()
            _check(_vcall(op, 3, [POINTER(c_long), POINTER(c_void_p)], byref(act_hr), byref(client)),
                   "GetActivateResult")
            _check(act_hr.value, "process loopback activation")
            self._client = client
        finally:
            _release(op)

        fmt = pcm_format(self.samplerate)
        _check(_vcall(self._client, 3, [c_int, wt.DWORD, c_longlong, c_longlong, POINTER(WAVEFORMATEX), c_void_p],
                      _AUDCLNT_SHAREMODE_SHARED, _STREAMFLAGS, 2_000_000, 0, byref(fmt), None),
               "IAudioClient.Initialize")
        self._event = self._k32.CreateEventW(None, False, False, None)
        if not self._event:
            raise ProcessCaptureError("CreateEvent failed")
        _check(_vcall(self._client, 13, [wt.HANDLE], self._event), "IAudioClient.SetEventHandle")
        cap = c_void_p()
        _check(_vcall(self._client, 14, [POINTER(GUID), POINTER(c_void_p)], byref(IID_IAudioCaptureClient),
                      byref(cap)), "IAudioClient.GetService")
        self._capture = cap
        _check(_vcall(self._client, 10, []), "IAudioClient.Start")

    def close(self):
        if self._client:
            _vcall(self._client, 11, [])  # Stop
        _release(self._capture)
        _release(self._client)
        self._capture = self._client = None
        if self._event:
            self._k32.CloseHandle(self._event)
            self._event = None
        if self._com:
            ctypes.WinDLL("ole32").CoUninitialize()
            self._com = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- reading
    def _drain(self):
        chunks = []
        block = _CHANNELS * _BITS // 8
        size = c_uint()
        while True:
            _check(_vcall(self._capture, 5, [POINTER(c_uint)], byref(size)), "GetNextPacketSize")
            if not size.value:
                break
            data = c_void_p()
            frames = c_uint()
            flags = wt.DWORD()
            _check(_vcall(self._capture, 3, [POINTER(c_void_p), POINTER(c_uint), POINTER(wt.DWORD),
                                             POINTER(c_ulonglong), POINTER(c_ulonglong)],
                          byref(data), byref(frames), byref(flags), None, None), "GetBuffer")
            n = frames.value
            if flags.value & _BUFFERFLAGS_SILENT or not data.value:
                chunks.append(np.zeros((n, _CHANNELS), dtype=np.float32))
            else:
                chunks.append(pcm16_to_float(ctypes.string_at(data.value, n * block)))
            _vcall(self._capture, 4, [c_uint], n)  # ReleaseBuffer
        if chunks:
            self._pending = np.concatenate([self._pending, *chunks])

    def record(self, numframes: int) -> np.ndarray:
        """Block for about ``numframes`` of audio. Gaps where the app plays nothing come back as
        silence, so the caller keeps a steady real-time pace and can check for quit."""
        timeout_ms = int(numframes * 1000 / self.samplerate) + 50
        while len(self._pending) < numframes:
            res = self._k32.WaitForSingleObject(self._event, timeout_ms)
            self._drain()
            if res != _WAIT_OBJECT_0 and len(self._pending) < numframes:
                pad = np.zeros((numframes - len(self._pending), _CHANNELS), dtype=np.float32)
                self._pending = np.concatenate([self._pending, pad])
        out, self._pending = self._pending[:numframes], self._pending[numframes:]
        return out


def open_process_recorder(name: str, samplerate: int = 48000) -> ProcessLoopbackRecorder:
    """Start capturing ``name`` (e.g. "Discord.exe"). Call from the thread that will record.

    Raises ProcessCaptureError if this Windows is too old, the app isn't running, or activation fails.
    """
    if os.name != "nt":
        raise ProcessCaptureError("process capture needs Windows")
    build = sys.getwindowsversion().build
    if build < MIN_BUILD:
        raise ProcessCaptureError(f"process capture needs Windows 10 2004 (build {MIN_BUILD}) or later; "
                                  f"this is build {build}")
    pid = pick_root_pid(list_processes(), name)
    if pid is None:
        raise ProcessCaptureError(f"'{name}' is not running")
    return ProcessLoopbackRecorder(pid, samplerate).open()
