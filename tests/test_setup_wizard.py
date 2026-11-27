"""First-run wizard logic (the Tk window itself is not tested) and model URL selection."""

from lingo_overlay import setup_wizard as w
from lingo_overlay.asr import whispercpp_backend as wc


def test_needs_setup_only_on_installed_first_run(tmp_path):
    assert w.needs_setup([], tmp_path, frozen=True)
    assert not w.needs_setup([], tmp_path, frozen=False)  # source checkout: unchanged behaviour
    w.mark_done(tmp_path)
    assert not w.needs_setup([], tmp_path, frozen=True)
    assert w.needs_setup(["--setup"], tmp_path, frozen=True)
    assert w.needs_setup(["--setup"], tmp_path, frozen=False)


def test_has_vb_cable():
    assert w.has_vb_cable(["Microphone (Realtek)", "CABLE Output (VB-Audio Virtual Cable)"])
    assert not w.has_vb_cable(["Microphone (Realtek)"])
    assert not w.has_vb_cable([])


def test_audio_setup_command(tmp_path):
    cmd = w.audio_setup_command(tmp_path / "audio_setup.ps1", "HyperX", "Blue Yeti")
    assert cmd[:4] == ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass"]
    assert cmd[cmd.index("-SpeakerMatch") + 1] == "HyperX"
    assert cmd[cmd.index("-MicMatch") + 1] == "Blue Yeti"


def test_model_urls_and_mirror(monkeypatch, tmp_path):
    monkeypatch.delenv(wc.MIRROR_ENV, raising=False)
    assert wc.model_urls("small", "q8_0") == [
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin"]
    monkeypatch.setenv(wc.MIRROR_ENV, "https://example.org/models/")
    assert wc.model_urls("small", "")[1] == "https://example.org/models/ggml-small.bin"
    assert wc.model_path(tmp_path, "small", "q8_0") == tmp_path / "ggml-small-q8_0.bin"
