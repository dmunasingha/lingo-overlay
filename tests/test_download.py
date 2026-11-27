"""Model download: resume after stalls/drops, fall back to the next URL, never leave a bad file."""

import http.server
import threading
import time

import pytest

from lingo_overlay.download import DownloadError, download, text_progress

DATA = bytes(range(256)) * 4096  # 1 MiB


class Handler(http.server.BaseHTTPRequestHandler):
    # Behaviour per path, set by tests: "ok", "404", "drop" (dies halfway once), "stall" (hangs once),
    # "norange" (ignores Range).
    calls: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        mode = self.path.strip("/").split("/")[0]
        n = Handler.calls[self.path] = Handler.calls.get(self.path, 0) + 1
        if mode == "404":
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and mode != "norange":
            start = int(rng.split("=")[1].rstrip("-"))
        body = DATA[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if mode == "drop" and n == 1:
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.connection.shutdown(2)  # close mid-body
            return
        if mode == "stall" and n == 1:
            self.wfile.write(body[: len(body) // 3])
            self.wfile.flush()
            time.sleep(1.5)  # longer than the test's stall_timeout
            return
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def fresh_calls():
    Handler.calls = {}


def quiet(*a):
    pass


def test_plain_download_with_progress(server, tmp_path):
    seen = []
    dest = download([f"{server}/ok/m.bin"], tmp_path / "m.bin", progress=lambda d, t: seen.append((d, t)),
                    log=quiet)
    assert dest.read_bytes() == DATA
    assert seen[-1] == (len(DATA), len(DATA))
    assert not (tmp_path / "m.bin.part").exists()


def test_existing_file_is_not_downloaded_again(server, tmp_path):
    (tmp_path / "m.bin").write_bytes(b"already here")
    download([f"{server}/ok/x.bin"], tmp_path / "m.bin", log=quiet)
    assert Handler.calls == {}


def test_resumes_after_dropped_connection(server, tmp_path):
    dest = download([f"{server}/drop/m.bin"], tmp_path / "m.bin", log=quiet, retries=3)
    assert dest.read_bytes() == DATA
    assert Handler.calls["/drop/m.bin"] == 2


def test_stall_is_detected_and_resumed(server, tmp_path):
    t0 = time.time()
    dest = download([f"{server}/stall/m.bin"], tmp_path / "m.bin", log=quiet, stall_timeout=0.5)
    assert dest.read_bytes() == DATA
    assert Handler.calls["/stall/m.bin"] == 2
    assert time.time() - t0 < 10


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    (tmp_path / "m.bin.part").write_bytes(b"garbage from an old attempt")
    dest = download([f"{server}/norange/m.bin"], tmp_path / "m.bin", log=quiet)
    assert dest.read_bytes() == DATA


def test_falls_back_to_next_url(server, tmp_path):
    dest = download([f"{server}/404/m.bin", f"{server}/ok/m.bin"], tmp_path / "m.bin", log=quiet)
    assert dest.read_bytes() == DATA
    assert Handler.calls["/404/m.bin"] == 1  # a 404 isn't retried


def test_all_urls_fail(server, tmp_path):
    with pytest.raises(DownloadError, match="404"):
        download([f"{server}/404/a.bin", f"{server}/404/b.bin"], tmp_path / "m.bin", log=quiet)
    assert not (tmp_path / "m.bin").exists()


def test_cancel_keeps_partial_for_resume(server, tmp_path):
    with pytest.raises(DownloadError, match="cancelled"):
        download([f"{server}/ok/m.bin"], tmp_path / "m.bin", log=quiet, cancel=lambda: True)
    assert not (tmp_path / "m.bin").exists()
    assert (tmp_path / "m.bin.part").exists()


def test_text_progress_logs_each_step_once():
    lines = []
    cb = text_progress(lines.append, steps=4)
    for done in range(0, 101, 5):
        cb(done << 20, 100 << 20)
    assert lines == [f"  {p}% of 100 MB" for p in (0, 25, 50, 75, 100)]
