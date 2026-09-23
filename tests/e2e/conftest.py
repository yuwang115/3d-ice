"""Playwright fixtures for 3D ICE E2E tests."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@pytest.fixture(scope="session")
def playwright_browser():
    """Launch one headless Chromium instance for E2E modules."""
    from playwright.sync_api import sync_playwright

    runtime = sync_playwright().start()
    browser = runtime.chromium.launch(headless=True)
    try:
        yield browser
    finally:
        browser.close()
        runtime.stop()


SERVER_STARTUP_TIMEOUT_S = 90


def _server_log_tail(log_path: Path, max_lines: int = 20) -> str:
    """Read back what the server wrote, for a startup-failure message."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return "<server log unreadable>"
    return "\n".join(lines[-max_lines:]).strip() or "<no server output>"


def _wait_for_port(proc: subprocess.Popen, port: int, timeout_s: float, log_path: Path) -> None:
    """Block until the server accepts connections, or fail with a useful message.

    A fixed sleep is not enough: on a machine where static/ lives in iCloud Drive the
    interpreter can take half a minute to bind, which surfaced as ERR_CONNECTION_TIMED_OUT
    in whichever tests happened to run first.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Could not start the local E2E server: {_server_log_tail(log_path)}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    _terminate(proc)
    raise RuntimeError(
        f"Local E2E server did not accept connections within {timeout_s:.0f}s: "
        f"{_server_log_tail(log_path)}"
    )


def _terminate(proc: subprocess.Popen) -> None:
    """Stop the server, escalating if it ignores SIGTERM."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="session")
def server(tmp_path_factory: pytest.TempPathFactory):
    """Start a local HTTP server serving the static/ directory."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    # The server's stderr must never go to a pipe that nothing drains. http.server logs
    # one line per request, and ThreadingHTTPServer adds a whole BrokenPipeError traceback
    # (~2.4 kB) every time the browser aborts an in-flight download -- roughly 150 kB over
    # a full suite run, against a 64 KiB OS pipe buffer. Once that buffer fills, the server
    # blocks forever inside write() *before* sending any response, so it stops answering
    # entirely and every later navigation times out. Writing to a file cannot block, and
    # keeps the access log around for diagnosing a failure.
    log_path = tmp_path_factory.mktemp("e2e-server") / "http-server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                "python3", "-m", "http.server", str(port),
                "--bind", "127.0.0.1",
                "--directory", str(STATIC_DIR),
            ],
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
    try:
        _wait_for_port(proc, port, SERVER_STARTUP_TIMEOUT_S, log_path)
        yield f"http://127.0.0.1:{port}"
    finally:
        _terminate(proc)


@pytest.fixture
def explorer_url(server: str) -> str:
    return f"{server}/tools/3D-interactive-cryosphere-explorer.html"


@pytest.fixture
def home_url(server: str) -> str:
    return f"{server}/index.html"
