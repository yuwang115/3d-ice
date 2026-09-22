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


def _wait_for_port(proc: subprocess.Popen, port: int, timeout_s: float) -> None:
    """Block until the server accepts connections, or fail with a useful message.

    A fixed sleep is not enough: on a machine where static/ lives in iCloud Drive the
    interpreter can take half a minute to bind, which surfaced as ERR_CONNECTION_TIMED_OUT
    in whichever tests happened to run first.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            raise RuntimeError(f"Could not start the local E2E server: {stderr.strip()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError(f"Local E2E server did not accept connections within {timeout_s:.0f}s")


@pytest.fixture(scope="session")
def server():
    """Start a local HTTP server serving the static/ directory."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(STATIC_DIR)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for_port(proc, port, SERVER_STARTUP_TIMEOUT_S)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.fixture
def explorer_url(server: str) -> str:
    return f"{server}/tools/3D-interactive-cryosphere-explorer.html"


@pytest.fixture
def home_url(server: str) -> str:
    return f"{server}/index.html"
