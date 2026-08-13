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
    time.sleep(1)  # give the server a moment to start
    if proc.poll() is not None:
        _, stderr = proc.communicate()
        raise RuntimeError(f"Could not start the local E2E server: {stderr.strip()}")
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
