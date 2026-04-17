"""Playwright fixtures for 3D ICE E2E tests."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
SERVER_PORT = 8765


@pytest.fixture(scope="session")
def server():
    """Start a local HTTP server serving the static/ directory."""
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(SERVER_PORT), "--directory", str(STATIC_DIR)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)  # give the server a moment to start
    yield f"http://localhost:{SERVER_PORT}"
    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def explorer_url(server: str) -> str:
    return f"{server}/tools/3D-interactive-cryosphere-explorer.html"


@pytest.fixture
def home_url(server: str) -> str:
    return f"{server}/index.html"
