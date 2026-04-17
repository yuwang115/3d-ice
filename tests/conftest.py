"""Shared fixtures and configuration for 3D ICE tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make the scripts/ directory importable so tests can `from conftest import`
# the helpers they need without restructuring the project layout.
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
DATA_DIR = Path(__file__).resolve().parent.parent / "static" / "tools" / "data"


def _load_script_module(name: str) -> object:
    """Import a standalone script from scripts/ as a module."""
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Avoid executing argparse when loading
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def bedmachine_antarctica_module():
    return _load_script_module("prepare_bedmachine_antarctica.py")


@pytest.fixture(scope="session")
def bedmachine_greenland_module():
    return _load_script_module("prepare_bedmachine_greenland.py")


@pytest.fixture(scope="session")
def velocity_module():
    pytest.importorskip("netCDF4", reason="netCDF4 not installed")
    return _load_script_module("prepare_antarctica_velocity.py")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR
