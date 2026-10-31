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
def bedmap3_antarctica_module():
    return _load_script_module("prepare_bedmap3_antarctica.py")


@pytest.fixture(scope="session")
def bedmap3_overlays_module():
    return _load_script_module("prepare_bedmap3_antarctica_overlays.py")


@pytest.fixture(scope="session")
def qrf_greenland_module():
    return _load_script_module("prepare_qrf_greenland.py")


@pytest.fixture(scope="session")
def polar_features_module():
    return _load_script_module("prepare_polar_features.py")


@pytest.fixture(scope="session")
def ice_cores_module():
    # prepare_ice_cores.py imports its projection helpers from prepare_polar_features.py.
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return _load_script_module("prepare_ice_cores.py")


@pytest.fixture(scope="session")
def ismip6_projection_module():
    return _load_script_module("prepare_ismip6_projection.py")


@pytest.fixture(scope="session")
def ismip6_regrid_module():
    pytest.importorskip("netCDF4", reason="netCDF4 not installed")
    return _load_script_module("regrid_ismip6_2300_run.py")


@pytest.fixture(scope="session")
def cpom_land_ice_module():
    return _load_script_module("prepare_cpom_land_ice.py")


@pytest.fixture(scope="session")
def nsidc_sea_ice_module():
    return _load_script_module("prepare_nsidc_sea_ice.py")


@pytest.fixture(scope="session")
def ismip6_combine_module():
    pytest.importorskip("netCDF4", reason="netCDF4 not installed")
    return _load_script_module("combine_ismip6_2300_mean.py")


@pytest.fixture(scope="session")
def velocity_module():
    pytest.importorskip("netCDF4", reason="netCDF4 not installed")
    return _load_script_module("prepare_antarctica_velocity.py")


@pytest.fixture(scope="session")
def isostatic_response_module():
    pytest.importorskip("netCDF4", reason="netCDF4 not installed")
    return _load_script_module("prepare_isostatic_response.py")


@pytest.fixture(scope="session")
def subglacial_hydrology_module():
    pytest.importorskip("netCDF4", reason="netCDF4 not installed")
    return _load_script_module("prepare_subglacial_hydrology.py")


@pytest.fixture(scope="session")
def antarctica_ocean_currents_module():
    pytest.importorskip("netCDF4", reason="netCDF4 not installed")
    return _load_script_module("prepare_antarctica_ocean_currents.py")


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR
