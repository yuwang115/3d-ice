"""Regression tests for the Bedmap3 Antarctica terrain package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "static" / "tools" / "data"
EXPLORER_PATHS = (
    ROOT_DIR / "static" / "tools" / "3D-interactive-cryosphere-explorer.html",
    ROOT_DIR / "static" / "zh" / "tools" / "3D-interactive-cryosphere-explorer.html",
)
BEDMAP3_META_PATH = DATA_DIR / "bedmap3_antarctica_10km.meta.json"
BEDMAP3_BIN_PATH = DATA_DIR / "bedmap3_antarctica_10km.bin"
BEDMAP3_HD_META_PATH = DATA_DIR / "bedmap3_antarctica_4km.meta.json"
BEDMAP3_HD_BIN_PATH = DATA_DIR / "bedmap3_antarctica_4km.bin"
BEDMAP3_OVERLAY_PACKAGES = {
    "bedmap3_antarctica_10km": (
        "bedmap3_antarctica_velocity_10km",
        "bedmap3_antarctica_basal_friction_10km",
        "bedmap3_antarctica_subglacial_hydrology_10km",
    ),
    "bedmap3_antarctica_4km": (
        "bedmap3_antarctica_velocity_4km",
        "bedmap3_antarctica_basal_friction_4km",
        "bedmap3_antarctica_subglacial_hydrology_4km",
    ),
}


@pytest.mark.integration
class TestBedmap3Integration:
    """Ensure Bedmap3 is a real, selectable Antarctica terrain source."""

    def test_explorer_registers_bedmap3_as_a_dataset(self):
        for explorer_path in EXPLORER_PATHS:
            explorer = explorer_path.read_text(encoding="utf-8")

            assert "bedmap3: {" in explorer
            assert 'id: "bedmap3"' in explorer
            assert 'label: "Bedmap3 — Balanced"' in explorer
            assert 'metaUrl: assetUrl("data/bedmap3_antarctica_10km.meta.json")' in explorer
            assert 'binUrl: assetUrl("data/bedmap3_antarctica_10km.bin")' in explorer
            assert '"bedmap3-hd": {' in explorer
            assert 'id: "bedmap3-hd"' in explorer
            assert 'label: "Bedmap3 — HD"' in explorer
            assert 'metaUrl: assetUrl("data/bedmap3_antarctica_4km.meta.json")' in explorer
            assert 'binUrl: assetUrl("data/bedmap3_antarctica_4km.bin")' in explorer
            assert "2d0e4791-8e20-46a3-80e4-f5f6716025d2" in explorer

    def test_bedmap3_enables_requested_overlay_packages(self):
        for explorer_path in EXPLORER_PATHS:
            explorer = explorer_path.read_text(encoding="utf-8")

            for dataset_key, resolution in (("bedmap3", "10km"), ('"bedmap3-hd"', "4km")):
                config = explorer.split(f"{dataset_key}: {{", maxsplit=1)[1].split("\n          },", maxsplit=1)[0]
                for capability in ("velocity", "basalFriction", "flowline", "oceanCurrents", "hydrology"):
                    assert f"{capability}: true" in config
                assert "rise: false" in config
                assert f'bedmap3_antarctica_velocity_{resolution}.meta.json' in config
                assert f'bedmap3_antarctica_basal_friction_{resolution}.meta.json' in config
                assert f'bedmap3_antarctica_subglacial_hydrology_{resolution}.meta.json' in config
                assert "antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean" in config

    def test_bedmap3_package_uses_its_native_antarctic_grid(self):
        with BEDMAP3_META_PATH.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        assert meta["grid"] == {
            "nx": 667,
            "ny": 667,
            "x0_m": -3333250,
            "y0_m": 3333250,
            "dx_m": 10000,
            "dy_m": -10000,
        }

    def test_bedmap3_hd_package_uses_its_native_four_kilometre_grid(self):
        with BEDMAP3_HD_META_PATH.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        assert meta["grid"] == {
            "nx": 1667,
            "ny": 1667,
            "x0_m": -3333250,
            "y0_m": 3333250,
            "dx_m": 4000,
            "dy_m": -4000,
        }

    def test_bedmap3_package_preserves_viewer_fields_and_provenance(self):
        with BEDMAP3_META_PATH.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        fields = {field["name"]: field for field in meta["fields"]}
        assert meta["title"] == "Bedmap3 Antarctica"
        assert meta["license"] == "CC-BY-4.0"
        assert "10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2" in meta["reference"]
        assert set(("bed", "surface", "thickness", "mask")).issubset(fields)
        assert fields["mask"]["flags"] == {
            "0": "ocean_or_no_data",
            "1": "ice_free_land_or_rock",
            "2": "grounded_ice",
            "3": "floating_ice_or_transiently_grounded_ice",
        }

    def test_bedmap3_binary_length_matches_metadata(self):
        with BEDMAP3_META_PATH.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        expected_length = max(field["byte_offset"] + field["byte_length"] for field in meta["fields"])
        assert BEDMAP3_BIN_PATH.stat().st_size == expected_length

    def test_bedmap3_hd_binary_length_matches_metadata(self):
        with BEDMAP3_HD_META_PATH.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        expected_length = max(field["byte_offset"] + field["byte_length"] for field in meta["fields"])
        assert BEDMAP3_HD_BIN_PATH.stat().st_size == expected_length

    @pytest.mark.parametrize("terrain_basename,overlay_basenames", BEDMAP3_OVERLAY_PACKAGES.items())
    def test_bedmap3_overlay_packages_share_their_terrain_grid_and_binary_contract(
        self, terrain_basename, overlay_basenames
    ):
        terrain_meta = json.loads((DATA_DIR / f"{terrain_basename}.meta.json").read_text(encoding="utf-8"))

        for overlay_basename in overlay_basenames:
            overlay_meta_path = DATA_DIR / f"{overlay_basename}.meta.json"
            overlay_bin_path = DATA_DIR / f"{overlay_basename}.bin"
            overlay_meta = json.loads(overlay_meta_path.read_text(encoding="utf-8"))

            assert overlay_meta["grid"] == terrain_meta["grid"]
            assert overlay_meta["resampled_to"].startswith("Bedmap3 —")
            binary_fields = [field for field in overlay_meta["fields"] if "byte_length" in field]
            expected_length = max(field["byte_offset"] + field["byte_length"] for field in binary_fields)
            assert overlay_bin_path.stat().st_size == expected_length
