"""Regression tests for QRF 2025 Greenland terrain packages."""

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
QRF_PACKAGES = {
    "greenland_qrf_2025_3km": {
        "nx": 511,
        "ny": 918,
        "x0_m": -652925,
        "y0_m": -632675,
        "dx_m": 3000,
        "dy_m": -3000,
    },
    "greenland_qrf_2025_1km": {
        "nx": 1533,
        "ny": 2752,
        "x0_m": -652925,
        "y0_m": -632675,
        "dx_m": 1000,
        "dy_m": -1000,
    },
}


@pytest.mark.integration
class TestQrfGreenlandIntegration:
    def test_explorers_register_qrf_greenland_modes(self):
        for explorer_path in EXPLORER_PATHS:
            explorer = explorer_path.read_text(encoding="utf-8")

            assert "qrf: {" in explorer
            assert 'id: "qrf"' in explorer
            assert 'label: "QRF 2025 — Balanced"' in explorer
            assert 'metaUrl: assetUrl("data/greenland_qrf_2025_3km.meta.json")' in explorer
            assert 'binUrl: assetUrl("data/greenland_qrf_2025_3km.bin")' in explorer
            assert '"qrf-hd": {' in explorer
            assert 'id: "qrf-hd"' in explorer
            assert 'label: "QRF 2025 — HD"' in explorer
            assert 'metaUrl: assetUrl("data/greenland_qrf_2025_1km.meta.json")' in explorer
            assert 'binUrl: assetUrl("data/greenland_qrf_2025_1km.bin")' in explorer
            assert "10.1017/jog.2025.10071" in explorer

    @pytest.mark.parametrize("basename,expected_grid", QRF_PACKAGES.items())
    def test_qrf_package_has_a_complete_viewer_contract(self, basename, expected_grid):
        meta_path = DATA_DIR / f"{basename}.meta.json"
        bin_path = DATA_DIR / f"{basename}.bin"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert meta["grid"] == expected_grid
        assert meta["source_sampling"]["pixel_anchor"] == "cell_center"
        assert meta["source_sampling"]["x0_m"] == -635750
        assert meta["source_sampling"]["y0_m"] == -666050
        assert meta["hybridization"]["qrf_bed_applies_to"] == "grounded_ice_only"
        assert "BedMachine Greenland v6 surface and mask" in meta["hybridization"]["fallback"]
        assert {field["name"] for field in meta["fields"]} == {"bed", "surface", "thickness", "mask"}
        expected_length = max(field["byte_offset"] + field["byte_length"] for field in meta["fields"])
        assert bin_path.stat().st_size == expected_length
