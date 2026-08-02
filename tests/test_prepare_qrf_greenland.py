"""Unit tests for the QRF Greenland hybrid-terrain preparer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
class TestPrepareQrfGreenland:
    @pytest.fixture(autouse=True)
    def _load(self, qrf_greenland_module):
        self.module = qrf_greenland_module

    def test_resolve_indices_respects_the_qrf_north_up_grid_orientation(self):
        target_axis = np.array([-635900, -635600, -635300], dtype=np.float64)

        indices, valid = self.module.resolve_indices(target_axis, -635900, 300, 3)

        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_array_equal(valid, [True, True, True])

    def test_sample_qrf_band_uses_fill_for_cells_outside_the_source_extent(self):
        source = np.array([[1, 2], [3, 4]], dtype=np.int16)
        target = self.module.sample_qrf_band(
            source,
            source_grid={"nx": 2, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10},
            target_grid={"nx": 3, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10},
        )

        np.testing.assert_array_equal(target, [[1, 2, -32768], [3, 4, -32768]])

    def test_qrf_sampling_uses_pixel_centres_not_the_geotiff_edge_tiepoint(self):
        source_grid = {
            "nx": 2,
            "ny": 2,
            "x0_m": self.module.QRF_SOURCE_GRID["x0_m"],
            "y0_m": self.module.QRF_SOURCE_GRID["y0_m"],
            "dx_m": self.module.QRF_SOURCE_GRID["dx_m"],
            "dy_m": self.module.QRF_SOURCE_GRID["dy_m"],
        }
        source = np.array([[11, 12], [21, 22]], dtype=np.int16)
        target = self.module.sample_qrf_band(
            source,
            source_grid=source_grid,
            target_grid={
                "nx": 1,
                "ny": 1,
                "x0_m": source_grid["x0_m"] + source_grid["dx_m"],
                "y0_m": source_grid["y0_m"] + source_grid["dy_m"],
                "dx_m": source_grid["dx_m"],
                "dy_m": source_grid["dy_m"],
            },
        )

        assert self.module.QRF_SOURCE_GRID["x0_m"] == -635750
        assert self.module.QRF_SOURCE_GRID["y0_m"] == -666050
        np.testing.assert_array_equal(target, [[22]])

    def test_build_hybrid_fields_uses_qrf_only_for_grounded_ice_with_coherent_thickness(self):
        base_bed = np.array([[10, 20], [30, 40]], dtype=np.int16)
        surface = np.array([[100, 200], [300, 400]], dtype=np.int16)
        base_thickness = np.array([[90, 180], [270, 360]], dtype=np.int16)
        mask = np.array([[2, 3], [2, 1]], dtype=np.uint8)
        qrf_bed = np.array([[25, -10], [350, -32768]], dtype=np.int16)
        qrf_thickness = np.array([[75, 210], [0, -32768]], dtype=np.int16)

        bed, thickness, applied = self.module.build_hybrid_fields(
            base_bed=base_bed,
            surface=surface,
            base_thickness=base_thickness,
            mask=mask,
            qrf_bed=qrf_bed,
            qrf_thickness=qrf_thickness,
        )

        np.testing.assert_array_equal(bed, [[25, 20], [30, 40]])
        np.testing.assert_array_equal(thickness, [[75, 180], [270, 360]])
        np.testing.assert_array_equal(applied, [[True, False], [False, False]])

    def test_prepare_target_writes_a_complete_hybrid_viewer_package(self, tmp_path, monkeypatch):
        grid = {"nx": 2, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10}
        count = grid["nx"] * grid["ny"]
        fields = [
            {"name": "bed", "dtype": "int16", "byte_offset": 0, "byte_length": count * 2},
            {"name": "surface", "dtype": "int16", "byte_offset": count * 2, "byte_length": count * 2},
            {"name": "thickness", "dtype": "int16", "byte_offset": count * 4, "byte_length": count * 2},
            {"name": "mask", "dtype": "uint8", "byte_offset": count * 6, "byte_length": count},
        ]
        (tmp_path / "base.meta.json").write_text(json.dumps({"grid": grid, "fields": fields}), encoding="utf-8")
        base_fields = (
            np.array([[10, 20], [30, 40]], dtype=np.int16),
            np.array([[100, 200], [300, 400]], dtype=np.int16),
            np.array([[90, 180], [270, 360]], dtype=np.int16),
            np.array([[2, 3], [2, 2]], dtype=np.uint8),
        )
        (tmp_path / "base.bin").write_bytes(b"".join(field.tobytes() for field in base_fields))
        monkeypatch.setattr(self.module, "QRF_SOURCE_GRID", grid)
        qrf_values = np.array(
            [
                [[50, 50, 10], [20, 190, 20]],
                [[80, 220, 30], [-32768, -32768, -32768]],
            ],
            dtype=np.int16,
        )

        self.module.prepare_target(tmp_path, qrf_values, self.module.Target("fixture", "base", "qrf"))

        metadata = json.loads((tmp_path / "qrf.meta.json").read_text(encoding="utf-8"))
        binary = (tmp_path / "qrf.bin").read_bytes()
        bed = self.module.decode_field(metadata, binary, "bed")
        thickness = self.module.decode_field(metadata, binary, "thickness")
        np.testing.assert_array_equal(bed, [[50, 20], [220, 40]])
        np.testing.assert_array_equal(thickness, [[50, 180], [80, 360]])
        assert metadata["qrf_coverage"]["applied_cell_count"] == 2
        assert metadata["qrf_coverage"]["applied_grounded_ice_fraction"] == pytest.approx(2 / 3)

    def test_read_qrf_geotiff_accepts_the_published_grid_contract(self, tmp_path, monkeypatch):
        source_grid = {"nx": 2, "ny": 2, "x0_m": -635900, "y0_m": -665900, "dx_m": 300, "dy_m": -300}
        monkeypatch.setattr(self.module, "QRF_SOURCE_GRID", source_grid)
        path = Path(tmp_path) / "qrf.tif"
        values = np.arange(12, dtype=np.int16).reshape(2, 2, 3)
        self.module.tifffile.imwrite(
            path,
            values,
            metadata=None,
            extratags=[
                (33550, "d", 3, (300.0, 300.0, 0.0), False),
                (33922, "d", 6, (0.0, 0.0, 0.0, -635900.0, -665900.0, 0.0), False),
                (
                    42112,
                    "s",
                    1,
                    "<GDALMetadata>"
                    '<Item name="DESCRIPTION" sample="0" role="description">qrf_pred_ice_thickness</Item>'
                    '<Item name="DESCRIPTION" sample="1" role="description">qrf_pred_bed_elev</Item>'
                    '<Item name="DESCRIPTION" sample="2" role="description">qrf_pred_uncertainty_SD</Item>'
                    "</GDALMetadata>",
                    False,
                ),
                (42113, "s", 7, "-32768", False),
            ],
        )

        actual = self.module.read_qrf_geotiff(path)

        np.testing.assert_array_equal(actual, values)
