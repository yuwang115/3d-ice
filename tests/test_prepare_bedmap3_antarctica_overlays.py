"""Unit tests for translating BedMachine overlay packages to Bedmap3 grids."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest


@pytest.mark.unit
class TestPrepareBedmap3AntarcticaOverlays:
    @pytest.fixture(autouse=True)
    def _load(self, bedmap3_overlays_module):
        self.module = bedmap3_overlays_module

    def test_resolve_indices_handles_the_bedmap3_quarter_kilometre_origin_offset(self):
        target_axis = np.array([-3333250, -3323250, 3326750], dtype=np.float64)

        indices, valid = self.module.resolve_indices(target_axis, -3333000, 10000, 667)

        np.testing.assert_array_equal(indices, [0, 1, 666])
        np.testing.assert_array_equal(valid, [True, True, True])

    def test_resample_grid_uses_nearest_source_cells_and_preserves_fill_outside_coverage(self):
        source = np.array([[1, 2], [3, 4]], dtype=np.int16)
        output = self.module.resample_grid(
            source,
            source_grid={"nx": 2, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10},
            target_grid={"nx": 3, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10},
            fill_value=-32768,
        )

        np.testing.assert_array_equal(output, [[1, 2, -32768], [3, 4, -32768]])

    def test_reproject_channels_collapses_reversed_duplicate_edges(self):
        channels = self.module.reproject_channels(
            source_grid={"nx": 3, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10},
            target_grid={"nx": 3, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10},
            col1=np.array([0, 1], dtype=np.uint16),
            row1=np.array([0, 0], dtype=np.uint16),
            col2=np.array([1, 0], dtype=np.uint16),
            row2=np.array([0, 0], dtype=np.uint16),
            discharge=np.array([2.0, 5.0], dtype=np.float32),
        )

        col1, row1, col2, row2, discharge = channels
        np.testing.assert_array_equal(col1, [0])
        np.testing.assert_array_equal(row1, [0])
        np.testing.assert_array_equal(col2, [1])
        np.testing.assert_array_equal(row2, [0])
        np.testing.assert_array_equal(discharge, [5.0])

    def test_finite_quantiles_uses_the_viewer_metadata_keys(self):
        quantiles = self.module.finite_quantiles(np.array([1.0, 2.0, 3.0]), (0.5, 0.9, 0.995))

        assert set(quantiles) == {"median", "q90", "q995"}

    def test_prepare_target_writes_complete_bedmap3_overlay_packages(self, tmp_path):
        grid = {"nx": 2, "ny": 2, "x0_m": 0, "y0_m": 10, "dx_m": 10, "dy_m": -10}
        (tmp_path / "bedmap3_antarctica_10km.meta.json").write_text(
            json.dumps({"grid": grid}), encoding="utf-8"
        )
        fill = -32768

        velocity_meta = {
            "grid": grid,
            "quantization": {"int16_fill_value": fill, "scale": 1.0, "offset": 0.0},
            "fields": [
                {"name": "vx", "dtype": "int16"},
                {"name": "vy", "dtype": "int16"},
                {"name": "speed", "unit": "m/year"},
            ],
        }
        self.module.write_package(
            tmp_path,
            "antarctic_ice_velocity_phase_v01_480",
            velocity_meta,
            {
                "vx": np.array([1, 2, fill, 4], dtype=np.int16),
                "vy": np.array([5, 6, fill, 8], dtype=np.int16),
            },
        )

        friction_meta = {
            "grid": grid,
            "coverage": {},
            "fields": [{"name": "basal_friction", "dtype": "float32", "unit": "MPa"}],
        }
        self.module.write_package(
            tmp_path,
            "antarctica_basal_friction_480",
            friction_meta,
            {"basal_friction": np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float32)},
        )

        hydrology_meta = {
            "grid": grid,
            "quantization": {
                "int16_fill_value": fill,
                "effective_pressure_scale_pa_per_int16": 1000.0,
                "effective_pressure_offset_pa": 0.0,
            },
            "coverage": {"channel_segment_count_raw": 2},
            "fields": [
                {"name": "effective_pressure", "dtype": "int16", "unit": "Pa"},
                {"name": "channel_col1", "dtype": "uint16"},
                {"name": "channel_row1", "dtype": "uint16"},
                {"name": "channel_col2", "dtype": "uint16"},
                {"name": "channel_row2", "dtype": "uint16"},
                {"name": "channel_discharge", "dtype": "float32", "unit": "m3/s"},
            ],
        }
        self.module.write_package(
            tmp_path,
            "antarctica_subglacial_hydrology_480",
            hydrology_meta,
            {
                "effective_pressure": np.array([1, 2, fill, 4], dtype=np.int16),
                "channel_col1": np.array([0, 1], dtype=np.uint16),
                "channel_row1": np.array([0, 0], dtype=np.uint16),
                "channel_col2": np.array([1, 0], dtype=np.uint16),
                "channel_row2": np.array([0, 0], dtype=np.uint16),
                "channel_discharge": np.array([2.0, 5.0], dtype=np.float32),
            },
        )

        target = self.module.Target("Bedmap3 fixture", "bedmap3_antarctica_10km", "10km", "480")
        self.module.prepare_target(tmp_path, target)

        velocity_out, velocity_fields = self.module.read_package(
            tmp_path / "bedmap3_antarctica_velocity_10km.meta.json",
            tmp_path / "bedmap3_antarctica_velocity_10km.bin",
        )
        assert velocity_out["grid"] == grid
        assert velocity_out["resampled_to"] == "Bedmap3 fixture"
        assert velocity_out["fields"][-1]["quantiles_m_per_year"]["median"] > 0
        np.testing.assert_array_equal(velocity_fields["vx"], [1, 2, fill, 4])

        friction_out, friction_fields = self.module.read_package(
            tmp_path / "bedmap3_antarctica_basal_friction_10km.meta.json",
            tmp_path / "bedmap3_antarctica_basal_friction_10km.bin",
        )
        assert friction_out["coverage"]["valid_count"] == 3
        assert np.isnan(friction_fields["basal_friction"][1])

        hydrology_out, hydrology_fields = self.module.read_package(
            tmp_path / "bedmap3_antarctica_subglacial_hydrology_10km.meta.json",
            tmp_path / "bedmap3_antarctica_subglacial_hydrology_10km.bin",
        )
        assert hydrology_out["coverage"]["channel_segment_count_unique"] == 1
        np.testing.assert_array_equal(hydrology_fields["channel_discharge"], [5.0])

    def test_main_prepares_every_configured_target(self, tmp_path, monkeypatch):
        target = self.module.Target("fixture", "terrain", "fixture", "source")
        monkeypatch.setattr(self.module, "TARGETS", (target,))
        monkeypatch.setattr(self.module, "parse_args", lambda: argparse.Namespace(data_dir=tmp_path))
        calls: list[tuple[object, object]] = []
        monkeypatch.setattr(self.module, "prepare_target", lambda directory, selected: calls.append((directory, selected)))

        self.module.main()

        assert calls == [(tmp_path, target)]
