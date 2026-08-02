"""Unit tests for Bedmap3 grid and mask conversion."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest


@pytest.mark.unit
class TestPrepareBedmap3Antarctica:
    @pytest.fixture(autouse=True)
    def _load(self, bedmap3_antarctica_module):
        self.build_axis_sampling = bedmap3_antarctica_module.build_axis_sampling
        self.translate_mask = bedmap3_antarctica_module.translate_mask
        self.module = bedmap3_antarctica_module

    def test_build_axis_sampling_preserves_source_grid_orientation(self):
        axis = np.array([3333250, 3332750, 3332250, 3331750, 3331250], dtype=np.int64)

        sampled_axis, indices = self.build_axis_sampling(axis, 1000)

        np.testing.assert_array_equal(sampled_axis, [3333250, 3332250, 3331250])
        np.testing.assert_array_equal(indices, [0, 2, 4])

    def test_build_axis_sampling_rejects_invalid_spacing(self):
        axis = np.array([0, 500, 1000], dtype=np.int64)

        with pytest.raises(ValueError, match="multiple"):
            self.build_axis_sampling(axis, 750)

    def test_build_axis_sampling_creates_the_hd_four_kilometre_grid(self):
        axis = np.arange(-3333250, 3333251, 500, dtype=np.int64)

        sampled_axis, indices = self.build_axis_sampling(axis, 4000)

        assert len(sampled_axis) == 1667
        assert sampled_axis[0] == -3333250
        assert sampled_axis[1] - sampled_axis[0] == 4000
        assert indices[-1] == 13328

    def test_translate_mask_adapts_bedmap3_categories_for_the_viewer(self):
        source_mask = np.array([[-9999, 0, 1, 2, 3, 4]], dtype=np.int16)

        translated = self.translate_mask(source_mask)

        np.testing.assert_array_equal(translated, [[0, 0, 2, 3, 3, 1]])

    def test_quantize_elevation_and_stats_respect_no_data(self):
        values = np.array([[-9999.0, -12.4, 42.7, np.nan]], dtype=np.float32)

        quantized = self.module.quantize_elevation(values)
        stats = self.module.field_stats(values)

        np.testing.assert_array_equal(quantized, [[-32768, -12, 43, -32768]])
        assert stats == {"min": -12.399999618530273, "max": 42.70000076293945, "mean": 15.15000057220459}

    def test_read_geotiff_validates_the_expected_bedmap3_shape(self, tmp_path, monkeypatch):
        tifffile = pytest.importorskip("tifffile")
        monkeypatch.setattr(self.module, "SOURCE_GRID_SIZE", 2)
        source_path = tmp_path / "source.tif"
        values = np.array([[1, 2], [3, 4]], dtype=np.int16)
        tifffile.imwrite(source_path, values)

        decoded = self.module.read_geotiff(source_path)

        np.testing.assert_array_equal(decoded, values)

    def test_main_writes_a_complete_package(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.module, "SOURCE_GRID_SIZE", 3)
        monkeypatch.setattr(self.module, "SOURCE_GRID_SPACING_M", 500)
        monkeypatch.setattr(self.module, "SOURCE_X0_M", -500)
        monkeypatch.setattr(self.module, "SOURCE_Y0_M", 500)
        monkeypatch.setattr(
            self.module,
            "parse_args",
            lambda: argparse.Namespace(
                input_dir=tmp_path / "source",
                output_dir=tmp_path / "output",
                resolution_m=1000,
                basename="bedmap3_test",
            ),
        )
        source_fields = iter(
            (
                np.array([[1, 2], [3, 4]], dtype=np.int16),
                np.array([[5, 6], [7, 8]], dtype=np.int16),
                np.array([[9, 10], [11, 12]], dtype=np.int16),
                np.array([[1, 2], [3, 4]], dtype=np.int16),
            )
        )
        monkeypatch.setattr(self.module, "load_and_sample_field", lambda *_args: next(source_fields))

        self.module.main()

        meta_path = tmp_path / "output" / "bedmap3_test.meta.json"
        bin_path = tmp_path / "output" / "bedmap3_test.bin"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["grid"] == {"nx": 2, "ny": 2, "x0_m": -500, "y0_m": 500, "dx_m": 1000, "dy_m": -1000}
        assert bin_path.stat().st_size == 28
