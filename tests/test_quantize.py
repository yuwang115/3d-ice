"""Tests for quantize_to_int16 and to_quantized_int16 functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

FILL_FLOAT = -9999.0
FILL_INT16 = -32768


class TestQuantizeToInt16Antarctica:
    """Tests for the Antarctica BedMachine variant (positional fill_value)."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_antarctica_module):
        self.quantize = bedmachine_antarctica_module.quantize_to_int16

    @pytest.mark.unit
    def test_normal_values(self):
        arr = np.array([100.0, -200.0, 0.0, 500.7], dtype=np.float32)
        result = self.quantize(arr)
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, [100, -200, 0, 501])

    @pytest.mark.unit
    def test_fill_value_mapped_to_int16_fill(self):
        arr = np.array([10.0, FILL_FLOAT, 20.0], dtype=np.float32)
        result = self.quantize(arr)
        assert result[1] == FILL_INT16
        assert result[0] == 10
        assert result[2] == 20

    @pytest.mark.unit
    def test_nan_mapped_to_int16_fill(self):
        arr = np.array([10.0, np.nan, 20.0], dtype=np.float32)
        result = self.quantize(arr)
        assert result[1] == FILL_INT16

    @pytest.mark.unit
    def test_clips_at_int16_boundaries(self):
        arr = np.array([50000.0, -50000.0], dtype=np.float32)
        result = self.quantize(arr)
        assert result[0] == 32767
        assert result[1] == -32767

    @pytest.mark.unit
    def test_all_fill(self):
        arr = np.full(5, FILL_FLOAT, dtype=np.float32)
        result = self.quantize(arr)
        np.testing.assert_array_equal(result, [FILL_INT16] * 5)

    @pytest.mark.unit
    def test_preserves_shape(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = self.quantize(arr)
        assert result.shape == (2, 2)

    @pytest.mark.unit
    def test_custom_fill_value(self):
        arr = np.array([1.0, -1e30, 2.0], dtype=np.float32)
        result = self.quantize(arr, fill_value=-1e30)
        assert result[0] == 1
        assert result[1] == FILL_INT16
        assert result[2] == 2


class TestQuantizeToInt16Greenland:
    """Tests for the Greenland BedMachine variant (keyword-only fill_value)."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_greenland_module):
        self.quantize = bedmachine_greenland_module.quantize_to_int16

    @pytest.mark.unit
    def test_basic(self):
        arr = np.array([10.0, -20.0, 300.4], dtype=np.float32)
        result = self.quantize(arr, fill_value=FILL_FLOAT)
        np.testing.assert_array_equal(result, [10, -20, 300])

    @pytest.mark.unit
    def test_custom_clip_range(self):
        arr = np.array([500.0, -500.0], dtype=np.float32)
        result = self.quantize(arr, fill_value=FILL_FLOAT, clip_min=-100, clip_max=100)
        assert result[0] == 100
        assert result[1] == -100


class TestToQuantizedInt16:
    """Tests for the velocity script's scale-based quantization."""

    @pytest.fixture(autouse=True)
    def _load(self, velocity_module):
        self.to_quantized = velocity_module.to_quantized_int16

    @pytest.mark.unit
    def test_with_scale_1(self):
        arr = np.array([100.0, -200.0], dtype=np.float64)
        result = self.to_quantized(arr, scale=1.0)
        np.testing.assert_array_equal(result, [100, -200])

    @pytest.mark.unit
    def test_with_scale_factor(self):
        arr = np.array([200.0, -400.0], dtype=np.float64)
        result = self.to_quantized(arr, scale=2.0)
        np.testing.assert_array_equal(result, [100, -200])

    @pytest.mark.unit
    def test_clips_large_values(self):
        arr = np.array([1e6], dtype=np.float64)
        result = self.to_quantized(arr, scale=1.0)
        assert result[0] == 32767
