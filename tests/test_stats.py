"""Tests for stats and finite_quantiles functions."""

from __future__ import annotations

import math

import numpy as np
import pytest

FILL_FLOAT = -9999.0


class TestStatsAntarctica:
    """Tests for the Antarctica variant (positional fill_value)."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_antarctica_module):
        self.stats = bedmachine_antarctica_module.stats

    @pytest.mark.unit
    def test_basic_stats(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = self.stats(arr)
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["mean"] == 3.0

    @pytest.mark.unit
    def test_excludes_fill_values(self):
        arr = np.array([10.0, FILL_FLOAT, 20.0, FILL_FLOAT], dtype=np.float32)
        result = self.stats(arr)
        assert result["min"] == 10.0
        assert result["max"] == 20.0
        assert result["mean"] == 15.0

    @pytest.mark.unit
    def test_excludes_nan(self):
        arr = np.array([5.0, np.nan, 15.0], dtype=np.float32)
        result = self.stats(arr)
        assert result["min"] == 5.0
        assert result["max"] == 15.0
        assert result["mean"] == 10.0

    @pytest.mark.unit
    def test_single_valid_value(self):
        arr = np.array([42.0, FILL_FLOAT, np.nan], dtype=np.float32)
        result = self.stats(arr)
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert result["mean"] == 42.0

    @pytest.mark.unit
    def test_negative_values(self):
        arr = np.array([-100.0, -50.0, -200.0], dtype=np.float32)
        result = self.stats(arr)
        assert result["min"] == -200.0
        assert result["max"] == -50.0


class TestStatsGreenland:
    """Tests for the Greenland variant (keyword-only fill_value)."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_greenland_module):
        self.stats = bedmachine_greenland_module.stats

    @pytest.mark.unit
    def test_custom_fill_value(self):
        arr = np.array([1.0, -1e30, 2.0], dtype=np.float32)
        result = self.stats(arr, fill_value=-1e30)
        assert result["min"] == 1.0
        assert result["max"] == 2.0


class TestFiniteQuantiles:
    """Tests for the velocity script's finite_quantiles function."""

    @pytest.fixture(autouse=True)
    def _load(self, velocity_module):
        self.finite_quantiles = velocity_module.finite_quantiles

    @pytest.mark.unit
    def test_basic_quantiles(self):
        arr = np.arange(1.0, 101.0)
        result = self.finite_quantiles(arr)
        assert "median" in result
        assert "q90" in result
        assert "q95" in result
        assert "q99" in result
        assert result["median"] == pytest.approx(50.5, abs=0.5)
        assert result["q90"] > result["median"]
        assert result["q99"] > result["q95"]

    @pytest.mark.unit
    def test_excludes_nan(self):
        arr = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        result = self.finite_quantiles(arr)
        assert result["median"] == pytest.approx(2.0)

    @pytest.mark.unit
    def test_all_nan_returns_nan(self):
        arr = np.array([np.nan, np.nan])
        result = self.finite_quantiles(arr)
        assert math.isnan(result["median"])
