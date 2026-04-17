"""Tests for axis sampling, grid construction, and index resolution."""

from __future__ import annotations

import numpy as np
import pytest


class TestBuildAxisSampling:
    """Tests for build_axis_sampling from the Greenland BedMachine script."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_greenland_module):
        self.build = bedmachine_greenland_module.build_axis_sampling

    @pytest.mark.unit
    def test_ascending_axis(self):
        axis = np.arange(0, 10000, 100, dtype=np.int64)  # 0..9900, step 100
        target, indices = self.build(axis, 500)
        # Target spacing should be 500m
        assert target[1] - target[0] == 500
        assert target[0] == 0
        assert all(0 <= i < len(axis) for i in indices)

    @pytest.mark.unit
    def test_descending_axis(self):
        axis = np.arange(10000, 0, -100, dtype=np.int64)  # 10000..100, step -100
        target, indices = self.build(axis, 500)
        assert target[0] == 10000
        assert target[1] < target[0]  # descending

    @pytest.mark.unit
    def test_invalid_resolution_raises(self):
        axis = np.arange(0, 1000, 10, dtype=np.int64)
        with pytest.raises(ValueError):
            self.build(axis, 0)
        with pytest.raises(ValueError):
            self.build(axis, -100)

    @pytest.mark.unit
    def test_single_element_axis_raises(self):
        with pytest.raises(ValueError):
            self.build(np.array([0], dtype=np.int64), 100)

    @pytest.mark.unit
    def test_indices_within_bounds(self):
        axis = np.arange(-3333000, 3333000, 500, dtype=np.int64)
        _, indices = self.build(axis, 10000)
        assert np.all(indices >= 0)
        assert np.all(indices < len(axis))


class TestBuildTargetAxis:
    """Tests for build_target_axis from the velocity script."""

    @pytest.fixture(autouse=True)
    def _load(self, velocity_module):
        self.build_target_axis = velocity_module.build_target_axis

    @pytest.mark.unit
    def test_basic_grid(self):
        grid = {"nx": 5, "ny": 3, "x0_m": 0, "y0_m": 100, "dx_m": 10, "dy_m": -10}
        x, y = self.build_target_axis(grid)
        assert len(x) == 5
        assert len(y) == 3
        np.testing.assert_array_equal(x, [0, 10, 20, 30, 40])
        np.testing.assert_array_equal(y, [100, 90, 80])


class TestResolveIndices:
    """Tests for resolve_indices from the velocity script."""

    @pytest.fixture(autouse=True)
    def _load(self, velocity_module):
        self.resolve = velocity_module.resolve_indices

    @pytest.mark.unit
    def test_aligned_grids(self):
        target = np.array([0.0, 100.0, 200.0])
        idx, valid = self.resolve(target, src0=0.0, src_step=100.0, src_count=3)
        np.testing.assert_array_equal(idx, [0, 1, 2])
        assert all(valid)

    @pytest.mark.unit
    def test_out_of_bounds_marked_invalid(self):
        target = np.array([-100.0, 0.0, 100.0, 500.0])
        idx, valid = self.resolve(target, src0=0.0, src_step=100.0, src_count=3)
        assert not valid[0]  # -100 out of bounds
        assert valid[1]
        assert valid[2]
        assert not valid[3]  # 500 out of bounds

    @pytest.mark.unit
    def test_indices_clipped_safely(self):
        target = np.array([-1000.0, 5000.0])
        idx, valid = self.resolve(target, src0=0.0, src_step=10.0, src_count=100)
        # All out of bounds but indices should be safe (clipped)
        assert np.all(idx >= 0)
        assert np.all(idx < 100)
