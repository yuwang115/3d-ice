"""Tests for decode_attr and get_fill_value functions."""

from __future__ import annotations

import numpy as np
import pytest

FILL_FLOAT = -9999.0


class TestDecodeAttrAntarctica:
    """Tests for the basic decode_attr variant."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_antarctica_module):
        self.decode_attr = bedmachine_antarctica_module.decode_attr

    @pytest.mark.unit
    def test_none_returns_default(self):
        assert self.decode_attr(None) == ""
        assert self.decode_attr(None, "fallback") == "fallback"

    @pytest.mark.unit
    def test_bytes_decoded(self):
        assert self.decode_attr(b"hello") == "hello"

    @pytest.mark.unit
    def test_string_passthrough(self):
        assert self.decode_attr("world") == "world"

    @pytest.mark.unit
    def test_number_stringified(self):
        assert self.decode_attr(42) == "42"

    @pytest.mark.unit
    def test_utf8_bytes(self):
        assert self.decode_attr("BedMachine".encode("utf-8")) == "BedMachine"


class TestDecodeAttrGreenland:
    """Tests for the Greenland variant (handles numpy 0-d arrays)."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_greenland_module):
        self.decode_attr = bedmachine_greenland_module.decode_attr

    @pytest.mark.unit
    def test_numpy_scalar_bytes(self):
        val = np.array(b"v6", dtype=object)
        assert self.decode_attr(val) == "v6"

    @pytest.mark.unit
    def test_numpy_scalar_string(self):
        val = np.array("hello")
        assert self.decode_attr(val) == "hello"

    @pytest.mark.unit
    def test_none_returns_default(self):
        assert self.decode_attr(None, "default") == "default"


class TestGetFillValue:
    """Tests for get_fill_value from the Greenland script."""

    @pytest.fixture(autouse=True)
    def _load(self, bedmachine_greenland_module):
        self.get_fill_value = bedmachine_greenland_module.get_fill_value

    @pytest.mark.unit
    def test_returns_fallback_when_no_attr(self):
        class MockDS:
            attrs = {}

        assert self.get_fill_value(MockDS(), fallback=-9999.0) == -9999.0

    @pytest.mark.unit
    def test_extracts_scalar(self):
        class MockDS:
            attrs = {"_FillValue": -32768.0}

        assert self.get_fill_value(MockDS()) == -32768.0

    @pytest.mark.unit
    def test_extracts_from_numpy_array(self):
        class MockDS:
            attrs = {"_FillValue": np.array([-9999.0])}

        assert self.get_fill_value(MockDS()) == -9999.0

    @pytest.mark.unit
    def test_empty_array_returns_fallback(self):
        class MockDS:
            attrs = {"_FillValue": np.array([])}

        assert self.get_fill_value(MockDS(), fallback=-1.0) == -1.0
