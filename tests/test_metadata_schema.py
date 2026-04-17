"""Validate that all .meta.json files conform to the expected schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "static" / "tools" / "data"


def collect_meta_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*.meta.json"))


@pytest.fixture(params=collect_meta_files(), ids=lambda p: p.stem)
def meta_path(request) -> Path:
    return request.param


@pytest.fixture
def meta(meta_path) -> dict:
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.unit
class TestMetadataSchema:
    """Validate structural integrity of every .meta.json file."""

    def test_is_valid_json(self, meta_path):
        with meta_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_has_grid_or_streamline_geometry(self, meta):
        # Streamline files use geometry_type instead of a grid section
        if meta.get("geometry_type") in ("streamlines_3d",):
            assert "sampling" in meta
            return
        assert "grid" in meta
        grid = meta["grid"]
        for key in ("nx", "ny", "x0_m", "y0_m", "dx_m", "dy_m"):
            assert key in grid, f"Missing grid key: {key}"
            assert isinstance(grid[key], (int, float)), f"grid.{key} should be numeric"

    def test_grid_dimensions_positive(self, meta):
        if "grid" not in meta:
            pytest.skip("streamline geometry, no grid")
        assert meta["grid"]["nx"] > 0
        assert meta["grid"]["ny"] > 0

    def test_has_fields_array(self, meta):
        assert "fields" in meta
        assert isinstance(meta["fields"], list)
        assert len(meta["fields"]) > 0

    def test_fields_have_name(self, meta):
        for field in meta["fields"]:
            assert "name" in field, f"Field missing 'name': {field}"
            assert isinstance(field["name"], str)

    def test_binary_fields_have_dtype_and_offsets(self, meta):
        for field in meta["fields"]:
            if "dtype" not in field:
                continue  # computed fields (e.g. speed) may lack dtype
            assert field["dtype"] in ("int16", "uint8", "uint16", "float32"), (
                f"Unexpected dtype: {field['dtype']}"
            )
            assert "byte_offset" in field
            assert "byte_length" in field
            assert field["byte_length"] > 0

    def test_has_quantization_or_alternative(self, meta):
        # Streamline files don't use standard quantization
        if meta.get("geometry_type") in ("streamlines_3d",):
            return
        if "quantization" not in meta:
            return
        q = meta["quantization"]
        # Standard quantization has unit+scale, but some products use
        # domain-specific keys (e.g. effective_pressure_unit)
        has_standard = "unit" in q and "scale" in q
        has_domain_specific = any(k.endswith("_unit") for k in q)
        assert has_standard or has_domain_specific, (
            f"quantization section lacks unit/scale or domain-specific keys: {list(q.keys())}"
        )

    def test_has_title(self, meta):
        assert "title" in meta or "product_version" in meta


@pytest.mark.unit
class TestMetadataConsistency:
    """Cross-field consistency checks."""

    def test_field_offsets_do_not_overlap(self, meta):
        binary_fields = [f for f in meta["fields"] if "byte_offset" in f and "byte_length" in f]
        if len(binary_fields) < 2:
            return
        sorted_fields = sorted(binary_fields, key=lambda f: f["byte_offset"])
        for i in range(len(sorted_fields) - 1):
            end = sorted_fields[i]["byte_offset"] + sorted_fields[i]["byte_length"]
            next_start = sorted_fields[i + 1]["byte_offset"]
            assert end <= next_start, (
                f"Fields overlap: {sorted_fields[i]['name']} ends at {end}, "
                f"{sorted_fields[i+1]['name']} starts at {next_start}"
            )

    def test_grid_cell_count_matches_first_field_size(self, meta):
        if "grid" not in meta:
            pytest.skip("no grid section (streamline geometry)")
        grid = meta["grid"]
        cell_count = grid["nx"] * grid["ny"]
        # Only check the first grid-aligned field; channel/segment fields
        # have a different element count by design.
        for field in meta["fields"]:
            if "byte_length" not in field or "dtype" not in field:
                continue
            dtype = field["dtype"]
            bpe = {"int16": 2, "uint8": 1, "uint16": 2, "float32": 4}.get(dtype)
            if bpe is None:
                continue
            expected = cell_count * bpe
            if field["byte_length"] == expected:
                return  # at least one grid-aligned field found
        # If no field matched, that's acceptable for products with only
        # non-grid-aligned data (e.g. channel segments in hydrology)
        has_non_grid = any(
            "channel" in f.get("name", "") or "segment" in f.get("name", "")
            for f in meta["fields"]
        )
        if not has_non_grid:
            pytest.fail("No field matched expected grid cell count")
