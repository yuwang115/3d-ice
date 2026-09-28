"""Unit tests for packaging the published isostatic response (Paxman et al. 2022, grids v3)."""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

EUSTATIC_M = 65.3
SOURCE_STEP_M = 500
SOURCE_N = 41


def _write_terrain_package(directory: Path, basename: str, grid: dict, mask: np.ndarray) -> None:
    """A minimal terrain package: only the mask is read by the response packager."""
    cells = grid["nx"] * grid["ny"]
    assert mask.size == cells
    placeholder = np.zeros(cells, dtype="<i2")
    fields = []
    offset = 0
    payload = b""
    for name in ("bed", "surface", "thickness"):
        fields.append({"name": name, "dtype": "int16", "byte_offset": offset, "byte_length": placeholder.nbytes})
        payload += placeholder.tobytes()
        offset += placeholder.nbytes
    fields.append({"name": "mask", "dtype": "uint8", "byte_offset": offset, "byte_length": mask.size})
    payload += mask.astype(np.uint8).tobytes()
    meta = {"title": basename, "grid": grid, "fields": fields}
    (directory / f"{basename}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / f"{basename}.bin").write_bytes(payload)


def _synthetic_fields(ny: int, nx: int) -> dict[str, np.ndarray]:
    """Fields that satisfy the published identities T = R - G and T = IU + LGM + WL.

    The outer ring is distal ocean, where the ice-unloading term reduces to the eustatic
    sea-surface rise alone, exactly as in the published grids.
    """
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    centre = (ny - 1) / 2
    radius = np.hypot(yy - centre, xx - centre)
    solid = np.where(radius < 12, 900.0 - 60.0 * radius, -20.0 - 0.01 * radius)
    residual_geoid = 20.0 + 0.1 * xx
    ssh = EUSTATIC_M + residual_geoid
    topography = solid - ssh
    post_lgm = 5.0 + 0.2 * yy
    water = np.where(radius < 12, -3.0, -36.0)
    ice = topography - post_lgm - water
    distal = radius >= 12
    ice[distal] = -EUSTATIC_M
    # Keep T = IU + LGM + WL exact in the distal ring by absorbing the residual in WL.
    water[distal] = topography[distal] - post_lgm[distal] - ice[distal]
    spread = 10.0 + 0.5 * radius
    return {
        "solid surface change": solid,
        "ssh change": ssh,
        "topography change": topography,
        "standard deviation": spread,
        "ice unloading": ice,
        "post-LGM disequilibrium": post_lgm,
        "water loading": water,
    }


def _write_source(
    path: Path, fields: dict[str, np.ndarray], *, x0: float, y0: float, version: str | None = "23-Jan-2026"
) -> None:
    ny, nx = next(iter(fields.values())).shape
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)
        x = ds.createVariable("x", "f4", ("x",))
        y = ds.createVariable("y", "f4", ("y",))
        x[:] = x0 + SOURCE_STEP_M * np.arange(nx)
        y[:] = y0 - SOURCE_STEP_M * np.arange(ny)
        for name, values in fields.items():
            variable = ds.createVariable(name, "f4", ("y", "x"))
            variable[:] = values.astype(np.float32)
        if version is not None:
            ds.setncattr("Version", version)


@pytest.mark.unit
class TestPrepareIsostaticResponse:
    @pytest.fixture(autouse=True)
    def _load(self, isostatic_response_module):
        self.module = isostatic_response_module

    # ------------------------------------------------------------------ sampling

    def test_source_indices_land_on_identical_nodes_for_a_stride_subset(self):
        source = -3333000.0 + 500.0 * np.arange(13333)
        target = -3333000.0 + 10000.0 * np.arange(667)

        indices, offset = self.module.source_indices(source, target)

        np.testing.assert_array_equal(indices, 20 * np.arange(667))
        assert offset == 0.0

    def test_source_indices_pick_the_same_nodes_as_the_greenland_terrain_sampler(
        self, bedmachine_greenland_module
    ):
        source = -652925.0 + 150.0 * np.arange(10218)
        _, terrain_index = bedmachine_greenland_module.build_axis_sampling(source, 1000)
        target = -652925.0 + 1000.0 * np.arange(terrain_index.size)

        indices, offset = self.module.source_indices(source, target)

        np.testing.assert_array_equal(indices, terrain_index)
        assert 0 < offset <= 75.0

    def test_source_indices_follow_a_descending_axis(self):
        source = 3333000.0 - 500.0 * np.arange(101)
        target = np.array([3333000.0, 3323000.0, 3283000.0])

        indices, offset = self.module.source_indices(source, target)

        np.testing.assert_array_equal(indices, [0, 20, 100])
        assert offset == 0.0

    def test_source_indices_reject_nodes_outside_the_source_grid(self):
        source = 500.0 * np.arange(10)
        with pytest.raises(ValueError, match="outside the published grid"):
            self.module.source_indices(source, np.array([0.0, 5000.0]))

    # ------------------------------------------------------------------ validation

    def test_validate_decomposition_reports_the_published_identities(self):
        fields = {
            self.module.PACKAGE_NAMES[name]: values
            for name, values in _synthetic_fields(9, 9).items()
        }

        residuals = self.module.validate_decomposition(fields)

        assert residuals["topography_minus_solid_plus_ssh_m"] < 1e-9
        assert residuals["topography_minus_components_m"] < 1e-9

    def test_validate_decomposition_rejects_inconsistent_fields(self):
        fields = {
            self.module.PACKAGE_NAMES[name]: values
            for name, values in _synthetic_fields(9, 9).items()
        }
        fields["ssh_change"] = fields["ssh_change"] + 1.0

        with pytest.raises(ValueError, match="T = R - G"):
            self.module.validate_decomposition(fields)

    def test_estimate_eustatic_rise_reads_the_distal_ice_unloading_term(self):
        fields = _synthetic_fields(SOURCE_N, SOURCE_N)
        centre = (SOURCE_N - 1) / 2
        yy, xx = np.mgrid[0:SOURCE_N, 0:SOURCE_N]
        mask = np.where(np.hypot(yy - centre, xx - centre) < 12, 2, 0).astype(np.uint8)

        estimate = self.module.estimate_eustatic_rise(fields["ice unloading"], mask)

        assert estimate == pytest.approx(EUSTATIC_M, abs=0.05)

    # ------------------------------------------------------------------ quantization

    def test_quantize_decimetres_rounds_half_to_even_and_maps_nan_to_fill(self):
        codes = self.module.quantize_decimetres(np.array([0.04, 0.05, 0.15, -12.34, np.nan]))

        np.testing.assert_array_equal(codes, [0, 0, 2, -123, -32768])
        assert codes.dtype == np.int16

    def test_quantize_decimetres_refuses_values_the_codes_cannot_hold(self):
        with pytest.raises(ValueError, match="int16"):
            self.module.quantize_decimetres(np.array([3300.0]))

    # ------------------------------------------------------------------ packaging

    def _prepare(self, tmp_path, *, terrain_x0=-10000.0, terrain_y0=10000.0, stride=4, version="23-Jan-2026"):
        fields = _synthetic_fields(SOURCE_N, SOURCE_N)
        source_path = tmp_path / "source.nc"
        _write_source(source_path, fields, x0=-10000.0, y0=10000.0, version=version)

        count = (SOURCE_N - 1) // stride + 1
        grid = {
            "nx": count,
            "ny": count,
            "x0_m": int(terrain_x0),
            "y0_m": int(terrain_y0),
            "dx_m": SOURCE_STEP_M * stride,
            "dy_m": -SOURCE_STEP_M * stride,
        }
        yy, xx = np.mgrid[0:count, 0:count]
        centre = (count - 1) / 2
        mask = np.where(np.hypot(yy - centre, xx - centre) * stride < 12, 2, 0).astype(np.uint8)
        _write_terrain_package(tmp_path, "terrain", grid, mask.ravel())

        target = self.module.Target(
            terrain_basename="terrain",
            output_basename="terrain_isostatic_response",
            source_key="antarctica_bedmachine",
            label="Synthetic (2 km)",
        )
        meta = self.module.prepare_target(
            data_dir=tmp_path, target=target, source_path=source_path, verify_checksum=False
        )
        return meta, fields, grid, stride

    def test_prepare_target_writes_a_package_on_the_terrain_grid(self, tmp_path):
        meta, fields, grid, stride = self._prepare(tmp_path)

        assert meta["grid"] == grid
        payload = (tmp_path / "terrain_isostatic_response.bin").read_bytes()
        stored = [field for field in meta["fields"] if "dtype" in field]
        assert [field["name"] for field in stored] == [
            "topography_change",
            "solid_surface_change",
            "standard_deviation",
        ]
        cursor = 0
        for field in stored:
            assert field["dtype"] == "int16"
            assert field["scale"] == pytest.approx(0.1)
            assert field["offset"] == 0.0
            assert field["fill_value"] == -32768
            assert field["unit"] == "m"
            assert field["byte_offset"] == cursor
            assert field["byte_length"] == 2 * grid["nx"] * grid["ny"]
            cursor += field["byte_length"]
        assert cursor == len(payload)

        # Every stored value is the published value at the same node, to within half a step.
        for field, source_name in zip(
            stored, ("topography change", "solid surface change", "standard deviation")
        ):
            codes = np.frombuffer(
                payload, dtype="<i2", count=grid["nx"] * grid["ny"], offset=field["byte_offset"]
            ).reshape(grid["ny"], grid["nx"])
            expected = fields[source_name][::stride, ::stride]
            np.testing.assert_allclose(codes * 0.1, expected, atol=0.05 + 1e-6)
            assert field["stats_m"]["max"] == pytest.approx(float(expected.max()), abs=1e-4)
            assert field["stats_m"]["min"] == pytest.approx(float(expected.min()), abs=1e-4)
            assert field["stats_m"]["mean"] == pytest.approx(float(expected.mean()), abs=1e-4)

    def test_prepare_target_records_provenance_and_the_earth_model(self, tmp_path):
        meta, *_ = self._prepare(tmp_path)

        dataset = meta["source_dataset"]
        assert dataset["doi"] == "10.18739/A22Z12R8C"
        assert dataset["url"] == "https://doi.org/10.18739/A22Z12R8C"
        assert "Arctic Data Center" in dataset["citation"]
        assert meta["license"] == "CC-BY-4.0"
        assert "10.1038/s41598-022-15440-y" in meta["reference"]
        assert meta["source_file"] == "Total_Isostatic_Adjustment_Antarctica_BedMachinev4.nc"
        assert meta["product_version"].startswith("Grid files version 3")
        assert meta["source_grid_version"] == "23-Jan-2026"
        assert meta["source_package"]["metadata"] == "terrain.meta.json"
        assert meta["resampling"]["max_node_offset_m"] == 0.0

        earth = meta["earth_model"]
        assert earth["eustatic_sea_level_rise_m"] == pytest.approx(EUSTATIC_M)
        assert earth["mantle_density_kg_m3"] == 3330
        assert earth["seawater_density_kg_m3"] == 1028
        assert earth["ice_density_kg_m3"] == 917
        assert "Swain" in earth["effective_elastic_thickness"]

        assert meta["validation"]["eustatic_estimate_m"] == pytest.approx(EUSTATIC_M, abs=0.05)
        assert meta["quantization"] == {"unit": "m", "scale": 0.1, "offset": 0.0, "int16_fill_value": -32768}

    def test_prepare_target_summarises_the_components_under_grounded_ice(self, tmp_path):
        meta, fields, grid, stride = self._prepare(tmp_path)

        centre = (grid["ny"] - 1) / 2
        yy, xx = np.mgrid[0 : grid["ny"], 0 : grid["nx"]]
        grounded = np.hypot(yy - centre, xx - centre) * stride < 12
        summary = meta["grounded_ice_summary_m"]
        for key, source_name in (
            ("ice_unloading", "ice unloading"),
            ("post_lgm_disequilibrium", "post-LGM disequilibrium"),
            ("water_loading", "water loading"),
            ("ssh_change", "ssh change"),
            ("topography_change", "topography change"),
            ("solid_surface_change", "solid surface change"),
            ("standard_deviation", "standard deviation"),
        ):
            values = fields[source_name][::stride, ::stride][grounded]
            assert summary[key]["mean"] == pytest.approx(float(values.mean()), abs=1e-4), key
            assert summary[key]["max"] == pytest.approx(float(values.max()), abs=1e-4), key
            assert summary[key]["min"] == pytest.approx(float(values.min()), abs=1e-4), key
        assert summary["cell_count"] == int(grounded.sum())

    def test_prepare_target_refuses_a_terrain_grid_the_source_does_not_cover(self, tmp_path):
        with pytest.raises(ValueError, match="outside the published grid"):
            self._prepare(tmp_path, terrain_x0=-12000.0)

    def test_prepare_target_refuses_grid_files_of_another_version(self, tmp_path):
        # The package would otherwise claim version-3 provenance for, say, a v2 file
        # passed with --skip-md5; the physical identities hold in every version.
        with pytest.raises(ValueError, match="grid version"):
            self._prepare(tmp_path, version="15-Nov-2022")

    def test_prepare_target_refuses_grid_files_without_a_version(self, tmp_path):
        with pytest.raises(ValueError, match="grid version"):
            self._prepare(tmp_path, version=None)

    def test_every_source_names_the_grid_version_it_expects(self):
        assert self.module.SOURCES["antarctica_bedmachine"].grid_version == "23-Jan-2026"
        assert self.module.SOURCES["greenland_bedmachine"].grid_version == "23-Jan-2026"
        assert self.module.SOURCES["antarctica_bedmap3"].grid_version == "18-Mar-2025"

    def test_verify_md5_detects_a_corrupted_download(self, tmp_path):
        path = tmp_path / "file.nc"
        path.write_bytes(b"not the published file")

        with pytest.raises(ValueError, match="MD5"):
            self.module.verify_md5(path, "c153d1bcfae390a93dde352122d96844")

    def test_every_target_names_a_committed_terrain_package_and_a_known_source(self):
        data_dir = Path(__file__).resolve().parent.parent / "static" / "tools" / "data"
        for target in self.module.TARGETS:
            assert (data_dir / f"{target.terrain_basename}.meta.json").exists(), target
            assert target.source_key in self.module.SOURCES, target
