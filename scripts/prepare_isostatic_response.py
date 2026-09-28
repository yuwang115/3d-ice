#!/usr/bin/env python3
"""Package the published isostatic response to complete deglaciation for the explorer grids.

Paxman, Austermann & Hollyday (2022) computed the fully re-equilibrated response of the
solid Earth and the sea surface to removing both the Greenland and Antarctic ice sheets:
flexure of an elastic plate with laterally variable effective elastic thickness, a
correction for the post-LGM disequilibrium that is still to come (the mean of 24
self-gravitating viscoelastic Earth models), and the water-loading feedback. Version 3 of
their grid files is computed on the same BedMachine Antarctica v4, Bedmap3 and BedMachine
Greenland v6 grids that the explorer's terrain packages are point-sampled from, so this
script samples it at exactly the terrain nodes and writes one package per terrain package.

The grid files are 5.0-5.3 GB each and are not committed. Download them from the NSF
Arctic Data Center (https://doi.org/10.18739/A22Z12R8C); each is checked against the MD5
checksum the repository records before it is read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np


DATASET_DOI = "10.18739/A22Z12R8C"
DATASET_URL = f"https://doi.org/{DATASET_DOI}"
DATASET_OBJECT_URL = "https://arcticdata.io/metacat/d1/mn/v2/object/"
PRODUCT = "Total isostatic response to the complete unloading of the Greenland and Antarctic Ice Sheets"
PRODUCT_VERSION = "Grid files version 3 (NSF Arctic Data Center, published 2026-01-27)"
DATASET_CITATION = (
    "Paxman, G., Austermann, J. & Hollyday, A. (2026). Grid files of the total isostatic response "
    "to the complete unloading of the Greenland and Antarctic Ice Sheets (version 3). "
    f"NSF Arctic Data Center. {DATASET_URL}"
)
REFERENCE = (
    "Paxman, G. J. G., Austermann, J. & Hollyday, A. (2022). Total isostatic response to the "
    "complete unloading of the Greenland and Antarctic Ice Sheets. Scientific Reports 12, 11399. "
    "https://doi.org/10.1038/s41598-022-15440-y"
)
LICENSE = "CC-BY-4.0"
LICENSE_NOTE = (
    "The Arctic Data Center record licenses the dataset under CC BY 4.0; the NetCDF global "
    "attribute reads 'No restrictions on access or use'. Changes made here: point samples of "
    "the published grids at the terrain-package nodes, quantized to 0.1 m."
)

# Methods of Paxman et al. (2022): 57.9 m from Antarctica plus 7.42 m from Greenland. Both ice
# sheets are removed in every published grid, so the Greenland grids carry the Antarctic term
# and vice versa.
EUSTATIC_SEA_LEVEL_RISE_M = 65.3
EUSTATIC_TOLERANCE_M = 0.1
# The published components tile the total exactly; float32 storage leaves ~1e-4 m.
DECOMPOSITION_TOLERANCE_M = 0.01

CODES_PER_METRE = 10  # 0.1 m quantization step; multiplying avoids 0.15 / 0.1 = 1.4999...
SCALE_M = 1.0 / CODES_PER_METRE
INT16_FILL = -32768
INT16_MAX_CODE = 32767
GROUNDED_MASK_CODES = (2, 4)
OCEAN_MASK_CODE = 0

# Published variable name -> package key.
PACKAGE_NAMES = {
    "topography change": "topography_change",
    "solid surface change": "solid_surface_change",
    "standard deviation": "standard_deviation",
    "ice unloading": "ice_unloading",
    "post-LGM disequilibrium": "post_lgm_disequilibrium",
    "water loading": "water_loading",
    "ssh change": "ssh_change",
}

STORED_FIELDS = (
    (
        "topography_change",
        "Total isostatic response T = R - G: change in bed elevation relative to the sea surface. "
        "Present bed + T is the fully rebounded bed above the ice-free sea surface.",
    ),
    (
        "solid_surface_change",
        "Vertical displacement R of the solid Earth surface. Present bed + R is the rebounded bed "
        "above present-day sea level.",
    ),
    (
        "standard_deviation",
        "Standard deviation of the total isostatic response across the published Earth-model suite "
        "(four elastic-thickness models and 24 viscoelastic models).",
    ),
)

SWAIN_KIRBY_2021 = (
    "Swain, C. J. & Kirby, J. F. (2021). Effective elastic thickness map reveals subglacial structure "
    "of East Antarctica. Geophysical Research Letters 48, e2020GL091576. https://doi.org/10.1029/2020GL091576"
)
STEFFEN_2018 = (
    "Steffen, R., Audet, P. & Lund, B. (2018). Weakened lithosphere beneath Greenland inferred from "
    "effective elastic thickness: a hot spot effect? Geophysical Research Letters 45, 4733-4742. "
    "https://doi.org/10.1029/2017GL076885"
)


@dataclass(frozen=True)
class Source:
    file_name: str
    object_id: str
    md5: str
    # The NetCDF global `Version` attribute of the version-3 file. Checked on every run, so
    # a file from another release cannot be packaged under version-3 provenance even when
    # --skip-md5 bypasses the checksum.
    grid_version: str
    terrain_product: str
    effective_elastic_thickness: str
    projection: str


@dataclass(frozen=True)
class Target:
    terrain_basename: str
    output_basename: str
    source_key: str
    label: str


SOURCES = {
    "antarctica_bedmachine": Source(
        file_name="Total_Isostatic_Adjustment_Antarctica_BedMachinev4.nc",
        object_id="urn:uuid:3a859542-675e-40c8-aa34-db5e05927c91",
        md5="c153d1bcfae390a93dde352122d96844",
        grid_version="23-Jan-2026",
        terrain_product="BedMachine Antarctica v4",
        effective_elastic_thickness=SWAIN_KIRBY_2021,
        projection="EPSG:3031",
    ),
    "antarctica_bedmap3": Source(
        file_name="Total_Isostatic_Adjustment_Bedmap3.nc",
        object_id="urn:uuid:ab853a15-a695-4331-990a-e094b447a9b2",
        md5="3d3bb35027ae749688ee3d798397c60a",
        grid_version="18-Mar-2025",
        terrain_product="Bedmap3",
        effective_elastic_thickness=SWAIN_KIRBY_2021,
        projection="EPSG:3031",
    ),
    "greenland_bedmachine": Source(
        file_name="Total_Isostatic_Adjustment_Greenland_BedMachinev6.nc",
        object_id="urn:uuid:0a3c7dfe-c39f-4376-a636-7dc57b49bc70",
        md5="79b45bd9ca7a65758cb392d69c0d5d96",
        grid_version="23-Jan-2026",
        terrain_product="BedMachine Greenland v6",
        effective_elastic_thickness=STEFFEN_2018,
        projection="EPSG:3413",
    ),
}

TARGETS = (
    Target("bedmachine_antarctica_v4_480", "antarctica_isostatic_response_480", "antarctica_bedmachine",
           "BedMachine Antarctica v4 - Balanced (10 km)"),
    Target("bedmachine_antarctica_v4_741", "antarctica_isostatic_response_741", "antarctica_bedmachine",
           "BedMachine Antarctica v4 - HD (4 km)"),
    Target("bedmap3_antarctica_10km", "bedmap3_antarctica_isostatic_response_10km", "antarctica_bedmap3",
           "Bedmap3 - Balanced (10 km)"),
    Target("bedmap3_antarctica_4km", "bedmap3_antarctica_isostatic_response_4km", "antarctica_bedmap3",
           "Bedmap3 - HD (4 km)"),
    Target("bedmachine_greenland_v6_3km", "greenland_isostatic_response_3km", "greenland_bedmachine",
           "BedMachine Greenland v6 - Balanced (3 km)"),
    Target("bedmachine_greenland_v6_1km", "greenland_isostatic_response_1km", "greenland_bedmachine",
           "BedMachine Greenland v6 - HD (1 km)"),
)

# ---------------------------------------------------------------------------- sampling


def normalize_grid(grid: dict[str, Any]) -> dict[str, int]:
    return {key: int(grid[key]) for key in ("nx", "ny", "x0_m", "y0_m", "dx_m", "dy_m")}


def source_indices(source_axis: np.ndarray, target_axis: np.ndarray) -> tuple[np.ndarray, float]:
    """Nearest published node for every terrain node, and the largest distance between them.

    The terrain packages are point samples of the same native grids, so for Antarctica every
    node coincides exactly; the Greenland 1 km package samples the 150 m grid at its nearest
    node, which this reproduces.
    """
    axis = np.asarray(source_axis, dtype=np.float64)
    targets = np.asarray(target_axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("The published coordinate axis must be 1-D with at least two values.")
    step = axis[1] - axis[0]
    if step == 0 or not np.allclose(np.diff(axis), step, rtol=0, atol=1e-3 * abs(step)):
        raise ValueError("The published coordinate axis is not uniformly spaced.")

    indices = np.rint((targets - axis[0]) / step).astype(np.int64)
    outside = (indices < 0) | (indices >= axis.size)
    if np.any(outside):
        raise ValueError(
            f"{int(outside.sum())} terrain nodes fall outside the published grid "
            f"({axis[0]:.0f} to {axis[-1]:.0f} m)."
        )
    offset = float(np.max(np.abs(axis[indices] - targets))) if targets.size else 0.0
    if offset > abs(step) / 2 + 1e-6:
        raise ValueError(f"Terrain nodes sit {offset:.1f} m from the nearest published node.")
    return indices, offset


def sample_variables(
    dataset: netCDF4.Dataset, row_index: np.ndarray, col_index: np.ndarray
) -> dict[str, np.ndarray]:
    """Read every published field at the selected nodes, one source row at a time.

    The fields are contiguous float32 arrays of up to 13,334 x 13,334 values; reading whole
    rows and discarding columns keeps memory to one row per field.
    """
    sampled: dict[str, np.ndarray] = {}
    for source_name, key in PACKAGE_NAMES.items():
        if source_name not in dataset.variables:
            raise ValueError(f"The published file has no '{source_name}' variable.")
        variable = dataset.variables[source_name]
        variable.set_auto_mask(False)
        fill = getattr(variable, "_FillValue", None)
        out = np.empty((row_index.size, col_index.size), dtype=np.float64)
        for out_row, source_row in enumerate(row_index):
            out[out_row] = np.asarray(variable[int(source_row), :], dtype=np.float64)[col_index]
        if fill is not None:
            out[out == float(fill)] = np.nan
        sampled[key] = out
    return sampled


# ---------------------------------------------------------------------------- validation


def validate_decomposition(
    fields: dict[str, np.ndarray], tolerance_m: float = DECOMPOSITION_TOLERANCE_M
) -> dict[str, float]:
    """Check the two identities the published grids satisfy, and report the residuals."""
    topography = fields["topography_change"]
    against_ssh = float(np.nanmax(np.abs(topography - (fields["solid_surface_change"] - fields["ssh_change"]))))
    components = fields["ice_unloading"] + fields["post_lgm_disequilibrium"] + fields["water_loading"]
    against_components = float(np.nanmax(np.abs(topography - components)))
    if against_ssh > tolerance_m:
        raise ValueError(f"The published fields break T = R - G by up to {against_ssh:.3f} m.")
    if against_components > tolerance_m:
        raise ValueError(
            "The published fields break T = ice unloading + post-LGM + water loading by up to "
            f"{against_components:.3f} m."
        )
    return {
        "topography_minus_solid_plus_ssh_m": against_ssh,
        "topography_minus_components_m": against_components,
    }


def estimate_eustatic_rise(ice_unloading: np.ndarray, mask: np.ndarray) -> float:
    """The eustatic term as the grids carry it: the ice-unloading term over distal ocean.

    Far from the ice the flexural response has decayed, so the published ice-unloading term
    (a topography change, R - G) reduces to minus the eustatic sea-surface rise over most of
    the open ocean. Its modal value to 0.1 m therefore recovers the rise the grids used.
    """
    values = np.asarray(ice_unloading, dtype=np.float64)
    ocean = (np.asarray(mask) == OCEAN_MASK_CODE) & np.isfinite(values)
    if not np.any(ocean):
        raise ValueError("The terrain package has no ocean nodes to read the eustatic term from.")
    rounded = np.round(values[ocean] * CODES_PER_METRE) / CODES_PER_METRE
    levels, counts = np.unique(rounded, return_counts=True)
    return float(-levels[np.argmax(counts)])


def verify_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"MD5 of {path.name} is {actual}, but the Arctic Data Center records {expected}; "
            "the download is incomplete or is a different version."
        )


# ---------------------------------------------------------------------------- packaging


def quantize_decimetres(values: np.ndarray) -> np.ndarray:
    """int16 codes at 0.1 m, rounding half to even, with NaN mapped to the fill code."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    codes = np.full(array.shape, INT16_FILL, dtype=np.int16)
    scaled = np.rint(array[finite] * CODES_PER_METRE)
    if scaled.size and np.max(np.abs(scaled)) > INT16_MAX_CODE:
        raise ValueError(
            f"Values up to {np.max(np.abs(array[finite])):.1f} m exceed the int16 range at 0.1 m."
        )
    codes[finite] = scaled.astype(np.int16)
    return codes


def summarise(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}


def grounded_summary(fields: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    grounded = np.isin(mask, GROUNDED_MASK_CODES)
    summary: dict[str, Any] = {
        "note": (
            "Statistics over the terrain package's grounded-ice nodes (mask 2 or 4). The three "
            "components are topography-type terms that sum to topography_change, as in Table 1 "
            "of Paxman et al. (2022)."
        ),
        "cell_count": int(grounded.sum()),
    }
    for key in PACKAGE_NAMES.values():
        summary[key] = summarise(fields[key][grounded])
    return summary


def read_terrain_package(data_dir: Path, basename: str) -> tuple[dict[str, Any], np.ndarray]:
    meta = json.loads((data_dir / f"{basename}.meta.json").read_text(encoding="utf-8"))
    grid = normalize_grid(meta["grid"])
    field = next((item for item in meta["fields"] if item["name"] == "mask"), None)
    if field is None or field.get("dtype") != "uint8":
        raise ValueError(f"{basename} has no uint8 mask field.")
    payload = (data_dir / f"{basename}.bin").read_bytes()
    count = grid["nx"] * grid["ny"]
    if int(field["byte_length"]) != count:
        raise ValueError(f"{basename}: mask does not cover the declared grid.")
    mask = np.frombuffer(payload, dtype=np.uint8, count=count, offset=int(field["byte_offset"]))
    return meta, mask.reshape(grid["ny"], grid["nx"]).copy()


def package_metadata(
    *,
    target: Target,
    source: Source,
    grid: dict[str, int],
    grid_version: str | None,
    max_node_offset_m: float,
    fields: dict[str, np.ndarray],
    mask: np.ndarray,
    residuals: dict[str, float],
    eustatic_estimate_m: float,
    md5_verified: bool,
) -> dict[str, Any]:
    stored = []
    for name, description in STORED_FIELDS:
        stored.append(
            {
                "name": name,
                "dtype": "int16",
                "unit": "m",
                "scale": SCALE_M,
                "offset": 0.0,
                "fill_value": INT16_FILL,
                "description": description,
                "stats_m": summarise(fields[name]),
            }
        )
    valid = int(np.isfinite(fields["topography_change"]).sum())
    return {
        "title": f"Total isostatic response to complete deglaciation on the {target.label} grid",
        "product": PRODUCT,
        "product_version": PRODUCT_VERSION,
        "source_grid_version": grid_version,
        "source_file": source.file_name,
        "source_dataset": {
            "doi": DATASET_DOI,
            "url": DATASET_URL,
            "citation": DATASET_CITATION,
            "object_url": f"{DATASET_OBJECT_URL}{source.object_id}",
            "md5": source.md5,
            "md5_verified": md5_verified,
        },
        "reference": REFERENCE,
        "source_url": DATASET_URL,
        "license": LICENSE,
        "license_note": LICENSE_NOTE,
        "projection": source.projection,
        "grid": grid,
        "earth_model": {
            "method": (
                "Fully re-equilibrated response to removing both ice sheets: flexure of an elastic "
                "plate over an inviscid mantle with laterally variable effective elastic thickness, "
                "plus the post-LGM disequilibrium still to come and the water-loading feedback."
            ),
            "terrain_product": source.terrain_product,
            "effective_elastic_thickness": source.effective_elastic_thickness,
            "young_modulus_pa": 1.0e11,
            "poisson_ratio": 0.25,
            "ice_density_kg_m3": 917,
            "seawater_density_kg_m3": 1028,
            "mantle_density_kg_m3": 3330,
            "post_lgm_disequilibrium": (
                "Mean of 24 self-gravitating viscoelastic Earth models (lower-mantle viscosity "
                "0.3-3e22 Pa s, upper-mantle viscosity 3-5e20 Pa s, elastic lithosphere 71 or "
                "96 km) driven by ICE-6G_C deglaciation after a eustatic build-up from 122 ka."
            ),
            "eustatic_sea_level_rise_m": EUSTATIC_SEA_LEVEL_RISE_M,
            "eustatic_note": "57.9 m from Antarctica plus 7.42 m from Greenland: both ice sheets are removed.",
            "sea_surface": (
                "ssh_change G is the eustatic rise plus the residual post-LGM geoid change; "
                "topography_change T = R - G, so present bed + T is elevation above the ice-free "
                "sea surface and present bed + R is elevation above present-day sea level."
            ),
            "water_loading": (
                "Seawater loads every node whose rebounded bed lies below the ice-free sea "
                "surface; the paper describes no ocean-connectivity test."
            ),
        },
        "source_package": {
            "metadata": f"{target.terrain_basename}.meta.json",
            "binary": f"{target.terrain_basename}.bin",
            "grid": grid,
        },
        "resampling": {
            "method": "nearest published node (point sample) at every terrain-package node",
            "max_node_offset_m": max_node_offset_m,
            "reason": (
                "The published grids share the terrain products' native grids, and the terrain "
                "packages are point samples of those grids, so the response is sampled at the "
                "same nodes rather than averaged."
            ),
        },
        "validation": {
            **residuals,
            "eustatic_estimate_m": eustatic_estimate_m,
        },
        "coverage": {"cell_count": int(fields["topography_change"].size), "valid_count": valid},
        "grounded_ice_summary_m": grounded_summary(fields, mask),
        "quantization": {"unit": "m", "scale": SCALE_M, "offset": 0.0, "int16_fill_value": INT16_FILL},
        "fields": stored,
    }


def write_package(
    data_dir: Path, basename: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Write the payload and metadata; returns the metadata with the byte layout filled in."""
    output = copy.deepcopy(meta)
    offset = 0
    for field in output["fields"]:
        values = arrays[field["name"]]
        field["byte_offset"] = offset
        field["byte_length"] = int(values.size * np.dtype("<i2").itemsize)
        offset += field["byte_length"]
    with (data_dir / f"{basename}.bin").open("wb") as handle:
        for field in output["fields"]:
            handle.write(arrays[field["name"]].astype("<i2", copy=False).tobytes(order="C"))
    (data_dir / f"{basename}.meta.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def prepare_target(
    *,
    data_dir: Path,
    target: Target,
    source_path: Path,
    verify_checksum: bool = True,
    md5_verified: bool = False,
) -> dict[str, Any]:
    """Sample one published grid at one terrain package's nodes and write the package.

    `md5_verified` records a checksum the caller already confirmed for this file, so a
    source shared by several targets is hashed once.
    """
    source = SOURCES[target.source_key]
    if verify_checksum:
        verify_md5(source_path, source.md5)
        md5_verified = True

    terrain_meta, mask = read_terrain_package(data_dir, target.terrain_basename)
    grid = normalize_grid(terrain_meta["grid"])
    target_x = grid["x0_m"] + grid["dx_m"] * np.arange(grid["nx"], dtype=np.float64)
    target_y = grid["y0_m"] + grid["dy_m"] * np.arange(grid["ny"], dtype=np.float64)

    with netCDF4.Dataset(source_path) as dataset:
        dataset.set_auto_mask(False)
        col_index, x_offset = source_indices(np.asarray(dataset.variables["x"][:]), target_x)
        row_index, y_offset = source_indices(np.asarray(dataset.variables["y"][:]), target_y)
        grid_version = str(dataset.getncattr("Version")) if "Version" in dataset.ncattrs() else None
        if grid_version != source.grid_version:
            raise ValueError(
                f"{source_path.name} reports grid version {grid_version!r}, but the version-3 "
                f"{source.file_name} is dated {source.grid_version!r}; download the current release."
            )
        fields = sample_variables(dataset, row_index, col_index)

    residuals = validate_decomposition(fields)
    eustatic = estimate_eustatic_rise(fields["ice_unloading"], mask)
    if abs(eustatic - EUSTATIC_SEA_LEVEL_RISE_M) > EUSTATIC_TOLERANCE_M:
        raise ValueError(
            f"The distal ice-unloading term implies a {eustatic:.2f} m eustatic rise, not the "
            f"{EUSTATIC_SEA_LEVEL_RISE_M} m the paper states; check the source version."
        )

    arrays = {name: quantize_decimetres(fields[name]).ravel() for name, _ in STORED_FIELDS}
    meta = package_metadata(
        target=target,
        source=source,
        grid=grid,
        grid_version=grid_version,
        max_node_offset_m=max(x_offset, y_offset),
        fields=fields,
        mask=mask,
        residuals=residuals,
        eustatic_estimate_m=eustatic,
        md5_verified=md5_verified,
    )
    return write_package(data_dir, target.output_basename, meta, arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--antarctica-bedmachine", type=Path, help=SOURCES["antarctica_bedmachine"].file_name)
    parser.add_argument("--antarctica-bedmap3", type=Path, help=SOURCES["antarctica_bedmap3"].file_name)
    parser.add_argument("--greenland-bedmachine", type=Path, help=SOURCES["greenland_bedmachine"].file_name)
    parser.add_argument("--data-dir", type=Path, default=Path("static/tools/data"))
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip the checksum (for a local copy already verified against the repository).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # argparse turns --antarctica-bedmachine into antarctica_bedmachine, the source key.
    paths = {key: getattr(args, key) for key in SOURCES}
    if not any(paths.values()):
        raise SystemExit("Pass at least one published grid file; see --help.")
    for key, path in paths.items():
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        if not args.skip_md5:
            verify_md5(path, SOURCES[key].md5)
    for target in TARGETS:
        source_path = paths[target.source_key]
        if source_path is None:
            continue
        meta = prepare_target(
            data_dir=args.data_dir,
            target=target,
            source_path=source_path,
            verify_checksum=False,
            md5_verified=not args.skip_md5,
        )
        summary = meta["grounded_ice_summary_m"]
        print(
            f"{target.output_basename}: T max {summary['topography_change']['max']:.1f} m, "
            f"R max {summary['solid_surface_change']['max']:.1f} m over grounded ice; "
            f"residuals {meta['validation']['topography_minus_components_m']:.1e} m"
        )


if __name__ == "__main__":
    main()
