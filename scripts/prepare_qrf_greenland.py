#!/usr/bin/env python3
"""Prepare QRF 2025 Greenland hybrid terrain packages for the browser.

The QRF source supplies predicted bed elevation, ice thickness, and prediction
uncertainty, but no surface elevation or ice/ocean mask.  This preparer places
the QRF bed on the established BedMachine Greenland v6 viewer grids, keeps the
v6 surface and mask, and derives thickness from the combined fields.  It uses
BedMachine bed/thickness where the QRF is not scientifically applicable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

FILL_INT16 = -32768
# The GeoTIFF tiepoint locates the upper-left *edge* of its first pixel.  The
# viewer grids describe sample centres, so sampling must use a half-cell shift.
QRF_GEOTIFF_TIEPOINT = (-635900.0, -665900.0)
QRF_GEOTIFF_PIXEL_SCALE = (300.0, 300.0)
QRF_SOURCE_GRID = {
    "nx": 4950,
    "ny": 8954,
    "x0_m": int(QRF_GEOTIFF_TIEPOINT[0] + QRF_GEOTIFF_PIXEL_SCALE[0] / 2),
    "y0_m": int(QRF_GEOTIFF_TIEPOINT[1] - QRF_GEOTIFF_PIXEL_SCALE[1] / 2),
    "dx_m": 300,
    "dy_m": -300,
}
QRF_BAND_DESCRIPTIONS = (
    "qrf_pred_ice_thickness",
    "qrf_pred_bed_elev",
    "qrf_pred_uncertainty_SD",
)
QRF_SOURCE_URL = (
    "https://raw.githubusercontent.com/charliekirkwood/greenlandice/"
    "5fba8ad8332752ed3b780f15ac5f70580fe0acaf/QRF_greenland_ice_predictions_300m.tif"
)
QRF_SOURCE_COMMIT = "5fba8ad8332752ed3b780f15ac5f70580fe0acaf"
QRF_REFERENCE = "Palmer et al. (2025), Journal of Glaciology, https://doi.org/10.1017/jog.2025.10071"


@dataclass(frozen=True)
class Target:
    label: str
    base_basename: str
    output_basename: str


TARGETS = (
    Target("QRF 2025 — Balanced (3 km)", "bedmachine_greenland_v6_3km", "greenland_qrf_2025_3km"),
    Target("QRF 2025 — HD (1 km)", "bedmachine_greenland_v6_1km", "greenland_qrf_2025_1km"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare QRF 2025 Greenland terrain packages.")
    parser.add_argument("--input", required=True, help="Path to QRF_greenland_ice_predictions_300m.tif.")
    parser.add_argument("--data-dir", default="static/tools/data", help="Viewer data-package directory.")
    return parser.parse_args()


def normalize_grid(grid: dict[str, Any]) -> dict[str, int]:
    return {
        "nx": int(grid["nx"]),
        "ny": int(grid["ny"]),
        "x0_m": int(grid["x0_m"]),
        "y0_m": int(grid["y0_m"]),
        "dx_m": int(grid["dx_m"]),
        "dy_m": int(grid["dy_m"]),
    }


def axis_for_grid(grid: dict[str, int], axis: str) -> np.ndarray:
    if axis == "x":
        count, origin, step = grid["nx"], grid["x0_m"], grid["dx_m"]
    elif axis == "y":
        count, origin, step = grid["ny"], grid["y0_m"], grid["dy_m"]
    else:
        raise ValueError(f"Unsupported axis {axis!r}.")
    return origin + np.arange(count, dtype=np.float64) * step


def resolve_indices(
    target_axis: np.ndarray, src0: float, src_step: float, src_count: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.rint((target_axis - src0) / src_step).astype(np.int64)
    valid = (indices >= 0) & (indices < src_count)
    return np.clip(indices, 0, src_count - 1), valid


def sample_qrf_band(
    source: np.ndarray,
    *,
    source_grid: dict[str, int],
    target_grid: dict[str, int],
) -> np.ndarray:
    """Resample a QRF 300 m band to a viewer grid by nearest projected cell."""
    if source.shape != (source_grid["ny"], source_grid["nx"]):
        raise ValueError(f"QRF source shape {source.shape} does not match source-grid metadata.")
    source_cols, valid_cols = resolve_indices(
        axis_for_grid(target_grid, "x"), source_grid["x0_m"], source_grid["dx_m"], source_grid["nx"]
    )
    source_rows, valid_rows = resolve_indices(
        axis_for_grid(target_grid, "y"), source_grid["y0_m"], source_grid["dy_m"], source_grid["ny"]
    )
    output = np.full((target_grid["ny"], target_grid["nx"]), FILL_INT16, dtype=np.int16)
    target_cols = np.flatnonzero(valid_cols)
    target_rows = np.flatnonzero(valid_rows)
    if target_cols.size and target_rows.size:
        output[np.ix_(target_rows, target_cols)] = source[np.ix_(source_rows[target_rows], source_cols[target_cols])]
    return output


def build_hybrid_fields(
    *,
    base_bed: np.ndarray,
    surface: np.ndarray,
    base_thickness: np.ndarray,
    mask: np.ndarray,
    qrf_bed: np.ndarray,
    qrf_thickness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use QRF predictions only where they are coherent grounded-ice terrain."""
    shapes = {array.shape for array in (base_bed, surface, base_thickness, mask, qrf_bed, qrf_thickness)}
    if len(shapes) != 1:
        raise ValueError("Base and QRF fields must have matching shapes.")

    candidate_thickness = surface.astype(np.int32) - qrf_bed.astype(np.int32)
    applies = (
        (mask == 2)
        & (qrf_bed != FILL_INT16)
        & (qrf_thickness != FILL_INT16)
        & (qrf_thickness > 0)
        & (candidate_thickness > 0)
        & (candidate_thickness <= 32767)
    )
    bed = base_bed.copy()
    thickness = base_thickness.copy()
    bed[applies] = qrf_bed[applies]
    thickness[applies] = candidate_thickness[applies].astype(np.int16)
    return bed, thickness, applies


def read_qrf_geotiff(path: Path) -> np.ndarray:
    """Read and validate the published three-band QRF GeoTIFF."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        pixel_scale = tuple(float(value) for value in page.tags["ModelPixelScaleTag"].value[:2])
        tiepoint = tuple(float(value) for value in page.tags["ModelTiepointTag"].value[3:5])
        nodata = int(page.tags["GDAL_NODATA"].value)
        gdal_metadata = str(page.tags["GDAL_METADATA"].value)
        values = np.asarray(tif.asarray(), dtype=np.int16)

    expected_shape = (QRF_SOURCE_GRID["ny"], QRF_SOURCE_GRID["nx"], 3)
    if values.shape != expected_shape:
        raise ValueError(f"Expected QRF source shape {expected_shape}, got {values.shape}.")
    if pixel_scale != QRF_GEOTIFF_PIXEL_SCALE or tiepoint != QRF_GEOTIFF_TIEPOINT:
        raise ValueError("Unexpected QRF GeoTIFF georeferencing.")
    if nodata != FILL_INT16:
        raise ValueError(f"Unexpected QRF NoData value {nodata}.")
    for sample, description in enumerate(QRF_BAND_DESCRIPTIONS):
        expected_item = f'<Item name="DESCRIPTION" sample="{sample}" role="description">{description}</Item>'
        if expected_item not in gdal_metadata:
            raise ValueError(f"Unexpected QRF GeoTIFF band {sample} description.")
    return values


def decode_field(meta: dict[str, Any], binary: bytes, name: str) -> np.ndarray:
    field = next((field for field in meta["fields"] if field["name"] == name), None)
    if field is None:
        raise KeyError(f"Missing {name!r} in source package metadata.")
    dtypes = {"int16": "<i2", "uint8": "u1"}
    dtype = np.dtype(dtypes[field["dtype"]])
    count = int(field["byte_length"]) // dtype.itemsize
    values = np.frombuffer(binary, dtype=dtype, count=count, offset=int(field["byte_offset"])).copy()
    grid = normalize_grid(meta["grid"])
    return values.reshape(grid["ny"], grid["nx"])


def stats(values: np.ndarray, *, fill_value: int = FILL_INT16) -> dict[str, float]:
    valid = values[np.isfinite(values) & (values != fill_value)]
    return {"min": float(np.min(valid)), "max": float(np.max(valid)), "mean": float(np.mean(valid))}


def build_metadata(
    *,
    target: Target,
    grid: dict[str, int],
    bed: np.ndarray,
    surface: np.ndarray,
    thickness: np.ndarray,
    mask: np.ndarray,
    applies: np.ndarray,
    qrf_uncertainty: np.ndarray,
) -> dict[str, Any]:
    count = int(bed.size)
    uncertainty_values = qrf_uncertainty[applies & (qrf_uncertainty != FILL_INT16)]
    return {
        "title": "Greenland QRF subglacial topography (2025)",
        "product_version": "QRF 2025 hybridized with BedMachine Greenland v6 support fields",
        "reference": QRF_REFERENCE,
        "source_url": QRF_SOURCE_URL,
        "source_commit": QRF_SOURCE_COMMIT,
        "source_file": "QRF_greenland_ice_predictions_300m.tif",
        "source_projection": "EPSG:3413 (specified by the published QRF code and paper)",
        "source_sampling": {
            "pixel_anchor": "cell_center",
            "x0_m": QRF_SOURCE_GRID["x0_m"],
            "y0_m": QRF_SOURCE_GRID["y0_m"],
            "dx_m": QRF_SOURCE_GRID["dx_m"],
            "dy_m": QRF_SOURCE_GRID["dy_m"],
            "geotiff_tiepoint_is": "upper_left_pixel_edge",
        },
        "source_license": "No standalone data license is stated in the upstream repository; preserve article attribution and confirm redistribution terms before release.",
        "resampled_to": target.label,
        "grid": grid,
        "quantization": {
            "float_fill_value": -9999.0,
            "int16_fill_value": FILL_INT16,
            "unit": "m",
            "scale": 1.0,
            "offset": 0.0,
        },
        "hybridization": {
            "qrf_bed_applies_to": "grounded_ice_only",
            "surface": "BedMachine Greenland v6 surface elevation",
            "thickness": "surface elevation minus QRF bed elevation where QRF applies",
            "fallback": "BedMachine Greenland v6 surface and mask; BedMachine v6 bed/thickness are retained for floating ice, non-ice, QRF NoData, and incoherent QRF samples.",
        },
        "qrf_coverage": {
            "applied_cell_count": int(applies.sum()),
            "grounded_ice_cell_count": int((mask == 2).sum()),
            "applied_grounded_ice_fraction": float(applies.sum() / max(1, (mask == 2).sum())),
            "uncertainty_sd_m": stats(uncertainty_values) if uncertainty_values.size else None,
            "uncertainty_rendered": False,
        },
        "fields": [
            {"name": "bed", "dtype": "int16", "byte_offset": 0, "byte_length": count * 2, "stats_m": stats(bed)},
            {"name": "surface", "dtype": "int16", "byte_offset": count * 2, "byte_length": count * 2, "stats_m": stats(surface)},
            {"name": "thickness", "dtype": "int16", "byte_offset": count * 4, "byte_length": count * 2, "stats_m": stats(thickness)},
            {
                "name": "mask",
                "dtype": "uint8",
                "byte_offset": count * 6,
                "byte_length": count,
                "flags": {"0": "ocean", "1": "ice_free_land", "2": "grounded_ice", "3": "floating_ice"},
            },
        ],
    }


def prepare_target(data_dir: Path, qrf_values: np.ndarray, target: Target) -> None:
    base_meta_path = data_dir / f"{target.base_basename}.meta.json"
    base_bin_path = data_dir / f"{target.base_basename}.bin"
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    base_binary = base_bin_path.read_bytes()
    grid = normalize_grid(base_meta["grid"])
    base_bed = decode_field(base_meta, base_binary, "bed")
    surface = decode_field(base_meta, base_binary, "surface")
    base_thickness = decode_field(base_meta, base_binary, "thickness")
    mask = decode_field(base_meta, base_binary, "mask")

    qrf_thickness = sample_qrf_band(qrf_values[:, :, 0], source_grid=QRF_SOURCE_GRID, target_grid=grid)
    qrf_bed = sample_qrf_band(qrf_values[:, :, 1], source_grid=QRF_SOURCE_GRID, target_grid=grid)
    qrf_uncertainty = sample_qrf_band(qrf_values[:, :, 2], source_grid=QRF_SOURCE_GRID, target_grid=grid)
    bed, thickness, applies = build_hybrid_fields(
        base_bed=base_bed,
        surface=surface,
        base_thickness=base_thickness,
        mask=mask,
        qrf_bed=qrf_bed,
        qrf_thickness=qrf_thickness,
    )
    metadata = build_metadata(
        target=target,
        grid=grid,
        bed=bed,
        surface=surface,
        thickness=thickness,
        mask=mask,
        applies=applies,
        qrf_uncertainty=qrf_uncertainty,
    )
    with (data_dir / f"{target.output_basename}.bin").open("wb") as output:
        output.write(bed.tobytes(order="C"))
        output.write(surface.tobytes(order="C"))
        output.write(thickness.tobytes(order="C"))
        output.write(mask.tobytes(order="C"))
    (data_dir / f"{target.output_basename}.meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing QRF input: {input_path}")
    data_dir = Path(args.data_dir)
    qrf_values = read_qrf_geotiff(input_path)
    for target in TARGETS:
        prepare_target(data_dir, qrf_values, target)
        print(f"Prepared {target.output_basename}.")


if __name__ == "__main__":
    main()
