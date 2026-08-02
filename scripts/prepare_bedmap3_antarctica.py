#!/usr/bin/env python3
"""Prepare a browser-ready Bedmap3 Antarctica terrain package from GeoTIFF grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SOURCE_FILL_VALUE = -9999
VIEWER_FILL_VALUE = -32768
SOURCE_GRID_SIZE = 13_334
SOURCE_GRID_SPACING_M = 500
SOURCE_X0_M = -3_333_250
SOURCE_Y0_M = 3_333_250
BEDMAP3_REFERENCE = (
    "Pritchard, H.D. et al. (2025) Bedmap3 updated ice bed, surface and thickness "
    "gridded datasets for Antarctica. Scientific Data 12, 414. "
    "https://doi.org/10.1038/s41597-025-04672-y. Data: "
    "https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare downsampled Bedmap3 Antarctica binary + metadata from GeoTIFF grids."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing bm3_bed.tif, bm3_surface.tif, bm3_thickness.tif, and bm3_masks.tif.",
    )
    parser.add_argument(
        "--resolution-m",
        type=int,
        default=10_000,
        help="Target spacing in metres; must be a multiple of Bedmap3's 500 m grid spacing.",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--basename",
        default="bedmap3_antarctica_10km",
        help="Basename for output files.",
    )
    return parser.parse_args()


def build_axis_sampling(axis: np.ndarray, resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
    """Return evenly spaced source-axis values and their indices for a target resolution."""
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("Bedmap3 source axis must contain at least two values.")
    if resolution_m <= 0:
        raise ValueError("Target resolution must be positive.")

    spacing_m = int(round(abs(float(axis[1] - axis[0]))))
    if spacing_m <= 0 or not np.allclose(np.diff(axis), axis[1] - axis[0]):
        raise ValueError("Bedmap3 source axis must be uniformly spaced.")
    if resolution_m % spacing_m != 0:
        raise ValueError("Target resolution must be a multiple of source grid spacing.")

    stride = resolution_m // spacing_m
    indices = np.arange(0, axis.size, stride, dtype=np.intp)
    return axis[indices], indices


def translate_mask(source_mask: np.ndarray) -> np.ndarray:
    """Map Bedmap3 masks to the viewer's ocean/land/grounded/floating contract."""
    mask = np.zeros(source_mask.shape, dtype=np.uint8)
    mask[source_mask == 1] = 2
    mask[(source_mask == 2) | (source_mask == 3)] = 3
    mask[source_mask == 4] = 1
    return mask


def quantize_elevation(values: np.ndarray) -> np.ndarray:
    """Convert source elevation values to the viewer's int16 representation."""
    valid = np.isfinite(values) & (values != SOURCE_FILL_VALUE)
    quantized = np.full(values.shape, VIEWER_FILL_VALUE, dtype=np.int16)
    quantized[valid] = np.rint(np.clip(values[valid], -32767, 32767)).astype(np.int16)
    return quantized


def field_stats(values: np.ndarray) -> dict[str, float]:
    """Return elevation statistics excluding the Bedmap3 no-data sentinel."""
    valid = values[np.isfinite(values) & (values != SOURCE_FILL_VALUE)]
    if not valid.size:
        raise ValueError("Bedmap3 field has no valid values after sampling.")
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
    }


def read_geotiff(path: Path) -> np.ndarray:
    """Read one Bedmap3 GeoTIFF grid with the optional tifffile dependency."""
    try:
        import tifffile
    except ImportError as error:  # pragma: no cover - depends on local environment
        raise RuntimeError("Install tifffile to prepare Bedmap3 GeoTIFF data: pip install tifffile") from error

    if not path.exists():
        raise FileNotFoundError(f"Missing Bedmap3 source file: {path}")
    values = tifffile.imread(path)
    if values.shape != (SOURCE_GRID_SIZE, SOURCE_GRID_SIZE):
        raise ValueError(
            f"Unexpected Bedmap3 grid shape for {path.name}: {values.shape}; "
            f"expected {(SOURCE_GRID_SIZE, SOURCE_GRID_SIZE)}."
        )
    return values


def load_and_sample_field(path: Path, y_indices: np.ndarray, x_indices: np.ndarray) -> np.ndarray:
    """Read one source field then sample it onto the browser grid."""
    values = read_geotiff(path)
    return values[np.ix_(y_indices, x_indices)]


def build_metadata(
    *,
    x: np.ndarray,
    y: np.ndarray,
    cell_count: int,
    resolution_m: int,
    source_files: list[str],
    bed: np.ndarray,
    surface: np.ndarray,
    thickness: np.ndarray,
) -> dict:
    """Build metadata that conforms to the browser terrain-package contract."""
    return {
        "title": "Bedmap3 Antarctica",
        "product_version": "v1.0",
        "reference": BEDMAP3_REFERENCE,
        "license": "CC-BY-4.0",
        "source_files": source_files,
        "source_projection": "EPSG:3031 (WGS 1984 Antarctic Polar Stereographic)",
        "source_grid": {
            "nx": SOURCE_GRID_SIZE,
            "ny": SOURCE_GRID_SIZE,
            "x0_m": SOURCE_X0_M,
            "y0_m": SOURCE_Y0_M,
            "dx_m": SOURCE_GRID_SPACING_M,
            "dy_m": -SOURCE_GRID_SPACING_M,
        },
        "downsample_resolution_m": resolution_m,
        "grid": {
            "nx": int(x.size),
            "ny": int(y.size),
            "x0_m": int(x[0]),
            "y0_m": int(y[0]),
            "dx_m": int(x[1] - x[0]),
            "dy_m": int(y[1] - y[0]),
        },
        "quantization": {
            "float_fill_value": SOURCE_FILL_VALUE,
            "int16_fill_value": VIEWER_FILL_VALUE,
            "unit": "m",
            "scale": 1.0,
            "offset": 0.0,
        },
        "fields": [
            {
                "name": "bed",
                "dtype": "int16",
                "byte_offset": 0,
                "byte_length": cell_count * 2,
                "stats_m": field_stats(bed),
            },
            {
                "name": "surface",
                "dtype": "int16",
                "byte_offset": cell_count * 2,
                "byte_length": cell_count * 2,
                "stats_m": field_stats(surface),
            },
            {
                "name": "thickness",
                "dtype": "int16",
                "byte_offset": cell_count * 4,
                "byte_length": cell_count * 2,
                "stats_m": field_stats(thickness),
            },
            {
                "name": "mask",
                "dtype": "uint8",
                "byte_offset": cell_count * 6,
                "byte_length": cell_count,
                "flags": {
                    "0": "ocean_or_no_data",
                    "1": "ice_free_land_or_rock",
                    "2": "grounded_ice",
                    "3": "floating_ice_or_transiently_grounded_ice",
                },
            },
        ],
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_axis = SOURCE_X0_M + np.arange(SOURCE_GRID_SIZE, dtype=np.int64) * SOURCE_GRID_SPACING_M
    y_axis = SOURCE_Y0_M - np.arange(SOURCE_GRID_SIZE, dtype=np.int64) * SOURCE_GRID_SPACING_M
    x, x_indices = build_axis_sampling(x_axis, args.resolution_m)
    y, y_indices = build_axis_sampling(y_axis, args.resolution_m)

    source_paths = {
        "bed": input_dir / "bm3_bed.tif",
        "surface": input_dir / "bm3_surface.tif",
        "thickness": input_dir / "bm3_thickness.tif",
        "mask": input_dir / "bm3_masks.tif",
    }
    bed = load_and_sample_field(source_paths["bed"], y_indices, x_indices)
    surface = load_and_sample_field(source_paths["surface"], y_indices, x_indices)
    thickness = load_and_sample_field(source_paths["thickness"], y_indices, x_indices)
    source_mask = load_and_sample_field(source_paths["mask"], y_indices, x_indices)

    bed_quantized = quantize_elevation(bed)
    surface_quantized = quantize_elevation(surface)
    thickness_quantized = quantize_elevation(thickness)
    mask = translate_mask(source_mask)
    cell_count = int(bed_quantized.size)

    bin_path = output_dir / f"{args.basename}.bin"
    with bin_path.open("wb") as fh:
        fh.write(bed_quantized.tobytes(order="C"))
        fh.write(surface_quantized.tobytes(order="C"))
        fh.write(thickness_quantized.tobytes(order="C"))
        fh.write(mask.tobytes(order="C"))

    meta = build_metadata(
        x=x,
        y=y,
        cell_count=cell_count,
        resolution_m=args.resolution_m,
        source_files=[path.name for path in source_paths.values()],
        bed=bed,
        surface=surface,
        thickness=thickness,
    )
    meta_path = output_dir / f"{args.basename}.meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {bin_path} ({bin_path.stat().st_size} bytes)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
