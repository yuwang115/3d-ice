#!/usr/bin/env python3
"""Resample BedMachine Greenland netCDF data for browser visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

FILL_FLOAT = -9999.0
FILL_INT16 = -32768


def decode_attr(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_attr(value.item(), default)
    return str(value)


def get_fill_value(dataset: h5py.Dataset, fallback: float = FILL_FLOAT) -> float:
    raw = dataset.attrs.get("_FillValue")
    if raw is None:
        return fallback
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return fallback
        raw = raw.reshape(-1)[0]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def parse_mask_flags(mask_ds: h5py.Dataset) -> dict[str, str]:
    raw_values = mask_ds.attrs.get("flag_values")
    raw_meanings = decode_attr(mask_ds.attrs.get("flag_meanings"), "")
    if raw_values is None or not raw_meanings:
        return {
            "0": "ocean",
            "1": "ice_free_land",
            "2": "grounded_ice",
            "3": "floating_ice",
        }

    if isinstance(raw_values, np.ndarray):
        values = raw_values.reshape(-1).tolist()
    else:
        values = [raw_values]
    meanings = raw_meanings.split()
    return {str(int(value)): meaning for value, meaning in zip(values, meanings)}


def quantize_to_int16(
    arr: np.ndarray,
    *,
    fill_value: float,
    clip_min: float = -32767,
    clip_max: float = 32767,
) -> np.ndarray:
    valid = np.isfinite(arr) & (arr != fill_value)
    out = np.full(arr.shape, FILL_INT16, dtype=np.int16)
    rounded = np.rint(np.clip(arr[valid], clip_min, clip_max)).astype(np.int16)
    out[valid] = rounded
    return out


def stats(arr: np.ndarray, *, fill_value: float) -> dict[str, float]:
    valid = arr[np.isfinite(arr) & (arr != fill_value)]
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare resampled BedMachine Greenland binary + metadata."
    )
    parser.add_argument(
        "--input",
        default="BedMachineGreenland-v6.nc",
        help="Path to BedMachineGreenland netCDF4 file.",
    )
    parser.add_argument(
        "--resolution-m",
        type=int,
        default=3000,
        help="Target grid spacing in meters.",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--basename",
        default="bedmachine_greenland_v6_3km",
        help="Basename for output files.",
    )
    return parser.parse_args()


def build_axis_sampling(axis: np.ndarray, resolution_m: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.int64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("Axis must be 1D with at least two values.")
    if resolution_m <= 0:
        raise ValueError("Resolution must be positive.")

    step_m = abs(int(axis[1] - axis[0]))
    if step_m <= 0:
        raise ValueError("Source axis spacing must be non-zero.")

    extent_m = abs(int(axis[-1] - axis[0]))
    count = extent_m // resolution_m + 1
    direction = 1 if axis[-1] >= axis[0] else -1
    target = axis[0] + direction * resolution_m * np.arange(count, dtype=np.int64)
    source_index = np.rint(np.abs(target - axis[0]) / step_m).astype(np.int64)
    source_index = np.clip(source_index, 0, axis.size - 1)
    return target, source_index


def sample_rows(dataset: h5py.Dataset, row_index: np.ndarray, col_index: np.ndarray, dtype: np.dtype) -> np.ndarray:
    out = np.empty((row_index.size, col_index.size), dtype=dtype)
    for out_row, src_row in enumerate(row_index):
        out[out_row] = dataset[int(src_row), col_index]
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    with h5py.File(input_path, "r") as ds:
        x_full = ds["x"][:]
        y_full = ds["y"][:]
        bed_ds = ds["bed"]
        surface_ds = ds["surface"]
        thickness_ds = ds["thickness"]
        mask_ds = ds["mask"]

        x, x_index = build_axis_sampling(x_full, args.resolution_m)
        y, y_index = build_axis_sampling(y_full, args.resolution_m)

        bed = sample_rows(bed_ds, y_index, x_index, np.float32)
        surface = sample_rows(surface_ds, y_index, x_index, np.float32)
        thickness = sample_rows(thickness_ds, y_index, x_index, np.float32)
        mask = sample_rows(mask_ds, y_index, x_index, np.uint8)

        bed_fill = get_fill_value(bed_ds)
        surface_fill = get_fill_value(surface_ds)
        thickness_fill = get_fill_value(thickness_ds)

        bed_q = quantize_to_int16(bed, fill_value=bed_fill)
        surface_q = quantize_to_int16(surface, fill_value=surface_fill)
        thickness_q = quantize_to_int16(thickness, fill_value=thickness_fill)

        count = int(bed_q.size)
        bin_path = output_dir / f"{args.basename}.bin"
        with bin_path.open("wb") as fh:
            fh.write(bed_q.tobytes(order="C"))
            fh.write(surface_q.tobytes(order="C"))
            fh.write(thickness_q.tobytes(order="C"))
            fh.write(mask.tobytes(order="C"))

        meta = {
            "title": decode_attr(ds.attrs.get("title"), "BedMachine Greenland v6"),
            "product_version": decode_attr(ds.attrs.get("product_version"), "v6"),
            "date_modified": decode_attr(ds.attrs.get("date_modified"), ""),
            "reference": decode_attr(ds.attrs.get("references"), ""),
            "license": decode_attr(ds.attrs.get("license"), ""),
            "summary": decode_attr(ds.attrs.get("summary"), ""),
            "source_file": str(input_path.name),
            "target_resolution_m": int(args.resolution_m),
            "grid": {
                "nx": int(x.shape[0]),
                "ny": int(y.shape[0]),
                "x0_m": int(x[0]),
                "y0_m": int(y[0]),
                "dx_m": int(args.resolution_m),
                "dy_m": -int(args.resolution_m) if y[-1] < y[0] else int(args.resolution_m),
            },
            "quantization": {
                "float_fill_value": FILL_FLOAT,
                "int16_fill_value": FILL_INT16,
                "unit": "m",
                "scale": 1.0,
                "offset": 0.0,
            },
            "fields": [
                {
                    "name": "bed",
                    "dtype": "int16",
                    "byte_offset": 0,
                    "byte_length": count * 2,
                    "stats_m": stats(bed, fill_value=bed_fill),
                },
                {
                    "name": "surface",
                    "dtype": "int16",
                    "byte_offset": count * 2,
                    "byte_length": count * 2,
                    "stats_m": stats(surface, fill_value=surface_fill),
                },
                {
                    "name": "thickness",
                    "dtype": "int16",
                    "byte_offset": count * 4,
                    "byte_length": count * 2,
                    "stats_m": stats(thickness, fill_value=thickness_fill),
                },
                {
                    "name": "mask",
                    "dtype": "uint8",
                    "byte_offset": count * 6,
                    "byte_length": count,
                    "flags": parse_mask_flags(mask_ds),
                },
            ],
        }

    meta_path = output_dir / f"{args.basename}.meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {bin_path} ({bin_path.stat().st_size} bytes)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
