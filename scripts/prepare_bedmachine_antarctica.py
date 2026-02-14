#!/usr/bin/env python3
"""Downsample BedMachine Antarctica netCDF data for browser visualization."""

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
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare downsampled BedMachine Antarctica binary + metadata."
    )
    parser.add_argument(
        "--input",
        default="BedMachineAntarctica_V4.nc",
        help="Path to BedMachineAntarctica netCDF4 file.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=28,
        help="Stride for downsampling in x/y directions.",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--basename",
        default="bedmachine_antarctica_v4_480",
        help="Basename for output files.",
    )
    return parser.parse_args()


def quantize_to_int16(arr: np.ndarray, fill_value: float = FILL_FLOAT) -> np.ndarray:
    valid = np.isfinite(arr) & (arr != fill_value)
    out = np.full(arr.shape, FILL_INT16, dtype=np.int16)
    rounded = np.rint(np.clip(arr[valid], -32767, 32767)).astype(np.int16)
    out[valid] = rounded
    return out


def stats(arr: np.ndarray, fill_value: float = FILL_FLOAT) -> dict[str, float]:
    valid = arr[(np.isfinite(arr)) & (arr != fill_value)]
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    with h5py.File(input_path, "r") as ds:
        x = ds["x"][:: args.step]
        y = ds["y"][:: args.step]
        bed = ds["bed"][:: args.step, :: args.step].astype(np.float32)
        surface = ds["surface"][:: args.step, :: args.step].astype(np.float32)
        thickness = ds["thickness"][:: args.step, :: args.step].astype(np.float32)
        mask = ds["mask"][:: args.step, :: args.step].astype(np.uint8)

        bed_q = quantize_to_int16(bed)
        surface_q = quantize_to_int16(surface)
        thickness_q = quantize_to_int16(thickness)

        count = int(bed_q.size)
        bin_path = output_dir / f"{args.basename}.bin"
        with bin_path.open("wb") as fh:
            fh.write(bed_q.tobytes(order="C"))
            fh.write(surface_q.tobytes(order="C"))
            fh.write(thickness_q.tobytes(order="C"))
            fh.write(mask.tobytes(order="C"))

        meta = {
            "title": decode_attr(ds.attrs.get("title"), "BedMachine Antarctica v4"),
            "product_version": decode_attr(ds.attrs.get("product_version"), "v4"),
            "date_modified": decode_attr(ds.attrs.get("date_modified"), ""),
            "reference": decode_attr(ds.attrs.get("references"), ""),
            "license": decode_attr(ds.attrs.get("license"), ""),
            "source_file": str(input_path.name),
            "downsample_step": int(args.step),
            "grid": {
                "nx": int(x.shape[0]),
                "ny": int(y.shape[0]),
                "x0_m": int(x[0]),
                "y0_m": int(y[0]),
                "dx_m": int(x[1] - x[0]),
                "dy_m": int(y[1] - y[0]),
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
                    "stats_m": stats(bed),
                },
                {
                    "name": "surface",
                    "dtype": "int16",
                    "byte_offset": count * 2,
                    "byte_length": count * 2,
                    "stats_m": stats(surface),
                },
                {
                    "name": "thickness",
                    "dtype": "int16",
                    "byte_offset": count * 4,
                    "byte_length": count * 2,
                    "stats_m": stats(thickness),
                },
                {
                    "name": "mask",
                    "dtype": "uint8",
                    "byte_offset": count * 6,
                    "byte_length": count,
                    "flags": {
                        "0": "ocean",
                        "1": "ice_free_land",
                        "2": "grounded_ice",
                        "3": "floating_ice",
                        "4": "lake_vostok",
                    },
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
