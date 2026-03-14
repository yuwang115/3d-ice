#!/usr/bin/env python3
"""Prepare BedMachine-aligned basal-friction packages for Antarctica or Greenland."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROW_CHUNK_SIZE = 64


@dataclass(frozen=True)
class TargetGrid:
    meta_path: Path
    output_basename: str
    label: str


TARGETS_BY_REGION = {
    "antarctica": (
        TargetGrid(
            meta_path=Path("static/tools/data/bedmachine_antarctica_v4_480.meta.json"),
            output_basename="antarctica_basal_friction_480",
            label="Balanced",
        ),
        TargetGrid(
            meta_path=Path("static/tools/data/bedmachine_antarctica_v4_741.meta.json"),
            output_basename="antarctica_basal_friction_741",
            label="HD",
        ),
    ),
    "greenland": (
        TargetGrid(
            meta_path=Path("static/tools/data/bedmachine_greenland_v6_3km.meta.json"),
            output_basename="greenland_basal_friction_3km",
            label="3 km",
        ),
        TargetGrid(
            meta_path=Path("static/tools/data/bedmachine_greenland_v6_1km.meta.json"),
            output_basename="greenland_basal_friction_1km",
            label="1 km",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample basal-friction netCDF data onto BedMachine viewer grids."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to taub_med.nc",
    )
    parser.add_argument(
        "--region",
        required=True,
        choices=tuple(TARGETS_BY_REGION.keys()),
        help="Target region whose BedMachine grids should be used.",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output .bin/.meta.json files.",
    )
    parser.add_argument(
        "--field",
        default="taub",
        help="Source variable name inside the netCDF file.",
    )
    return parser.parse_args()


def decode_attr(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_attr(value.item(), default)
    return str(value)


def build_target_axis(grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    x0 = float(grid["x0_m"])
    y0 = float(grid["y0_m"])
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])
    x = x0 + np.arange(nx, dtype=np.float64) * dx
    y = y0 + np.arange(ny, dtype=np.float64) * dy
    return x, y


def resolve_indices(
    target_axis: np.ndarray,
    src0: float,
    src_step: float,
    src_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.rint((target_axis - src0) / src_step).astype(np.int64)
    valid = (idx >= 0) & (idx < src_count)
    idx_safe = np.clip(idx, 0, src_count - 1)
    return idx_safe, valid


def choose_visualization(_quantiles: dict[str, float]) -> dict[str, Any]:
    return {
        "display_range_mpa": [0.0, 0.30],
        "knee_mpa": 0.015,
        "ticks_mpa": [0.0, 0.02, 0.04, 0.08, 0.15, 0.30],
        "note": "",
    }


def prepare_target(
    ds: h5py.File,
    source_ds: h5py.Dataset,
    target: TargetGrid,
    output_dir: Path,
    *,
    region: str,
    source_field_name: str,
) -> None:
    with target.meta_path.open("r", encoding="utf-8") as fh:
        bed_meta = json.load(fh)

    grid = bed_meta["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    cell_count = nx * ny

    src_x = np.asarray(ds["x"][:], dtype=np.float64)
    src_y = np.asarray(ds["y"][:], dtype=np.float64)
    src_dx = float(src_x[1] - src_x[0])
    src_dy = float(src_y[1] - src_y[0])

    target_x, target_y = build_target_axis(grid)
    src_ix, src_ix_valid = resolve_indices(target_x, float(src_x[0]), src_dx, src_x.shape[0])
    src_iy, src_iy_valid = resolve_indices(target_y, float(src_y[0]), src_dy, src_y.shape[0])

    out = np.full((ny, nx), np.nan, dtype=np.float32)
    col_positions = np.flatnonzero(src_ix_valid)
    src_col_positions = src_ix[col_positions]
    row_positions = np.flatnonzero(src_iy_valid)
    src_row_positions = src_iy[row_positions]

    for start in range(0, row_positions.size, ROW_CHUNK_SIZE):
        stop = min(row_positions.size, start + ROW_CHUNK_SIZE)
        target_rows = row_positions[start:stop]
        source_rows = src_row_positions[start:stop]
        chunk = np.full((target_rows.size, col_positions.size), np.nan, dtype=np.float32)

        for chunk_row, source_row in enumerate(source_rows):
            sampled = np.asarray(source_ds[int(source_row), src_col_positions], dtype=np.float32)
            valid = np.isfinite(sampled) & (sampled >= 0)
            if np.any(valid):
                chunk[chunk_row, valid] = sampled[valid]

        out[target_rows[:, None], col_positions] = chunk

    finite_values = out[np.isfinite(out)]
    if finite_values.size == 0:
        raise RuntimeError(f"No valid basal-friction samples for target {target.label}.")

    stats = {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values, dtype=np.float64)),
    }
    quantile_values = np.quantile(finite_values, [0.5, 0.95, 0.99, 0.995])
    quantiles = {
        "median": float(quantile_values[0]),
        "q95": float(quantile_values[1]),
        "q99": float(quantile_values[2]),
        "q995": float(quantile_values[3]),
    }

    out_bin = output_dir / f"{target.output_basename}.bin"
    out_meta = output_dir / f"{target.output_basename}.meta.json"
    with out_bin.open("wb") as fh:
        fh.write(out.astype(np.float32, copy=False).tobytes(order="C"))

    source_title = decode_attr(ds.attrs.get("title"), "")
    if not source_title:
        source_title = f"{region.title()} basal friction (taub)"

    meta = {
        "title": source_title,
        "product_version": "taub_med",
        "source_file": Path(ds.filename).name,
        "source_variable": source_field_name,
        "region": region,
        "resampled_to": target.label,
        "units_inferred": True,
        "inference_note": "MPa units are inferred from the tau_b variable name and value range because the source file does not expose unit metadata.",
        "netcdf_properties": decode_attr(ds.attrs.get("_NCProperties"), ""),
        "grid": {
            "nx": nx,
            "ny": ny,
            "x0_m": int(grid["x0_m"]),
            "y0_m": int(grid["y0_m"]),
            "dx_m": int(grid["dx_m"]),
            "dy_m": int(grid["dy_m"]),
        },
        "coverage": {
            "valid_count": int(finite_values.size),
            "cell_count": cell_count,
            "valid_fraction": float(finite_values.size / cell_count),
            "mask_hint": "Render on grounded BedMachine ice cells only.",
        },
        "visualization": choose_visualization(quantiles),
        "fields": [
            {
                "name": "basal_friction",
                "dtype": "float32",
                "byte_offset": 0,
                "byte_length": cell_count * 4,
                "unit": "MPa",
                "stats_mpa": stats,
                "quantiles_mpa": quantiles,
                "source_name": source_field_name,
            }
        ],
    }

    with out_meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {out_bin} ({out_bin.stat().st_size} bytes)")
    print(f"Wrote {out_meta}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as ds:
        if args.field not in ds:
            raise KeyError(f"Missing variable '{args.field}' in {input_path}")
        source_ds = ds[args.field]
        if source_ds.ndim != 2:
            raise ValueError(f"Expected a 2D variable for '{args.field}', got shape {source_ds.shape}")

        for target in TARGETS_BY_REGION[args.region]:
            prepare_target(
                ds,
                source_ds,
                target,
                output_dir,
                region=args.region,
                source_field_name=args.field,
            )


if __name__ == "__main__":
    main()
