#!/usr/bin/env python3
"""Prepare web-ready Antarctica ice velocity tiles aligned to BedMachine grids.

This script reads a NETCDF3 velocity product (vx, vy in m/year) and resamples it
to the same grids used by the BedMachine browser datasets.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

FILL_INT16 = -32768
DEFAULT_SCALE = 1.0  # int16 value * scale = velocity in m/year


@dataclass(frozen=True)
class TargetGrid:
    meta_path: Path
    output_basename: str
    label: str


DEFAULT_TARGETS = (
    TargetGrid(
        meta_path=Path("static/tools/data/bedmachine_antarctica_v4_480.meta.json"),
        output_basename="antarctic_ice_velocity_phase_v01_480",
        label="Balanced",
    ),
    TargetGrid(
        meta_path=Path("static/tools/data/bedmachine_antarctica_v4_741.meta.json"),
        output_basename="antarctic_ice_velocity_phase_v01_741",
        label="HD",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample MEaSUREs Antarctic velocity data onto BedMachine grids."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to MEaSUREs velocity netCDF file (vx/vy).",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output .bin/.meta.json files.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Quantization scale (m/year per int16 unit).",
    )
    return parser.parse_args()


def decode_attr(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def to_quantized_int16(values: np.ndarray, scale: float) -> np.ndarray:
    q = np.rint(values / scale)
    q = np.clip(q, -32767, 32767)
    return q.astype(np.int16)


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
    target_axis: np.ndarray, src0: float, src_step: float, src_count: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.rint((target_axis - src0) / src_step).astype(np.int64)
    valid = (idx >= 0) & (idx < src_count)
    idx_safe = np.clip(idx, 0, src_count - 1)
    return idx_safe, valid


def prepare_target(
    ds: Dataset,
    target: TargetGrid,
    output_dir: Path,
    scale: float,
) -> None:
    with target.meta_path.open("r", encoding="utf-8") as fh:
        bed_meta = json.load(fh)

    grid = bed_meta["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    cell_count = nx * ny

    src_x = np.asarray(ds.variables["x"][:], dtype=np.float64)
    src_y = np.asarray(ds.variables["y"][:], dtype=np.float64)
    src_nx = src_x.shape[0]
    src_ny = src_y.shape[0]
    src_dx = float(src_x[1] - src_x[0])
    src_dy = float(src_y[1] - src_y[0])

    target_x, target_y = build_target_axis(grid)
    src_ix, src_ix_valid = resolve_indices(target_x, float(src_x[0]), src_dx, src_nx)
    src_iy, src_iy_valid = resolve_indices(target_y, float(src_y[0]), src_dy, src_ny)

    vx_var = ds.variables["vx"]
    vy_var = ds.variables["vy"]
    src_fill = float(getattr(vx_var, "_FillValue", -99999.9))
    fill_eps = 1e-3

    vx_out = np.full((ny, nx), FILL_INT16, dtype=np.int16)
    vy_out = np.full((ny, nx), FILL_INT16, dtype=np.int16)

    col_positions = np.flatnonzero(src_ix_valid)

    vx_min = math.inf
    vx_max = -math.inf
    vx_sum = 0.0
    vy_min = math.inf
    vy_max = -math.inf
    vy_sum = 0.0
    speed_min = math.inf
    speed_max = -math.inf
    speed_sum = 0.0
    valid_count = 0

    for row in range(ny):
        if not src_iy_valid[row]:
            continue

        sy = int(src_iy[row])
        src_row_vx = np.asarray(vx_var[sy, :], dtype=np.float64)
        src_row_vy = np.asarray(vy_var[sy, :], dtype=np.float64)

        vx_sample = src_row_vx[src_ix[col_positions]]
        vy_sample = src_row_vy[src_ix[col_positions]]

        finite = np.isfinite(vx_sample) & np.isfinite(vy_sample)
        not_fill = (np.abs(vx_sample - src_fill) > fill_eps) & (
            np.abs(vy_sample - src_fill) > fill_eps
        )
        valid = finite & not_fill
        if not np.any(valid):
            continue

        valid_cols = col_positions[valid]
        vx_valid = vx_sample[valid]
        vy_valid = vy_sample[valid]

        vx_out[row, valid_cols] = to_quantized_int16(vx_valid, scale)
        vy_out[row, valid_cols] = to_quantized_int16(vy_valid, scale)

        speed_valid = np.hypot(vx_valid, vy_valid)

        vx_min = min(vx_min, float(np.min(vx_valid)))
        vx_max = max(vx_max, float(np.max(vx_valid)))
        vx_sum += float(np.sum(vx_valid))
        vy_min = min(vy_min, float(np.min(vy_valid)))
        vy_max = max(vy_max, float(np.max(vy_valid)))
        vy_sum += float(np.sum(vy_valid))
        speed_min = min(speed_min, float(np.min(speed_valid)))
        speed_max = max(speed_max, float(np.max(speed_valid)))
        speed_sum += float(np.sum(speed_valid))
        valid_count += int(vx_valid.size)

    if valid_count == 0:
        raise RuntimeError(f"No valid velocity samples for target {target.label}.")

    out_bin = output_dir / f"{target.output_basename}.bin"
    out_meta = output_dir / f"{target.output_basename}.meta.json"
    with out_bin.open("wb") as fh:
        fh.write(vx_out.tobytes(order="C"))
        fh.write(vy_out.tobytes(order="C"))

    source_title = decode_attr(getattr(ds, "Title", None), "MEaSUREs Antarctica velocity")
    source_name = decode_attr(getattr(ds, "Source", None), "")
    source_proj = decode_attr(getattr(ds, "Coordinate system", None), "")
    source_remark = decode_attr(getattr(ds, "Remark", None), "")

    meta = {
        "title": source_title,
        "product_version": "v1",
        "source_file": Path(ds.filepath()).name,
        "source_url": source_name,
        "coordinate_system": source_proj,
        "remark": source_remark,
        "resampled_to": target.label,
        "grid": {
            "nx": nx,
            "ny": ny,
            "x0_m": int(grid["x0_m"]),
            "y0_m": int(grid["y0_m"]),
            "dx_m": int(grid["dx_m"]),
            "dy_m": int(grid["dy_m"]),
        },
        "quantization": {
            "int16_fill_value": FILL_INT16,
            "unit": "m/year",
            "scale": scale,
            "offset": 0.0,
        },
        "coverage": {
            "valid_count": valid_count,
            "cell_count": cell_count,
            "valid_fraction": float(valid_count / cell_count),
        },
        "fields": [
            {
                "name": "vx",
                "dtype": "int16",
                "byte_offset": 0,
                "byte_length": cell_count * 2,
                "stats_m_per_year": {
                    "min": vx_min,
                    "max": vx_max,
                    "mean": vx_sum / valid_count,
                },
            },
            {
                "name": "vy",
                "dtype": "int16",
                "byte_offset": cell_count * 2,
                "byte_length": cell_count * 2,
                "stats_m_per_year": {
                    "min": vy_min,
                    "max": vy_max,
                    "mean": vy_sum / valid_count,
                },
            },
            {
                "name": "speed",
                "unit": "m/year",
                "stats_m_per_year": {
                    "min": speed_min,
                    "max": speed_max,
                    "mean": speed_sum / valid_count,
                },
            },
        ],
    }

    with out_meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(
        f"[{target.label}] wrote {out_bin} ({out_bin.stat().st_size / (1024 * 1024):.2f} MB), "
        f"valid {valid_count}/{cell_count} ({(valid_count / cell_count) * 100:.1f}%)"
    )
    print(f"[{target.label}] wrote {out_meta}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")
    if args.scale <= 0:
        raise ValueError("--scale must be > 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Dataset(input_path, "r") as ds:
        if "vx" not in ds.variables or "vy" not in ds.variables:
            raise RuntimeError("Input file must contain variables: vx, vy")

        for target in DEFAULT_TARGETS:
            if not target.meta_path.exists():
                raise FileNotFoundError(f"Missing BedMachine target meta: {target.meta_path}")
            prepare_target(ds=ds, target=target, output_dir=output_dir, scale=args.scale)


if __name__ == "__main__":
    main()
