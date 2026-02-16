#!/usr/bin/env python3
"""Prepare web-ready Antarctic subglacial hydrology overlays.

Outputs are resampled to the same BedMachine-derived grids used by the 3D viewer.
Each target package contains:
1) effective_pressure grid (int16 quantized, Pa)
2) subglacial channel segments collapsed to target-cell edges, colored by channel_discharge
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
EFFECTIVE_PRESSURE_SCALE_PA = 1000.0


@dataclass(frozen=True)
class TargetGrid:
    meta_path: Path
    output_basename: str
    label: str


DEFAULT_TARGETS = (
    TargetGrid(
        meta_path=Path("static/tools/data/bedmachine_antarctica_v4_480.meta.json"),
        output_basename="antarctica_subglacial_hydrology_480",
        label="Balanced",
    ),
    TargetGrid(
        meta_path=Path("static/tools/data/bedmachine_antarctica_v4_741.meta.json"),
        output_basename="antarctica_subglacial_hydrology_741",
        label="HD",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample Antarctic subglacial hydrology data to BedMachine-aligned grids."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Antarctica_SubglacialHydrology.nc",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output .bin/.meta.json files.",
    )
    parser.add_argument(
        "--effective-pressure-scale",
        type=float,
        default=EFFECTIVE_PRESSURE_SCALE_PA,
        help="Quantization scale (Pa per int16 unit) for effective pressure.",
    )
    parser.add_argument(
        "--min-channel-discharge",
        type=float,
        default=1e-5,
        help="Minimum channel_discharge (m3/s) to keep before edge aggregation.",
    )
    return parser.parse_args()


def decode_attr(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
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
    target_axis: np.ndarray, src0: float, src_step: float, src_count: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.rint((target_axis - src0) / src_step).astype(np.int64)
    valid = (idx >= 0) & (idx < src_count)
    idx_safe = np.clip(idx, 0, src_count - 1)
    return idx_safe, valid


def quantize_to_int16(values: np.ndarray, scale: float) -> np.ndarray:
    q = np.rint(values / scale)
    q = np.clip(q, -32767, 32767)
    return q.astype(np.int16)


def reduce_channel_edges(
    *,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    discharge: np.ndarray,
    min_discharge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    col1 = np.rint((x1 - x0) / dx).astype(np.int64)
    row1 = np.rint((y1 - y0) / dy).astype(np.int64)
    col2 = np.rint((x2 - x0) / dx).astype(np.int64)
    row2 = np.rint((y2 - y0) / dy).astype(np.int64)

    in_bounds = (
        (col1 >= 0)
        & (col1 < nx)
        & (row1 >= 0)
        & (row1 < ny)
        & (col2 >= 0)
        & (col2 < nx)
        & (row2 >= 0)
        & (row2 < ny)
    )
    finite = np.isfinite(discharge)
    nonzero = discharge >= min_discharge
    same_cell = (col1 == col2) & (row1 == row2)

    keep = in_bounds & finite & nonzero & (~same_cell)
    kept_count = int(np.sum(keep))
    if kept_count == 0:
        return (
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.uint16),
            np.empty(0, dtype=np.float32),
            0,
        )

    idx1 = row1[keep] * nx + col1[keep]
    idx2 = row2[keep] * nx + col2[keep]
    discharge_kept = discharge[keep]

    a = np.minimum(idx1, idx2).astype(np.int64)
    b = np.maximum(idx1, idx2).astype(np.int64)

    cell_count = int(nx * ny)
    edge_key = a.astype(np.uint64) * np.uint64(cell_count) + b.astype(np.uint64)

    order = np.argsort(edge_key)
    edge_sorted = edge_key[order]
    a_sorted = a[order]
    b_sorted = b[order]
    q_sorted = discharge_kept[order]

    group_start = np.empty(edge_sorted.shape, dtype=bool)
    group_start[0] = True
    group_start[1:] = edge_sorted[1:] != edge_sorted[:-1]
    group_idx = np.flatnonzero(group_start)

    a_unique = a_sorted[group_idx]
    b_unique = b_sorted[group_idx]
    q_unique = np.maximum.reduceat(q_sorted, group_idx).astype(np.float32)

    col1_u = (a_unique % nx).astype(np.uint16)
    row1_u = (a_unique // nx).astype(np.uint16)
    col2_u = (b_unique % nx).astype(np.uint16)
    row2_u = (b_unique // nx).astype(np.uint16)
    return col1_u, row1_u, col2_u, row2_u, q_unique, kept_count


def prepare_target(
    *,
    ds: Dataset,
    target: TargetGrid,
    output_dir: Path,
    effective_pressure_scale: float,
    min_channel_discharge: float,
    channel_src: dict[str, np.ndarray],
) -> None:
    with target.meta_path.open("r", encoding="utf-8") as fh:
        bed_meta = json.load(fh)

    grid = bed_meta["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    cell_count = nx * ny
    x0 = float(grid["x0_m"])
    y0 = float(grid["y0_m"])
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])

    src_x = np.asarray(ds.variables["x"][:], dtype=np.float64)
    src_y = np.asarray(ds.variables["y"][:], dtype=np.float64)
    src_nx = src_x.shape[0]
    src_ny = src_y.shape[0]
    src_dx = float(src_x[1] - src_x[0])
    src_dy = float(src_y[1] - src_y[0])

    target_x, target_y = build_target_axis(grid)
    src_ix, src_ix_valid = resolve_indices(target_x, float(src_x[0]), src_dx, src_nx)
    src_iy, src_iy_valid = resolve_indices(target_y, float(src_y[0]), src_dy, src_ny)
    target_cols = np.flatnonzero(src_ix_valid)

    ep_var = ds.variables["effective_pressure"]
    ep_out = np.full((ny, nx), FILL_INT16, dtype=np.int16)

    ep_min = math.inf
    ep_max = -math.inf
    ep_sum = 0.0
    ep_valid_count = 0

    for row in range(ny):
        if not src_iy_valid[row]:
            continue

        sy = int(src_iy[row])
        src_row = np.asarray(ep_var[sy, :], dtype=np.float64)
        sample = src_row[src_ix[target_cols]]

        valid = np.isfinite(sample) & (sample > 0)
        if not np.any(valid):
            continue

        values = sample[valid]
        quantized = quantize_to_int16(values, effective_pressure_scale)

        valid_cols = target_cols[valid]
        ep_out[row, valid_cols] = quantized

        ep_min = min(ep_min, float(np.min(values)))
        ep_max = max(ep_max, float(np.max(values)))
        ep_sum += float(np.sum(values))
        ep_valid_count += int(values.size)

    if ep_valid_count == 0:
        raise RuntimeError(f"No valid effective_pressure values for target {target.label}.")

    col1, row1, col2, row2, q, kept_count = reduce_channel_edges(
        nx=nx,
        ny=ny,
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        x1=channel_src["x1"],
        y1=channel_src["y1"],
        x2=channel_src["x2"],
        y2=channel_src["y2"],
        discharge=channel_src["q"],
        min_discharge=min_channel_discharge,
    )

    if q.size == 0:
        raise RuntimeError(f"No channel segments remain after filtering for target {target.label}.")

    q_min = float(np.min(q))
    q_max = float(np.max(q))
    q_mean = float(np.mean(q))

    out_bin = output_dir / f"{target.output_basename}.bin"
    out_meta = output_dir / f"{target.output_basename}.meta.json"

    ep_bytes = cell_count * 2
    channel_count = int(q.size)

    offset_col1 = ep_bytes
    offset_row1 = offset_col1 + channel_count * 2
    offset_col2 = offset_row1 + channel_count * 2
    offset_row2 = offset_col2 + channel_count * 2
    offset_q = offset_row2 + channel_count * 2

    with out_bin.open("wb") as fh:
        fh.write(ep_out.tobytes(order="C"))
        fh.write(col1.tobytes(order="C"))
        fh.write(row1.tobytes(order="C"))
        fh.write(col2.tobytes(order="C"))
        fh.write(row2.tobytes(order="C"))
        fh.write(q.astype(np.float32).tobytes(order="C"))

    meta = {
        "title": decode_attr(getattr(ds, "Title", None), "Subglacial Hydrology Antarctica"),
        "product_version": decode_attr(getattr(ds, "version", None), "v1.0"),
        "source_file": Path(ds.filepath()).name,
        "resampled_to": target.label,
        "grid": {
            "nx": nx,
            "ny": ny,
            "x0_m": int(x0),
            "y0_m": int(y0),
            "dx_m": int(dx),
            "dy_m": int(dy),
        },
        "quantization": {
            "int16_fill_value": FILL_INT16,
            "effective_pressure_unit": "Pa",
            "effective_pressure_scale_pa_per_int16": effective_pressure_scale,
            "effective_pressure_offset_pa": 0.0,
        },
        "channel_filter": {
            "min_discharge_m3_per_s": float(min_channel_discharge),
            "aggregation": "max discharge on collapsed undirected target-cell edge",
        },
        "coverage": {
            "effective_pressure_valid_count": ep_valid_count,
            "cell_count": cell_count,
            "effective_pressure_valid_fraction": float(ep_valid_count / cell_count),
            "channel_segment_count_raw": int(channel_src["q"].size),
            "channel_segment_count_after_filter": kept_count,
            "channel_segment_count_unique": channel_count,
        },
        "fields": [
            {
                "name": "effective_pressure",
                "dtype": "int16",
                "byte_offset": 0,
                "byte_length": ep_bytes,
                "unit": "Pa",
                "stats_pa": {
                    "min": ep_min,
                    "max": ep_max,
                    "mean": ep_sum / ep_valid_count,
                },
            },
            {
                "name": "channel_col1",
                "dtype": "uint16",
                "byte_offset": offset_col1,
                "byte_length": channel_count * 2,
                "unit": "grid_col",
            },
            {
                "name": "channel_row1",
                "dtype": "uint16",
                "byte_offset": offset_row1,
                "byte_length": channel_count * 2,
                "unit": "grid_row",
            },
            {
                "name": "channel_col2",
                "dtype": "uint16",
                "byte_offset": offset_col2,
                "byte_length": channel_count * 2,
                "unit": "grid_col",
            },
            {
                "name": "channel_row2",
                "dtype": "uint16",
                "byte_offset": offset_row2,
                "byte_length": channel_count * 2,
                "unit": "grid_row",
            },
            {
                "name": "channel_discharge",
                "dtype": "float32",
                "byte_offset": offset_q,
                "byte_length": channel_count * 4,
                "unit": "m3/s",
                "stats_m3_per_s": {
                    "min": q_min,
                    "max": q_max,
                    "mean": q_mean,
                },
            },
        ],
    }

    with out_meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(
        f"[{target.label}] wrote {out_bin} ({out_bin.stat().st_size / (1024 * 1024):.2f} MB), "
        f"effective_pressure valid {ep_valid_count}/{cell_count} "
        f"({(ep_valid_count / cell_count) * 100:.1f}%), channels {channel_count}"
    )
    print(f"[{target.label}] wrote {out_meta}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")
    if args.effective_pressure_scale <= 0:
        raise ValueError("--effective-pressure-scale must be > 0")
    if args.min_channel_discharge < 0:
        raise ValueError("--min-channel-discharge must be >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Dataset(input_path, "r") as ds:
        required = (
            "x",
            "y",
            "effective_pressure",
            "x1_S",
            "y1_S",
            "x2_S",
            "y2_S",
            "channel_discharge",
        )
        for name in required:
            if name not in ds.variables:
                raise RuntimeError(f"Input file missing variable: {name}")

        channel_src = {
            "x1": np.asarray(ds.variables["x1_S"][:], dtype=np.float64),
            "y1": np.asarray(ds.variables["y1_S"][:], dtype=np.float64),
            "x2": np.asarray(ds.variables["x2_S"][:], dtype=np.float64),
            "y2": np.asarray(ds.variables["y2_S"][:], dtype=np.float64),
            "q": np.asarray(ds.variables["channel_discharge"][:], dtype=np.float64),
        }

        for target in DEFAULT_TARGETS:
            if not target.meta_path.exists():
                raise FileNotFoundError(f"Missing BedMachine target meta: {target.meta_path}")
            prepare_target(
                ds=ds,
                target=target,
                output_dir=output_dir,
                effective_pressure_scale=args.effective_pressure_scale,
                min_channel_discharge=args.min_channel_discharge,
                channel_src=channel_src,
            )


if __name__ == "__main__":
    main()
