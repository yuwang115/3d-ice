#!/usr/bin/env python3
"""Translate Antarctica overlay packages onto Bedmap3's native viewer grids.

The source velocity, basal-friction, and hydrology packages were prepared for
BedMachine grids whose origins are offset by 250 m from Bedmap3.  This utility
resamples gridded fields by nearest projected grid cell and reprojects the
subglacial-channel endpoints before writing standalone Bedmap3 packages.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DTYPES = {
    "int16": np.dtype("<i2"),
    "uint16": np.dtype("<u2"),
    "float32": np.dtype("<f4"),
}


@dataclass(frozen=True)
class Target:
    label: str
    terrain_basename: str
    suffix: str
    source_suffix: str


TARGETS = (
    Target("Bedmap3 — Balanced (10 km)", "bedmap3_antarctica_10km", "10km", "480"),
    Target("Bedmap3 — HD (4 km)", "bedmap3_antarctica_4km", "4km", "741"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample existing Antarctica overlays to Bedmap3's native grids."
    )
    parser.add_argument(
        "--data-dir",
        default="static/tools/data",
        help="Directory containing source and destination viewer packages.",
    )
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
        raise ValueError(f"Unsupported axis: {axis}")
    return origin + np.arange(count, dtype=np.float64) * step


def resolve_indices(
    target_axis: np.ndarray, src0: float, src_step: float, src_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest source indices plus a mask of in-coverage samples."""
    indices = np.rint((target_axis - src0) / src_step).astype(np.int64)
    valid = (indices >= 0) & (indices < src_count)
    return np.clip(indices, 0, src_count - 1), valid


def resample_grid(
    source: np.ndarray,
    *,
    source_grid: dict[str, int],
    target_grid: dict[str, int],
    fill_value: float | int,
) -> np.ndarray:
    """Nearest-neighbour resample of a 2D projected grid onto a target grid."""
    if source.shape != (source_grid["ny"], source_grid["nx"]):
        raise ValueError(f"Source array shape {source.shape} does not match its metadata grid.")

    source_x, valid_x = resolve_indices(
        axis_for_grid(target_grid, "x"),
        source_grid["x0_m"],
        source_grid["dx_m"],
        source_grid["nx"],
    )
    source_y, valid_y = resolve_indices(
        axis_for_grid(target_grid, "y"),
        source_grid["y0_m"],
        source_grid["dy_m"],
        source_grid["ny"],
    )
    out = np.full((target_grid["ny"], target_grid["nx"]), fill_value, dtype=source.dtype)
    target_cols = np.flatnonzero(valid_x)
    target_rows = np.flatnonzero(valid_y)
    if target_cols.size and target_rows.size:
        out[np.ix_(target_rows, target_cols)] = source[np.ix_(source_y[target_rows], source_x[target_cols])]
    return out


def reproject_channels(
    *,
    source_grid: dict[str, int],
    target_grid: dict[str, int],
    col1: np.ndarray,
    row1: np.ndarray,
    col2: np.ndarray,
    row2: np.ndarray,
    discharge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproject channel endpoints and reduce duplicate undirected edges by peak discharge."""
    source_col1 = np.asarray(col1, dtype=np.int64)
    source_row1 = np.asarray(row1, dtype=np.int64)
    source_col2 = np.asarray(col2, dtype=np.int64)
    source_row2 = np.asarray(row2, dtype=np.int64)
    source_q = np.asarray(discharge, dtype=np.float32)

    def map_axis(indices: np.ndarray, source_origin: int, source_step: int, target_origin: int, target_step: int) -> np.ndarray:
        coords = source_origin + indices * source_step
        return np.rint((coords - target_origin) / target_step).astype(np.int64)

    target_col1 = map_axis(source_col1, source_grid["x0_m"], source_grid["dx_m"], target_grid["x0_m"], target_grid["dx_m"])
    target_row1 = map_axis(source_row1, source_grid["y0_m"], source_grid["dy_m"], target_grid["y0_m"], target_grid["dy_m"])
    target_col2 = map_axis(source_col2, source_grid["x0_m"], source_grid["dx_m"], target_grid["x0_m"], target_grid["dx_m"])
    target_row2 = map_axis(source_row2, source_grid["y0_m"], source_grid["dy_m"], target_grid["y0_m"], target_grid["dy_m"])

    in_bounds = (
        (target_col1 >= 0)
        & (target_col1 < target_grid["nx"])
        & (target_row1 >= 0)
        & (target_row1 < target_grid["ny"])
        & (target_col2 >= 0)
        & (target_col2 < target_grid["nx"])
        & (target_row2 >= 0)
        & (target_row2 < target_grid["ny"])
    )
    non_degenerate = (target_col1 != target_col2) | (target_row1 != target_row2)
    keep = in_bounds & non_degenerate & np.isfinite(source_q) & (source_q >= 0)
    if not np.any(keep):
        empty_u16 = np.empty(0, dtype=np.uint16)
        return empty_u16, empty_u16, empty_u16, empty_u16, np.empty(0, dtype=np.float32)

    endpoint1 = target_row1[keep] * target_grid["nx"] + target_col1[keep]
    endpoint2 = target_row2[keep] * target_grid["nx"] + target_col2[keep]
    lower = np.minimum(endpoint1, endpoint2)
    upper = np.maximum(endpoint1, endpoint2)
    key = lower.astype(np.uint64) * np.uint64(target_grid["nx"] * target_grid["ny"]) + upper.astype(np.uint64)
    order = np.argsort(key)
    sorted_key = key[order]
    starts = np.empty(sorted_key.shape, dtype=bool)
    starts[0] = True
    starts[1:] = sorted_key[1:] != sorted_key[:-1]
    group_starts = np.flatnonzero(starts)
    unique_lower = lower[order][group_starts]
    unique_upper = upper[order][group_starts]
    unique_discharge = np.maximum.reduceat(source_q[keep][order], group_starts).astype(np.float32)

    nx = target_grid["nx"]
    return (
        (unique_lower % nx).astype(np.uint16),
        (unique_lower // nx).astype(np.uint16),
        (unique_upper % nx).astype(np.uint16),
        (unique_upper // nx).astype(np.uint16),
        unique_discharge,
    )


def read_package(meta_path: Path, bin_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = bin_path.read_bytes()
    fields: dict[str, np.ndarray] = {}
    for field in meta["fields"]:
        if "dtype" not in field:
            continue
        dtype = DTYPES[field["dtype"]]
        byte_length = int(field["byte_length"])
        if byte_length % dtype.itemsize:
            raise ValueError(f"Invalid byte length for {field['name']} in {meta_path.name}.")
        offset = int(field["byte_offset"])
        if offset + byte_length > len(raw):
            raise ValueError(f"Field {field['name']} exceeds {bin_path.name}.")
        fields[field["name"]] = np.frombuffer(raw, dtype=dtype, count=byte_length // dtype.itemsize, offset=offset).copy()
    return meta, fields


def add_array_metadata(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    fields = copy.deepcopy(meta["fields"])
    offset = 0
    for field in fields:
        if field["name"] not in arrays:
            continue
        values = arrays[field["name"]]
        dtype = DTYPES[field["dtype"]]
        byte_length = values.size * dtype.itemsize
        field["byte_offset"] = offset
        field["byte_length"] = byte_length
        offset += byte_length
    return fields


def finite_quantiles(values: np.ndarray, percentiles: tuple[float, ...]) -> dict[str, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    def quantile_key(percentile: float) -> str:
        if percentile == 0.5:
            return "median"
        return f"q{(percentile * 100):g}".replace(".", "")

    if not finite.size:
        return {quantile_key(percentile): math.nan for percentile in percentiles}
    quantiles = np.quantile(finite, percentiles)
    return {quantile_key(percentile): float(value) for percentile, value in zip(percentiles, quantiles, strict=True)}


def update_velocity_metadata(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    quantization = meta["quantization"]
    fill = int(quantization["int16_fill_value"])
    scale = float(quantization["scale"])
    offset = float(quantization.get("offset", 0.0))
    vx = arrays["vx"].astype(np.int16, copy=False)
    vy = arrays["vy"].astype(np.int16, copy=False)
    valid = (vx != fill) & (vy != fill)
    vx_values = vx[valid].astype(np.float64) * scale + offset
    vy_values = vy[valid].astype(np.float64) * scale + offset
    speed = np.hypot(vx_values, vy_values)
    meta["coverage"] = {
        "valid_count": int(valid.sum()),
        "cell_count": int(vx.size),
        "valid_fraction": float(valid.mean()),
    }
    fields = {field["name"]: field for field in meta["fields"]}
    for name, values in (("vx", vx_values), ("vy", vy_values)):
        fields[name]["stats_m_per_year"] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    speed_metadata = {
        "min": float(np.min(speed)),
        "max": float(np.max(speed)),
        "mean": float(np.mean(speed)),
    }
    fields["speed"]["stats_m_per_year"] = speed_metadata
    fields["speed"]["quantiles_m_per_year"] = finite_quantiles(speed, (0.5, 0.9, 0.95, 0.99))


def update_basal_friction_metadata(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    values = arrays["basal_friction"].astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    meta["coverage"] = {
        **meta.get("coverage", {}),
        "valid_count": int(finite.size),
        "cell_count": int(values.size),
        "valid_fraction": float(finite.size / values.size),
    }
    field = next(field for field in meta["fields"] if field["name"] == "basal_friction")
    field["stats_mpa"] = {"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite))}
    field["quantiles_mpa"] = finite_quantiles(finite, (0.5, 0.9, 0.95, 0.99, 0.995))


def update_hydrology_metadata(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    quantization = meta["quantization"]
    fill = int(quantization["int16_fill_value"])
    scale = float(quantization["effective_pressure_scale_pa_per_int16"])
    offset = float(quantization.get("effective_pressure_offset_pa", 0.0))
    pressure = arrays["effective_pressure"].astype(np.int16, copy=False)
    values = pressure[pressure != fill].astype(np.float64) * scale + offset
    discharge = arrays["channel_discharge"].astype(np.float32, copy=False)
    meta["coverage"] = {
        **meta.get("coverage", {}),
        "effective_pressure_valid_count": int(values.size),
        "cell_count": int(pressure.size),
        "effective_pressure_valid_fraction": float(values.size / pressure.size),
        "channel_segment_count_raw": int(meta.get("coverage", {}).get("channel_segment_count_raw", discharge.size)),
        "channel_segment_count_after_filter": int(discharge.size),
        "channel_segment_count_unique": int(discharge.size),
    }
    fields = {field["name"]: field for field in meta["fields"]}
    fields["effective_pressure"]["stats_pa"] = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }
    fields["effective_pressure"]["quantiles_pa"] = finite_quantiles(values, (0.5, 0.9, 0.95, 0.99))
    fields["channel_discharge"]["stats_m3_per_s"] = {
        "min": float(np.min(discharge)),
        "max": float(np.max(discharge)),
        "mean": float(np.mean(discharge)),
    }
    fields["channel_discharge"]["quantiles_m3_per_s"] = finite_quantiles(discharge, (0.5, 0.9, 0.95, 0.99))


def write_package(path: Path, basename: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    meta["fields"] = add_array_metadata(meta, arrays)
    with (path / f"{basename}.bin").open("wb") as output:
        for field in meta["fields"]:
            if field["name"] not in arrays:
                continue
            values = arrays[field["name"]]
            output.write(values.astype(DTYPES[field["dtype"]], copy=False).tobytes(order="C"))
    (path / f"{basename}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def package_metadata(source_meta: dict[str, Any], source_basename: str, target: Target, target_grid: dict[str, int]) -> dict[str, Any]:
    meta = copy.deepcopy(source_meta)
    meta["grid"] = target_grid
    meta["resampled_to"] = target.label
    meta["source_package"] = {
        "metadata": f"{source_basename}.meta.json",
        "binary": f"{source_basename}.bin",
        "source_grid": normalize_grid(source_meta["grid"]),
    }
    meta["resampling"] = {
        "method": "nearest projected source grid cell",
        "reason": "Bedmap3 grid origin differs from the BedMachine-aligned source package by 250 m.",
    }
    return meta


def prepare_target(data_dir: Path, target: Target) -> None:
    terrain_meta = json.loads((data_dir / f"{target.terrain_basename}.meta.json").read_text(encoding="utf-8"))
    target_grid = normalize_grid(terrain_meta["grid"])

    source_velocity = f"antarctic_ice_velocity_phase_v01_{target.source_suffix}"
    velocity_meta, velocity_fields = read_package(data_dir / f"{source_velocity}.meta.json", data_dir / f"{source_velocity}.bin")
    velocity_grid = normalize_grid(velocity_meta["grid"])
    velocity_arrays = {
        name: resample_grid(values.reshape(velocity_grid["ny"], velocity_grid["nx"]), source_grid=velocity_grid, target_grid=target_grid, fill_value=velocity_meta["quantization"]["int16_fill_value"]).ravel()
        for name, values in velocity_fields.items()
    }
    velocity_output = package_metadata(velocity_meta, source_velocity, target, target_grid)
    update_velocity_metadata(velocity_output, velocity_arrays)
    write_package(data_dir, f"bedmap3_antarctica_velocity_{target.suffix}", velocity_output, velocity_arrays)

    source_friction = f"antarctica_basal_friction_{target.source_suffix}"
    friction_meta, friction_fields = read_package(data_dir / f"{source_friction}.meta.json", data_dir / f"{source_friction}.bin")
    friction_grid = normalize_grid(friction_meta["grid"])
    friction_arrays = {
        "basal_friction": resample_grid(
            friction_fields["basal_friction"].reshape(friction_grid["ny"], friction_grid["nx"]),
            source_grid=friction_grid,
            target_grid=target_grid,
            fill_value=np.nan,
        ).ravel()
    }
    friction_output = package_metadata(friction_meta, source_friction, target, target_grid)
    update_basal_friction_metadata(friction_output, friction_arrays)
    write_package(data_dir, f"bedmap3_antarctica_basal_friction_{target.suffix}", friction_output, friction_arrays)

    source_hydrology = f"antarctica_subglacial_hydrology_{target.source_suffix}"
    hydrology_meta, hydrology_fields = read_package(data_dir / f"{source_hydrology}.meta.json", data_dir / f"{source_hydrology}.bin")
    hydrology_grid = normalize_grid(hydrology_meta["grid"])
    hydrology_arrays = {
        "effective_pressure": resample_grid(
            hydrology_fields["effective_pressure"].reshape(hydrology_grid["ny"], hydrology_grid["nx"]),
            source_grid=hydrology_grid,
            target_grid=target_grid,
            fill_value=hydrology_meta["quantization"]["int16_fill_value"],
        ).ravel()
    }
    channel_arrays = reproject_channels(
        source_grid=hydrology_grid,
        target_grid=target_grid,
        col1=hydrology_fields["channel_col1"],
        row1=hydrology_fields["channel_row1"],
        col2=hydrology_fields["channel_col2"],
        row2=hydrology_fields["channel_row2"],
        discharge=hydrology_fields["channel_discharge"],
    )
    hydrology_arrays.update(
        dict(zip(("channel_col1", "channel_row1", "channel_col2", "channel_row2", "channel_discharge"), channel_arrays, strict=True))
    )
    hydrology_output = package_metadata(hydrology_meta, source_hydrology, target, target_grid)
    update_hydrology_metadata(hydrology_output, hydrology_arrays)
    write_package(data_dir, f"bedmap3_antarctica_subglacial_hydrology_{target.suffix}", hydrology_output, hydrology_arrays)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    for target in TARGETS:
        prepare_target(data_dir, target)
        print(f"Prepared Bedmap3 overlay packages for {target.label}.")


if __name__ == "__main__":
    main()
