#!/usr/bin/env python3
"""Prepare 3D Greenland ocean streamlines from Copernicus monthly Arctic physics data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_SAMPLE_STRIDE = 6
DEFAULT_BEDMACHINE_MARGIN_M = 450_000.0
DEFAULT_TARGET_STREAMLINES = 1200
DEFAULT_MIN_SPEED_MPS = 0.0
DEFAULT_MIN_SEED_SPEED_MPS = 0.0
DEFAULT_MIN_TRACE_SPEED_MPS = 0.001
DEFAULT_WATER_COLUMN_CLEARANCE_M = 20.0
DEFAULT_FLOWLINE_STEP_CELLS = 0.75
DEFAULT_FLOWLINE_MAX_STEPS = 72
DEFAULT_MIN_STREAMLINE_SEGMENTS = 5
DEFAULT_MIN_UNIQUE_XY_CELLS = 4
DEFAULT_MIN_NET_DISPLACEMENT_CELLS = 1.0
DEFAULT_VERTICAL_SEED_SEPARATION_M = 120.0
DEFAULT_SEED_DEPTH_ATTEMPTS = 6
DEFAULT_SECTOR_COUNT = 8
DEFAULT_RANDOM_SEED = 3413
DEFAULT_SPATIAL_BIN_SIZE_MIN = 4
DEFAULT_SPATIAL_BIN_SIZE_MAX = 10
GREENLAND_LAYER_BUCKETS = (
    {
        "key": "greenland_surface",
        "label": "Greenland ocean surface layer",
        "retain_fraction": 0.25,
        "candidate_factor": 5.0,
        "min_total_depth_m": 60.0,
        "depth_fraction_range": (0.03, 0.18),
        "proxy_fraction": 0.10,
        "depth_beta": (1.0, 1.0),
        "min_speed_mps": 0.0,
        "min_seed_speed_mps": 0.0,
        "min_trace_speed_mps": 0.001,
        "min_streamline_segments": 4,
        "min_unique_xy_cells": 3,
        "min_net_displacement_cells": 0.5,
    },
    {
        "key": "greenland_upper",
        "label": "Greenland ocean upper layer",
        "retain_fraction": 0.25,
        "candidate_factor": 5.0,
        "min_total_depth_m": 80.0,
        "depth_fraction_range": (0.18, 0.40),
        "proxy_fraction": 0.30,
        "depth_beta": (1.0, 1.0),
        "min_speed_mps": 0.0,
        "min_seed_speed_mps": 0.0,
        "min_trace_speed_mps": 0.001,
        "min_streamline_segments": 4,
        "min_unique_xy_cells": 3,
        "min_net_displacement_cells": 0.5,
    },
    {
        "key": "greenland_mid",
        "label": "Greenland ocean mid-water layer",
        "retain_fraction": 0.25,
        "candidate_factor": 5.0,
        "min_total_depth_m": 120.0,
        "depth_fraction_range": (0.40, 0.70),
        "proxy_fraction": 0.55,
        "depth_beta": (1.0, 1.0),
        "min_speed_mps": 0.0,
        "min_seed_speed_mps": 0.0,
        "min_trace_speed_mps": 0.001,
        "min_streamline_segments": 4,
        "min_unique_xy_cells": 3,
        "min_net_displacement_cells": 0.5,
    },
    {
        "key": "greenland_lower",
        "label": "Greenland ocean lower layer",
        "retain_fraction": 0.25,
        "candidate_factor": 5.0,
        "min_total_depth_m": 180.0,
        "depth_fraction_range": (0.70, 0.95),
        "proxy_fraction": 0.82,
        "depth_beta": (1.0, 1.0),
        "min_speed_mps": 0.0,
        "min_seed_speed_mps": 0.0,
        "min_trace_speed_mps": 0.001,
        "min_streamline_segments": 4,
        "min_unique_xy_cells": 3,
        "min_net_displacement_cells": 0.5,
    },
)
WGS84_A = 6_378_137.0
WGS84_E = 0.08181919084262149
PS70_LAT_TS_RAD = math.radians(70.0)
PS3413_LON0_RAD = math.radians(-45.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 3D Greenland ocean streamlines from Copernicus monthly Arctic physics data."
    )
    parser.add_argument("--input", required=True, help="Path to the Copernicus monthly NetCDF file.")
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output .bin/.meta.json files.",
    )
    parser.add_argument(
        "--bedmachine-meta",
        default="static/tools/data/bedmachine_greenland_v6_3km.meta.json",
        help="BedMachine Greenland metadata used to clip/project the ocean points.",
    )
    parser.add_argument(
        "--bedmachine-bin",
        default="static/tools/data/bedmachine_greenland_v6_3km.bin",
        help="BedMachine Greenland binary used to read the ocean/land mask.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=DEFAULT_SAMPLE_STRIDE,
        help="Subsample every Nth latitude/longitude grid cell before tracing streamlines.",
    )
    parser.add_argument(
        "--margin-m",
        type=float,
        default=DEFAULT_BEDMACHINE_MARGIN_M,
        help="Extra padding around the BedMachine Greenland extent in EPSG:3413 meters.",
    )
    parser.add_argument(
        "--min-speed-mps",
        type=float,
        default=DEFAULT_MIN_SPEED_MPS,
        help="Minimum horizontal speed required for a valid ocean sample.",
    )
    parser.add_argument(
        "--min-seed-speed-mps",
        type=float,
        default=DEFAULT_MIN_SEED_SPEED_MPS,
        help="Minimum horizontal speed required for a streamline seed.",
    )
    parser.add_argument(
        "--min-trace-speed-mps",
        type=float,
        default=DEFAULT_MIN_TRACE_SPEED_MPS,
        help="Minimum horizontal speed required to continue tracing a streamline.",
    )
    parser.add_argument(
        "--target-streamlines",
        type=int,
        default=DEFAULT_TARGET_STREAMLINES,
        help="Approximate target number of retained Greenland streamlines across the four depth layers.",
    )
    parser.add_argument(
        "--flowline-step-cells",
        type=float,
        default=DEFAULT_FLOWLINE_STEP_CELLS,
        help="Target horizontal tracing step in sampled-grid cells.",
    )
    parser.add_argument(
        "--flowline-max-steps",
        type=int,
        default=DEFAULT_FLOWLINE_MAX_STEPS,
        help="Maximum traced steps in each direction from a seed.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for Greenland ocean seeding and selection.",
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


def t_func(phi_rad: np.ndarray) -> np.ndarray:
    sin_phi = np.sin(phi_rad)
    ratio = (1 - WGS84_E * sin_phi) / (1 + WGS84_E * sin_phi)
    return np.tan(np.pi / 4 - phi_rad / 2) / np.power(ratio, WGS84_E / 2)


PS70_MC = math.cos(PS70_LAT_TS_RAD) / math.sqrt(1 - (WGS84_E * math.sin(PS70_LAT_TS_RAD)) ** 2)
PS70_TC = float(t_func(np.array([PS70_LAT_TS_RAD], dtype=np.float64))[0])


def project_epsg3413(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = np.deg2rad(lat_deg.astype(np.float64))
    lam = np.deg2rad(lon_deg.astype(np.float64))
    t = t_func(phi)
    rho = WGS84_A * PS70_MC * t / PS70_TC
    dlam = lam - PS3413_LON0_RAD
    x = rho * np.sin(dlam)
    y = -rho * np.cos(dlam)
    return x.astype(np.float32), y.astype(np.float32)


def load_bedmachine_mask(meta_path: Path, bin_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mask_field = next(field for field in meta["fields"] if field["name"] == "mask")
    raw = bin_path.read_bytes()
    mask = np.frombuffer(
        raw,
        dtype=np.uint8,
        count=int(mask_field["byte_length"]),
        offset=int(mask_field["byte_offset"]),
    ).copy()
    grid = meta["grid"]
    expected = int(grid["nx"]) * int(grid["ny"])
    if mask.size != expected:
        raise RuntimeError("BedMachine mask length does not match BedMachine grid.")
    return meta, mask.reshape((int(grid["ny"]), int(grid["nx"])))


def project_to_bedmachine_grid(grid: dict[str, Any], x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col = (x_m - float(grid["x0_m"])) / float(grid["dx_m"])
    row = (y_m - float(grid["y0_m"])) / float(grid["dy_m"])
    return col, row


def hash_unit_interval(row: int, col: int, salt: int = 0) -> float:
    value = math.sin((row + 1) * 12.9898 + (col + 1) * 78.233 + (salt + 1) * 37.719) * 43758.5453123
    return value - math.floor(value)


def percentile_range(values: np.ndarray, lo: float, hi: float) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0, 1.0]
    lo_val, hi_val = np.percentile(finite, [lo, hi])
    if not np.isfinite(lo_val) or not np.isfinite(hi_val) or hi_val <= lo_val:
        return [float(np.nanmin(finite)), float(np.nanmax(finite))]
    return [float(lo_val), float(hi_val)]


def stats_dict(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def build_default_basename(source_time_iso: str) -> str:
    stamp = source_time_iso[:7].replace("-", "")
    return f"greenland_ocean_currents_cmems_{stamp}"


def load_sparse_volume(dataset: h5py.Dataset, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    depth_count = int(dataset.shape[1])
    out = np.empty((depth_count, lat_idx.size, lon_idx.size), dtype=np.float32)
    for depth_idx in range(depth_count):
        out[depth_idx] = np.asarray(dataset[0, depth_idx], dtype=np.float32)[np.ix_(lat_idx, lon_idx)]
    return out


def bilinear_sample(array2d: np.ndarray, row: float, col: float) -> float:
    if row < 0 or col < 0 or row > array2d.shape[0] - 1 or col > array2d.shape[1] - 1:
        return float("nan")
    r0 = int(math.floor(row))
    c0 = int(math.floor(col))
    r1 = min(array2d.shape[0] - 1, r0 + 1)
    c1 = min(array2d.shape[1] - 1, c0 + 1)
    tx = col - c0
    ty = row - r0
    values = (
        float(array2d[r0, c0]),
        float(array2d[r0, c1]),
        float(array2d[r1, c0]),
        float(array2d[r1, c1]),
    )
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty
    return values[0] * w00 + values[1] * w10 + values[2] * w01 + values[3] * w11


def bilinear_mask_valid(mask2d: np.ndarray, row: float, col: float) -> bool:
    if row < 0 or col < 0 or row >= mask2d.shape[0] - 1 or col >= mask2d.shape[1] - 1:
        return False
    r0 = int(math.floor(row))
    c0 = int(math.floor(col))
    r1 = min(mask2d.shape[0] - 1, r0 + 1)
    c1 = min(mask2d.shape[1] - 1, c0 + 1)
    return bool(mask2d[r0, c0] and mask2d[r0, c1] and mask2d[r1, c0] and mask2d[r1, c1])


def depth_index_fraction(depth_axis: np.ndarray, depth_m: float) -> tuple[int, int, float] | None:
    if not math.isfinite(depth_m):
        return None
    if depth_m < float(depth_axis[0]) or depth_m > float(depth_axis[-1]):
        return None
    upper = int(np.searchsorted(depth_axis, depth_m, side="right"))
    if upper <= 0:
        return 0, 1, 0.0
    if upper >= depth_axis.size:
        return depth_axis.size - 2, depth_axis.size - 1, 1.0
    lower = upper - 1
    z0 = float(depth_axis[lower])
    z1 = float(depth_axis[upper])
    if z1 <= z0:
        return lower, upper, 0.0
    return lower, upper, (depth_m - z0) / (z1 - z0)


def trilinear_sample(volume: np.ndarray, depth_axis: np.ndarray, depth_m: float, row: float, col: float) -> float:
    depth_state = depth_index_fraction(depth_axis, depth_m)
    if depth_state is None:
        return float("nan")
    lower, upper, tz = depth_state
    value0 = bilinear_sample(volume[lower], row, col)
    value1 = bilinear_sample(volume[upper], row, col)
    if not math.isfinite(value0) or not math.isfinite(value1):
        return float("nan")
    return value0 * (1 - tz) + value1 * tz


def interpolate_depth_plane(volume: np.ndarray, depth_axis: np.ndarray, depth_m: float) -> np.ndarray:
    depth_state = depth_index_fraction(depth_axis, depth_m)
    if depth_state is None:
        raise ValueError(f"Requested depth {depth_m} m is outside the source depth axis.")
    lower, upper, tz = depth_state
    if lower == upper or tz <= 1e-9:
        return np.asarray(volume[lower], dtype=np.float32)
    return (np.asarray(volume[lower], dtype=np.float32) * (1 - tz) + np.asarray(volume[upper], dtype=np.float32) * tz).astype(
        np.float32
    )


def meters_per_degree_lat(lat_deg: float) -> float:
    phi = math.radians(lat_deg)
    return (
        111132.92
        - 559.82 * math.cos(2 * phi)
        + 1.175 * math.cos(4 * phi)
        - 0.0023 * math.cos(6 * phi)
    )


def meters_per_degree_lon(lat_deg: float) -> float:
    phi = math.radians(lat_deg)
    return (
        111412.84 * math.cos(phi)
        - 93.5 * math.cos(3 * phi)
        + 0.118 * math.cos(5 * phi)
    )


def collect_seeds(speed2d: np.ndarray, valid2d: np.ndarray, min_seed_speed: float, seed_target: int) -> list[tuple[int, int]]:
    valid_count = int(np.count_nonzero(valid2d))
    if valid_count == 0:
        return []

    seed_spacing = max(3, int(round(math.sqrt(valid_count / max(1, seed_target)))))
    half_offset = max(1, seed_spacing // 2)
    seeds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for offset in (0, half_offset):
        row_start = offset
        col_start = offset
        for row_block in range(row_start, speed2d.shape[0] - 1, seed_spacing):
            row_end = min(speed2d.shape[0] - 1, row_block + seed_spacing)
            for col_block in range(col_start, speed2d.shape[1] - 1, seed_spacing):
                col_end = min(speed2d.shape[1] - 1, col_block + seed_spacing)
                best: tuple[int, int] | None = None
                best_speed = min_seed_speed
                for row in range(row_block, row_end):
                    for col in range(col_block, col_end):
                        if not valid2d[row, col]:
                            continue
                        candidate = float(speed2d[row, col])
                        if not math.isfinite(candidate) or candidate < best_speed:
                            continue
                        best_speed = candidate
                        best = (row, col)
                if best and best not in seen:
                    seen.add(best)
                    seeds.append(best)
    return seeds


def collect_mask_aware_poisson_disk_cells(
    valid2d: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
    *,
    search_iterations: int = 14,
) -> list[tuple[int, int]]:
    valid_rows, valid_cols = np.nonzero(valid2d)
    valid_count = int(valid_rows.size)
    if valid_count == 0 or target_count <= 0:
        return []
    if valid_count <= target_count:
        order = rng.permutation(valid_count)
        return [(int(valid_rows[index]), int(valid_cols[index])) for index in order.tolist()]

    order = rng.permutation(valid_count)
    rows = valid_rows[order].astype(np.int32, copy=False)
    cols = valid_cols[order].astype(np.int32, copy=False)

    def greedy(radius_cells: float, *, stop_at_target: bool) -> list[tuple[int, int]] | None:
        if radius_cells <= 1e-6:
            limit = min(target_count, valid_count) if stop_at_target else valid_count
            return [(int(rows[index]), int(cols[index])) for index in range(limit)]

        bin_size = max(1.0, float(radius_cells))
        radius_sq = float(radius_cells) * float(radius_cells)
        bins: dict[tuple[int, int], list[tuple[int, int]]] = {}
        accepted: list[tuple[int, int]] = []

        for row, col in zip(rows.tolist(), cols.tolist()):
            bin_row = int(math.floor(float(row) / bin_size))
            bin_col = int(math.floor(float(col) / bin_size))
            allowed = True
            for neighbor_row in range(bin_row - 1, bin_row + 2):
                for neighbor_col in range(bin_col - 1, bin_col + 2):
                    for other_row, other_col in bins.get((neighbor_row, neighbor_col), []):
                        drow = float(row - other_row)
                        dcol = float(col - other_col)
                        if drow * drow + dcol * dcol < radius_sq:
                            allowed = False
                            break
                    if not allowed:
                        break
                if not allowed:
                    break
            if not allowed:
                continue

            accepted.append((int(row), int(col)))
            bins.setdefault((bin_row, bin_col), []).append((int(row), int(col)))
            if stop_at_target and len(accepted) >= target_count:
                return accepted

        return accepted

    low_radius = 0.0
    high_radius = float(max(valid2d.shape))
    best_radius = 0.0

    for _ in range(max(4, int(search_iterations))):
        trial_radius = 0.5 * (low_radius + high_radius)
        accepted = greedy(trial_radius, stop_at_target=True)
        if accepted is not None and len(accepted) >= target_count:
            best_radius = trial_radius
            low_radius = trial_radius
        else:
            high_radius = trial_radius

    final_seeds = greedy(best_radius, stop_at_target=True)
    if final_seeds is None:
        return []
    return final_seeds[:target_count]


def compute_bucket_targets(target_streamlines: int, seed_buckets: tuple[dict[str, Any], ...]) -> list[int]:
    raw_bucket_targets = [float(bucket["retain_fraction"]) * float(target_streamlines) for bucket in seed_buckets]
    bucket_targets = [int(math.floor(value)) for value in raw_bucket_targets]
    remaining_slots = max(0, int(target_streamlines) - sum(bucket_targets))
    remainders = sorted(
        ((raw - base, index) for index, (raw, base) in enumerate(zip(raw_bucket_targets, bucket_targets))),
        reverse=True,
    )
    for _, index in remainders[:remaining_slots]:
        bucket_targets[index] += 1
    return bucket_targets


def compute_weighted_targets(weights: list[int] | np.ndarray, target_count: int) -> list[int]:
    weights_array = np.asarray(weights, dtype=np.float64)
    if target_count <= 0 or weights_array.size == 0:
        return [0 for _ in range(int(weights_array.size))]

    positive = np.isfinite(weights_array) & (weights_array > 0)
    if not np.any(positive):
        return [0 for _ in range(int(weights_array.size))]

    targets = np.zeros(weights_array.shape, dtype=np.int32)
    positive_indices = np.flatnonzero(positive)
    guaranteed = min(int(target_count), int(positive_indices.size))
    if guaranteed > 0:
        ranked_positive = sorted(
            positive_indices.tolist(),
            key=lambda index: (float(weights_array[index]), -int(index)),
            reverse=True,
        )
        for index in ranked_positive[:guaranteed]:
            targets[index] += 1

    remaining = int(target_count) - int(np.sum(targets))
    if remaining <= 0:
        return targets.astype(int).tolist()

    positive_weights = weights_array[positive]
    weight_sum = float(np.sum(positive_weights))
    if weight_sum <= 0:
        return targets.astype(int).tolist()

    raw_additional = (positive_weights / weight_sum) * float(remaining)
    base_additional = np.floor(raw_additional).astype(np.int32)
    targets[positive] += base_additional
    leftover = int(target_count) - int(np.sum(targets))
    if leftover > 0:
        remainders = raw_additional - base_additional.astype(np.float64)
        order = np.argsort(-remainders, kind="stable")
        for rank in order[:leftover].tolist():
            targets[positive_indices[rank]] += 1
    return targets.astype(int).tolist()


def sector_indices_from_xy(x_values: np.ndarray, y_values: np.ndarray, sector_count: int = DEFAULT_SECTOR_COUNT) -> np.ndarray:
    count = max(1, int(sector_count))
    angles = np.mod(np.arctan2(y_values, x_values), 2.0 * np.pi)
    sector_width = (2.0 * np.pi) / float(count)
    indices = np.floor(angles / sector_width).astype(np.int32)
    return np.clip(indices, 0, count - 1)


def mask_sector_counts(mask2d: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, sector_count: int = DEFAULT_SECTOR_COUNT) -> list[int]:
    count = max(1, int(sector_count))
    rows, cols = np.nonzero(mask2d)
    if rows.size == 0:
        return [0 for _ in range(count)]
    sector_indices = sector_indices_from_xy(x_grid[rows, cols], y_grid[rows, cols], count)
    counts = np.bincount(sector_indices, minlength=count)
    return counts.astype(int).tolist()


def spatial_bin_key(row: int, col: int, bin_size: int) -> tuple[int, int]:
    safe_size = max(1, int(bin_size))
    return int(row) // safe_size, int(col) // safe_size


def compute_spatial_bin_size(valid2d: np.ndarray, target_count: int) -> int:
    valid_count = int(np.count_nonzero(valid2d))
    if valid_count <= 0 or target_count <= 0:
        return DEFAULT_SPATIAL_BIN_SIZE_MAX
    raw_size = int(round(math.sqrt(valid_count / max(1, target_count))))
    return max(DEFAULT_SPATIAL_BIN_SIZE_MIN, min(DEFAULT_SPATIAL_BIN_SIZE_MAX, raw_size))


def compute_spatial_bin_size_for_streamlines(candidates: list[dict[str, Any]], target_count: int) -> int:
    if not candidates or target_count <= 0:
        return DEFAULT_SPATIAL_BIN_SIZE_MAX
    unique_seed_cells = {
        (int(streamline["seed_row"]), int(streamline["seed_col"]))
        for streamline in candidates
    }
    raw_size = int(round(math.sqrt(len(unique_seed_cells) / max(1, target_count))))
    return max(DEFAULT_SPATIAL_BIN_SIZE_MIN, min(DEFAULT_SPATIAL_BIN_SIZE_MAX, raw_size))


def streamline_sector_index(streamline: dict[str, Any], sector_count: int = DEFAULT_SECTOR_COUNT) -> int:
    count = max(1, int(sector_count))
    seed_x = float(streamline["seed_x_m"])
    seed_y = float(streamline["seed_y_m"])
    if not math.isfinite(seed_x) or not math.isfinite(seed_y):
        return 0
    angle = (math.atan2(seed_y, seed_x) + 2.0 * math.pi) % (2.0 * math.pi)
    sector_width = (2.0 * math.pi) / float(count)
    return min(count - 1, max(0, int(math.floor(angle / sector_width))))


def retained_seed_sector_counts(streamlines: list[dict[str, Any]], sector_count: int = DEFAULT_SECTOR_COUNT) -> list[int]:
    counts = [0 for _ in range(max(1, int(sector_count)))]
    for streamline in streamlines:
        counts[streamline_sector_index(streamline, len(counts))] += 1
    return counts


def streamline_rank_key(streamline: dict[str, Any]) -> tuple[int, float]:
    return int(streamline["segment_count"]), float(streamline.get("selection_rank_secondary", streamline.get("seed_speed", 0.0)))


def select_streamlines_spatially(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
    selection_bin_size: int,
    depth_bin_size: int,
    retained_depths_by_bin: dict[tuple[int, int], list[float]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if target_count <= 0 or not candidates:
        return [], list(candidates)

    bin_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for candidate in candidates:
        bin_key = spatial_bin_key(int(candidate["seed_row"]), int(candidate["seed_col"]), selection_bin_size)
        bin_groups.setdefault(bin_key, []).append(candidate)

    ordered_bin_keys = list(bin_groups.keys())
    if ordered_bin_keys:
        permutation = rng.permutation(len(ordered_bin_keys)).tolist()
        ordered_bin_keys = [ordered_bin_keys[index] for index in permutation]
    for bin_key in ordered_bin_keys:
        bin_groups[bin_key] = sorted(bin_groups[bin_key], key=streamline_rank_key, reverse=True)

    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    made_progress = True
    while len(selected) < target_count and made_progress:
        made_progress = False
        for bin_key in ordered_bin_keys:
            queue = bin_groups[bin_key]
            while queue:
                candidate = queue.pop(0)
                depth_key = spatial_bin_key(int(candidate["seed_row"]), int(candidate["seed_col"]), depth_bin_size)
                existing_depths = retained_depths_by_bin.setdefault(depth_key, [])
                if any(
                    abs(float(candidate["seed_depth_m"]) - float(existing_depth)) < DEFAULT_VERTICAL_SEED_SEPARATION_M
                    for existing_depth in existing_depths
                ):
                    leftovers.append(candidate)
                    continue
                selected.append(candidate)
                existing_depths.append(float(candidate["seed_depth_m"]))
                made_progress = True
                break
            if len(selected) >= target_count:
                break

    for queue in bin_groups.values():
        leftovers.extend(queue)

    if len(selected) >= target_count:
        return selected[:target_count], leftovers

    if not leftovers:
        return selected, leftovers

    leftovers.sort(key=streamline_rank_key, reverse=True)
    remaining: list[dict[str, Any]] = []
    for candidate in leftovers:
        if len(selected) >= target_count:
            remaining.append(candidate)
            continue
        depth_key = spatial_bin_key(int(candidate["seed_row"]), int(candidate["seed_col"]), depth_bin_size)
        existing_depths = retained_depths_by_bin.setdefault(depth_key, [])
        if any(
            abs(float(candidate["seed_depth_m"]) - float(existing_depth)) < DEFAULT_VERTICAL_SEED_SEPARATION_M
            for existing_depth in existing_depths
        ):
            remaining.append(candidate)
            continue
        selected.append(candidate)
        existing_depths.append(float(candidate["seed_depth_m"]))

    return selected, remaining


def select_streamlines_spatially_by_sector_targets(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
    sector_targets: list[int],
    selection_bin_size: int,
    depth_bin_size: int,
    retained_depths_by_bin: dict[tuple[int, int], list[float]],
    retained_sector_counts: list[int],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if target_count <= 0 or not candidates:
        return [], list(candidates)

    sector_count = max(1, int(DEFAULT_SECTOR_COUNT))
    candidates_by_sector: dict[int, list[dict[str, Any]]] = {sector: [] for sector in range(sector_count)}
    for candidate in candidates:
        candidates_by_sector[streamline_sector_index(candidate, sector_count)].append(candidate)

    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    sector_order = rng.permutation(sector_count).tolist()
    for sector in sector_order:
        sector_target = int(sector_targets[sector]) if sector < len(sector_targets) else 0
        sector_candidates = candidates_by_sector.get(sector, [])
        if sector_target <= 0:
            leftovers.extend(sector_candidates)
            continue
        sector_selection_bin_size = compute_spatial_bin_size_for_streamlines(sector_candidates, sector_target)
        selected_sector, leftovers_sector = select_streamlines_spatially(
            sector_candidates,
            target_count=sector_target,
            selection_bin_size=sector_selection_bin_size,
            depth_bin_size=depth_bin_size,
            retained_depths_by_bin=retained_depths_by_bin,
            rng=rng,
        )
        selected.extend(selected_sector)
        retained_sector_counts[sector] += len(selected_sector)
        leftovers.extend(leftovers_sector)

    return selected, leftovers


def count_unique_xy_cells(row_values: np.ndarray, col_values: np.ndarray) -> int:
    cells = {
        (int(round(float(row_value))), int(round(float(col_value))))
        for row_value, col_value in zip(row_values.tolist(), col_values.tolist())
    }
    return len(cells)


def net_displacement_cells(row_values: np.ndarray, col_values: np.ndarray) -> float:
    if row_values.size < 2 or col_values.size < 2:
        return 0.0
    drow = float(row_values[-1]) - float(row_values[0])
    dcol = float(col_values[-1]) - float(col_values[0])
    return math.hypot(drow, dcol)


def sample_stream_state(
    *,
    depth_axis: np.ndarray,
    depth_m: float,
    row: float,
    col: float,
    water_mask_2d: np.ndarray,
    model_depth_2d: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u_volume: np.ndarray,
    v_volume: np.ndarray,
    w_volume: np.ndarray,
    theta_volume: np.ndarray,
    sal_volume: np.ndarray,
    clearance_m: float,
    lon_step_deg: float,
    lat_step_deg: float,
) -> dict[str, float] | None:
    if not bilinear_mask_valid(water_mask_2d, row, col):
        return None

    model_depth_here = bilinear_sample(model_depth_2d, row, col)
    if not math.isfinite(model_depth_here) or model_depth_here < depth_m + clearance_m:
        return None

    lat = bilinear_sample(lat_grid, row, col)
    lon = bilinear_sample(lon_grid, row, col)
    x = bilinear_sample(x_grid, row, col)
    y = bilinear_sample(y_grid, row, col)
    u = trilinear_sample(u_volume, depth_axis, depth_m, row, col)
    v = trilinear_sample(v_volume, depth_axis, depth_m, row, col)
    w = trilinear_sample(w_volume, depth_axis, depth_m, row, col)
    theta = trilinear_sample(theta_volume, depth_axis, depth_m, row, col)
    sal = trilinear_sample(sal_volume, depth_axis, depth_m, row, col)
    if not all(math.isfinite(value) for value in (lat, lon, x, y, u, v, w, theta, sal)):
        return None

    speed = math.hypot(u, v)
    if not math.isfinite(speed):
        return None

    lon_metric = max(1.0, meters_per_degree_lon(lat) * lon_step_deg)
    lat_metric = max(1.0, meters_per_degree_lat(lat) * lat_step_deg)
    dcol_dt = u / lon_metric
    drow_dt = v / lat_metric
    horizontal_cell_speed = math.hypot(dcol_dt, drow_dt)
    if not math.isfinite(horizontal_cell_speed) or horizontal_cell_speed < 1e-10:
        return None

    return {
        "x": x,
        "y": y,
        "depth": depth_m,
        "sample_row": row,
        "sample_col": col,
        "theta": theta,
        "sal": sal,
        "u": u,
        "v": v,
        "w": w,
        "speed": speed,
        "dcol_dt": dcol_dt,
        "drow_dt": drow_dt,
        "horizontal_cell_speed": horizontal_cell_speed,
    }


def trace_streamline_direction(
    *,
    depth_axis: np.ndarray,
    seed_depth_m: float,
    seed_row: int,
    seed_col: int,
    direction: float,
    water_mask_2d: np.ndarray,
    model_depth_2d: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u_volume: np.ndarray,
    v_volume: np.ndarray,
    w_volume: np.ndarray,
    theta_volume: np.ndarray,
    sal_volume: np.ndarray,
    clearance_m: float,
    lon_step_deg: float,
    lat_step_deg: float,
    step_cells: float,
    min_trace_speed: float,
    max_steps: int,
) -> list[dict[str, float]]:
    states: list[dict[str, float]] = []
    row = float(seed_row)
    col = float(seed_col)
    depth_m = float(seed_depth_m)

    for _ in range(max_steps):
        state = sample_stream_state(
            depth_axis=depth_axis,
            depth_m=depth_m,
            row=row,
            col=col,
            water_mask_2d=water_mask_2d,
            model_depth_2d=model_depth_2d,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            u_volume=u_volume,
            v_volume=v_volume,
            w_volume=w_volume,
            theta_volume=theta_volume,
            sal_volume=sal_volume,
            clearance_m=clearance_m,
            lon_step_deg=lon_step_deg,
            lat_step_deg=lat_step_deg,
        )
        if state is None or state["speed"] < min_trace_speed:
            break

        states.append(state)
        dt_seconds = step_cells / state["horizontal_cell_speed"]
        next_row = row + direction * state["drow_dt"] * dt_seconds
        next_col = col + direction * state["dcol_dt"] * dt_seconds
        next_depth_m = depth_m - direction * state["w"] * dt_seconds

        if (
            not math.isfinite(next_row)
            or not math.isfinite(next_col)
            or not math.isfinite(next_depth_m)
            or next_depth_m < float(depth_axis[0])
            or next_depth_m > float(depth_axis[-1])
        ):
            break

        if (
            abs(next_row - row) < 1e-6
            and abs(next_col - col) < 1e-6
            and abs(next_depth_m - depth_m) < 1e-4
        ):
            break

        row = next_row
        col = next_col
        depth_m = next_depth_m

    return states


def choose_seed_depth(
    model_depth_m: float,
    *,
    depth_fraction_range: tuple[float, float],
    proxy_fraction: float,
    rng: np.random.Generator,
    beta_shape: tuple[float, float] = (1.0, 1.0),
    clearance_m: float = DEFAULT_WATER_COLUMN_CLEARANCE_M,
    attempts: int = DEFAULT_SEED_DEPTH_ATTEMPTS,
) -> list[float]:
    if not math.isfinite(model_depth_m) or model_depth_m <= 2.0 * clearance_m:
        return []
    lower = max(clearance_m, model_depth_m * float(depth_fraction_range[0]))
    upper = min(model_depth_m - clearance_m, model_depth_m * float(depth_fraction_range[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return []

    candidate_depths: list[float] = []
    alpha = max(1e-6, float(beta_shape[0]))
    beta = max(1e-6, float(beta_shape[1]))
    for _ in range(max(1, int(attempts))):
        depth_fraction = float(rng.beta(alpha, beta))
        candidate_depths.append(lower + depth_fraction * (upper - lower))

    proxy_depth = min(model_depth_m - clearance_m, max(clearance_m, model_depth_m * float(proxy_fraction)))
    if all(abs(proxy_depth - candidate) >= 1e-3 for candidate in candidate_depths):
        candidate_depths.append(proxy_depth)
    return candidate_depths


def build_traced_streamline(
    *,
    depth_axis: np.ndarray,
    seed_depth_m: float,
    seed_row: int,
    seed_col: int,
    seed_state: dict[str, float],
    water_mask_2d: np.ndarray,
    model_depth_2d: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u_volume: np.ndarray,
    v_volume: np.ndarray,
    w_volume: np.ndarray,
    theta_volume: np.ndarray,
    sal_volume: np.ndarray,
    clearance_m: float,
    lon_step_deg: float,
    lat_step_deg: float,
    step_cells: float,
    min_trace_speed: float,
    max_steps: int,
    min_segment_count: int,
    min_unique_xy_cells: int,
    min_net_displacement_cells: float,
) -> dict[str, Any] | None:
    forward = trace_streamline_direction(
        depth_axis=depth_axis,
        seed_depth_m=seed_depth_m,
        seed_row=seed_row,
        seed_col=seed_col,
        direction=1.0,
        water_mask_2d=water_mask_2d,
        model_depth_2d=model_depth_2d,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        u_volume=u_volume,
        v_volume=v_volume,
        w_volume=w_volume,
        theta_volume=theta_volume,
        sal_volume=sal_volume,
        clearance_m=clearance_m,
        lon_step_deg=lon_step_deg,
        lat_step_deg=lat_step_deg,
        step_cells=step_cells,
        min_trace_speed=min_trace_speed,
        max_steps=max_steps,
    )
    backward = trace_streamline_direction(
        depth_axis=depth_axis,
        seed_depth_m=seed_depth_m,
        seed_row=seed_row,
        seed_col=seed_col,
        direction=-1.0,
        water_mask_2d=water_mask_2d,
        model_depth_2d=model_depth_2d,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        u_volume=u_volume,
        v_volume=v_volume,
        w_volume=w_volume,
        theta_volume=theta_volume,
        sal_volume=sal_volume,
        clearance_m=clearance_m,
        lon_step_deg=lon_step_deg,
        lat_step_deg=lat_step_deg,
        step_cells=step_cells,
        min_trace_speed=min_trace_speed,
        max_steps=max_steps,
    )
    merged = list(reversed(backward)) + forward[1:]
    if len(merged) < max(2, int(min_segment_count) + 1):
        return None

    row_values = np.asarray(
        [float(state["y"]) for state in merged],
        dtype=np.float32,
    )
    col_values = np.asarray(
        [float(state["x"]) for state in merged],
        dtype=np.float32,
    )
    row_values = np.asarray([float(seed_row) if not np.isfinite(v) else v for v in row_values], dtype=np.float32)
    col_values = np.asarray([float(seed_col) if not np.isfinite(v) else v for v in col_values], dtype=np.float32)
    unique_xy_cells = count_unique_xy_cells(
        np.asarray([float(state.get("sample_row", seed_row)) for state in merged], dtype=np.float32),
        np.asarray([float(state.get("sample_col", seed_col)) for state in merged], dtype=np.float32),
    )
    if unique_xy_cells < max(1, int(min_unique_xy_cells)):
        return None

    if (
        net_displacement_cells(
            np.asarray([float(state.get("sample_row", seed_row)) for state in merged], dtype=np.float32),
            np.asarray([float(state.get("sample_col", seed_col)) for state in merged], dtype=np.float32),
        )
        < float(min_net_displacement_cells)
    ):
        return None

    return {
        "x": np.asarray([point["x"] for point in merged], dtype=np.float32),
        "y": np.asarray([point["y"] for point in merged], dtype=np.float32),
        "depth": np.asarray([point["depth"] for point in merged], dtype=np.float32),
        "theta": np.asarray([point["theta"] for point in merged], dtype=np.float32),
        "sal": np.asarray([point["sal"] for point in merged], dtype=np.float32),
        "speed": np.asarray([point["speed"] for point in merged], dtype=np.float32),
        "sample_row": np.asarray([point.get("sample_row", seed_row) for point in merged], dtype=np.float32),
        "sample_col": np.asarray([point.get("sample_col", seed_col) for point in merged], dtype=np.float32),
        "segment_count": len(merged) - 1,
        "seed_speed": float(seed_state["speed"]),
        "seed_theta": float(seed_state["theta"]),
        "seed_sal": float(seed_state["sal"]),
        "seed_x_m": float(seed_state["x"]),
        "seed_y_m": float(seed_state["y"]),
        "seed_row": int(seed_row),
        "seed_col": int(seed_col),
        "seed_depth_m": float(seed_depth_m),
        "unique_xy_cells": int(unique_xy_cells),
    }


def append_streamline_segments(
    streamline: dict[str, Any],
    *,
    segment_x0: list[float],
    segment_y0: list[float],
    segment_depth0: list[float],
    segment_x1: list[float],
    segment_y1: list[float],
    segment_depth1: list[float],
    segment_theta0: list[float],
    segment_sal0: list[float],
    segment_theta1: list[float],
    segment_sal1: list[float],
    segment_terminal_flag: list[int],
) -> None:
    point_count = int(streamline["x"].size)
    for segment_index in range(1, point_count):
        segment_x0.append(float(streamline["x"][segment_index - 1]))
        segment_y0.append(float(streamline["y"][segment_index - 1]))
        segment_depth0.append(float(streamline["depth"][segment_index - 1]))
        segment_x1.append(float(streamline["x"][segment_index]))
        segment_y1.append(float(streamline["y"][segment_index]))
        segment_depth1.append(float(streamline["depth"][segment_index]))
        segment_theta0.append(float(streamline["theta"][segment_index - 1]))
        segment_sal0.append(float(streamline["sal"][segment_index - 1]))
        segment_theta1.append(float(streamline["theta"][segment_index]))
        segment_sal1.append(float(streamline["sal"][segment_index]))
        segment_terminal_flag.append(1 if segment_index == point_count - 1 else 0)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.random_seed))

    bed_meta, bed_mask = load_bedmachine_mask(Path(args.bedmachine_meta), Path(args.bedmachine_bin))
    bed_grid = bed_meta["grid"]
    x_min = float(bed_grid["x0_m"]) - args.margin_m
    x_max = float(bed_grid["x0_m"]) + (int(bed_grid["nx"]) - 1) * float(bed_grid["dx_m"]) + args.margin_m
    y_min = float(bed_grid["y0_m"]) + (int(bed_grid["ny"]) - 1) * float(bed_grid["dy_m"]) - args.margin_m
    y_max = float(bed_grid["y0_m"]) + args.margin_m

    segment_x0: list[float] = []
    segment_y0: list[float] = []
    segment_depth0: list[float] = []
    segment_x1: list[float] = []
    segment_y1: list[float] = []
    segment_depth1: list[float] = []
    segment_theta0: list[float] = []
    segment_sal0: list[float] = []
    segment_theta1: list[float] = []
    segment_sal1: list[float] = []
    segment_terminal_flag: list[int] = []

    with h5py.File(input_path, "r") as ds:
        lat_axis = np.asarray(ds["latitude"][:], dtype=np.float32)
        lon_axis = np.asarray(ds["longitude"][:], dtype=np.float32)
        depth_axis = np.asarray(ds["depth"][:], dtype=np.float32)
        model_depth = np.asarray(ds["model_depth"][:], dtype=np.float32)
        time_seconds = float(ds["time"][0])
        source_time_iso = dt.datetime.fromtimestamp(time_seconds, dt.UTC).isoformat().replace("+00:00", "Z")
        source_dataset = decode_attr(ds.attrs.get("subset:datasetId"), "")

        lat_idx = np.arange(0, lat_axis.size, max(1, int(args.sample_stride)))
        lon_idx = np.arange(0, lon_axis.size, max(1, int(args.sample_stride)))
        lat_sparse = lat_axis[lat_idx]
        lon_sparse = lon_axis[lon_idx]

        lat_grid = np.repeat(lat_sparse[:, None], lon_sparse.size, axis=1)
        lon_grid = np.repeat(lon_sparse[None, :], lat_sparse.size, axis=0)
        x_grid, y_grid = project_epsg3413(lat_grid, lon_grid)

        sampled_model_depth = model_depth[np.ix_(lat_idx, lon_idx)]
        sampled_col, sampled_row = project_to_bedmachine_grid(bed_grid, x_grid, y_grid)
        nearest_col = np.rint(sampled_col).astype(np.int32)
        nearest_row = np.rint(sampled_row).astype(np.int32)
        in_bed_extent = (
            (nearest_col >= 0)
            & (nearest_row >= 0)
            & (nearest_col < int(bed_grid["nx"]))
            & (nearest_row < int(bed_grid["ny"]))
        )
        bed_ocean_mask = np.zeros(lat_grid.shape, dtype=bool)
        bed_ocean_mask[in_bed_extent] = bed_mask[nearest_row[in_bed_extent], nearest_col[in_bed_extent]] == 0
        region_mask = (x_grid >= x_min) & (x_grid <= x_max) & (y_grid >= y_min) & (y_grid <= y_max)
        water_mask_2d = region_mask & in_bed_extent & bed_ocean_mask & np.isfinite(sampled_model_depth)
        if not np.any(water_mask_2d):
            raise RuntimeError("No Greenland ocean cells remain after clipping and masking.")

        lon_step_deg = float(np.mean(np.abs(np.diff(lon_sparse)))) if lon_sparse.size > 1 else 1.0
        lat_step_deg = float(np.mean(np.abs(np.diff(lat_sparse)))) if lat_sparse.size > 1 else 1.0

        u_volume = load_sparse_volume(ds["vxo"], lat_idx, lon_idx)
        v_volume = load_sparse_volume(ds["vyo"], lat_idx, lon_idx)
        w_volume = load_sparse_volume(ds["wo"], lat_idx, lon_idx)
        theta_volume = load_sparse_volume(ds["thetao"], lat_idx, lon_idx)
        sal_volume = load_sparse_volume(ds["so"], lat_idx, lon_idx)
        seed_buckets = tuple(GREENLAND_LAYER_BUCKETS)
        bucket_targets = compute_bucket_targets(int(args.target_streamlines), seed_buckets)
        bucket_streamlines: dict[str, list[dict[str, Any]]] = {str(bucket["key"]): [] for bucket in seed_buckets}
        bucket_labels = {str(bucket["key"]): str(bucket["label"]) for bucket in seed_buckets}
        bucket_valid_masks: dict[str, np.ndarray] = {}
        bucket_valid_sector_counts: dict[str, list[int]] = {}

        for bucket, bucket_target in zip(seed_buckets, bucket_targets):
            bucket_key = str(bucket["key"])
            valid_seed_mask = water_mask_2d.copy()
            valid_seed_mask &= sampled_model_depth >= float(bucket["min_total_depth_m"])
            bucket_valid_masks[bucket_key] = valid_seed_mask.copy()
            bucket_valid_sector_counts[bucket_key] = mask_sector_counts(valid_seed_mask, x_grid, y_grid)
            if not np.any(valid_seed_mask):
                continue

            candidate_seed_target = max(
                int(bucket_target),
                int(math.ceil(float(bucket["candidate_factor"]) * max(1, int(bucket_target)))),
            )
            seeds = collect_mask_aware_poisson_disk_cells(valid_seed_mask, candidate_seed_target, rng)

            for seed_row, seed_col in seeds:
                model_depth_here = float(sampled_model_depth[seed_row, seed_col])
                candidate_depths = choose_seed_depth(
                    model_depth_here,
                    depth_fraction_range=tuple(bucket["depth_fraction_range"]),
                    proxy_fraction=float(bucket["proxy_fraction"]),
                    rng=rng,
                    beta_shape=tuple(bucket["depth_beta"]),
                    clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                )
                seed_state: dict[str, float] | None = None
                seed_depth_m = float("nan")
                for candidate_depth in candidate_depths:
                    state = sample_stream_state(
                        depth_axis=depth_axis,
                        depth_m=float(candidate_depth),
                        row=float(seed_row),
                        col=float(seed_col),
                        water_mask_2d=water_mask_2d,
                        model_depth_2d=sampled_model_depth,
                        lat_grid=lat_grid,
                        lon_grid=lon_grid,
                        x_grid=x_grid,
                        y_grid=y_grid,
                        u_volume=u_volume,
                        v_volume=v_volume,
                        w_volume=w_volume,
                        theta_volume=theta_volume,
                        sal_volume=sal_volume,
                        clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                        lon_step_deg=lon_step_deg,
                        lat_step_deg=lat_step_deg,
                    )
                    if state is None or float(state["speed"]) < float(bucket["min_seed_speed_mps"]):
                        continue
                    seed_state = state
                    seed_depth_m = float(candidate_depth)
                    break

                if seed_state is None:
                    continue

                streamline = build_traced_streamline(
                    depth_axis=depth_axis,
                    seed_depth_m=seed_depth_m,
                    seed_row=int(seed_row),
                    seed_col=int(seed_col),
                    seed_state=seed_state,
                    water_mask_2d=water_mask_2d,
                    model_depth_2d=sampled_model_depth,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    u_volume=u_volume,
                    v_volume=v_volume,
                    w_volume=w_volume,
                    theta_volume=theta_volume,
                    sal_volume=sal_volume,
                    clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                    lon_step_deg=lon_step_deg,
                    lat_step_deg=lat_step_deg,
                    step_cells=float(args.flowline_step_cells),
                    min_trace_speed=float(bucket["min_trace_speed_mps"]),
                    max_steps=int(args.flowline_max_steps),
                    min_segment_count=int(bucket["min_streamline_segments"]),
                    min_unique_xy_cells=int(bucket["min_unique_xy_cells"]),
                    min_net_displacement_cells=float(bucket["min_net_displacement_cells"]),
                )
                if streamline is None:
                    continue

                streamline["bucket_key"] = bucket_key
                streamline["bucket_label"] = bucket_labels[bucket_key]
                streamline["selection_rank_secondary"] = float(rng.random())
                bucket_streamlines[bucket_key].append(streamline)

        selected_streamlines: list[dict[str, Any]] = []
        retained_depths_by_global_bin: dict[tuple[int, int], list[float]] = {}
        retained_sector_counts = [0 for _ in range(DEFAULT_SECTOR_COUNT)]

        for bucket, bucket_target in zip(seed_buckets, bucket_targets):
            bucket_key = str(bucket["key"])
            selection_bin_size = compute_spatial_bin_size(bucket_valid_masks.get(bucket_key, water_mask_2d), int(bucket_target))
            sector_targets = compute_weighted_targets(bucket_valid_sector_counts.get(bucket_key, []), int(bucket_target))
            selected_bucket, _ = select_streamlines_spatially_by_sector_targets(
                bucket_streamlines[bucket_key],
                target_count=int(bucket_target),
                sector_targets=sector_targets,
                selection_bin_size=selection_bin_size,
                depth_bin_size=selection_bin_size,
                retained_depths_by_bin=retained_depths_by_global_bin,
                retained_sector_counts=retained_sector_counts,
                rng=rng,
            )
            selected_streamlines.extend(selected_bucket)

    if not selected_streamlines:
        raise RuntimeError("No valid Greenland 3D ocean streamlines were retained.")

    counts_by_seed_bucket: dict[str, int] = {str(bucket["key"]): 0 for bucket in GREENLAND_LAYER_BUCKETS}
    flowlines_by_seed_bucket: dict[str, int] = {str(bucket["key"]): 0 for bucket in GREENLAND_LAYER_BUCKETS}
    for streamline in selected_streamlines:
        bucket_key = str(streamline["bucket_key"])
        flowlines_by_seed_bucket[bucket_key] += 1
        counts_by_seed_bucket[bucket_key] += int(streamline["segment_count"])
        append_streamline_segments(
            streamline,
            segment_x0=segment_x0,
            segment_y0=segment_y0,
            segment_depth0=segment_depth0,
            segment_x1=segment_x1,
            segment_y1=segment_y1,
            segment_depth1=segment_depth1,
            segment_theta0=segment_theta0,
            segment_sal0=segment_sal0,
            segment_theta1=segment_theta1,
            segment_sal1=segment_sal1,
            segment_terminal_flag=segment_terminal_flag,
        )

    x0_all = np.asarray(segment_x0, dtype=np.float32)
    y0_all = np.asarray(segment_y0, dtype=np.float32)
    depth0_all = np.asarray(segment_depth0, dtype=np.float32)
    x1_all = np.asarray(segment_x1, dtype=np.float32)
    y1_all = np.asarray(segment_y1, dtype=np.float32)
    depth1_all = np.asarray(segment_depth1, dtype=np.float32)
    theta0_all = np.asarray(segment_theta0, dtype=np.float32)
    sal0_all = np.asarray(segment_sal0, dtype=np.float32)
    theta1_all = np.asarray(segment_theta1, dtype=np.float32)
    sal1_all = np.asarray(segment_sal1, dtype=np.float32)
    terminal_all = np.asarray(segment_terminal_flag, dtype=np.uint8)

    speed_all = np.concatenate([np.asarray(streamline["speed"], dtype=np.float32) for streamline in selected_streamlines])
    theta_all = np.concatenate([np.asarray(streamline["theta"], dtype=np.float32) for streamline in selected_streamlines])
    sal_all = np.concatenate([np.asarray(streamline["sal"], dtype=np.float32) for streamline in selected_streamlines])
    seed_depth_all = np.asarray([float(streamline["seed_depth_m"]) for streamline in selected_streamlines], dtype=np.float32)

    streamline_count = int(sum(flowlines_by_seed_bucket.values()))
    segment_count = int(x0_all.size)
    basename = build_default_basename(source_time_iso)
    out_bin = output_dir / f"{basename}.bin"
    out_meta = output_dir / f"{basename}.meta.json"

    with out_bin.open("wb") as fh:
        offset = 0
        fields: list[dict[str, Any]] = []
        for name, array in (
            ("x0_ps_m", x0_all),
            ("y0_ps_m", y0_all),
            ("depth0_m", depth0_all),
            ("x1_ps_m", x1_all),
            ("y1_ps_m", y1_all),
            ("depth1_m", depth1_all),
            ("theta0_c", theta0_all),
            ("sal0_psu", sal0_all),
            ("theta1_c", theta1_all),
            ("sal1_psu", sal1_all),
            ("terminal_flag", terminal_all),
        ):
            raw = array.tobytes(order="C")
            fh.write(raw)
            fields.append(
                {
                    "name": name,
                    "dtype": "uint8" if array.dtype == np.uint8 else "float32",
                    "byte_offset": offset,
                    "byte_length": len(raw),
                }
            )
            offset += len(raw)

    x_bounds = np.concatenate([x0_all, x1_all])
    y_bounds = np.concatenate([y0_all, y1_all])
    depth_bounds = np.concatenate([depth0_all, depth1_all])
    meta = {
        "title": "Arctic Ocean Physics Analysis, 6.25km monthly mean",
        "product_version": "ARCTIC_ANALYSISFORECAST_PHY_002_001",
        "source_file": input_path.name,
        "source_dataset": source_dataset,
        "source_reference": "https://marine.copernicus.eu/",
        "source_time_utc": source_time_iso,
        "projection": "EPSG:3413",
        "geometry_type": "streamlines_3d",
        "sampling": {
            "sample_stride": int(args.sample_stride),
            "streamline_class": "greenland_open_ocean",
            "streamline_class_label": "Greenland ocean domain clipped to the BedMachine Greenland ocean mask",
            "seed_strategy": "uniform_spatial_scattered",
            "depth_sampling_summary": "Mask-aware Poisson-disk seeding across the Greenland ocean domain, with four vertically stratified seed layers (surface, upper, mid, lower) and sector-proportional spatial selection.",
            "min_speed_mps": float(min(float(bucket["min_speed_mps"]) for bucket in seed_buckets)),
            "min_seed_speed_mps": float(min(float(bucket["min_seed_speed_mps"]) for bucket in seed_buckets)),
            "min_trace_speed_mps": float(min(float(bucket["min_trace_speed_mps"]) for bucket in seed_buckets)),
            "flowline_step_cells": float(args.flowline_step_cells),
            "flowline_max_steps": int(args.flowline_max_steps),
            "target_streamline_count": int(args.target_streamlines),
            "min_streamline_segments": int(min(int(bucket["min_streamline_segments"]) for bucket in seed_buckets)),
            "random_seed": int(args.random_seed),
            "bedmachine_margin_m": float(args.margin_m),
            "clearance_m": float(DEFAULT_WATER_COLUMN_CLEARANCE_M),
            "selection_strategy": "sector_proportional_spatial_round_robin",
            "selection_sector_targets": "valid_seed_area",
            "vertical_seed_separation_m": float(DEFAULT_VERTICAL_SEED_SEPARATION_M),
            "region_definition": {
                "type": "bedmachine_greenland_ocean_mask",
                "bedmachine_margin_m": float(args.margin_m),
            },
            "seed_bucket_labels": bucket_labels,
            "seed_buckets": [
                {
                    "key": str(bucket["key"]),
                    "label": str(bucket["label"]),
                    "retain_fraction": float(bucket["retain_fraction"]),
                    "candidate_factor": float(bucket["candidate_factor"]),
                    "min_total_depth_m": float(bucket["min_total_depth_m"]),
                    "depth_fraction_range": [float(value) for value in bucket["depth_fraction_range"]],
                    "proxy_fraction": float(bucket["proxy_fraction"]),
                    "depth_beta": [float(value) for value in bucket["depth_beta"]],
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_speed_mps": float(bucket["min_speed_mps"]),
                    "min_seed_speed_mps": float(bucket["min_seed_speed_mps"]),
                    "min_trace_speed_mps": float(bucket["min_trace_speed_mps"]),
                    "min_streamline_segments": int(bucket["min_streamline_segments"]),
                    "min_unique_xy_cells": int(bucket["min_unique_xy_cells"]),
                    "min_net_displacement_cells": float(bucket["min_net_displacement_cells"]),
                }
                for bucket in seed_buckets
            ],
        },
        "streamline_count": streamline_count,
        "segment_count": segment_count,
        "coverage": {
            "x_min_m": float(np.min(x_bounds)),
            "x_max_m": float(np.max(x_bounds)),
            "y_min_m": float(np.min(y_bounds)),
            "y_max_m": float(np.max(y_bounds)),
            "depth_min_m": float(np.min(depth_bounds)),
            "depth_max_m": float(np.max(depth_bounds)),
            "seed_depth_min_m": float(np.min(seed_depth_all)),
            "seed_depth_max_m": float(np.max(seed_depth_all)),
            "valid_seed_sector_counts_8": mask_sector_counts(water_mask_2d, x_grid, y_grid, DEFAULT_SECTOR_COUNT),
            "retained_seed_sector_counts_8": retained_seed_sector_counts(selected_streamlines, DEFAULT_SECTOR_COUNT),
            "segments_by_seed_bucket": counts_by_seed_bucket,
            "streamlines_by_seed_bucket": flowlines_by_seed_bucket,
        },
        "visualization": {
            "temperature_range_c": percentile_range(theta_all, 5, 95),
            "salinity_range_psu": percentile_range(sal_all, 5, 95),
        },
        "fields": fields
        + [
            {
                "name": "speed_mps",
                "unit": "m/s",
                "stats": stats_dict(speed_all),
            },
            {
                "name": "theta_c_summary",
                "unit": "degC",
                "stats": stats_dict(theta_all),
            },
            {
                "name": "sal_psu_summary",
                "unit": "psu",
                "stats": stats_dict(sal_all),
            },
        ],
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Kept {streamline_count} Greenland 3D streamlines / {segment_count} segments across "
        f"{', '.join(str(bucket['key']) for bucket in seed_buckets)}"
    )


if __name__ == "__main__":
    main()
