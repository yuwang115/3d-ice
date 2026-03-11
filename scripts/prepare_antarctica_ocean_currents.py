#!/usr/bin/env python3
"""Prepare Antarctica ocean streamlines from WAOM2 annual-mean ROMS output."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np

DEFAULT_SAMPLE_STRIDE = 1
DEFAULT_MODEL_X0_M = -3_000_000.0
DEFAULT_MODEL_Y0_M = -3_000_000.0
DEFAULT_MODEL_DX_M = 2_000.0
DEFAULT_MODEL_DY_M = 2_000.0
DEFAULT_MIN_SPEED_MPS = 0.01
DEFAULT_MIN_SEED_SPEED_MPS = 0.03
DEFAULT_MIN_TRACE_SPEED_MPS = 0.01
DEFAULT_WATER_COLUMN_CLEARANCE_M = 20.0
DEFAULT_FLOWLINE_STEP_CELLS = 1.2
DEFAULT_FLOWLINE_MAX_STEPS = 150
DEFAULT_TARGET_STREAMLINES = 4200
DEFAULT_CAVITY_MARGIN_TARGET_STREAMLINES = 5000
DEFAULT_MIN_STREAMLINE_SEGMENTS = 5
DEFAULT_RANDOM_SEED = 115
DEFAULT_SEED_DEPTH_ATTEMPTS = 6
DEFAULT_FRONT_RADIUS_CELLS = 2
DEFAULT_CAVITY_MARGIN_RADIUS_M = 80_000.0
DEFAULT_STATE_DEPTH_BAND_M = 100.0
DEFAULT_REVISIT_WINDOW_STEPS = 6
DEFAULT_REVERSAL_WINDOW_STEPS = 5
DEFAULT_REVERSAL_ANGLE_DEG = 150.0
DEFAULT_MIN_UNIQUE_XY_CELLS = 6
DEFAULT_MIN_NET_DISPLACEMENT_CELLS = 2.0
DEFAULT_VERTICAL_SEED_SEPARATION_M = 120.0
DEFAULT_SPATIAL_BIN_SIZE_MIN = 6
DEFAULT_SPATIAL_BIN_SIZE_MAX = 12
DEFAULT_SECTOR_COUNT = 8
DEFAULT_STREAMLINE_CLASS = "legacy_multi_bucket"
FRONT_BUCKET_KEYS = frozenset({"cavity_front", "open_front"})
LEGACY_SEED_BUCKETS = (
    {
        "key": "cavity_front",
        "label": "Ice-shelf front cavity exchange",
        "mask_name": "cavity_front",
        "retain_fraction": 0.25,
        "candidate_factor": 3.5,
        "min_total_depth_m": 180.0,
        "depth_fraction_range": (0.32, 0.92),
        "proxy_fraction": 0.72,
        "depth_beta": (1.6, 1.0),
        "min_seed_speed_mps": 0.004,
    },
    {
        "key": "cavity_interior",
        "label": "Interior ice-shelf cavity circulation",
        "mask_name": "cavity_interior",
        "retain_fraction": 0.15,
        "candidate_factor": 3.5,
        "min_total_depth_m": 140.0,
        "depth_fraction_range": (0.22, 0.86),
        "proxy_fraction": 0.58,
        "depth_beta": (1.45, 1.15),
        "min_seed_speed_mps": 0.004,
    },
    {
        "key": "open_front",
        "label": "Open-ocean intrusion pathways",
        "mask_name": "open_front",
        "retain_fraction": 0.25,
        "candidate_factor": 3.5,
        "min_total_depth_m": 250.0,
        "depth_fraction_range": (0.40, 0.92),
        "proxy_fraction": 0.68,
        "depth_beta": (1.55, 1.0),
        "min_seed_speed_mps": 0.006,
    },
    {
        "key": "open_mid",
        "label": "Open-ocean mid-water",
        "mask_name": "open_mid",
        "retain_fraction": 0.15,
        "candidate_factor": 3.5,
        "min_total_depth_m": 300.0,
        "depth_fraction_range": (0.35, 0.72),
        "proxy_fraction": 0.56,
        "depth_beta": (1.35, 1.35),
        "min_seed_speed_mps": 0.008,
    },
    {
        "key": "open_lower",
        "label": "Open-ocean lower water",
        "mask_name": "open_lower",
        "retain_fraction": 0.20,
        "candidate_factor": 3.5,
        "min_total_depth_m": 700.0,
        "depth_fraction_range": (0.62, 0.94),
        "proxy_fraction": 0.80,
        "depth_beta": (2.1, 1.0),
        "min_seed_speed_mps": 0.006,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 3D Antarctica ocean streamlines from WAOM2 annual-mean ROMS output."
    )
    parser.add_argument("--input", required=True, help="Path to the WAOM2 annual-mean NetCDF file.")
    parser.add_argument(
        "--streamline-class",
        choices=("legacy_multi_bucket", "cavity_margin_50km", "cavity_margin_80km", "remote_open_ocean"),
        default=DEFAULT_STREAMLINE_CLASS,
        help="Streamline domain/class to export. Only cavity_margin_50km and cavity_margin_80km are implemented in the new two-class workflow.",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for output .bin/.meta.json files.",
    )
    parser.add_argument(
        "--output-basename",
        default="",
        help="Optional basename for the output .bin/.meta.json pair.",
    )
    parser.add_argument(
        "--bedmachine-meta",
        default="static/tools/data/bedmachine_antarctica_v4_741.meta.json",
        help="BedMachine Antarctica metadata used to clip/project the ocean points.",
    )
    parser.add_argument(
        "--bedmachine-bin",
        default="static/tools/data/bedmachine_antarctica_v4_741.bin",
        help="BedMachine Antarctica binary used to read the open-ocean mask.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=DEFAULT_SAMPLE_STRIDE,
        help="Subsample every Nth WAOM grid cell before tracing streamlines.",
    )
    parser.add_argument(
        "--model-x0-m",
        type=float,
        default=DEFAULT_MODEL_X0_M,
        help="Projected x-coordinate of the southwest WAOM rho-point in EPSG:3031 meters.",
    )
    parser.add_argument(
        "--model-y0-m",
        type=float,
        default=DEFAULT_MODEL_Y0_M,
        help="Projected y-coordinate of the southwest WAOM rho-point in EPSG:3031 meters.",
    )
    parser.add_argument(
        "--model-dx-m",
        type=float,
        default=DEFAULT_MODEL_DX_M,
        help="WAOM horizontal grid spacing in x at rho-points, in meters.",
    )
    parser.add_argument(
        "--model-dy-m",
        type=float,
        default=DEFAULT_MODEL_DY_M,
        help="WAOM horizontal grid spacing in y at rho-points, in meters.",
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
        default=None,
        help="Approximate number of Antarctica streamlines to retain after filtering. Defaults depend on the selected streamline class.",
    )
    parser.add_argument(
        "--min-streamline-segments",
        type=int,
        default=DEFAULT_MIN_STREAMLINE_SEGMENTS,
        help="Minimum number of segments required to retain a traced streamline.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Seed for deterministic randomized streamline placement.",
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
    return parser.parse_args()


def decode_attr(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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


def build_default_basename(source_name: str) -> str:
    stem = Path(source_name).stem.lower()
    year_match = re.search(r"yr(\d+)", stem)
    year_suffix = f"_yr{year_match.group(1)}" if year_match else ""
    annual_suffix = "_annual" if "annual" in stem else ""
    return f"antarctica_ocean_currents_waom2{year_suffix}{annual_suffix}"


def build_output_basename(source_name: str, streamline_class: str) -> str:
    base = build_default_basename(source_name)
    if streamline_class == "cavity_margin_50km":
        return f"{base}_cavity_margin50km"
    if streamline_class == "cavity_margin_80km":
        return f"{base}_cavity_margin80km"
    if streamline_class == "remote_open_ocean":
        return f"{base}_remote_open_ocean"
    return base


def extract_connected_component(mask: np.ndarray, seed_row: int, seed_col: int) -> np.ndarray:
    component = np.zeros_like(mask, dtype=bool)
    if seed_row < 0 or seed_col < 0 or seed_row >= mask.shape[0] or seed_col >= mask.shape[1]:
        return component
    if not bool(mask[seed_row, seed_col]):
        return component

    queue: deque[tuple[int, int]] = deque([(int(seed_row), int(seed_col))])
    component[seed_row, seed_col] = True
    ny, nx = mask.shape

    while queue:
        row, col = queue.popleft()
        row_start = max(0, row - 1)
        row_end = min(ny, row + 2)
        col_start = max(0, col - 1)
        col_end = min(nx, col + 2)
        for next_row in range(row_start, row_end):
            component_row = component[next_row]
            mask_row = mask[next_row]
            for next_col in range(col_start, col_end):
                if component_row[next_col] or not mask_row[next_col]:
                    continue
                component_row[next_col] = True
                queue.append((next_row, next_col))

    return component


def build_main_antarctic_ice_mask(bed_mask: np.ndarray, bed_grid: dict[str, Any]) -> np.ndarray:
    ice_mask = np.asarray(bed_mask != 0, dtype=bool)
    if not np.any(ice_mask):
        raise RuntimeError("BedMachine mask does not contain any Antarctic ice/land cells.")

    x_coords = float(bed_grid["x0_m"]) + np.arange(ice_mask.shape[1], dtype=np.float64) * float(bed_grid["dx_m"])
    y_coords = float(bed_grid["y0_m"]) + np.arange(ice_mask.shape[0], dtype=np.float64) * float(bed_grid["dy_m"])
    distance_sq = y_coords[:, None] ** 2 + x_coords[None, :] ** 2
    distance_sq[~ice_mask] = np.inf
    seed_index = int(np.argmin(distance_sq))
    seed_row, seed_col = divmod(seed_index, ice_mask.shape[1])

    main_component = extract_connected_component(ice_mask, seed_row, seed_col)
    if not np.any(main_component):
        raise RuntimeError("Failed to isolate the main Antarctic ice component from BedMachine.")
    return main_component


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


def collect_random_spatial_cells(
    valid2d: np.ndarray,
    weight2d: np.ndarray | None,
    target_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    valid_count = int(np.count_nonzero(valid2d))
    if valid_count == 0 or target_count <= 0:
        return []

    seed_spacing = max(2, int(round(math.sqrt(valid_count / max(1, target_count)))))
    seeds: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    passes = max(2, min(8, int(math.ceil(target_count / max(1, valid_count / max(1, seed_spacing * seed_spacing))))))

    for _ in range(passes):
        row_offset = int(rng.integers(0, seed_spacing))
        col_offset = int(rng.integers(0, seed_spacing))
        for row_block in range(row_offset, valid2d.shape[0], seed_spacing):
            row_end = min(valid2d.shape[0], row_block + seed_spacing)
            for col_block in range(col_offset, valid2d.shape[1], seed_spacing):
                col_end = min(valid2d.shape[1], col_block + seed_spacing)
                block_mask = valid2d[row_block:row_end, col_block:col_end]
                if not np.any(block_mask):
                    continue
                block_rows, block_cols = np.nonzero(block_mask)
                rows = block_rows + row_block
                cols = block_cols + col_block
                if weight2d is not None:
                    weights = weight2d[rows, cols].astype(np.float64, copy=False)
                    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
                    if float(np.sum(weights)) > 0:
                        weights = np.power(weights, 1.5)
                        choice = int(rng.choice(rows.size, p=weights / float(np.sum(weights))))
                    else:
                        choice = int(rng.integers(0, rows.size))
                else:
                    choice = int(rng.integers(0, rows.size))
                selected = (int(rows[choice]), int(cols[choice]))
                if selected not in seen:
                    seen.add(selected)
                    seeds.append(selected)
                    if len(seeds) >= target_count:
                        return seeds

    if len(seeds) >= target_count:
        return seeds[:target_count]

    remaining_rows, remaining_cols = np.nonzero(valid2d)
    order = rng.permutation(remaining_rows.size)
    for index in order.tolist():
        selected = (int(remaining_rows[index]), int(remaining_cols[index]))
        if selected in seen:
            continue
        seen.add(selected)
        seeds.append(selected)
        if len(seeds) >= target_count:
            break
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


def expand_mask_8_connected(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    expanded = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(radius_cells))):
        padded = np.pad(expanded, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(expanded, dtype=bool)
        for row_offset in range(3):
            for col_offset in range(3):
                grown |= padded[row_offset : row_offset + expanded.shape[0], col_offset : col_offset + expanded.shape[1]]
        expanded = grown
    return expanded


def build_seed_masks(open_water_mask: np.ndarray, cavity_water_mask: np.ndarray) -> dict[str, np.ndarray]:
    cavity_front = cavity_water_mask & expand_mask_8_connected(open_water_mask, DEFAULT_FRONT_RADIUS_CELLS)
    open_front = open_water_mask & expand_mask_8_connected(cavity_water_mask, DEFAULT_FRONT_RADIUS_CELLS)
    return {
        "cavity_front": cavity_front,
        "cavity_interior": cavity_water_mask & ~cavity_front,
        "open_front": open_front,
        "open_mid": open_water_mask & ~open_front,
        "open_lower": open_water_mask.copy(),
    }


def build_cavity_margin_mask(
    open_water_mask: np.ndarray,
    cavity_water_mask: np.ndarray,
    *,
    sample_dx_m: float,
    sample_dy_m: float,
    radius_m: float,
) -> np.ndarray:
    cell_size_m = max(1.0, min(float(sample_dx_m), float(sample_dy_m)))
    radius_cells = max(0, int(math.ceil(float(radius_m) / cell_size_m)))
    if radius_cells <= 0:
        return np.asarray(cavity_water_mask, dtype=bool).copy()
    coastal_open_mask = np.asarray(open_water_mask, dtype=bool) & expand_mask_8_connected(
        np.asarray(cavity_water_mask, dtype=bool),
        radius_cells,
    )
    return np.asarray(cavity_water_mask, dtype=bool) | coastal_open_mask


def resolve_target_streamlines(args: argparse.Namespace) -> int:
    if args.target_streamlines is not None:
        return max(1, int(args.target_streamlines))
    if str(args.streamline_class) in {"cavity_margin_50km", "cavity_margin_80km"}:
        return DEFAULT_CAVITY_MARGIN_TARGET_STREAMLINES
    return DEFAULT_TARGET_STREAMLINES


def build_streamline_class_definition(
    streamline_class: str,
    *,
    open_water_mask: np.ndarray,
    cavity_water_mask: np.ndarray,
    sample_dx_m: float,
    sample_dy_m: float,
) -> dict[str, Any]:
    class_key = str(streamline_class)
    if class_key == "legacy_multi_bucket":
        return {
            "key": class_key,
            "label": "Legacy mixed cavity/open-ocean buckets",
            "seed_strategy": "random_spatial_scattered",
            "depth_sampling_summary": (
                "Randomized cavity/open-ocean seeding with front-focused retention near ice-shelf margins, "
                "sector-balanced extra fill, and vertically separated XY-bin balancing."
            ),
            "selection_strategy": "spatial_bin_round_robin",
            "prefer_front_extra_fill": True,
            "seed_masks": build_seed_masks(open_water_mask, cavity_water_mask),
            "seed_buckets": LEGACY_SEED_BUCKETS,
            "region_metadata": {
                "type": "legacy_mixed_bucket_domain",
            },
        }
    if class_key in {"cavity_margin_50km", "cavity_margin_80km"}:
        radius_m = 50_000.0 if class_key == "cavity_margin_50km" else float(DEFAULT_CAVITY_MARGIN_RADIUS_M)
        class_label = (
            "Ice-shelf cavities and surrounding 50 km ocean"
            if class_key == "cavity_margin_50km"
            else "Ice-shelf cavities and surrounding 80 km ocean"
        )
        bucket_label = (
            "Ice-shelf cavity and surrounding 50 km ocean"
            if class_key == "cavity_margin_50km"
            else "Ice-shelf cavity and surrounding 80 km ocean"
        )
        cavity_margin_mask = build_cavity_margin_mask(
            open_water_mask,
            cavity_water_mask,
            sample_dx_m=sample_dx_m,
            sample_dy_m=sample_dy_m,
            radius_m=radius_m,
        )
        return {
            "key": class_key,
            "label": class_label,
            "seed_strategy": "uniform_spatial_scattered",
            "depth_sampling_summary": (
                f"Uniform random spatial seeding across ice-shelf cavities and the surrounding {int(radius_m / 1000)} km open-ocean band, "
                "with four vertically stratified seed layers (surface, upper, mid, lower) and sector-proportional spatial selection."
            ),
            "selection_strategy": "sector_proportional_spatial_round_robin",
            "selection_sector_targets": "valid_seed_area",
            "prefer_front_extra_fill": False,
            "seed_masks": {
                class_key: cavity_margin_mask,
            },
            "seed_buckets": (
                {
                    "key": f"{class_key}_surface",
                    "label": f"{bucket_label} surface layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.4,
                    "min_total_depth_m": 60.0,
                    "depth_fraction_range": (0.03, 0.18),
                    "proxy_fraction": 0.10,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
                {
                    "key": f"{class_key}_upper",
                    "label": f"{bucket_label} upper layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.4,
                    "min_total_depth_m": 80.0,
                    "depth_fraction_range": (0.18, 0.40),
                    "proxy_fraction": 0.30,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
                {
                    "key": f"{class_key}_mid",
                    "label": f"{bucket_label} mid-water layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.4,
                    "min_total_depth_m": 120.0,
                    "depth_fraction_range": (0.40, 0.70),
                    "proxy_fraction": 0.55,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
                {
                    "key": f"{class_key}_lower",
                    "label": f"{bucket_label} lower layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.4,
                    "min_total_depth_m": 180.0,
                    "depth_fraction_range": (0.70, 0.95),
                    "proxy_fraction": 0.82,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
            ),
            "region_metadata": {
                "type": class_key,
                "radius_m": float(radius_m),
                "cavity_cell_count": int(np.count_nonzero(cavity_water_mask)),
                "open_ocean_margin_cell_count": int(np.count_nonzero(cavity_margin_mask & open_water_mask)),
                "total_region_cell_count": int(np.count_nonzero(cavity_margin_mask)),
            },
        }
    if class_key == "remote_open_ocean":
        exclusion_mask = build_cavity_margin_mask(
            open_water_mask,
            cavity_water_mask,
            sample_dx_m=sample_dx_m,
            sample_dy_m=sample_dy_m,
            radius_m=float(DEFAULT_CAVITY_MARGIN_RADIUS_M),
        )
        remote_open_mask = np.asarray(open_water_mask, dtype=bool) & ~np.asarray(exclusion_mask, dtype=bool)
        return {
            "key": class_key,
            "label": "Remote open ocean beyond the ice-shelf cavity + 80 km coastal band",
            "seed_strategy": "uniform_spatial_scattered",
            "depth_sampling_summary": (
                "Mask-aware Poisson-disk seeding across remote open-ocean cells outside the cavity + 80 km coastal band, "
                "with four vertically stratified seed layers (surface, upper, mid, lower) and sector-proportional spatial selection."
            ),
            "selection_strategy": "sector_proportional_spatial_round_robin",
            "selection_sector_targets": "valid_seed_area",
            "prefer_front_extra_fill": False,
            "seed_masks": {
                class_key: remote_open_mask,
            },
            "seed_buckets": (
                {
                    "key": "remote_surface",
                    "label": "Remote open ocean surface layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.35,
                    "min_total_depth_m": 60.0,
                    "depth_fraction_range": (0.03, 0.18),
                    "proxy_fraction": 0.10,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
                {
                    "key": "remote_upper",
                    "label": "Remote open ocean upper layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.35,
                    "min_total_depth_m": 80.0,
                    "depth_fraction_range": (0.18, 0.40),
                    "proxy_fraction": 0.30,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
                {
                    "key": "remote_mid",
                    "label": "Remote open ocean mid-water layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.35,
                    "min_total_depth_m": 120.0,
                    "depth_fraction_range": (0.40, 0.70),
                    "proxy_fraction": 0.55,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
                {
                    "key": "remote_lower",
                    "label": "Remote open ocean lower layer",
                    "mask_name": class_key,
                    "retain_fraction": 0.25,
                    "candidate_factor": 1.35,
                    "min_total_depth_m": 180.0,
                    "depth_fraction_range": (0.70, 0.95),
                    "proxy_fraction": 0.82,
                    "depth_beta": (1.0, 1.0),
                    "min_seed_speed_mps": 0.0,
                    "min_proxy_speed_mps": 0.0,
                    "min_trace_speed_mps": 0.001,
                    "use_speed_filter_in_seed_mask": False,
                    "seed_layout_mode": "mask_aware_poisson_disk",
                    "seed_weight_mode": "uniform",
                    "seed_depth_mode": "first_valid_with_proxy_fallback",
                    "selection_rank_mode": "segment_random",
                    "min_streamline_segments": 5,
                    "min_unique_xy_cells": 4,
                    "min_net_displacement_cells": 1.0,
                },
            ),
            "region_metadata": {
                "type": class_key,
                "excluded_radius_m": float(DEFAULT_CAVITY_MARGIN_RADIUS_M),
                "remote_open_ocean_cell_count": int(np.count_nonzero(remote_open_mask)),
                "excluded_cavity_margin_cell_count": int(np.count_nonzero(exclusion_mask)),
            },
        }
    raise ValueError(f"Unsupported streamline class: {class_key}")


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


def compute_bucket_targets(target_streamlines: int, seed_buckets: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> list[int]:
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


def streamline_rank_key(streamline: dict[str, Any]) -> tuple[int, float]:
    return int(streamline["segment_count"]), float(streamline.get("selection_rank_secondary", streamline["seed_speed"]))


def preferred_streamline_rank_key(streamline: dict[str, Any], *, prefer_front: bool) -> tuple[int, int, float]:
    front_bonus = 1 if prefer_front and str(streamline.get("bucket_key", "")) in FRONT_BUCKET_KEYS else 0
    return (
        front_bonus,
        int(streamline["segment_count"]),
        float(streamline.get("selection_rank_secondary", streamline["seed_speed"])),
    )


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

    bin_groups: dict[tuple[int, int], deque[dict[str, Any]]] = {}
    for candidate in candidates:
        bin_key = spatial_bin_key(int(candidate["seed_row"]), int(candidate["seed_col"]), selection_bin_size)
        queue = bin_groups.get(bin_key)
        if queue is None:
            queue = deque()
            bin_groups[bin_key] = queue
        queue.append(candidate)

    for bin_key, queue in list(bin_groups.items()):
        ordered = sorted(list(queue), key=streamline_rank_key, reverse=True)
        bin_groups[bin_key] = deque(ordered)

    bin_keys = list(bin_groups.keys())
    if bin_keys:
        permutation = rng.permutation(len(bin_keys)).tolist()
        ordered_bin_keys = [bin_keys[index] for index in permutation]
    else:
        ordered_bin_keys = []

    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    made_progress = True
    while len(selected) < target_count and made_progress:
        made_progress = False
        for bin_key in ordered_bin_keys:
            queue = bin_groups[bin_key]
            while queue:
                candidate = queue.popleft()
                depth_key = spatial_bin_key(int(candidate["seed_row"]), int(candidate["seed_col"]), depth_bin_size)
                existing_depths = retained_depths_by_bin[depth_key]
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
        leftovers.extend(list(queue))

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
        existing_depths = retained_depths_by_bin[depth_key]
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

    if len(selected) < target_count and leftovers:
        extra_selected, leftovers = select_streamlines_sector_balanced(
            leftovers,
            target_count=int(target_count) - len(selected),
            selection_bin_size=selection_bin_size,
            depth_bin_size=depth_bin_size,
            retained_depths_by_bin=retained_depths_by_bin,
            retained_sector_counts=retained_sector_counts,
            rng=rng,
            prefer_front=False,
        )
        selected.extend(extra_selected)

    return selected, leftovers


def streamline_sector_index(streamline: dict[str, Any], sector_count: int = DEFAULT_SECTOR_COUNT) -> int:
    count = max(1, int(sector_count))
    seed_x = float(streamline["seed_x_m"])
    seed_y = float(streamline["seed_y_m"])
    if not math.isfinite(seed_x) or not math.isfinite(seed_y):
        return 0
    angle = (math.atan2(seed_y, seed_x) + 2.0 * math.pi) % (2.0 * math.pi)
    sector_width = (2.0 * math.pi) / float(count)
    return min(count - 1, max(0, int(math.floor(angle / sector_width))))


def select_streamlines_sector_balanced(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
    selection_bin_size: int,
    depth_bin_size: int,
    retained_depths_by_bin: dict[tuple[int, int], list[float]],
    retained_sector_counts: list[int],
    rng: np.random.Generator,
    prefer_front: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if target_count <= 0 or not candidates:
        return [], list(candidates)

    sector_bin_groups: dict[int, dict[tuple[int, int], deque[dict[str, Any]]]] = {
        sector: {} for sector in range(max(1, int(DEFAULT_SECTOR_COUNT)))
    }
    sector_bin_orders: dict[int, list[tuple[int, int]]] = {}
    sector_bin_indices: dict[int, int] = {}

    for candidate in candidates:
        sector = streamline_sector_index(candidate)
        bin_key = spatial_bin_key(int(candidate["seed_row"]), int(candidate["seed_col"]), selection_bin_size)
        sector_groups = sector_bin_groups[sector]
        queue = sector_groups.get(bin_key)
        if queue is None:
            queue = deque()
            sector_groups[bin_key] = queue
        queue.append(candidate)

    for sector, sector_groups in sector_bin_groups.items():
        for bin_key, queue in list(sector_groups.items()):
            ordered = sorted(
                list(queue),
                key=lambda item: preferred_streamline_rank_key(item, prefer_front=prefer_front),
                reverse=True,
            )
            sector_groups[bin_key] = deque(ordered)
        bin_keys = list(sector_groups.keys())
        if bin_keys:
            permutation = rng.permutation(len(bin_keys)).tolist()
            sector_bin_orders[sector] = [bin_keys[index] for index in permutation]
        else:
            sector_bin_orders[sector] = []
        sector_bin_indices[sector] = 0

    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []

    made_progress = True
    while len(selected) < target_count and made_progress:
        made_progress = False
        sector_order = list(range(max(1, int(DEFAULT_SECTOR_COUNT))))
        random_tie_break = {
            sector: tie_index
            for tie_index, sector in enumerate(rng.permutation(len(sector_order)).tolist())
        }
        sector_order.sort(key=lambda sector: (retained_sector_counts[sector], random_tie_break.get(sector, sector)))

        for sector in sector_order:
            bin_order = sector_bin_orders.get(sector, [])
            if not bin_order:
                continue

            bin_index = sector_bin_indices[sector]
            selected_in_sector = False
            for _ in range(len(bin_order)):
                bin_key = bin_order[bin_index % len(bin_order)]
                bin_index = (bin_index + 1) % len(bin_order)
                queue = sector_bin_groups[sector].get(bin_key)
                if not queue:
                    continue

                while queue:
                    candidate = queue.popleft()
                    depth_key = spatial_bin_key(int(candidate["seed_row"]), int(candidate["seed_col"]), depth_bin_size)
                    existing_depths = retained_depths_by_bin[depth_key]
                    if any(
                        abs(float(candidate["seed_depth_m"]) - float(existing_depth)) < DEFAULT_VERTICAL_SEED_SEPARATION_M
                        for existing_depth in existing_depths
                    ):
                        leftovers.append(candidate)
                        continue

                    selected.append(candidate)
                    retained_sector_counts[sector] += 1
                    existing_depths.append(float(candidate["seed_depth_m"]))
                    made_progress = True
                    selected_in_sector = True
                    break

                if selected_in_sector:
                    break

            sector_bin_indices[sector] = bin_index
            if len(selected) >= target_count:
                break

    for sector, bin_order in sector_bin_orders.items():
        for bin_key in bin_order:
            queue = sector_bin_groups[sector].get(bin_key)
            if queue:
                leftovers.extend(list(queue))

    return selected, leftovers


def quantize_state_key(x_m: float, y_m: float, depth_m: float, x0_m: float, y0_m: float, dx_m: float, dy_m: float) -> tuple[int, int, int]:
    col = int(round((x_m - x0_m) / dx_m))
    row = int(round((y_m - y0_m) / dy_m))
    depth_band = int(round(depth_m / DEFAULT_STATE_DEPTH_BAND_M))
    return row, col, depth_band


def count_unique_xy_cells(x_values: np.ndarray, y_values: np.ndarray, x0_m: float, y0_m: float, dx_m: float, dy_m: float) -> int:
    cells = {
        (int(round((float(x_m) - x0_m) / dx_m)), int(round((float(y_m) - y0_m) / dy_m)))
        for x_m, y_m in zip(x_values.tolist(), y_values.tolist())
    }
    return len(cells)


def net_displacement_cells(x_values: np.ndarray, y_values: np.ndarray, dx_m: float, dy_m: float) -> float:
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    dx_cells = (float(x_values[-1]) - float(x_values[0])) / dx_m
    dy_cells = (float(y_values[-1]) - float(y_values[0])) / dy_m
    return math.hypot(dx_cells, dy_cells)


def retained_seed_sector_counts(streamlines: list[dict[str, Any]], sector_count: int = DEFAULT_SECTOR_COUNT) -> list[int]:
    counts = [0 for _ in range(max(1, int(sector_count)))]
    for streamline in streamlines:
        counts[streamline_sector_index(streamline, len(counts))] += 1
    return counts


def read_masked_array(dataset: netCDF4.Variable, key: Any) -> np.ndarray:
    values = dataset[key]
    if np.ma.isMaskedArray(values):
        out = values.filled(np.nan)
    else:
        out = np.asarray(values)
    out = np.asarray(out, dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 1e20] = np.nan
    return out


def gather_rho_to_u(field: np.ndarray, row_idx: np.ndarray, u_col_idx: np.ndarray) -> np.ndarray:
    return 0.5 * (field[np.ix_(row_idx, u_col_idx)] + field[np.ix_(row_idx, u_col_idx + 1)])


def gather_rho_to_v(field: np.ndarray, v_row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    return 0.5 * (field[np.ix_(v_row_idx, col_idx)] + field[np.ix_(v_row_idx + 1, col_idx)])


def compute_depth_levels(
    h: np.ndarray,
    zice: np.ndarray,
    zeta: np.ndarray,
    s_axis: np.ndarray,
    c_axis: np.ndarray,
    hc: float,
) -> np.ndarray:
    if not np.isfinite(hc) or hc <= 0:
        raise RuntimeError("Unsupported ROMS critical depth (hc).")
    h64 = np.asarray(h, dtype=np.float64)
    zice64 = np.asarray(zice, dtype=np.float64)
    zeta64 = np.asarray(zeta, dtype=np.float64)
    out = np.empty((s_axis.size, h.shape[0], h.shape[1]), dtype=np.float32)
    denom = hc + h64
    for level, (s_value, c_value) in enumerate(zip(s_axis.tolist(), c_axis.tolist())):
        zo = (hc * float(s_value) + h64 * float(c_value)) / denom
        z = zeta64 + (zeta64 + h64) * zo + zice64 * (1.0 + zo)
        out[level] = np.maximum(0.0, -z).astype(np.float32)
    return out


def interpolate_plane_at_depth(volume: np.ndarray, depth_levels: np.ndarray, depth_m: float) -> np.ndarray:
    depth_top = depth_levels[0]
    depth_bottom = depth_levels[-1]
    in_range = np.isfinite(depth_top) & np.isfinite(depth_bottom) & (depth_m >= depth_top) & (depth_m <= depth_bottom)

    upper = np.sum(depth_levels < depth_m, axis=0)
    upper = np.clip(upper, 1, depth_levels.shape[0] - 1)
    lower = upper - 1

    depth0 = np.take_along_axis(depth_levels, lower[None, ...], axis=0)[0]
    depth1 = np.take_along_axis(depth_levels, upper[None, ...], axis=0)[0]
    value0 = np.take_along_axis(volume, lower[None, ...], axis=0)[0]
    value1 = np.take_along_axis(volume, upper[None, ...], axis=0)[0]
    frac = np.divide(
        depth_m - depth0,
        depth1 - depth0,
        out=np.zeros_like(depth0, dtype=np.float32),
        where=np.abs(depth1 - depth0) > 1e-6,
    )
    out = value0 * (1.0 - frac) + value1 * frac
    out[~in_range] = np.nan
    return out.astype(np.float32)


def interpolate_plane_at_target_depths(
    volume: np.ndarray,
    depth_levels: np.ndarray,
    target_depths: np.ndarray,
) -> np.ndarray:
    if target_depths.ndim != 2:
        raise ValueError("Target depths must be a 2D plane.")
    depth_top = depth_levels[0]
    depth_bottom = depth_levels[-1]
    in_range = (
        np.isfinite(target_depths)
        & np.isfinite(depth_top)
        & np.isfinite(depth_bottom)
        & (target_depths >= depth_top)
        & (target_depths <= depth_bottom)
    )
    upper = np.sum(depth_levels < target_depths[None, ...], axis=0)
    upper = np.clip(upper, 1, depth_levels.shape[0] - 1)
    lower = upper - 1

    depth0 = np.take_along_axis(depth_levels, lower[None, ...], axis=0)[0]
    depth1 = np.take_along_axis(depth_levels, upper[None, ...], axis=0)[0]
    value0 = np.take_along_axis(volume, lower[None, ...], axis=0)[0]
    value1 = np.take_along_axis(volume, upper[None, ...], axis=0)[0]
    frac = np.divide(
        target_depths - depth0,
        depth1 - depth0,
        out=np.zeros_like(target_depths, dtype=np.float32),
        where=np.abs(depth1 - depth0) > 1e-6,
    )
    out = value0 * (1.0 - frac) + value1 * frac
    out[~in_range] = np.nan
    return out.astype(np.float32)


def interpolate_volume_to_target_depths(
    source_volume: np.ndarray,
    source_depths: np.ndarray,
    target_depths: np.ndarray,
) -> np.ndarray:
    if source_volume.shape != source_depths.shape or source_volume.shape != target_depths.shape:
        raise ValueError("Source values, source depths, and target depths must share the same shape.")

    out = np.full(target_depths.shape, np.nan, dtype=np.float32)
    depth_top = source_depths[0]
    depth_bottom = source_depths[-1]

    for level in range(target_depths.shape[0]):
        target_plane = target_depths[level]
        in_range = (
            np.isfinite(target_plane)
            & np.isfinite(depth_top)
            & np.isfinite(depth_bottom)
            & (target_plane >= depth_top)
            & (target_plane <= depth_bottom)
        )
        upper = np.sum(source_depths < target_plane[None, ...], axis=0)
        upper = np.clip(upper, 1, source_depths.shape[0] - 1)
        lower = upper - 1

        depth0 = np.take_along_axis(source_depths, lower[None, ...], axis=0)[0]
        depth1 = np.take_along_axis(source_depths, upper[None, ...], axis=0)[0]
        value0 = np.take_along_axis(source_volume, lower[None, ...], axis=0)[0]
        value1 = np.take_along_axis(source_volume, upper[None, ...], axis=0)[0]
        frac = np.divide(
            target_plane - depth0,
            depth1 - depth0,
            out=np.zeros_like(target_plane, dtype=np.float32),
            where=np.abs(depth1 - depth0) > 1e-6,
        )
        plane = value0 * (1.0 - frac) + value1 * frac
        plane[~in_range] = np.nan
        out[level] = plane.astype(np.float32)

    return out


def interpolate_profile(depth_profile: np.ndarray, value_profile: np.ndarray, depth_m: float) -> float:
    if not np.isfinite(depth_m):
        return float("nan")
    if not np.all(np.isfinite(depth_profile)):
        return float("nan")
    if depth_m < float(depth_profile[0]) or depth_m > float(depth_profile[-1]):
        return float("nan")
    upper = int(np.searchsorted(depth_profile, depth_m, side="right"))
    if upper <= 0:
        return float(value_profile[0])
    if upper >= depth_profile.size:
        return float(value_profile[-1])
    lower = upper - 1
    d0 = float(depth_profile[lower])
    d1 = float(depth_profile[upper])
    v0 = float(value_profile[lower])
    v1 = float(value_profile[upper])
    if not all(math.isfinite(value) for value in (d0, d1, v0, v1)):
        return float("nan")
    if abs(d1 - d0) <= 1e-6:
        return v0
    frac = (depth_m - d0) / (d1 - d0)
    return v0 * (1.0 - frac) + v1 * frac


def choose_seed_depth(
    *,
    top_depth_m: float,
    bottom_depth_m: float,
    depth_fraction_range: tuple[float, float],
    depth_beta: tuple[float, float],
    clearance_m: float,
    rng: np.random.Generator,
) -> float:
    if not math.isfinite(top_depth_m) or not math.isfinite(bottom_depth_m):
        return float("nan")
    if bottom_depth_m <= top_depth_m + 2.0 * clearance_m:
        return float("nan")
    water_column_m = bottom_depth_m - top_depth_m
    lower = max(top_depth_m + clearance_m, top_depth_m + depth_fraction_range[0] * water_column_m)
    upper = min(bottom_depth_m - clearance_m, top_depth_m + depth_fraction_range[1] * water_column_m)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return float("nan")
    depth_alpha, depth_beta_param = depth_beta
    fraction = float(rng.beta(depth_alpha, depth_beta_param))
    return lower + fraction * (upper - lower)


def sample_stream_state(
    *,
    x_m: float,
    y_m: float,
    depth_m: float,
    water_mask_2d: np.ndarray,
    top_depth_2d: np.ndarray,
    bottom_depth_2d: np.ndarray,
    depth_levels: np.ndarray,
    w_depth_levels: np.ndarray,
    u_volume: np.ndarray,
    v_volume: np.ndarray,
    w_volume: np.ndarray,
    theta_volume: np.ndarray,
    sal_volume: np.ndarray,
    x0_m: float,
    y0_m: float,
    dx_m: float,
    dy_m: float,
    clearance_m: float,
) -> dict[str, float] | None:
    col_f = (x_m - x0_m) / dx_m
    row_f = (y_m - y0_m) / dy_m
    if not math.isfinite(col_f) or not math.isfinite(row_f):
        return None

    col = int(round(col_f))
    row = int(round(row_f))
    if row < 0 or col < 0 or row >= water_mask_2d.shape[0] or col >= water_mask_2d.shape[1]:
        return None
    if not water_mask_2d[row, col]:
        return None

    top_depth = float(top_depth_2d[row, col])
    bottom_depth = float(bottom_depth_2d[row, col])
    if not math.isfinite(top_depth) or not math.isfinite(bottom_depth):
        return None
    if depth_m < top_depth + clearance_m or depth_m > bottom_depth - clearance_m:
        return None

    depth_profile = depth_levels[:, row, col]
    w_depth_profile = w_depth_levels[:, row, col]
    u = interpolate_profile(depth_profile, u_volume[:, row, col], depth_m)
    v = interpolate_profile(depth_profile, v_volume[:, row, col], depth_m)
    w = interpolate_profile(w_depth_profile, w_volume[:, row, col], depth_m)
    theta = interpolate_profile(depth_profile, theta_volume[:, row, col], depth_m)
    sal = interpolate_profile(depth_profile, sal_volume[:, row, col], depth_m)
    if not all(math.isfinite(value) for value in (u, v, w, theta, sal)):
        return None

    speed = math.hypot(u, v)
    if not math.isfinite(speed):
        return None

    dcol_dt = u / dx_m
    drow_dt = v / dy_m
    horizontal_cell_speed = math.hypot(dcol_dt, drow_dt)
    if not math.isfinite(horizontal_cell_speed) or horizontal_cell_speed < 1e-12:
        return None

    return {
        "x": x_m,
        "y": y_m,
        "depth": depth_m,
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
    seed_x_m: float,
    seed_y_m: float,
    seed_depth_m: float,
    direction: float,
    water_mask_2d: np.ndarray,
    top_depth_2d: np.ndarray,
    bottom_depth_2d: np.ndarray,
    depth_levels: np.ndarray,
    w_depth_levels: np.ndarray,
    u_volume: np.ndarray,
    v_volume: np.ndarray,
    w_volume: np.ndarray,
    theta_volume: np.ndarray,
    sal_volume: np.ndarray,
    x0_m: float,
    y0_m: float,
    dx_m: float,
    dy_m: float,
    clearance_m: float,
    step_cells: float,
    min_trace_speed: float,
    max_steps: int,
) -> list[dict[str, float]]:
    states: list[dict[str, float]] = []
    recent_state_keys: deque[tuple[int, int, int]] = deque(maxlen=DEFAULT_REVISIT_WINDOW_STEPS)
    reversal_steps: deque[int] = deque(maxlen=2)
    previous_step_vector: tuple[float, float, float] | None = None
    x_m = float(seed_x_m)
    y_m = float(seed_y_m)
    depth_m = float(seed_depth_m)

    for step_index in range(max_steps):
        state = sample_stream_state(
            x_m=x_m,
            y_m=y_m,
            depth_m=depth_m,
            water_mask_2d=water_mask_2d,
            top_depth_2d=top_depth_2d,
            bottom_depth_2d=bottom_depth_2d,
            depth_levels=depth_levels,
            w_depth_levels=w_depth_levels,
            u_volume=u_volume,
            v_volume=v_volume,
            w_volume=w_volume,
            theta_volume=theta_volume,
            sal_volume=sal_volume,
            x0_m=x0_m,
            y0_m=y0_m,
            dx_m=dx_m,
            dy_m=dy_m,
            clearance_m=clearance_m,
        )
        if state is None or state["speed"] < min_trace_speed:
            break

        state_key = quantize_state_key(x_m, y_m, depth_m, x0_m, y0_m, dx_m, dy_m)
        if state_key in recent_state_keys:
            break

        states.append(state)
        recent_state_keys.append(state_key)
        dt_seconds = step_cells / state["horizontal_cell_speed"]
        next_x_m = x_m + direction * state["u"] * dt_seconds
        next_y_m = y_m + direction * state["v"] * dt_seconds
        next_depth_m = depth_m - direction * state["w"] * dt_seconds

        if (
            not math.isfinite(next_x_m)
            or not math.isfinite(next_y_m)
            or not math.isfinite(next_depth_m)
            or next_depth_m < 0
        ):
            break

        if (
            abs(next_x_m - x_m) < 1e-4
            and abs(next_y_m - y_m) < 1e-4
            and abs(next_depth_m - depth_m) < 1e-4
        ):
            break

        step_vector = (next_x_m - x_m, next_y_m - y_m, next_depth_m - depth_m)
        vector_norm = math.sqrt(step_vector[0] ** 2 + step_vector[1] ** 2 + step_vector[2] ** 2)
        if not math.isfinite(vector_norm) or vector_norm <= 1e-9:
            break

        if previous_step_vector is not None:
            prev_norm = math.sqrt(
                previous_step_vector[0] ** 2 + previous_step_vector[1] ** 2 + previous_step_vector[2] ** 2
            )
            if math.isfinite(prev_norm) and prev_norm > 1e-9:
                dot = (
                    previous_step_vector[0] * step_vector[0]
                    + previous_step_vector[1] * step_vector[1]
                    + previous_step_vector[2] * step_vector[2]
                ) / (prev_norm * vector_norm)
                dot = max(-1.0, min(1.0, dot))
                angle_deg = math.degrees(math.acos(dot))
                if angle_deg >= DEFAULT_REVERSAL_ANGLE_DEG:
                    reversal_steps.append(step_index)
                    if (
                        len(reversal_steps) >= 2
                        and reversal_steps[-1] - reversal_steps[-2] <= DEFAULT_REVERSAL_WINDOW_STEPS
                    ):
                        break

        previous_step_vector = step_vector
        x_m = next_x_m
        y_m = next_y_m
        depth_m = next_depth_m

    return states


def build_traced_streamline(
    *,
    seed_x_m: float,
    seed_y_m: float,
    seed_depth_m: float,
    water_mask_2d: np.ndarray,
    top_depth_2d: np.ndarray,
    bottom_depth_2d: np.ndarray,
    depth_levels: np.ndarray,
    w_depth_levels: np.ndarray,
    u_volume: np.ndarray,
    v_volume: np.ndarray,
    w_volume: np.ndarray,
    theta_volume: np.ndarray,
    sal_volume: np.ndarray,
    x0_m: float,
    y0_m: float,
    dx_m: float,
    dy_m: float,
    clearance_m: float,
    step_cells: float,
    min_trace_speed: float,
    max_steps: int,
    min_segment_count: int,
    min_unique_xy_cells: int,
    min_net_displacement_cells: float,
) -> dict[str, Any] | None:
    forward = trace_streamline_direction(
        seed_x_m=seed_x_m,
        seed_y_m=seed_y_m,
        seed_depth_m=seed_depth_m,
        direction=1.0,
        water_mask_2d=water_mask_2d,
        top_depth_2d=top_depth_2d,
        bottom_depth_2d=bottom_depth_2d,
        depth_levels=depth_levels,
        w_depth_levels=w_depth_levels,
        u_volume=u_volume,
        v_volume=v_volume,
        w_volume=w_volume,
        theta_volume=theta_volume,
        sal_volume=sal_volume,
        x0_m=x0_m,
        y0_m=y0_m,
        dx_m=dx_m,
        dy_m=dy_m,
        clearance_m=clearance_m,
        step_cells=step_cells,
        min_trace_speed=min_trace_speed,
        max_steps=max_steps,
    )
    backward = trace_streamline_direction(
        seed_x_m=seed_x_m,
        seed_y_m=seed_y_m,
        seed_depth_m=seed_depth_m,
        direction=-1.0,
        water_mask_2d=water_mask_2d,
        top_depth_2d=top_depth_2d,
        bottom_depth_2d=bottom_depth_2d,
        depth_levels=depth_levels,
        w_depth_levels=w_depth_levels,
        u_volume=u_volume,
        v_volume=v_volume,
        w_volume=w_volume,
        theta_volume=theta_volume,
        sal_volume=sal_volume,
        x0_m=x0_m,
        y0_m=y0_m,
        dx_m=dx_m,
        dy_m=dy_m,
        clearance_m=clearance_m,
        step_cells=step_cells,
        min_trace_speed=min_trace_speed,
        max_steps=max_steps,
    )
    merged = list(reversed(backward)) + forward[1:]
    if len(merged) < max(2, min_segment_count + 1):
        return None

    x_values = np.asarray([point["x"] for point in merged], dtype=np.float32)
    y_values = np.asarray([point["y"] for point in merged], dtype=np.float32)
    depth_values = np.asarray([point["depth"] for point in merged], dtype=np.float32)
    unique_xy_cells = count_unique_xy_cells(x_values, y_values, x0_m, y0_m, dx_m, dy_m)
    if unique_xy_cells < max(1, int(min_unique_xy_cells)):
        return None

    if net_displacement_cells(x_values, y_values, dx_m, dy_m) < float(min_net_displacement_cells):
        return None

    return {
        "x": x_values,
        "y": y_values,
        "depth": depth_values,
        "theta": np.asarray([point["theta"] for point in merged], dtype=np.float32),
        "sal": np.asarray([point["sal"] for point in merged], dtype=np.float32),
        "segment_count": len(merged) - 1,
        "seed_speed": float(forward[0]["speed"]) if forward else float("nan"),
        "seed_x_m": float(seed_x_m),
        "seed_y_m": float(seed_y_m),
        "seed_row": int(round((seed_y_m - y0_m) / dy_m)),
        "seed_col": int(round((seed_x_m - x0_m) / dx_m)),
        "seed_depth_m": float(seed_depth_m),
        "unique_xy_cells": unique_xy_cells,
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
    for index in range(1, point_count):
        segment_x0.append(float(streamline["x"][index - 1]))
        segment_y0.append(float(streamline["y"][index - 1]))
        segment_depth0.append(float(streamline["depth"][index - 1]))
        segment_x1.append(float(streamline["x"][index]))
        segment_y1.append(float(streamline["y"][index]))
        segment_depth1.append(float(streamline["depth"][index]))
        segment_theta0.append(float(streamline["theta"][index - 1]))
        segment_sal0.append(float(streamline["sal"][index - 1]))
        segment_theta1.append(float(streamline["theta"][index]))
        segment_sal1.append(float(streamline["sal"][index]))
        segment_terminal_flag.append(1 if index == point_count - 1 else 0)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bed_meta, bed_mask = load_bedmachine_mask(Path(args.bedmachine_meta), Path(args.bedmachine_bin))
    bed_grid = bed_meta["grid"]
    bed_main_ice_mask = build_main_antarctic_ice_mask(bed_mask, bed_grid)

    with netCDF4.Dataset(input_path) as ds:
        vtransform = int(ds.variables["Vtransform"][:].item())
        if vtransform != 2:
            raise RuntimeError(f"Unsupported ROMS Vtransform={vtransform}; expected 2.")

        hc = float(ds.variables["hc"][:].item())
        s_rho = np.asarray(ds.variables["s_rho"][:], dtype=np.float32)
        c_rho = np.asarray(ds.variables["Cs_r"][:], dtype=np.float32)
        s_w = np.asarray(ds.variables["s_w"][:], dtype=np.float32)
        c_w = np.asarray(ds.variables["Cs_w"][:], dtype=np.float32)

        h_full = read_masked_array(ds.variables["h"], (slice(None), slice(None)))
        zice_full = read_masked_array(ds.variables["zice"], (slice(None), slice(None)))
        zeta_full = read_masked_array(ds.variables["zeta"], (0, slice(None), slice(None)))
        mask_rho = read_masked_array(
            ds.variables["mask_rho"], (slice(None, None, args.sample_stride), slice(None, None, args.sample_stride))
        )
        angle = read_masked_array(
            ds.variables["angle"], (slice(None, None, args.sample_stride), slice(None, None, args.sample_stride))
        )

        row_idx = np.arange(0, int(ds.dimensions["eta_rho"].size), max(1, int(args.sample_stride)), dtype=np.int32)
        col_idx = np.arange(0, int(ds.dimensions["xi_rho"].size), max(1, int(args.sample_stride)), dtype=np.int32)
        h = h_full[np.ix_(row_idx, col_idx)].copy()
        zice = zice_full[np.ix_(row_idx, col_idx)].copy()
        zeta = zeta_full[np.ix_(row_idx, col_idx)].copy()
        sample_dx_m = float(args.model_dx_m) * int(args.sample_stride)
        sample_dy_m = float(args.model_dy_m) * int(args.sample_stride)
        sample_x = float(args.model_x0_m) + col_idx.astype(np.float64) * float(args.model_dx_m)
        sample_y = float(args.model_y0_m) + row_idx.astype(np.float64) * float(args.model_dy_m)
        x_grid, y_grid = np.meshgrid(sample_x.astype(np.float32), sample_y.astype(np.float32))
        cell_sector_indices = sector_indices_from_xy(x_grid.astype(np.float64), y_grid.astype(np.float64))

        bed_col = np.rint((x_grid - float(bed_grid["x0_m"])) / float(bed_grid["dx_m"])).astype(np.int32)
        bed_row = np.rint((y_grid - float(bed_grid["y0_m"])) / float(bed_grid["dy_m"])).astype(np.int32)
        in_bed_extent = (
            (bed_col >= 0)
            & (bed_row >= 0)
            & (bed_col < int(bed_grid["nx"]))
            & (bed_row < int(bed_grid["ny"]))
        )
        bed_region_mask = np.full(mask_rho.shape, 255, dtype=np.uint8)
        bed_region_mask[in_bed_extent] = bed_mask[bed_row[in_bed_extent], bed_col[in_bed_extent]]
        bed_main_ice_region = np.zeros(mask_rho.shape, dtype=bool)
        bed_main_ice_region[in_bed_extent] = bed_main_ice_mask[bed_row[in_bed_extent], bed_col[in_bed_extent]]
        bed_water_mask = np.isin(bed_region_mask, (0, 3))

        model_water_mask = (mask_rho > 0.5) & np.isfinite(h) & np.isfinite(zice)
        open_water_mask = model_water_mask & bed_water_mask & (bed_region_mask == 0)
        cavity_water_mask = model_water_mask & bed_water_mask & (bed_region_mask == 3) & bed_main_ice_region
        water_mask_2d = open_water_mask | cavity_water_mask
        if not np.any(water_mask_2d):
            raise RuntimeError("No Antarctica ocean cells remain after clipping and masking.")
        target_streamlines = resolve_target_streamlines(args)
        streamline_class_def = build_streamline_class_definition(
            str(args.streamline_class),
            open_water_mask=open_water_mask,
            cavity_water_mask=cavity_water_mask,
            sample_dx_m=sample_dx_m,
            sample_dy_m=sample_dy_m,
        )
        seed_masks = {str(key): np.asarray(value, dtype=bool) for key, value in streamline_class_def["seed_masks"].items()}
        seed_buckets = tuple(streamline_class_def["seed_buckets"])

        top_depth = np.maximum(0.0, -(zeta + np.minimum(zice, 0.0))).astype(np.float32)
        bottom_depth = h.astype(np.float32)
        depth_levels = compute_depth_levels(h, zice, zeta, s_rho, c_rho, hc)
        w_depth_levels = compute_depth_levels(h, zice, zeta, s_w, c_w, hc)

        ny_s, nx_s = water_mask_2d.shape
        level_count = s_rho.size
        w_level_count = s_w.size
        w_volume = np.full((w_level_count, ny_s, nx_s), np.nan, dtype=np.float32)
        theta_volume = np.full((level_count, ny_s, nx_s), np.nan, dtype=np.float32)
        sal_volume = np.full((level_count, ny_s, nx_s), np.nan, dtype=np.float32)

        xi_u_size = int(ds.dimensions["xi_u"].size)
        eta_v_size = int(ds.dimensions["eta_v"].size)
        u_left_native_cols = np.clip(col_idx - 1, 0, xi_u_size - 1).astype(np.int32)
        u_right_native_cols = np.clip(col_idx, 0, xi_u_size - 1).astype(np.int32)
        v_lower_native_rows = np.clip(row_idx - 1, 0, eta_v_size - 1).astype(np.int32)
        v_upper_native_rows = np.clip(row_idx, 0, eta_v_size - 1).astype(np.int32)

        needed_u_cols = np.unique(np.concatenate([u_left_native_cols, u_right_native_cols])).astype(np.int32)
        needed_v_rows = np.unique(np.concatenate([v_lower_native_rows, v_upper_native_rows])).astype(np.int32)

        u_col_lookup = {int(col): idx for idx, col in enumerate(needed_u_cols.tolist())}
        v_row_lookup = {int(row): idx for idx, row in enumerate(needed_v_rows.tolist())}
        u_left_lut = np.asarray([u_col_lookup[int(col)] for col in u_left_native_cols], dtype=np.int32)
        u_right_lut = np.asarray([u_col_lookup[int(col)] for col in u_right_native_cols], dtype=np.int32)
        v_lower_lut = np.asarray([v_row_lookup[int(row)] for row in v_lower_native_rows], dtype=np.int32)
        v_upper_lut = np.asarray([v_row_lookup[int(row)] for row in v_upper_native_rows], dtype=np.int32)

        depth_levels_u = compute_depth_levels(
            gather_rho_to_u(h_full, row_idx, needed_u_cols),
            gather_rho_to_u(zice_full, row_idx, needed_u_cols),
            gather_rho_to_u(zeta_full, row_idx, needed_u_cols),
            s_rho,
            c_rho,
            hc,
        )
        depth_levels_v = compute_depth_levels(
            gather_rho_to_v(h_full, needed_v_rows, col_idx),
            gather_rho_to_v(zice_full, needed_v_rows, col_idx),
            gather_rho_to_v(zeta_full, needed_v_rows, col_idx),
            s_rho,
            c_rho,
            hc,
        )
        del h_full, zice_full, zeta_full

        u_native_volume = np.full((level_count, ny_s, needed_u_cols.size), np.nan, dtype=np.float32)
        v_native_volume = np.full((level_count, needed_v_rows.size, nx_s), np.nan, dtype=np.float32)

        cos_angle = np.cos(angle).astype(np.float32)
        sin_angle = np.sin(angle).astype(np.float32)

        for level in range(level_count):
            temp_slice = read_masked_array(
                ds.variables["temp"], (0, level, slice(None, None, args.sample_stride), slice(None, None, args.sample_stride))
            )
            salt_slice = read_masked_array(
                ds.variables["salt"], (0, level, slice(None, None, args.sample_stride), slice(None, None, args.sample_stride))
            )
            u_slice = read_masked_array(ds.variables["u"], (0, level, slice(None), slice(None)))
            v_slice = read_masked_array(ds.variables["v"], (0, level, slice(None), slice(None)))

            theta_volume[level] = temp_slice
            sal_volume[level] = salt_slice
            u_native_volume[level] = u_slice[np.ix_(row_idx, needed_u_cols)]
            v_native_volume[level] = v_slice[np.ix_(needed_v_rows, col_idx)]
        for level in range(w_level_count):
            w_volume[level] = read_masked_array(
                ds.variables["w"], (0, level, slice(None, None, args.sample_stride), slice(None, None, args.sample_stride))
            )

        # ROMS stores rho-levels from the seabed upward; streamline sampling expects shallow to deep.
        depth_levels = depth_levels[::-1].copy()
        w_depth_levels = w_depth_levels[::-1].copy()
        depth_levels_u = depth_levels_u[::-1].copy()
        depth_levels_v = depth_levels_v[::-1].copy()
        u_native_volume = u_native_volume[::-1].copy()
        v_native_volume = v_native_volume[::-1].copy()
        w_volume = w_volume[::-1].copy()
        theta_volume = theta_volume[::-1].copy()
        sal_volume = sal_volume[::-1].copy()

        u_volume = np.full((level_count, ny_s, nx_s), np.nan, dtype=np.float32)
        v_volume = np.full((level_count, ny_s, nx_s), np.nan, dtype=np.float32)
        target_depths = depth_levels
        u_left = interpolate_volume_to_target_depths(
            u_native_volume[:, :, u_left_lut],
            depth_levels_u[:, :, u_left_lut],
            target_depths,
        )
        u_right = interpolate_volume_to_target_depths(
            u_native_volume[:, :, u_right_lut],
            depth_levels_u[:, :, u_right_lut],
            target_depths,
        )
        u_volume = 0.5 * (u_left + u_right)
        v_lower = interpolate_volume_to_target_depths(
            v_native_volume[:, v_lower_lut, :],
            depth_levels_v[:, v_lower_lut, :],
            target_depths,
        )
        v_upper = interpolate_volume_to_target_depths(
            v_native_volume[:, v_upper_lut, :],
            depth_levels_v[:, v_upper_lut, :],
            target_depths,
        )
        v_volume = 0.5 * (v_lower + v_upper)

        u_rot = u_volume * cos_angle[None, ...] - v_volume * sin_angle[None, ...]
        v_rot = u_volume * sin_angle[None, ...] + v_volume * cos_angle[None, ...]
        u_volume = u_rot.astype(np.float32)
        v_volume = v_rot.astype(np.float32)

        rng = np.random.default_rng(int(args.random_seed))
        water_column_thickness = bottom_depth - top_depth
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
        bucket_streamlines: dict[str, list[dict[str, Any]]] = {str(bucket["key"]): [] for bucket in seed_buckets}
        bucket_valid_masks: dict[str, np.ndarray] = {}
        bucket_valid_sector_counts: dict[str, list[int]] = {}
        bucket_labels = {str(bucket["key"]): str(bucket["label"]) for bucket in seed_buckets}

        for bucket in seed_buckets:
            bucket_key = str(bucket["key"])
            mask_name = str(bucket["mask_name"])
            base_mask = seed_masks.get(mask_name)
            if base_mask is None:
                base_mask = cavity_water_mask if "cavity" in mask_name else open_water_mask
            valid_seed_cells = base_mask.copy()
            valid_seed_cells &= np.isfinite(water_column_thickness)
            valid_seed_cells &= water_column_thickness >= float(bucket["min_total_depth_m"])

            proxy_depths = np.clip(
                top_depth + float(bucket["proxy_fraction"]) * water_column_thickness,
                top_depth + DEFAULT_WATER_COLUMN_CLEARANCE_M,
                bottom_depth - DEFAULT_WATER_COLUMN_CLEARANCE_M,
            ).astype(np.float32)
            u_proxy = interpolate_plane_at_target_depths(u_volume, depth_levels, proxy_depths)
            v_proxy = interpolate_plane_at_target_depths(v_volume, depth_levels, proxy_depths)
            speed_proxy = np.hypot(u_proxy, v_proxy)
            if bool(bucket.get("use_speed_filter_in_seed_mask", True)):
                min_proxy_speed_mps = float(bucket.get("min_proxy_speed_mps", args.min_speed_mps))
                valid_seed_cells &= np.isfinite(speed_proxy) & (speed_proxy >= min_proxy_speed_mps)
            bucket_valid_masks[bucket_key] = valid_seed_cells.copy()
            bucket_valid_sector_counts[bucket_key] = mask_sector_counts(valid_seed_cells, x_grid, y_grid)

            candidate_target = max(
                1,
                int(round(float(target_streamlines) * float(bucket["retain_fraction"]) * float(bucket["candidate_factor"]))),
            )
            seed_layout_mode = str(bucket.get("seed_layout_mode", "block_scattered"))
            seed_weight_mode = str(bucket.get("seed_weight_mode", "speed_proxy"))
            if seed_layout_mode == "mask_aware_poisson_disk":
                seeds = collect_mask_aware_poisson_disk_cells(
                    valid_seed_cells,
                    candidate_target,
                    rng,
                )
            else:
                if seed_weight_mode == "uniform":
                    seed_weights = None
                else:
                    seed_weights = np.where(np.isfinite(speed_proxy), np.maximum(speed_proxy, 0.0), 0.0)
                seeds = collect_random_spatial_cells(
                    valid_seed_cells,
                    seed_weights,
                    candidate_target,
                    rng,
                )

            for seed_row, seed_col in seeds:
                top_here = float(top_depth[seed_row, seed_col])
                bottom_here = float(bottom_depth[seed_row, seed_col])
                seed_state_best: dict[str, float] | None = None
                seed_depth_m = float("nan")
                seed_depth_mode = str(bucket.get("seed_depth_mode", "fastest_valid"))
                proxy_depth_here = float(proxy_depths[seed_row, seed_col])
                if seed_depth_mode == "proxy_only":
                    state = sample_stream_state(
                        x_m=float(x_grid[seed_row, seed_col]),
                        y_m=float(y_grid[seed_row, seed_col]),
                        depth_m=proxy_depth_here,
                        water_mask_2d=water_mask_2d,
                        top_depth_2d=top_depth,
                        bottom_depth_2d=bottom_depth,
                        depth_levels=depth_levels,
                        w_depth_levels=w_depth_levels,
                        u_volume=u_volume,
                        v_volume=v_volume,
                        w_volume=w_volume,
                        theta_volume=theta_volume,
                        sal_volume=sal_volume,
                        x0_m=float(args.model_x0_m),
                        y0_m=float(args.model_y0_m),
                        dx_m=sample_dx_m,
                        dy_m=sample_dy_m,
                        clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                    )
                    if state is not None and float(state["speed"]) >= float(bucket["min_seed_speed_mps"]):
                        seed_state_best = state
                        seed_depth_m = proxy_depth_here
                else:
                    for _ in range(DEFAULT_SEED_DEPTH_ATTEMPTS):
                        candidate_depth_m = choose_seed_depth(
                            top_depth_m=top_here,
                            bottom_depth_m=bottom_here,
                            depth_fraction_range=tuple(bucket["depth_fraction_range"]),
                            depth_beta=tuple(bucket["depth_beta"]),
                            clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                            rng=rng,
                        )
                        if not math.isfinite(candidate_depth_m):
                            continue
                        state = sample_stream_state(
                            x_m=float(x_grid[seed_row, seed_col]),
                            y_m=float(y_grid[seed_row, seed_col]),
                            depth_m=candidate_depth_m,
                            water_mask_2d=water_mask_2d,
                            top_depth_2d=top_depth,
                            bottom_depth_2d=bottom_depth,
                            depth_levels=depth_levels,
                            w_depth_levels=w_depth_levels,
                            u_volume=u_volume,
                            v_volume=v_volume,
                            w_volume=w_volume,
                            theta_volume=theta_volume,
                            sal_volume=sal_volume,
                            x0_m=float(args.model_x0_m),
                            y0_m=float(args.model_y0_m),
                            dx_m=sample_dx_m,
                            dy_m=sample_dy_m,
                            clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                        )
                        if state is None or float(state["speed"]) < float(bucket["min_seed_speed_mps"]):
                            continue
                        if seed_depth_mode in {"first_valid", "first_valid_with_proxy_fallback"}:
                            seed_state_best = state
                            seed_depth_m = candidate_depth_m
                            break
                        if seed_state_best is None or float(state["speed"]) > float(seed_state_best["speed"]):
                            seed_state_best = state
                            seed_depth_m = candidate_depth_m

                    if (
                        seed_state_best is None
                        and seed_depth_mode == "first_valid_with_proxy_fallback"
                        and math.isfinite(proxy_depth_here)
                    ):
                        state = sample_stream_state(
                            x_m=float(x_grid[seed_row, seed_col]),
                            y_m=float(y_grid[seed_row, seed_col]),
                            depth_m=proxy_depth_here,
                            water_mask_2d=water_mask_2d,
                            top_depth_2d=top_depth,
                            bottom_depth_2d=bottom_depth,
                            depth_levels=depth_levels,
                            w_depth_levels=w_depth_levels,
                            u_volume=u_volume,
                            v_volume=v_volume,
                            w_volume=w_volume,
                            theta_volume=theta_volume,
                            sal_volume=sal_volume,
                            x0_m=float(args.model_x0_m),
                            y0_m=float(args.model_y0_m),
                            dx_m=sample_dx_m,
                            dy_m=sample_dy_m,
                            clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                        )
                        if state is not None and float(state["speed"]) >= float(bucket["min_seed_speed_mps"]):
                            seed_state_best = state
                            seed_depth_m = proxy_depth_here

                if seed_state_best is None:
                    continue

                streamline = build_traced_streamline(
                    seed_x_m=float(x_grid[seed_row, seed_col]),
                    seed_y_m=float(y_grid[seed_row, seed_col]),
                    seed_depth_m=seed_depth_m,
                    water_mask_2d=water_mask_2d,
                    top_depth_2d=top_depth,
                    bottom_depth_2d=bottom_depth,
                    depth_levels=depth_levels,
                    w_depth_levels=w_depth_levels,
                    u_volume=u_volume,
                    v_volume=v_volume,
                    w_volume=w_volume,
                    theta_volume=theta_volume,
                    sal_volume=sal_volume,
                    x0_m=float(args.model_x0_m),
                    y0_m=float(args.model_y0_m),
                    dx_m=sample_dx_m,
                    dy_m=sample_dy_m,
                    clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                    step_cells=float(args.flowline_step_cells),
                    min_trace_speed=float(bucket.get("min_trace_speed_mps", args.min_trace_speed_mps)),
                    max_steps=int(args.flowline_max_steps),
                    min_segment_count=int(bucket.get("min_streamline_segments", args.min_streamline_segments)),
                    min_unique_xy_cells=int(bucket.get("min_unique_xy_cells", DEFAULT_MIN_UNIQUE_XY_CELLS)),
                    min_net_displacement_cells=float(
                        bucket.get("min_net_displacement_cells", DEFAULT_MIN_NET_DISPLACEMENT_CELLS)
                    ),
                )
                if streamline is None:
                    continue

                streamline["bucket_key"] = bucket_key
                streamline["bucket_label"] = bucket_labels[bucket_key]
                streamline["seed_speed"] = float(seed_state_best["speed"])
                streamline["seed_theta"] = float(seed_state_best["theta"])
                streamline["seed_sal"] = float(seed_state_best["sal"])
                selection_rank_mode = str(bucket.get("selection_rank_mode", "segment_speed"))
                if selection_rank_mode == "segment_random":
                    streamline["selection_rank_secondary"] = float(rng.random())
                else:
                    streamline["selection_rank_secondary"] = float(seed_state_best["speed"])
                bucket_streamlines[bucket_key].append(streamline)

            if str(streamline_class_def.get("sector_fallback_mode", "")) == "proxy_relaxed_if_empty":
                expected_bucket_target = max(
                    1,
                    int(round(float(target_streamlines) * float(bucket["retain_fraction"]))),
                )
                sector_targets = compute_weighted_targets(bucket_valid_sector_counts[bucket_key], expected_bucket_target)
                candidate_sector_counts = retained_seed_sector_counts(bucket_streamlines[bucket_key])
                for sector_index, sector_target in enumerate(sector_targets):
                    if sector_target <= 0 or candidate_sector_counts[sector_index] > 0:
                        continue
                    sector_valid_mask = valid_seed_cells & (cell_sector_indices == sector_index)
                    fallback_seed_target = min(
                        int(np.count_nonzero(sector_valid_mask)),
                        max(128, int(sector_target) * 8),
                    )
                    if fallback_seed_target <= 0:
                        continue
                    fallback_seeds = collect_random_spatial_cells(
                        sector_valid_mask,
                        None,
                        fallback_seed_target,
                        rng,
                    )
                    for seed_row, seed_col in fallback_seeds:
                        proxy_depth_here = float(proxy_depths[seed_row, seed_col])
                        state = sample_stream_state(
                            x_m=float(x_grid[seed_row, seed_col]),
                            y_m=float(y_grid[seed_row, seed_col]),
                            depth_m=proxy_depth_here,
                            water_mask_2d=water_mask_2d,
                            top_depth_2d=top_depth,
                            bottom_depth_2d=bottom_depth,
                            depth_levels=depth_levels,
                            w_depth_levels=w_depth_levels,
                            u_volume=u_volume,
                            v_volume=v_volume,
                            w_volume=w_volume,
                            theta_volume=theta_volume,
                            sal_volume=sal_volume,
                            x0_m=float(args.model_x0_m),
                            y0_m=float(args.model_y0_m),
                            dx_m=sample_dx_m,
                            dy_m=sample_dy_m,
                            clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                        )
                        if state is None or float(state["speed"]) < float(bucket["min_seed_speed_mps"]):
                            continue
                        streamline = build_traced_streamline(
                            seed_x_m=float(x_grid[seed_row, seed_col]),
                            seed_y_m=float(y_grid[seed_row, seed_col]),
                            seed_depth_m=proxy_depth_here,
                            water_mask_2d=water_mask_2d,
                            top_depth_2d=top_depth,
                            bottom_depth_2d=bottom_depth,
                            depth_levels=depth_levels,
                            w_depth_levels=w_depth_levels,
                            u_volume=u_volume,
                            v_volume=v_volume,
                            w_volume=w_volume,
                            theta_volume=theta_volume,
                            sal_volume=sal_volume,
                            x0_m=float(args.model_x0_m),
                            y0_m=float(args.model_y0_m),
                            dx_m=sample_dx_m,
                            dy_m=sample_dy_m,
                            clearance_m=DEFAULT_WATER_COLUMN_CLEARANCE_M,
                            step_cells=float(args.flowline_step_cells),
                            min_trace_speed=float(bucket.get("fallback_min_trace_speed_mps", bucket.get("min_trace_speed_mps", args.min_trace_speed_mps))),
                            max_steps=int(args.flowline_max_steps),
                            min_segment_count=int(bucket.get("fallback_min_streamline_segments", bucket.get("min_streamline_segments", args.min_streamline_segments))),
                            min_unique_xy_cells=int(bucket.get("fallback_min_unique_xy_cells", bucket.get("min_unique_xy_cells", DEFAULT_MIN_UNIQUE_XY_CELLS))),
                            min_net_displacement_cells=float(
                                bucket.get("fallback_min_net_displacement_cells", bucket.get("min_net_displacement_cells", DEFAULT_MIN_NET_DISPLACEMENT_CELLS))
                            ),
                        )
                        if streamline is None:
                            continue
                        streamline["bucket_key"] = bucket_key
                        streamline["bucket_label"] = bucket_labels[bucket_key]
                        streamline["seed_speed"] = float(state["speed"])
                        streamline["seed_theta"] = float(state["theta"])
                        streamline["seed_sal"] = float(state["sal"])
                        streamline["selection_rank_secondary"] = float(rng.random())
                        bucket_streamlines[bucket_key].append(streamline)
                    candidate_sector_counts = retained_seed_sector_counts(bucket_streamlines[bucket_key])

        bucket_targets = compute_bucket_targets(int(target_streamlines), seed_buckets)
        selection_domain_mask = np.zeros_like(water_mask_2d, dtype=bool)
        for valid_mask in bucket_valid_masks.values():
            selection_domain_mask |= valid_mask
        global_depth_bin_size = compute_spatial_bin_size(
            selection_domain_mask if np.any(selection_domain_mask) else water_mask_2d,
            int(target_streamlines),
        )
        selected_streamlines: list[dict[str, Any]] = []
        leftovers: list[dict[str, Any]] = []
        retained_depths_by_global_bin: dict[tuple[int, int], list[float]] = defaultdict(list)
        retained_sector_counts = [0 for _ in range(DEFAULT_SECTOR_COUNT)]
        selection_strategy = str(streamline_class_def.get("selection_strategy", "spatial_bin_round_robin"))
        front_target_count = 0
        if bool(streamline_class_def.get("prefer_front_extra_fill", False)):
            front_target_count = sum(
                bucket_target
                for bucket, bucket_target in zip(seed_buckets, bucket_targets)
                if str(bucket["key"]) in FRONT_BUCKET_KEYS
            )
        for bucket, bucket_target in zip(seed_buckets, bucket_targets):
            bucket_key = str(bucket["key"])
            selection_bin_size = compute_spatial_bin_size(bucket_valid_masks.get(bucket_key, water_mask_2d), bucket_target)
            if selection_strategy == "retain_all_filtered":
                selected_bucket = list(bucket_streamlines[bucket_key])
                leftovers_bucket = []
                for sector_index, count in enumerate(retained_seed_sector_counts(selected_bucket)):
                    retained_sector_counts[sector_index] += count
            elif str(streamline_class_def.get("selection_sector_targets", "")) == "valid_seed_area":
                sector_targets = compute_weighted_targets(bucket_valid_sector_counts.get(bucket_key, []), bucket_target)
                selected_bucket, leftovers_bucket = select_streamlines_spatially_by_sector_targets(
                    bucket_streamlines[bucket_key],
                    target_count=bucket_target,
                    sector_targets=sector_targets,
                    selection_bin_size=selection_bin_size,
                    depth_bin_size=selection_bin_size,
                    retained_depths_by_bin=retained_depths_by_global_bin,
                    retained_sector_counts=retained_sector_counts,
                    rng=rng,
                )
            else:
                selected_bucket, leftovers_bucket = select_streamlines_spatially(
                    bucket_streamlines[bucket_key],
                    target_count=bucket_target,
                    selection_bin_size=selection_bin_size,
                    depth_bin_size=selection_bin_size,
                    retained_depths_by_bin=retained_depths_by_global_bin,
                    rng=rng,
                )
                for sector_index, count in enumerate(retained_seed_sector_counts(selected_bucket)):
                    retained_sector_counts[sector_index] += count
            selected_streamlines.extend(selected_bucket)
            leftovers.extend(leftovers_bucket)

        if selection_strategy != "retain_all_filtered" and len(selected_streamlines) < int(target_streamlines):
            remaining_slots = int(target_streamlines) - len(selected_streamlines)
            if bool(streamline_class_def.get("prefer_front_extra_fill", False)) and front_target_count > 0:
                current_front_count = sum(
                    1 for streamline in selected_streamlines if str(streamline.get("bucket_key", "")) in FRONT_BUCKET_KEYS
                )
                front_needed = max(0, front_target_count - current_front_count)
                front_leftovers = [
                    streamline for streamline in leftovers if str(streamline.get("bucket_key", "")) in FRONT_BUCKET_KEYS
                ]
                non_front_leftovers = [
                    streamline for streamline in leftovers if str(streamline.get("bucket_key", "")) not in FRONT_BUCKET_KEYS
                ]
                selected_front, front_leftovers = select_streamlines_sector_balanced(
                    front_leftovers,
                    target_count=min(remaining_slots, front_needed),
                    selection_bin_size=global_depth_bin_size,
                    depth_bin_size=global_depth_bin_size,
                    retained_depths_by_bin=retained_depths_by_global_bin,
                    retained_sector_counts=retained_sector_counts,
                    rng=rng,
                    prefer_front=True,
                )
                selected_streamlines.extend(selected_front)
                remaining_slots = int(target_streamlines) - len(selected_streamlines)
                leftovers = non_front_leftovers + front_leftovers

            if remaining_slots > 0:
                extra_selected, leftovers = select_streamlines_sector_balanced(
                    leftovers,
                    target_count=remaining_slots,
                    selection_bin_size=global_depth_bin_size,
                    depth_bin_size=global_depth_bin_size,
                    retained_depths_by_bin=retained_depths_by_global_bin,
                    retained_sector_counts=retained_sector_counts,
                    rng=rng,
                    prefer_front=False,
                )
                selected_streamlines.extend(extra_selected)

        selected_seed_speeds: list[float] = []
        selected_seed_theta: list[float] = []
        selected_seed_sal: list[float] = []
        selected_seed_depths: list[float] = []
        selected_seed_sector_counts = retained_seed_sector_counts(selected_streamlines)
        counts_by_seed_bucket: dict[str, int] = {str(bucket["key"]): 0 for bucket in seed_buckets}
        flowlines_by_seed_bucket: dict[str, int] = {str(bucket["key"]): 0 for bucket in seed_buckets}
        seed_depth_stats_by_bucket: dict[str, dict[str, float]] = {}

        bucket_depth_samples_selected: dict[str, list[float]] = {str(bucket["key"]): [] for bucket in seed_buckets}
        for streamline in selected_streamlines:
            bucket_key = str(streamline["bucket_key"])
            flowlines_by_seed_bucket[bucket_key] += 1
            counts_by_seed_bucket[bucket_key] += int(streamline["segment_count"])
            selected_seed_speeds.append(float(streamline["seed_speed"]))
            selected_seed_theta.append(float(streamline["seed_theta"]))
            selected_seed_sal.append(float(streamline["seed_sal"]))
            selected_seed_depths.append(float(streamline["seed_depth_m"]))
            bucket_depth_samples_selected[bucket_key].append(float(streamline["seed_depth_m"]))
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

        for bucket in seed_buckets:
            bucket_key = str(bucket["key"])
            depth_samples = np.asarray(bucket_depth_samples_selected[bucket_key], dtype=np.float32)
            if depth_samples.size:
                seed_depth_stats_by_bucket[bucket_key] = {
                    "min_m": float(np.min(depth_samples)),
                    "max_m": float(np.max(depth_samples)),
                    "mean_m": float(np.mean(depth_samples)),
                }

    if not segment_x0:
        raise RuntimeError("No valid Antarctica 3D ocean streamlines were retained.")

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

    speed_all = np.asarray(selected_seed_speeds, dtype=np.float32) if selected_seed_speeds else np.array([0], dtype=np.float32)
    theta_all = (
        np.concatenate([theta0_all, theta1_all]) if theta0_all.size and theta1_all.size else np.asarray(selected_seed_theta, dtype=np.float32)
    )
    sal_all = np.concatenate([sal0_all, sal1_all]) if sal0_all.size and sal1_all.size else np.asarray(selected_seed_sal, dtype=np.float32)

    with netCDF4.Dataset(input_path) as ds:
        ocean_time_var = ds.variables["ocean_time"]
        time_value = float(ocean_time_var[0])
        time_units = decode_attr(getattr(ocean_time_var, "units", ""), "")
        time_calendar = decode_attr(getattr(ocean_time_var, "calendar", "gregorian"), "gregorian")
        source_time = netCDF4.num2date(time_value, units=time_units, calendar=time_calendar)
        source_time_iso = source_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        source_title = decode_attr(getattr(ds, "title", ""), "Whole Antarctic Ocean Application")

    year_match = re.search(r"yr(\d+)", input_path.stem.lower())
    source_time_label = f"Year {year_match.group(1)} annual mean" if year_match and "annual" in input_path.stem.lower() else source_time_iso

    streamline_count = int(sum(flowlines_by_seed_bucket.values()))
    segment_count = int(x0_all.size)
    valid_seed_sector_counts = [0 for _ in range(DEFAULT_SECTOR_COUNT)]
    for counts in bucket_valid_sector_counts.values():
        for sector_index, count in enumerate(counts[: DEFAULT_SECTOR_COUNT]):
            valid_seed_sector_counts[sector_index] += int(count)
    basename = str(args.output_basename).strip() or build_output_basename(input_path.name, str(streamline_class_def["key"]))
    out_bin = output_dir / f"{basename}.bin"
    out_meta = output_dir / f"{basename}.meta.json"
    effective_min_proxy_speed_mps = min(float(bucket.get("min_proxy_speed_mps", args.min_speed_mps)) for bucket in seed_buckets)
    effective_min_seed_speed_mps = min(float(bucket["min_seed_speed_mps"]) for bucket in seed_buckets)
    effective_min_trace_speed_mps = min(float(bucket.get("min_trace_speed_mps", args.min_trace_speed_mps)) for bucket in seed_buckets)
    effective_min_streamline_segments = min(int(bucket.get("min_streamline_segments", args.min_streamline_segments)) for bucket in seed_buckets)
    effective_min_unique_xy_cells = min(int(bucket.get("min_unique_xy_cells", DEFAULT_MIN_UNIQUE_XY_CELLS)) for bucket in seed_buckets)
    effective_min_net_displacement_cells = min(
        float(bucket.get("min_net_displacement_cells", DEFAULT_MIN_NET_DISPLACEMENT_CELLS)) for bucket in seed_buckets
    )

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
        "title": source_title,
        "product_version": "WAOM2",
        "source_file": input_path.name,
        "source_reference": "https://gmd.copernicus.org/articles/15/723/2022/",
        "source_time_utc": source_time_iso,
        "source_time_label": source_time_label,
        "projection": "EPSG:3031",
        "geometry_type": "streamlines_3d",
        "sampling": {
            "sample_stride": int(args.sample_stride),
            "streamline_class": str(streamline_class_def["key"]),
            "streamline_class_label": str(streamline_class_def["label"]),
            "seed_strategy": str(streamline_class_def["seed_strategy"]),
            "depth_sampling_summary": str(streamline_class_def["depth_sampling_summary"]),
            "min_speed_mps": effective_min_proxy_speed_mps,
            "min_seed_speed_mps": effective_min_seed_speed_mps,
            "min_trace_speed_mps": effective_min_trace_speed_mps,
            "flowline_step_cells": float(args.flowline_step_cells),
            "flowline_max_steps": int(args.flowline_max_steps),
            "target_streamline_count": int(target_streamlines),
            "min_streamline_segments": effective_min_streamline_segments,
            "random_seed": int(args.random_seed),
            "source_grid_dx_m": float(args.model_dx_m),
            "source_grid_dy_m": float(args.model_dy_m),
            "source_grid_x0_m": float(args.model_x0_m),
            "source_grid_y0_m": float(args.model_y0_m),
            "clearance_m": DEFAULT_WATER_COLUMN_CLEARANCE_M,
            "velocity_vertical_mapping": "native_u_v_to_rho_depths",
            "selection_strategy": str(streamline_class_def["selection_strategy"]),
            "selection_bin_size_cells_global": int(global_depth_bin_size),
            "vertical_seed_separation_m": DEFAULT_VERTICAL_SEED_SEPARATION_M,
            "front_radius_cells": DEFAULT_FRONT_RADIUS_CELLS,
            "region_definition": streamline_class_def.get("region_metadata", {}),
            "anti_zigzag": {
                "state_depth_band_m": DEFAULT_STATE_DEPTH_BAND_M,
                "revisit_window_steps": DEFAULT_REVISIT_WINDOW_STEPS,
                "reversal_window_steps": DEFAULT_REVERSAL_WINDOW_STEPS,
                "reversal_angle_deg": DEFAULT_REVERSAL_ANGLE_DEG,
                "min_unique_xy_cells": effective_min_unique_xy_cells,
                "min_net_displacement_cells": effective_min_net_displacement_cells,
            },
            "seed_bucket_labels": bucket_labels,
            "seed_buckets": [
                {
                    "key": str(bucket["key"]),
                    "label": str(bucket["label"]),
                    "mask_name": str(bucket["mask_name"]),
                    "retain_fraction": float(bucket["retain_fraction"]),
                    "candidate_factor": float(bucket["candidate_factor"]),
                    "min_total_depth_m": float(bucket["min_total_depth_m"]),
                    "depth_fraction_range": [float(value) for value in bucket["depth_fraction_range"]],
                    "proxy_fraction": float(bucket["proxy_fraction"]),
                    "min_seed_speed_mps": float(bucket["min_seed_speed_mps"]),
                    "min_proxy_speed_mps": float(bucket.get("min_proxy_speed_mps", args.min_speed_mps)),
                    "min_trace_speed_mps": float(bucket.get("min_trace_speed_mps", args.min_trace_speed_mps)),
                    "seed_layout_mode": str(bucket.get("seed_layout_mode", "block_scattered")),
                    "seed_weight_mode": str(bucket.get("seed_weight_mode", "speed_proxy")),
                    "seed_depth_mode": str(bucket.get("seed_depth_mode", "fastest_valid")),
                    "selection_rank_mode": str(bucket.get("selection_rank_mode", "segment_speed")),
                    "min_streamline_segments": int(bucket.get("min_streamline_segments", args.min_streamline_segments)),
                    "min_unique_xy_cells": int(bucket.get("min_unique_xy_cells", DEFAULT_MIN_UNIQUE_XY_CELLS)),
                    "min_net_displacement_cells": float(
                        bucket.get("min_net_displacement_cells", DEFAULT_MIN_NET_DISPLACEMENT_CELLS)
                    ),
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
            "seed_depth_min_m": float(np.min(selected_seed_depths)) if selected_seed_depths else 0.0,
            "seed_depth_max_m": float(np.max(selected_seed_depths)) if selected_seed_depths else 0.0,
            "valid_seed_sector_counts_8": valid_seed_sector_counts,
            "retained_seed_sector_counts_8": selected_seed_sector_counts,
            "segments_by_seed_bucket": counts_by_seed_bucket,
            "streamlines_by_seed_bucket": flowlines_by_seed_bucket,
            "seed_depth_stats_by_bucket": seed_depth_stats_by_bucket,
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
        f"Kept {streamline_count} Antarctica 3D streamlines / {segment_count} segments for "
        f"{streamline_class_def['label']}"
    )


if __name__ == "__main__":
    main()
