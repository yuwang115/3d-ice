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

DEFAULT_SEED_DEPTHS_M = (0.0, 50.0, 150.0, 300.0, 600.0, 1000.0)
DEFAULT_SAMPLE_STRIDE = 6
DEFAULT_BEDMACHINE_MARGIN_M = 450_000.0
DEFAULT_MIN_SPEED_MPS = 0.01
DEFAULT_MIN_SEED_SPEED_MPS = 0.03
DEFAULT_MIN_TRACE_SPEED_MPS = 0.015
DEFAULT_WATER_COLUMN_CLEARANCE_M = 20.0
DEFAULT_FLOWLINE_STEP_CELLS = 0.75
DEFAULT_FLOWLINE_MAX_STEPS = 72
DEFAULT_SEED_TARGET_PER_DEPTH = 220
DEFAULT_ADAPTIVE_BOTTOM_SEED_TARGET = 90
DEFAULT_ADAPTIVE_BOTTOM_MIN_MODEL_DEPTH_M = 1400.0
DEFAULT_ADAPTIVE_BOTTOM_MIN_SEED_DEPTH_M = 1200.0
DEFAULT_ADAPTIVE_BOTTOM_DEPTH_FRACTION = 0.82
DEFAULT_ADAPTIVE_BOTTOM_OFFSET_M = 120.0
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
        "--seed-target-per-depth",
        type=int,
        default=DEFAULT_SEED_TARGET_PER_DEPTH,
        help="Approximate target number of streamline seeds per seed depth.",
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


def select_depth_indices(depth_axis: np.ndarray, requested_depths_m: tuple[float, ...]) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    used: set[int] = set()
    for requested in requested_depths_m:
        idx = int(np.argmin(np.abs(depth_axis - requested)))
        if idx in used:
            continue
        used.add(idx)
        selected.append((idx, float(depth_axis[idx])))
    return selected


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


def append_traced_streamline(
    *,
    depth_axis: np.ndarray,
    seed_depth_m: float,
    seed_row: int,
    seed_col: int,
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
) -> bool:
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
    if len(merged) < 4:
        return False

    for segment_index in range(1, len(merged)):
        point0 = merged[segment_index - 1]
        point1 = merged[segment_index]
        segment_x0.append(point0["x"])
        segment_y0.append(point0["y"])
        segment_depth0.append(point0["depth"])
        segment_x1.append(point1["x"])
        segment_y1.append(point1["y"])
        segment_depth1.append(point1["depth"])
        segment_theta0.append(point0["theta"])
        segment_sal0.append(point0["sal"])
        segment_theta1.append(point1["theta"])
        segment_sal1.append(point1["sal"])
        segment_terminal_flag.append(1 if segment_index == len(merged) - 1 else 0)
    return True


def compute_adaptive_bottom_seed_depth(model_depth_m: float, max_supported_depth_m: float) -> float:
    if not math.isfinite(model_depth_m) or not math.isfinite(max_supported_depth_m):
        return float("nan")
    if model_depth_m < DEFAULT_ADAPTIVE_BOTTOM_MIN_MODEL_DEPTH_M:
        return float("nan")

    candidate = min(
        model_depth_m * DEFAULT_ADAPTIVE_BOTTOM_DEPTH_FRACTION,
        model_depth_m - DEFAULT_WATER_COLUMN_CLEARANCE_M - DEFAULT_ADAPTIVE_BOTTOM_OFFSET_M,
        max_supported_depth_m - DEFAULT_WATER_COLUMN_CLEARANCE_M,
    )
    candidate = max(DEFAULT_ADAPTIVE_BOTTOM_MIN_SEED_DEPTH_M, candidate)
    if candidate <= DEFAULT_SEED_DEPTHS_M[-1] + 40.0:
        return float("nan")
    if candidate >= model_depth_m - DEFAULT_WATER_COLUMN_CLEARANCE_M:
        return float("nan")
    return candidate


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    retained_speed_samples: list[np.ndarray] = []
    retained_theta_samples: list[np.ndarray] = []
    retained_sal_samples: list[np.ndarray] = []
    seed_depth_levels: list[float] = []
    counts_by_seed_depth: dict[str, int] = {}
    flowlines_by_seed_depth: dict[str, int] = {}

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
        max_supported_depth_m = float(depth_axis[-1])

        depth_selections = select_depth_indices(depth_axis, DEFAULT_SEED_DEPTHS_M)
        for depth_idx, seed_depth_m in depth_selections:
            u2d = u_volume[depth_idx]
            v2d = v_volume[depth_idx]
            theta2d = theta_volume[depth_idx]
            sal2d = sal_volume[depth_idx]
            speed2d = np.hypot(u2d, v2d)

            valid_seed_layer = water_mask_2d.copy()
            valid_seed_layer &= np.isfinite(u2d) & np.isfinite(v2d) & np.isfinite(theta2d) & np.isfinite(sal2d)
            valid_seed_layer &= np.isfinite(speed2d) & (speed2d >= float(args.min_speed_mps))
            valid_seed_layer &= sampled_model_depth >= seed_depth_m + DEFAULT_WATER_COLUMN_CLEARANCE_M

            if not np.any(valid_seed_layer):
                continue

            seed_depth_levels.append(seed_depth_m)
            retained_speed_samples.append(speed2d[valid_seed_layer].astype(np.float32))
            retained_theta_samples.append(theta2d[valid_seed_layer].astype(np.float32))
            retained_sal_samples.append(sal2d[valid_seed_layer].astype(np.float32))

            seeds = collect_seeds(
                speed2d,
                valid_seed_layer,
                float(args.min_seed_speed_mps),
                int(args.seed_target_per_depth),
            )
            depth_key = str(int(round(seed_depth_m)))
            segment_count_before = len(segment_x0)
            flowline_count = 0

            for seed_row, seed_col in seeds:
                appended = append_traced_streamline(
                    depth_axis=depth_axis,
                    seed_depth_m=seed_depth_m,
                    seed_row=seed_row,
                    seed_col=seed_col,
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
                    min_trace_speed=float(args.min_trace_speed_mps),
                    max_steps=int(args.flowline_max_steps),
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
                if appended:
                    flowline_count += 1

            flowlines_by_seed_depth[depth_key] = flowline_count
            counts_by_seed_depth[depth_key] = len(segment_x0) - segment_count_before

        adaptive_seed_depth_2d = np.full(sampled_model_depth.shape, np.nan, dtype=np.float32)
        adaptive_speed_2d = np.full(sampled_model_depth.shape, np.nan, dtype=np.float32)
        adaptive_theta_2d = np.full(sampled_model_depth.shape, np.nan, dtype=np.float32)
        adaptive_sal_2d = np.full(sampled_model_depth.shape, np.nan, dtype=np.float32)
        deep_water_candidates = water_mask_2d & (sampled_model_depth >= DEFAULT_ADAPTIVE_BOTTOM_MIN_MODEL_DEPTH_M)

        for seed_row, seed_col in np.argwhere(deep_water_candidates):
            seed_depth_m = compute_adaptive_bottom_seed_depth(
                float(sampled_model_depth[seed_row, seed_col]),
                max_supported_depth_m,
            )
            if not math.isfinite(seed_depth_m):
                continue
            state = sample_stream_state(
                depth_axis=depth_axis,
                depth_m=seed_depth_m,
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
            if state is None or state["speed"] < float(args.min_speed_mps):
                continue
            adaptive_seed_depth_2d[seed_row, seed_col] = state["depth"]
            adaptive_speed_2d[seed_row, seed_col] = state["speed"]
            adaptive_theta_2d[seed_row, seed_col] = state["theta"]
            adaptive_sal_2d[seed_row, seed_col] = state["sal"]

        valid_adaptive_seed_layer = (
            np.isfinite(adaptive_seed_depth_2d)
            & np.isfinite(adaptive_speed_2d)
            & np.isfinite(adaptive_theta_2d)
            & np.isfinite(adaptive_sal_2d)
        )
        if np.any(valid_adaptive_seed_layer):
            retained_speed_samples.append(adaptive_speed_2d[valid_adaptive_seed_layer].astype(np.float32))
            retained_theta_samples.append(adaptive_theta_2d[valid_adaptive_seed_layer].astype(np.float32))
            retained_sal_samples.append(adaptive_sal_2d[valid_adaptive_seed_layer].astype(np.float32))

            seeds = collect_seeds(
                adaptive_speed_2d,
                valid_adaptive_seed_layer,
                float(args.min_seed_speed_mps),
                DEFAULT_ADAPTIVE_BOTTOM_SEED_TARGET,
            )
            depth_key = "adaptive_bottom"
            segment_count_before = len(segment_x0)
            flowline_count = 0

            for seed_row, seed_col in seeds:
                seed_depth_m = float(adaptive_seed_depth_2d[seed_row, seed_col])
                appended = append_traced_streamline(
                    depth_axis=depth_axis,
                    seed_depth_m=seed_depth_m,
                    seed_row=seed_row,
                    seed_col=seed_col,
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
                    min_trace_speed=float(args.min_trace_speed_mps),
                    max_steps=int(args.flowline_max_steps),
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
                if appended:
                    flowline_count += 1

            flowlines_by_seed_depth[depth_key] = flowline_count
            counts_by_seed_depth[depth_key] = len(segment_x0) - segment_count_before

    if not segment_x0:
        raise RuntimeError("No valid Greenland 3D ocean streamlines were retained.")

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

    speed_all = np.concatenate(retained_speed_samples) if retained_speed_samples else np.array([0], dtype=np.float32)
    theta_all = np.concatenate(retained_theta_samples) if retained_theta_samples else np.array([0], dtype=np.float32)
    sal_all = np.concatenate(retained_sal_samples) if retained_sal_samples else np.array([0], dtype=np.float32)

    streamline_count = int(sum(flowlines_by_seed_depth.values()))
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
            "seed_depths_m": [float(depth) for depth in seed_depth_levels],
            "min_speed_mps": float(args.min_speed_mps),
            "min_seed_speed_mps": float(args.min_seed_speed_mps),
            "min_trace_speed_mps": float(args.min_trace_speed_mps),
            "flowline_step_cells": float(args.flowline_step_cells),
            "flowline_max_steps": int(args.flowline_max_steps),
            "seed_target_per_depth": int(args.seed_target_per_depth),
            "bedmachine_margin_m": float(args.margin_m),
            "adaptive_bottom_seeding": {
                "enabled": True,
                "seed_target": DEFAULT_ADAPTIVE_BOTTOM_SEED_TARGET,
                "min_model_depth_m": DEFAULT_ADAPTIVE_BOTTOM_MIN_MODEL_DEPTH_M,
                "min_seed_depth_m": DEFAULT_ADAPTIVE_BOTTOM_MIN_SEED_DEPTH_M,
                "depth_fraction": DEFAULT_ADAPTIVE_BOTTOM_DEPTH_FRACTION,
                "bottom_offset_m": DEFAULT_ADAPTIVE_BOTTOM_OFFSET_M,
            },
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
            "segments_by_seed_depth": counts_by_seed_depth,
            "streamlines_by_seed_depth": flowlines_by_seed_depth,
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
        f"Kept {streamline_count} 3D streamlines / {segment_count} segments across seed depths "
        f"{[float(depth) for depth in seed_depth_levels]}"
    )


if __name__ == "__main__":
    main()
