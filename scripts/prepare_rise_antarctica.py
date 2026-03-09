#!/usr/bin/env python3
"""Prepare RISE Antarctica preview rasters and BedMachine-aligned overlay packages."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

FILL_INT16 = -32768
MASK_FLAGS = {
    0: "no_data",
    1: "grounded",
    2: "iceshelf",
    3: "conshelf",
    4: "ocean",
}
MASK_COLORS = {
    0: "#000000",
    1: "#8f95a0",
    2: "#d8f5ff",
    3: "#65b8b5",
    4: "#113f70",
}
PREVIEW_VARIABLES = {
    "ismr": {
        "title": "Basal Melt",
        "unit": "m/yr",
        "description": "Average basal ice-shelf melt.",
        "palette": [
            (0.0, "#153760"),
            (0.22, "#4fa7d8"),
            (0.5, "#f6f7fb"),
            (0.78, "#f7a35b"),
            (1.0, "#8a1117"),
        ],
        "percentiles": [90.0, 90.0],
        "mode": "zero_centered_asymmetric",
        "preview_filename": "rise_antarctica_ismr.png",
        "meta_range_key": "display_range_m_per_year",
        "stats_key": "stats_m_per_year",
    },
    "tstar_zice": {
        "title": "Thermal Driving",
        "unit": "°C",
        "description": "Average thermal driving at the ice draft.",
        "palette": [
            (0.0, "#081d58"),
            (0.22, "#225ea8"),
            (0.52, "#1d91c0"),
            (0.78, "#7fcdbb"),
            (1.0, "#fff3a3"),
        ],
        "percentiles": [1.0, 99.0],
        "mode": "sequential_positive",
        "preview_filename": "rise_antarctica_tstar_zice.png",
        "meta_range_key": "display_range_c",
        "stats_key": "stats_c",
    },
    "zice": {
        "title": "Ice Draft Depth",
        "unit": "m",
        "description": "Ice draft depth.",
        "palette": [
            (0.0, "#edf8fb"),
            (0.24, "#bfd3e6"),
            (0.5, "#9ebcda"),
            (0.76, "#8c6bb1"),
            (1.0, "#5b1f6d"),
        ],
        "percentiles": [1.0, 99.0],
        "mode": "negative_depth",
        "preview_filename": "rise_antarctica_zice.png",
        "meta_range_key": "display_range_m",
        "stats_key": "stats_m",
    },
}
MASK_PREVIEW_FILENAME = "rise_antarctica_mask.png"
MASK_INDEX_FILENAME = "rise_antarctica_mask_index.png"
PREVIEW_INDEX_FILENAME = "rise_antarctica_preview_index.json"


@dataclass(frozen=True)
class TargetGrid:
    meta_path: Path
    output_basename: str
    label: str


DEFAULT_TARGETS = (
    TargetGrid(
        meta_path=Path("static/tools/data/bedmachine_antarctica_v4_480.meta.json"),
        output_basename="rise_antarctica_480",
        label="Balanced",
    ),
    TargetGrid(
        meta_path=Path("static/tools/data/bedmachine_antarctica_v4_741.meta.json"),
        output_basename="rise_antarctica_741",
        label="HD",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RISE Antarctica preview rasters and BedMachine-aligned overlay packages."
    )
    parser.add_argument("--input", required=True, help="Path to the RISE netCDF4/HDF5 file.")
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for generated preview rasters and BedMachine-aligned packages.",
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


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def interpolate_palette(stops: list[tuple[float, str]], t_values: np.ndarray) -> np.ndarray:
    positions = np.asarray([float(stop[0]) for stop in stops], dtype=np.float32)
    colors = np.asarray([hex_to_rgb(stop[1]) for stop in stops], dtype=np.float32)
    t = np.clip(np.asarray(t_values, dtype=np.float32), 0.0, 1.0)
    idx = np.searchsorted(positions, t, side="right") - 1
    idx = np.clip(idx, 0, len(stops) - 2)
    next_idx = idx + 1
    left = positions[idx]
    right = positions[next_idx]
    frac = np.divide(
        t - left,
        np.maximum(right - left, 1e-6),
        out=np.zeros_like(t, dtype=np.float32),
        where=np.abs(right - left) > 1e-6,
    )
    rgb = colors[idx] * (1.0 - frac[:, None]) + colors[next_idx] * frac[:, None]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def describe_mask(mask_values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    unique, counts = np.unique(mask_values.astype(np.uint8), return_counts=True)
    for value, count in zip(unique.tolist(), counts.tolist()):
        out[str(int(value))] = int(count)
    return out


def build_target_axis(grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    x0 = float(grid["x0_m"])
    y0 = float(grid["y0_m"])
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])
    target_x = x0 + np.arange(nx, dtype=np.float64) * dx
    target_y = y0 + np.arange(ny, dtype=np.float64) * dy
    return target_x, target_y


def resolve_indices(target_axis: np.ndarray, src_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_axis = np.asarray(src_axis, dtype=np.float64)
    if src_axis.ndim != 1 or src_axis.size < 2:
        raise ValueError("Source axis must be 1D with at least two values.")
    step = float(src_axis[1] - src_axis[0])
    if abs(step) < 1e-9:
        raise ValueError("Source axis spacing must be non-zero.")
    idx = np.rint((target_axis - float(src_axis[0])) / step).astype(np.int64)
    valid = (idx >= 0) & (idx < src_axis.size)
    return np.clip(idx, 0, src_axis.size - 1), valid


def sample_to_target(field: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    sampled = np.asarray(field[np.ix_(x_idx, y_idx)])
    return sampled.T.copy()


def quantize_to_int16(values: np.ndarray, scale: float) -> np.ndarray:
    out = np.full(values.shape, FILL_INT16, dtype=np.int16)
    valid = np.isfinite(values)
    if not np.any(valid):
        return out
    quantized = np.rint(values[valid] / scale)
    quantized = np.clip(quantized, -32767, 32767)
    out[valid] = quantized.astype(np.int16)
    return out


def compute_display_range(key: str, values: np.ndarray) -> tuple[list[float], list[float]]:
    spec = PREVIEW_VARIABLES[key]
    finite = values[np.isfinite(values)]
    percentiles = [float(spec["percentiles"][0]), float(spec["percentiles"][1])]

    if spec["mode"] == "zero_centered_asymmetric":
        negative = finite[finite < 0]
        positive = finite[finite > 0]
        if negative.size:
            low = float(np.percentile(negative, max(0.0, 100.0 - percentiles[0])))
        else:
            low = -0.25
        if positive.size:
            high = float(np.percentile(positive, min(100.0, percentiles[1])))
        else:
            high = 0.25
        low = min(low, -0.05)
        high = max(high, 0.25)
        return [low, high], [low, high]

    low, high = np.percentile(finite, percentiles)

    if spec["mode"] == "sequential_positive":
        lo = min(0.0, float(low))
        hi = max(float(high), lo + 1e-6)
        return [lo, hi], [float(low), float(high)]

    if spec["mode"] == "negative_depth":
        lo = float(low)
        hi = 0.0
        return [lo, hi], [float(low), float(high)]

    raise ValueError(f"Unsupported visualization mode: {spec['mode']}")


def values_to_t(key: str, values: np.ndarray, display_range: list[float]) -> np.ndarray:
    lo = float(display_range[0])
    hi = float(display_range[1])
    if key == "ismr":
        values_f = values.astype(np.float32)
        out = np.empty_like(values_f, dtype=np.float32)
        negative = values_f <= 0
        out[negative] = 0.5 * np.clip((values_f[negative] - lo) / max(1e-6, 0.0 - lo), 0.0, 1.0)
        out[~negative] = 0.5 + 0.5 * np.clip(values_f[~negative] / max(1e-6, hi), 0.0, 1.0)
        return out
    if key == "zice":
        depth = np.maximum(0.0, -values.astype(np.float32))
        depth_max = max(abs(lo), 1e-6)
        return np.clip(depth / depth_max, 0.0, 1.0)
    return np.clip((values.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def render_continuous_png(
    *,
    field_name: str,
    values_display: np.ndarray,
    display_range: list[float],
    output_path: Path,
) -> None:
    spec = PREVIEW_VARIABLES[field_name]
    rgba = np.zeros(values_display.shape + (4,), dtype=np.uint8)
    valid = np.isfinite(values_display)
    if np.any(valid):
        t = values_to_t(field_name, values_display[valid], display_range)
        rgba[valid, :3] = interpolate_palette(list(spec["palette"]), t)
        rgba[valid, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(output_path, optimize=True)


def render_mask_png(mask_display: np.ndarray, output_path: Path) -> None:
    rgba = np.zeros(mask_display.shape + (4,), dtype=np.uint8)
    for mask_value, color in MASK_COLORS.items():
        rgb = np.asarray(hex_to_rgb(color), dtype=np.uint8)
        hit = mask_display == mask_value
        if np.any(hit):
            rgba[hit, :3] = rgb
            rgba[hit, 3] = 255 if mask_value else 0
    Image.fromarray(rgba, mode="RGBA").save(output_path, optimize=True)


def render_mask_index(mask_display: np.ndarray, output_path: Path) -> None:
    Image.fromarray(mask_display.astype(np.uint8), mode="L").save(output_path, optimize=True)


def create_preview_index(
    *,
    ds: h5py.File,
    input_path: Path,
    output_dir: Path,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    mask_display: np.ndarray,
    variable_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    step_x = float(x_axis[1] - x_axis[0])
    step_y = float(y_axis[1] - y_axis[0])
    return {
        "title": "RISE Antarctica 2D Preview",
        "dataset_title": "Realistic ice-shelf/ocean state estimates (RISE) of basal melting and drivers: data",
        "dataset_source": "Australian Antarctic Data Centre (RISE multi-model mean, Version 1)",
        "citation": "Galton-Fenzi, B.K. (2025) Realistic ice-shelf/ocean state estimates (RISE) of basal melting and drivers: data.",
        "license": "CC BY 4.0",
        "source_file": input_path.name,
        "projection": "EPSG:3031 polar stereographic",
        "grid": {
            "width": int(mask_display.shape[1]),
            "height": int(mask_display.shape[0]),
            "x_min_m": float(np.min(x_axis)),
            "x_max_m": float(np.max(x_axis)),
            "y_min_m": float(np.min(y_axis)),
            "y_max_m": float(np.max(y_axis)),
            "dx_m": step_x,
            "dy_m": step_y,
        },
        "mask": {
            "image_url": f"/tools/data/{MASK_PREVIEW_FILENAME}",
            "index_image_url": f"/tools/data/{MASK_INDEX_FILENAME}",
            "categories": {
                str(mask_value): {
                    "label": label,
                    "color": MASK_COLORS[mask_value],
                }
                for mask_value, label in MASK_FLAGS.items()
                if mask_value != 0
            },
            "counts": describe_mask(mask_display),
        },
        "variables": variable_meta,
        "source_url": "https://data.aad.gov.au/metadata/RISE",
        "paper_url": "https://doi.org/10.5194/egusphere-2024-4047",
        "output_directory": str(output_dir),
    }


def write_preview_assets(ds: h5py.File, input_path: Path, output_dir: Path) -> None:
    mask_display = np.asarray(ds["mask"][:], dtype=np.uint8).T
    x_axis = np.asarray(ds["easting"][:, 0], dtype=np.float64)
    y_axis = np.asarray(ds["northing"][0, :], dtype=np.float64)

    render_mask_png(mask_display, output_dir / MASK_PREVIEW_FILENAME)
    render_mask_index(mask_display, output_dir / MASK_INDEX_FILENAME)

    variable_meta: dict[str, dict[str, Any]] = {
        "mask": {
            "title": "Mask",
            "unit": "category",
            "description": "RISE mask categories for grounded, ice-shelf, continental-shelf, and ocean cells.",
            "image_url": f"/tools/data/{MASK_PREVIEW_FILENAME}",
            "type": "categorical",
            "categories": {
                str(mask_value): {
                    "label": label,
                    "color": MASK_COLORS[mask_value],
                }
                for mask_value, label in MASK_FLAGS.items()
                if mask_value != 0
            },
        }
    }

    for field_name, spec in PREVIEW_VARIABLES.items():
        values_display = np.asarray(ds[field_name][:], dtype=np.float32).T
        display_range, percentile_range = compute_display_range(field_name, values_display)
        render_continuous_png(
            field_name=field_name,
            values_display=values_display,
            display_range=display_range,
            output_path=output_dir / str(spec["preview_filename"]),
        )
        variable_meta[field_name] = {
            "title": str(spec["title"]),
            "unit": str(spec["unit"]),
            "description": str(spec["description"]),
            "image_url": f"/tools/data/{spec['preview_filename']}",
            "type": "continuous",
            "display_range": display_range,
            "percentile_range": percentile_range,
            "recommended_percentiles": list(spec["percentiles"]),
            "stats": finite_stats(values_display),
            "palette_stops": [
                {"t": float(stop[0]), "color": str(stop[1])}
                for stop in spec["palette"]
            ],
        }

    preview_index = create_preview_index(
        ds=ds,
        input_path=input_path,
        output_dir=output_dir,
        x_axis=x_axis,
        y_axis=y_axis,
        mask_display=mask_display,
        variable_meta=variable_meta,
    )
    (output_dir / PREVIEW_INDEX_FILENAME).write_text(
        json.dumps(preview_index, indent=2),
        encoding="utf-8",
    )


def write_target_package(
    *,
    ds: h5py.File,
    input_path: Path,
    target: TargetGrid,
    output_dir: Path,
    src_x: np.ndarray,
    src_y: np.ndarray,
) -> None:
    bed_meta = json.loads(target.meta_path.read_text(encoding="utf-8"))
    grid = bed_meta["grid"]
    target_x, target_y = build_target_axis(grid)
    src_ix, valid_x = resolve_indices(target_x, src_x)
    src_iy, valid_y = resolve_indices(target_y, src_y)

    sampled_mask = sample_to_target(np.asarray(ds["mask"][:], dtype=np.uint8), src_ix, src_iy).astype(np.uint8)
    sampled_ids = sample_to_target(np.asarray(ds["iceshelf_id"][:], dtype=np.float32), src_ix, src_iy)
    sampled_zice = sample_to_target(np.asarray(ds["zice"][:], dtype=np.float32), src_ix, src_iy)
    sampled_ismr = sample_to_target(np.asarray(ds["ismr"][:], dtype=np.float32), src_ix, src_iy)
    sampled_tstar = sample_to_target(np.asarray(ds["tstar_zice"][:], dtype=np.float32), src_ix, src_iy)

    valid_grid = np.outer(valid_y, valid_x)
    sampled_mask = np.where(valid_grid, sampled_mask, 0).astype(np.uint8)
    sampled_ids = np.where(valid_grid, sampled_ids, 0.0)
    sampled_zice = np.where(sampled_mask == 2, sampled_zice, np.nan)
    sampled_ismr = np.where(sampled_mask == 2, sampled_ismr, np.nan)
    sampled_tstar = np.where(sampled_mask == 2, sampled_tstar, np.nan)

    mask_count = int(sampled_mask.size)
    zice_scale = 1.0
    ismr_scale = 0.001
    tstar_scale = 0.0001

    ids_u16 = np.clip(np.rint(np.nan_to_num(sampled_ids, nan=0.0)), 0, 65535).astype(np.uint16)
    zice_q = quantize_to_int16(sampled_zice, zice_scale)
    ismr_q = quantize_to_int16(sampled_ismr, ismr_scale)
    tstar_q = quantize_to_int16(sampled_tstar, tstar_scale)

    display_ismr, pct_ismr = compute_display_range("ismr", sampled_ismr)
    display_tstar, pct_tstar = compute_display_range("tstar_zice", sampled_tstar)
    display_zice, pct_zice = compute_display_range("zice", sampled_zice)

    bin_path = output_dir / f"{target.output_basename}.bin"
    with bin_path.open("wb") as fh:
        fh.write(sampled_mask.tobytes(order="C"))
        fh.write(ids_u16.tobytes(order="C"))
        fh.write(zice_q.tobytes(order="C"))
        fh.write(ismr_q.tobytes(order="C"))
        fh.write(tstar_q.tobytes(order="C"))

    offset_mask = 0
    offset_ids = offset_mask + mask_count
    offset_zice = offset_ids + mask_count * 2
    offset_ismr = offset_zice + mask_count * 2
    offset_tstar = offset_ismr + mask_count * 2

    meta = {
        "title": "RISE Antarctica ice-shelf basal melt and thermal driving",
        "product_version": "v1",
        "source_file": input_path.name,
        "source_dataset": "Australian Antarctic Data Centre RISE multi-model mean",
        "source_url": "https://data.aad.gov.au/metadata/RISE",
        "license": "CC BY 4.0",
        "resampled_to": target.label,
        "grid": {
            "nx": int(grid["nx"]),
            "ny": int(grid["ny"]),
            "x0_m": int(grid["x0_m"]),
            "y0_m": int(grid["y0_m"]),
            "dx_m": int(grid["dx_m"]),
            "dy_m": int(grid["dy_m"]),
        },
        "mask_flags": {str(key): value for key, value in MASK_FLAGS.items()},
        "coverage": {
            "cell_count": mask_count,
            "iceshelf_cell_count": int(np.count_nonzero(sampled_mask == 2)),
            "mask_counts": describe_mask(sampled_mask),
        },
        "visualization": {
            "basal_melt": {
                "display_range_m_per_year": display_ismr,
                "percentile_range_m_per_year": pct_ismr,
                "palette_stops": [
                    {"t": float(stop[0]), "color": str(stop[1])}
                    for stop in PREVIEW_VARIABLES["ismr"]["palette"]
                ],
                "recommended_percentiles": list(PREVIEW_VARIABLES["ismr"]["percentiles"]),
            },
            "thermal_driving": {
                "display_range_c": display_tstar,
                "percentile_range_c": pct_tstar,
                "palette_stops": [
                    {"t": float(stop[0]), "color": str(stop[1])}
                    for stop in PREVIEW_VARIABLES["tstar_zice"]["palette"]
                ],
                "recommended_percentiles": list(PREVIEW_VARIABLES["tstar_zice"]["percentiles"]),
            },
            "ice_draft": {
                "display_range_m": display_zice,
                "percentile_range_m": pct_zice,
                "palette_stops": [
                    {"t": float(stop[0]), "color": str(stop[1])}
                    for stop in PREVIEW_VARIABLES["zice"]["palette"]
                ],
                "recommended_percentiles": list(PREVIEW_VARIABLES["zice"]["percentiles"]),
            },
        },
        "fields": [
            {
                "name": "mask",
                "dtype": "uint8",
                "byte_offset": offset_mask,
                "byte_length": mask_count,
                "flags": {str(key): value for key, value in MASK_FLAGS.items()},
            },
            {
                "name": "iceshelf_id",
                "dtype": "uint16",
                "byte_offset": offset_ids,
                "byte_length": mask_count * 2,
                "stats": finite_stats(sampled_ids),
            },
            {
                "name": "zice",
                "dtype": "int16",
                "byte_offset": offset_zice,
                "byte_length": mask_count * 2,
                "scale": zice_scale,
                "offset": 0.0,
                "fill_value": FILL_INT16,
                "unit": "m",
                "stats_m": finite_stats(sampled_zice),
            },
            {
                "name": "ismr",
                "dtype": "int16",
                "byte_offset": offset_ismr,
                "byte_length": mask_count * 2,
                "scale": ismr_scale,
                "offset": 0.0,
                "fill_value": FILL_INT16,
                "unit": "m/yr",
                "stats_m_per_year": finite_stats(sampled_ismr),
            },
            {
                "name": "tstar_zice",
                "dtype": "int16",
                "byte_offset": offset_tstar,
                "byte_length": mask_count * 2,
                "scale": tstar_scale,
                "offset": 0.0,
                "fill_value": FILL_INT16,
                "unit": "celsius",
                "stats_c": finite_stats(sampled_tstar),
            },
        ],
    }

    meta_path = output_dir / f"{target.output_basename}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as ds:
        src_x = np.asarray(ds["easting"][:, 0], dtype=np.float64)
        src_y = np.asarray(ds["northing"][0, :], dtype=np.float64)

        write_preview_assets(ds, input_path, output_dir)

        for target in DEFAULT_TARGETS:
            write_target_package(
                ds=ds,
                input_path=input_path,
                target=target,
                output_dir=output_dir,
                src_x=src_x,
                src_y=src_y,
            )

    print(f"Wrote preview index to {output_dir / PREVIEW_INDEX_FILENAME}")
    for target in DEFAULT_TARGETS:
        print(f"Wrote {output_dir / f'{target.output_basename}.meta.json'}")


if __name__ == "__main__":
    main()
