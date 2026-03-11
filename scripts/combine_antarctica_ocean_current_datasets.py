#!/usr/bin/env python3
"""Combine precomputed Antarctica ocean streamline datasets into one package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from prepare_antarctica_ocean_currents import percentile_range, stats_dict


DEFAULT_COMPONENTS = (
    "static/tools/data/antarctica_ocean_currents_waom2_yr5_annual_cavity_margin80km.meta.json",
    "static/tools/data/antarctica_ocean_currents_waom2_yr5_annual_remote_open_ocean.meta.json",
)
DEFAULT_OUTPUT_BASENAME = "antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(DEFAULT_COMPONENTS),
        help="Input component .meta.json files to merge.",
    )
    parser.add_argument(
        "--output-dir",
        default="static/tools/data",
        help="Directory for the merged .bin/.meta.json outputs.",
    )
    parser.add_argument(
        "--output-basename",
        default=DEFAULT_OUTPUT_BASENAME,
        help="Basename for the merged .bin/.meta.json files.",
    )
    return parser.parse_args()


def load_binary_fields(meta_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    meta = json.loads(meta_path.read_text())
    if meta_path.name.endswith(".meta.json"):
        bin_path = meta_path.with_name(meta_path.name[: -len(".meta.json")] + ".bin")
    else:
        bin_path = meta_path.with_suffix(".bin")
    raw_fields = [field for field in meta.get("fields", []) if "byte_offset" in field and "byte_length" in field]
    arrays: dict[str, np.ndarray] = {}
    with bin_path.open("rb") as fh:
        for field in raw_fields:
            dtype_name = str(field["dtype"])
            dtype = np.uint8 if dtype_name == "uint8" else np.float32
            fh.seek(int(field["byte_offset"]))
            raw = fh.read(int(field["byte_length"]))
            arrays[str(field["name"])] = np.frombuffer(raw, dtype=dtype).copy()
    return meta, arrays, raw_fields


def combine_summary_stats(stats_list: list[dict[str, float]], weights: list[int]) -> dict[str, float]:
    finite_triplets = []
    for stats, weight in zip(stats_list, weights):
        if not stats:
            continue
        min_value = float(stats.get("min", float("nan")))
        max_value = float(stats.get("max", float("nan")))
        mean_value = float(stats.get("mean", float("nan")))
        if not all(np.isfinite([min_value, max_value, mean_value])):
            continue
        finite_triplets.append((min_value, max_value, mean_value, int(weight)))

    if not finite_triplets:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}

    total_weight = sum(weight for _, _, _, weight in finite_triplets)
    weighted_mean = (
        sum(mean * weight for _, _, mean, weight in finite_triplets) / float(total_weight)
        if total_weight > 0
        else 0.0
    )
    return {
        "min": float(min(min_value for min_value, _, _, _ in finite_triplets)),
        "max": float(max(max_value for _, max_value, _, _ in finite_triplets)),
        "mean": float(weighted_mean),
    }


def merge_count_dicts(dicts: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for data in dicts:
        for key, value in data.items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return merged


def merge_sector_counts(sectors: list[list[int]], sector_count: int = 8) -> list[int]:
    merged = [0 for _ in range(sector_count)]
    for counts in sectors:
        for index, value in enumerate(list(counts)[:sector_count]):
            merged[index] += int(value)
    return merged


def main() -> None:
    args = parse_args()
    component_meta_paths = [Path(path) for path in args.components]
    if len(component_meta_paths) < 2:
        raise RuntimeError("At least two component metadata files are required.")

    loaded = [load_binary_fields(path) for path in component_meta_paths]
    component_metas = [item[0] for item in loaded]
    component_arrays = [item[1] for item in loaded]
    component_raw_fields = [item[2] for item in loaded]

    raw_field_names = [str(field["name"]) for field in component_raw_fields[0]]
    for raw_fields in component_raw_fields[1:]:
        names = [str(field["name"]) for field in raw_fields]
        if names != raw_field_names:
            raise RuntimeError("Component datasets do not share the same binary field layout.")

    merged_arrays: dict[str, np.ndarray] = {}
    for field_name in raw_field_names:
        merged_arrays[field_name] = np.concatenate([arrays[field_name] for arrays in component_arrays])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_bin = output_dir / f"{args.output_basename}.bin"
    out_meta = output_dir / f"{args.output_basename}.meta.json"

    fields: list[dict[str, Any]] = []
    offset = 0
    with out_bin.open("wb") as fh:
        for field in component_raw_fields[0]:
            name = str(field["name"])
            array = merged_arrays[name]
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

    component_streamline_counts = [int(meta.get("streamline_count", 0)) for meta in component_metas]
    streamline_count = int(sum(component_streamline_counts))
    segment_count = int(merged_arrays["x0_ps_m"].size)

    theta_all = np.concatenate([merged_arrays["theta0_c"], merged_arrays["theta1_c"]]).astype(np.float32)
    sal_all = np.concatenate([merged_arrays["sal0_psu"], merged_arrays["sal1_psu"]]).astype(np.float32)
    x_bounds = np.concatenate([merged_arrays["x0_ps_m"], merged_arrays["x1_ps_m"]]).astype(np.float32)
    y_bounds = np.concatenate([merged_arrays["y0_ps_m"], merged_arrays["y1_ps_m"]]).astype(np.float32)
    depth_bounds = np.concatenate([merged_arrays["depth0_m"], merged_arrays["depth1_m"]]).astype(np.float32)

    component_seed_bucket_labels = [meta.get("sampling", {}).get("seed_bucket_labels", {}) for meta in component_metas]
    merged_seed_bucket_labels: dict[str, str] = {}
    for labels in component_seed_bucket_labels:
        for key, value in labels.items():
            merged_seed_bucket_labels[str(key)] = str(value)

    component_seed_buckets = [meta.get("sampling", {}).get("seed_buckets", []) for meta in component_metas]
    merged_seed_buckets: list[dict[str, Any]] = []
    for buckets in component_seed_buckets:
        merged_seed_buckets.extend(buckets)

    component_streamline_bucket_counts = [
        meta.get("coverage", {}).get("streamlines_by_seed_bucket", {}) for meta in component_metas
    ]
    merged_streamlines_by_bucket = merge_count_dicts(component_streamline_bucket_counts)
    component_segments_by_bucket = [meta.get("coverage", {}).get("segments_by_seed_bucket", {}) for meta in component_metas]
    merged_segments_by_bucket = merge_count_dicts(component_segments_by_bucket)

    component_valid_sector_counts = [meta.get("coverage", {}).get("valid_seed_sector_counts_8", []) for meta in component_metas]
    component_retained_sector_counts = [meta.get("coverage", {}).get("retained_seed_sector_counts_8", []) for meta in component_metas]

    component_seed_depth_stats = [meta.get("coverage", {}).get("seed_depth_stats_by_bucket", {}) for meta in component_metas]
    merged_seed_depth_stats: dict[str, dict[str, float]] = {}
    for stats_by_bucket, bucket_counts in zip(component_seed_depth_stats, component_streamline_bucket_counts):
        for key, stats in stats_by_bucket.items():
            bucket_key = str(key)
            if bucket_key not in merged_seed_depth_stats:
                merged_seed_depth_stats[bucket_key] = {
                    "min_m": float(stats["min_m"]),
                    "max_m": float(stats["max_m"]),
                    "mean_m": float(stats["mean_m"]),
                }
                continue
            previous = merged_seed_depth_stats[bucket_key]
            previous_count = int(merged_streamlines_by_bucket.get(bucket_key, 0) - bucket_counts.get(bucket_key, 0))
            current_count = int(bucket_counts.get(bucket_key, 0))
            total_count = max(1, previous_count + current_count)
            merged_seed_depth_stats[bucket_key] = {
                "min_m": float(min(previous["min_m"], float(stats["min_m"]))),
                "max_m": float(max(previous["max_m"], float(stats["max_m"]))),
                "mean_m": float(
                    (previous["mean_m"] * previous_count + float(stats["mean_m"]) * current_count) / float(total_count)
                ),
            }

    first_meta = component_metas[0]
    first_sampling = first_meta.get("sampling", {})
    first_coverage = first_meta.get("coverage", {})

    merged_meta = {
        "title": first_meta.get("title", "Whole Antarctic Ocean Application"),
        "product_version": first_meta.get("product_version", "WAOM2"),
        "source_file": "merged:" + ",".join(meta.get("source_file", "") for meta in component_metas),
        "source_reference": first_meta.get("source_reference", ""),
        "source_time_utc": first_meta.get("source_time_utc", ""),
        "source_time_label": first_meta.get("source_time_label", ""),
        "projection": first_meta.get("projection", "EPSG:3031"),
        "geometry_type": first_meta.get("geometry_type", "streamlines_3d"),
        "sampling": {
            "sample_stride": int(first_sampling.get("sample_stride", 1)),
            "streamline_class": "combined_cavity_margin80km_remote_open_ocean",
            "streamline_class_label": "Ice-shelf cavities + 80 km coastal ocean merged with remote open ocean",
            "seed_strategy": "merged_streamline_sets",
            "depth_sampling_summary": "Merged precomputed streamline sets: cavity + 80 km coastal ocean plus remote open ocean beyond that band.",
            "min_speed_mps": float(min(meta.get("sampling", {}).get("min_speed_mps", 0.0) for meta in component_metas)),
            "min_seed_speed_mps": float(min(meta.get("sampling", {}).get("min_seed_speed_mps", 0.0) for meta in component_metas)),
            "min_trace_speed_mps": float(min(meta.get("sampling", {}).get("min_trace_speed_mps", 0.0) for meta in component_metas)),
            "flowline_step_cells": float(first_sampling.get("flowline_step_cells", 1.2)),
            "flowline_max_steps": int(first_sampling.get("flowline_max_steps", 150)),
            "target_streamline_count": int(sum(meta.get("sampling", {}).get("target_streamline_count", 0) for meta in component_metas)),
            "min_streamline_segments": int(min(meta.get("sampling", {}).get("min_streamline_segments", 0) for meta in component_metas)),
            "random_seed": int(first_sampling.get("random_seed", 0)),
            "source_grid_dx_m": float(first_sampling.get("source_grid_dx_m", 2000.0)),
            "source_grid_dy_m": float(first_sampling.get("source_grid_dy_m", 2000.0)),
            "source_grid_x0_m": float(first_sampling.get("source_grid_x0_m", -3000000.0)),
            "source_grid_y0_m": float(first_sampling.get("source_grid_y0_m", -3000000.0)),
            "clearance_m": float(first_sampling.get("clearance_m", 20.0)),
            "velocity_vertical_mapping": first_sampling.get("velocity_vertical_mapping", "native_u_v_to_rho_depths"),
            "selection_strategy": "merged_precomputed_streamline_sets",
            "selection_bin_size_cells_global": int(first_sampling.get("selection_bin_size_cells_global", 0)),
            "vertical_seed_separation_m": float(first_sampling.get("vertical_seed_separation_m", 120.0)),
            "front_radius_cells": int(first_sampling.get("front_radius_cells", 2)),
            "region_definition": {
                "type": "combined_domains",
                "components": [
                    {
                        "streamline_class": meta.get("sampling", {}).get("streamline_class", ""),
                        "streamline_class_label": meta.get("sampling", {}).get("streamline_class_label", ""),
                        "region_definition": meta.get("sampling", {}).get("region_definition", {}),
                        "streamline_count": int(meta.get("streamline_count", 0)),
                    }
                    for meta in component_metas
                ],
            },
            "anti_zigzag": first_sampling.get("anti_zigzag", {}),
            "seed_bucket_labels": merged_seed_bucket_labels,
            "seed_buckets": merged_seed_buckets,
            "component_datasets": [
                {
                    "basename": component_meta_paths[index].stem.replace(".meta", ""),
                    "meta_file": component_meta_paths[index].name,
                    "streamline_count": component_streamline_counts[index],
                    "segment_count": int(component_metas[index].get("segment_count", 0)),
                }
                for index in range(len(component_meta_paths))
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
            "seed_depth_min_m": float(min(meta.get("coverage", {}).get("seed_depth_min_m", 0.0) for meta in component_metas)),
            "seed_depth_max_m": float(max(meta.get("coverage", {}).get("seed_depth_max_m", 0.0) for meta in component_metas)),
            "valid_seed_sector_counts_8": merge_sector_counts(component_valid_sector_counts),
            "retained_seed_sector_counts_8": merge_sector_counts(component_retained_sector_counts),
            "segments_by_seed_bucket": merged_segments_by_bucket,
            "streamlines_by_seed_bucket": merged_streamlines_by_bucket,
            "seed_depth_stats_by_bucket": merged_seed_depth_stats,
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
                "stats": combine_summary_stats(
                    [
                        next((field.get("stats", {}) for field in meta.get("fields", []) if field.get("name") == "speed_mps"), {})
                        for meta in component_metas
                    ],
                    component_streamline_counts,
                ),
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

    out_meta.write_text(json.dumps(merged_meta, indent=2), encoding="utf-8")
    print(out_bin)
    print(out_meta)
    print(f"Merged {streamline_count} streamlines / {segment_count} segments")


if __name__ == "__main__":
    main()
