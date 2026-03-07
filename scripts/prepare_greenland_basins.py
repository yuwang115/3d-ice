#!/usr/bin/env python3
"""Prepare Greenland basin boundary JSON for the 3D explorer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import shapefile

SUBREGION_NAME_MAP = {
    "CE": "Central East",
    "CW": "Central West",
    "NE": "Northeast",
    "NO": "North",
    "NW": "Northwest",
    "SE": "Southeast",
    "SW": "Southwest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Greenland basin shapefile boundaries to the explorer JSON format."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Greenland_Basins_PS_v1.4.2.shp",
    )
    parser.add_argument(
        "--output",
        default="static/tools/data/greenland_basins_ps_v1_4_2.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def normalize_name(subregion: str, raw_name: object) -> str:
    name = str(raw_name or "").strip()
    if name:
        return name
    return SUBREGION_NAME_MAP.get(subregion, subregion)


def ring_points(points: Iterable[tuple[float, float]]) -> list[list[int]]:
    return [[int(round(x)), int(round(y))] for x, y in points]


def closed_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    if points[0] == points[-1]:
        return points
    return points + [points[0]]


def polygon_area_and_centroid(points: list[tuple[float, float]]) -> tuple[float, list[int] | None]:
    ring = closed_ring(points)
    if len(ring) < 4:
        return 0.0, None

    cross_sum = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cross = x0 * y1 - x1 * y0
        cross_sum += cross
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross

    area = cross_sum * 0.5
    if abs(area) < 1e-9:
        return 0.0, None

    scale = 1.0 / (3.0 * cross_sum)
    return area, [int(round(centroid_x * scale)), int(round(centroid_y * scale))]


def average_point(points: list[tuple[float, float]]) -> list[int] | None:
    if not points:
        return None
    sx = 0.0
    sy = 0.0
    for x, y in points:
        sx += x
        sy += y
    n = float(len(points))
    return [int(round(sx / n)), int(round(sy / n))]


def extract_segments(shape: shapefile.Shape) -> tuple[list[list[list[int]]], list[int] | None, float]:
    parts = list(shape.parts) + [len(shape.points)]
    segments: list[list[list[int]]] = []
    label_xy_m: list[int] | None = None
    best_area = -math.inf
    total_area_m2 = 0.0

    for start, stop in zip(parts, parts[1:]):
        part_points = shape.points[start:stop]
        if len(part_points) < 2:
            continue
        segments.append(ring_points(part_points))
        area_m2, centroid = polygon_area_and_centroid(part_points)
        total_area_m2 += abs(area_m2)
        if abs(area_m2) > best_area and centroid is not None:
            best_area = abs(area_m2)
            label_xy_m = centroid

    if label_xy_m is None and shape.points:
        label_xy_m = average_point(shape.points)

    return segments, label_xy_m, total_area_m2 / 1_000_000.0


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input shapefile: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = shapefile.Reader(str(input_path))
    fields = [field[0] for field in reader.fields[1:]]
    basins = []

    for shape_record in reader.iterShapeRecords():
        record = dict(zip(fields, shape_record.record))
        subregion = str(record.get("SUBREGION1") or "").strip()
        if not subregion:
            continue
        segments_xy_m, label_xy_m, area_km2 = extract_segments(shape_record.shape)
        if not segments_xy_m:
            continue
        basins.append(
            {
                "id": subregion,
                "name": normalize_name(subregion, record.get("NAME")),
                "subregion": subregion,
                "label_xy_m": label_xy_m,
                "segments_xy_m": segments_xy_m,
                "area_km2": round(area_km2, 2),
            }
        )

    basins.sort(key=lambda basin: basin["id"])

    payload = {
        "dataset": "Greenland Basins PS",
        "dataset_key": "greenland_basins_ps_v1_4_2",
        "projection": "EPSG:3413",
        "source_shapefile": input_path.name,
        "note": "Catchment boundary sectors for the Greenland Ice Sheet in NSIDC Sea Ice Polar Stereographic North.",
        "basin_count": len(basins),
        "basins": basins,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.write("\n")

    print(f"Wrote {output_path} with {len(basins)} basins")


if __name__ == "__main__":
    main()
