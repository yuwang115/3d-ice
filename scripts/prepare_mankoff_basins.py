#!/usr/bin/env python3
"""Package the Mankoff et al. (2020) Greenland ice basins as drainage divides for the explorer.

Source: "Streams, Outlets, Basins, and Discharge [k=1.0]", GEUS Dataverse
doi:10.22008/FK2/XKQVL7 (version 8), file ``freshwater/ice/basins_filled.gpkg``, CC0 1.0.
Each basin is the set of 100 m cells that drain to one outlet at the ice margin, routed
under ice-overburden water pressure (k = 1.0). The "filled" file keeps nunataks inside
their basin, so the outlines are the divides themselves rather than rings around rock.

The 23,906 basins tile the ice sheet and the glaciers and ice caps around it, but most are
a few cells at the margin, far below the explorer's 1–3 km grid: drawn in full they fringe
the whole coast. So the package keeps only the divides between two neighbouring basins of
at least ``--min-area-km2`` each (the large basins, 97% of the ice area at 10 km²), and no
ice margin, which the explorer's ice surface already draws.

Neighbouring basins share their boundaries exactly on the 100 m lattice, so each divide is
stored once: the lattice edges are chained between junctions and each chain is simplified
with Douglas-Peucker, junctions fixed, so divides that meet still meet.
Points are delta-encoded in lattice units. The input is read with sqlite3 alone (a
GeoPackage geometry is a short header and standard WKB), so no GIS library is needed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "static/tools/data/greenland_mankoff_ice_basins_k1.json"
SOURCE = {
    "name": "Mankoff et al. (2020) Streams, Outlets, Basins, and Discharge [k=1.0]",
    "doi": "10.22008/FK2/XKQVL7",
    "version": "8.0",
    "file": "freshwater/ice/basins_filled.gpkg",
    "license": "CC0-1.0",
    "publication_doi": "10.5194/essd-12-2811-2020",
}
LATTICE_M = 100
DEFAULT_MIN_AREA_KM2 = 10.0
DEFAULT_TOLERANCE_M = 250.0
# GeoPackage envelope sizes (bytes) by the envelope-contents indicator in the header flags.
ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
WKB_POLYGON = 3
WKB_MULTIPOLYGON = 6
# Offsets that keep lattice indices of Greenland's EPSG:3413 extent non-negative.
NODE_SHIFT = 1 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=Path, help="Path to freshwater/ice/basins_filled.gpkg")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="Output JSON path.")
    parser.add_argument(
        "--min-area-km2", default=DEFAULT_MIN_AREA_KM2, type=float, help="Smallest basin whose divides are drawn."
    )
    parser.add_argument("--tolerance-m", default=DEFAULT_TOLERANCE_M, type=float, help="Douglas-Peucker tolerance.")
    return parser.parse_args()


# ------------------------------------------------------------------ reading


Ring = list[tuple[float, float]]
# One polygon: its outer ring, then its holes.
Polygon = list[Ring]


def gpkg_geometry_polygons(blob: bytes) -> list[Polygon]:
    """The polygons of a GeoPackage POLYGON or MULTIPOLYGON geometry blob, in order."""
    if blob[:2] != b"GP":
        raise ValueError("Not a GeoPackage geometry blob")
    envelope = (blob[3] >> 1) & 0b111
    if envelope not in ENVELOPE_BYTES:
        raise ValueError(f"Unknown GeoPackage envelope indicator {envelope}")
    polygons, _ = _wkb_polygons(blob, 8 + ENVELOPE_BYTES[envelope])
    return polygons


def _wkb_polygons(data: bytes, offset: int) -> tuple[list[Polygon], int]:
    order = "<" if data[offset] == 1 else ">"
    (kind,) = struct.unpack_from(order + "I", data, offset + 1)
    offset += 5
    if kind == WKB_MULTIPOLYGON:
        (count,) = struct.unpack_from(order + "I", data, offset)
        offset += 4
        polygons: list[Polygon] = []
        for _ in range(count):
            parts, offset = _wkb_polygons(data, offset)
            polygons.extend(parts)
        return polygons, offset
    if kind != WKB_POLYGON:
        raise ValueError(f"Unsupported WKB geometry type {kind}")
    (ring_count,) = struct.unpack_from(order + "I", data, offset)
    offset += 4
    rings: Polygon = []
    for _ in range(ring_count):
        (point_count,) = struct.unpack_from(order + "I", data, offset)
        offset += 4
        values = struct.unpack_from(order + f"{2 * point_count}d", data, offset)
        offset += 16 * point_count
        rings.append(list(zip(values[0::2], values[1::2], strict=True)))
    return [rings], offset


def read_basins(path: Path) -> Iterator[tuple[int, Polygon]]:
    """(basin id, rings) of every polygon in the GeoPackage's single feature table; each
    part of a multipolygon comes as a polygon of its own, under its basin's id."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        tables = connection.execute(
            "SELECT table_name, srs_id FROM gpkg_contents WHERE data_type = 'features'"
        ).fetchall()
        if len(tables) != 1:
            raise ValueError(f"Expected one feature table in {path}, found {len(tables)}")
        table, srs_id = tables[0]
        if srs_id != 3413:
            raise ValueError(f"Expected EPSG:3413 basins, found SRS {srs_id}")
        (geometry_column,) = connection.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?", (table,)
        ).fetchone()
        rows = connection.execute(f'SELECT "{geometry_column}", cat FROM "{table}"')
        for blob, basin_id in rows:
            if blob is None:
                continue
            for polygon in gpkg_geometry_polygons(blob):
                yield int(basin_id), polygon


# ------------------------------------------------------------------ lattice edges


def ring_area_m2(ring: Ring) -> float:
    """Signed shoelace area of a closed ring."""
    xs = np.asarray([p[0] for p in ring])
    ys = np.asarray([p[1] for p in ring])
    return 0.5 * float(np.dot(xs[:-1], ys[1:]) - np.dot(xs[1:], ys[:-1]))


def polygon_area_m2(rings: Polygon) -> float:
    """Area of a polygon whose first ring is the outer one and the rest its holes."""
    if not rings:
        return 0.0
    return abs(ring_area_m2(rings[0])) - sum(abs(ring_area_m2(ring)) for ring in rings[1:])


def lattice_index(value: float, lattice_m: int = LATTICE_M) -> int:
    index = round(value / lattice_m)
    if abs(index * lattice_m - value) > 1e-6:
        raise ValueError(f"Vertex {value} is not on the {lattice_m} m lattice")
    return index


def node_key(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    return (ix.astype(np.int64) + NODE_SHIFT) * (2 * NODE_SHIFT) + (iy.astype(np.int64) + NODE_SHIFT)


def node_xy(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return key // (2 * NODE_SHIFT) - NODE_SHIFT, key % (2 * NODE_SHIFT) - NODE_SHIFT


def ring_unit_edges(ring: Ring, lattice_m: int = LATTICE_M) -> np.ndarray:
    """Keys of the unit lattice edges along a ring: start node * 2 + (0 east, 1 north)."""
    ix = np.asarray([lattice_index(x, lattice_m) for x, _ in ring], dtype=np.int64)
    iy = np.asarray([lattice_index(y, lattice_m) for _, y in ring], dtype=np.int64)
    dx = np.diff(ix)
    dy = np.diff(iy)
    if np.any((dx != 0) & (dy != 0)):
        raise ValueError("Ring has a segment that is not along the lattice")
    lengths = np.abs(dx) + np.abs(dy)
    keep = lengths > 0
    starts_x = np.minimum(ix[:-1], ix[1:])[keep]
    starts_y = np.minimum(iy[:-1], iy[1:])[keep]
    vertical = (dx == 0)[keep]
    lengths = lengths[keep]
    step = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    vertical_rep = np.repeat(vertical, lengths)
    edge_x = np.repeat(starts_x, lengths) + np.where(vertical_rep, 0, step)
    edge_y = np.repeat(starts_y, lengths) + np.where(vertical_rep, step, 0)
    return node_key(edge_x, edge_y) * 2 + vertical_rep.astype(np.int64)


def collect_edges(
    basins: Iterable[tuple[int, Polygon]], lattice_m: int = LATTICE_M
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Every polygon's unit edges with the basin they bound, and each basin's area (m²)."""
    keys: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    areas: dict[int, float] = {}
    for basin_id, rings in basins:
        areas[basin_id] = areas.get(basin_id, 0.0) + polygon_area_m2(rings)
        for ring in rings:
            edges = ring_unit_edges(ring, lattice_m)
            keys.append(edges)
            owners.append(np.full(edges.size, basin_id, dtype=np.int64))
    if not keys:
        raise ValueError("No basin polygons were read")
    return np.concatenate(keys), np.concatenate(owners), areas


def classify_edges(keys: np.ndarray, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique edges with the basins on either side; -1 marks the outside of the domain.

    An edge inside one basin (two of its parts touching) has the same id on both sides.
    """
    order = np.lexsort((owners, keys))
    keys = keys[order]
    owners = owners[order]
    unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
    if counts.max() > 2:
        raise ValueError("A lattice edge borders more than two basin polygons")
    left = owners[first]
    right = np.where(counts == 2, owners[np.minimum(first + 1, owners.size - 1)], -1)
    return unique, left, right


def select_divides(unique: np.ndarray, left: np.ndarray, right: np.ndarray, large: set[int]) -> np.ndarray:
    """The edges between two different basins that are both large."""
    large_ids = np.fromiter(large, dtype=np.int64, count=len(large))
    between = (right != -1) & (left != right)
    return unique[between & np.isin(left, large_ids) & np.isin(right, large_ids)]


# ------------------------------------------------------------------ chains


def edge_nodes(edge_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = edge_keys // 2
    vertical = (edge_keys % 2).astype(bool)
    sx, sy = node_xy(start)
    end = node_key(sx + np.where(vertical, 0, 1), sy + np.where(vertical, 1, 0))
    return start, end


def chain_edges(edge_keys: np.ndarray) -> list[np.ndarray]:
    """Chain unit edges into polylines of node keys, broken at every node not of degree two.

    A closed loop with no junction comes back as one polyline whose ends are the same node.
    """
    if edge_keys.size == 0:
        return []
    start, end = edge_nodes(edge_keys)
    nodes, inverse = np.unique(np.concatenate([start, end]), return_inverse=True)
    u = inverse[: edge_keys.size]
    v = inverse[edge_keys.size :]
    degree = np.bincount(np.concatenate([u, v]), minlength=nodes.size)
    # Adjacency in CSR form: for each node, the edges that touch it.
    touching = np.concatenate([np.arange(u.size), np.arange(v.size)])
    order = np.argsort(np.concatenate([u, v]), kind="stable")
    touching = touching[order]
    pointer = np.concatenate([[0], np.cumsum(degree)])
    junctions = np.flatnonzero(degree != 2)
    # The walk is a scalar loop, which runs several times faster over Python lists.
    u, v, touching, pointer, degree = (array.tolist() for array in (u, v, touching, pointer, degree))
    used = [False] * len(u)
    chains: list[np.ndarray] = []

    def walk(first_node: int, first_edge: int) -> list[int]:
        path = [first_node]
        node, edge = first_node, first_edge
        while True:
            used[edge] = True
            node = v[edge] if u[edge] == node else u[edge]
            path.append(node)
            if degree[node] != 2:
                return path
            a, b = touching[pointer[node]], touching[pointer[node] + 1]
            edge = b if a == edge else a
            if used[edge]:
                return path

    for node in junctions.tolist():
        for slot in range(pointer[node], pointer[node + 1]):
            edge = touching[slot]
            if not used[edge]:
                chains.append(nodes[walk(node, edge)])
    for edge in range(len(u)):
        if not used[edge]:
            chains.append(nodes[walk(u[edge], edge)])
    return chains


def douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of the points Douglas-Peucker keeps, ends always included."""
    count = len(points)
    if count <= 2:
        return np.arange(count)
    keep = np.zeros(count, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        a = points[first]
        b = points[last]
        inner = points[first + 1 : last]
        ab = b - a
        length = float(np.hypot(*ab))
        if length == 0.0:
            distances = np.hypot(*(inner - a).T)
        else:
            distances = np.abs(ab[0] * (inner[:, 1] - a[1]) - ab[1] * (inner[:, 0] - a[0])) / length
        worst = int(np.argmax(distances))
        if distances[worst] > tolerance:
            middle = first + 1 + worst
            keep[middle] = True
            stack.append((first, middle))
            stack.append((middle, last))
    return np.flatnonzero(keep)


def simplify_chain(chain_nodes: np.ndarray, tolerance_cells: float) -> np.ndarray:
    """A chain's simplified points in lattice units, (n, 2) int64."""
    ix, iy = node_xy(chain_nodes)
    points = np.column_stack([ix, iy]).astype(np.int64)
    if len(points) > 2 and np.array_equal(points[0], points[-1]):
        # A closed loop: split it at its farthest point so both halves have distinct ends.
        far = int(np.argmax(np.hypot(*(points - points[0]).T)))
        head = points[: far + 1][douglas_peucker(points[: far + 1].astype(float), tolerance_cells)]
        tail = points[far:][douglas_peucker(points[far:].astype(float), tolerance_cells)]
        return np.vstack([head, tail[1:]])
    return points[douglas_peucker(points.astype(float), tolerance_cells)]


def delta_encode(points: np.ndarray) -> list[int]:
    """[x0, y0, dx1, dy1, ...] of a polyline in lattice units."""
    deltas = np.diff(points, axis=0)
    return [int(points[0, 0]), int(points[0, 1]), *(int(value) for value in deltas.ravel())]


def encode_arcs(edge_keys: np.ndarray, tolerance_cells: float) -> list[list[int]]:
    arcs = []
    for chain in chain_edges(edge_keys):
        points = simplify_chain(chain, tolerance_cells)
        if len(points) >= 2:
            arcs.append(delta_encode(points))
    # A stable order keeps rebuilds of an unchanged source byte-identical.
    arcs.sort()
    return arcs


# ------------------------------------------------------------------ package


def build_payload(
    basins: Iterable[tuple[int, Polygon]],
    *,
    min_area_km2: float = DEFAULT_MIN_AREA_KM2,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    lattice_m: int = LATTICE_M,
) -> dict:
    keys, owners, areas = collect_edges(basins, lattice_m)
    unique, left, right = classify_edges(keys, owners)
    large = {basin_id for basin_id, area in areas.items() if area >= min_area_km2 * 1e6}
    divides = encode_arcs(select_divides(unique, left, right, large), tolerance_m / lattice_m)
    total_area = sum(areas.values())
    large_area = sum(areas[basin_id] for basin_id in large)
    return {
        "schema": "3d-ice/basin-divides",
        "schema_version": 1,
        "dataset": "Mankoff et al. (2020) Greenland ice basins, k = 1.0",
        "dataset_key": "mankoff_2020_ice_basins_k1",
        "projection": "EPSG:3413",
        "source": SOURCE,
        "note": (
            "Divides between the basins that drain to one outlet each at the margin of the ice sheet "
            "or of a glacier around it, routed under ice-overburden water pressure on a 100 m grid. "
            "Only divides between two basins of at least min_area_km2 are kept; the ice margin is not."
        ),
        "lattice_m": lattice_m,
        "simplify_tolerance_m": tolerance_m,
        "min_area_km2": min_area_km2,
        "basin_count": len(areas),
        "large_basin_count": len(large),
        "total_area_km2": round(total_area / 1e6, 1),
        "large_area_fraction": round(large_area / total_area, 4) if total_area else 0.0,
        "divide_count": len(divides),
        "point_count": sum(len(arc) // 2 for arc in divides),
        "divides": divides,
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input GeoPackage: {args.input}")
    payload = build_payload(read_basins(args.input), min_area_km2=args.min_area_km2, tolerance_m=args.tolerance_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")
    size_kb = args.output.stat().st_size / 1024
    print(
        f"Wrote {args.output} ({size_kb:.0f} KB): divides of {payload['large_basin_count']} of "
        f"{payload['basin_count']} basins ({payload['large_area_fraction']:.1%} of the area), "
        f"{payload['divide_count']} arcs, {payload['point_count']} points"
    )


if __name__ == "__main__":
    main()
