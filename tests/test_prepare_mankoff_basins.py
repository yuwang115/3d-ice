"""Tests for scripts/prepare_mankoff_basins.py and the divide package it writes."""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_mankoff_basins as mankoff  # noqa: E402

PACKAGE = ROOT / "static" / "tools" / "data" / "greenland_mankoff_ice_basins_k1.json"
L = mankoff.LATTICE_M


def rect(x0: int, y0: int, x1: int, y1: int) -> list[tuple[float, float]]:
    """A closed counter-clockwise ring around lattice cells [x0, x1) x [y0, y1), in metres."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return [(float(x * L), float(y * L)) for x, y in corners]


def wkb_polygon(rings: list[list[tuple[float, float]]]) -> bytes:
    body = struct.pack("<BII", 1, mankoff.WKB_POLYGON, len(rings))
    for ring in rings:
        body += struct.pack("<I", len(ring)) + b"".join(struct.pack("<dd", x, y) for x, y in ring)
    return body


def gpkg_blob(wkb: bytes, envelope: int = 1) -> bytes:
    flags = (envelope << 1) | 1  # little-endian header
    return b"GP\x00" + bytes([flags]) + struct.pack("<i", 3413) + b"\x00" * mankoff.ENVELOPE_BYTES[envelope] + wkb


def decode(arc: list[int]) -> np.ndarray:
    return np.cumsum(np.asarray(arc, dtype=np.int64).reshape(-1, 2), axis=0)


def write_gpkg(path: Path, rows: list[tuple[int, bytes]], srs_id: int = 3413) -> Path:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT, srs_id INTEGER);
            CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT);
            CREATE TABLE basins_filled (fid INTEGER PRIMARY KEY, geom BLOB, cat INTEGER);
            """
        )
        connection.execute("INSERT INTO gpkg_contents VALUES ('basins_filled', 'features', ?)", (srs_id,))
        connection.execute("INSERT INTO gpkg_geometry_columns VALUES ('basins_filled', 'geom')")
        connection.executemany(
            "INSERT INTO basins_filled (geom, cat) VALUES (?, ?)", [(blob, cat) for cat, blob in rows]
        )
    return path


# ------------------------------------------------------------------ reading


def wkb_multipolygon(polygons: list[list[list[tuple[float, float]]]]) -> bytes:
    return struct.pack("<BII", 1, mankoff.WKB_MULTIPOLYGON, len(polygons)) + b"".join(map(wkb_polygon, polygons))


@pytest.mark.parametrize("envelope", [0, 1, 2, 4])
def test_geometry_blobs_are_read_past_any_envelope(envelope):
    ring = rect(0, 0, 2, 1)
    assert mankoff.gpkg_geometry_polygons(gpkg_blob(wkb_polygon([ring]), envelope)) == [[ring]]


def test_multipolygons_keep_each_part_with_its_own_holes():
    outer, hole, other = rect(0, 0, 4, 4), rect(1, 1, 2, 2), rect(10, 0, 11, 1)
    blob = gpkg_blob(wkb_multipolygon([[outer, hole], [other]]))
    assert mankoff.gpkg_geometry_polygons(blob) == [[outer, hole], [other]]


def test_a_blob_that_is_not_a_geopackage_geometry_is_refused():
    with pytest.raises(ValueError, match="GeoPackage"):
        mankoff.gpkg_geometry_polygons(wkb_polygon([rect(0, 0, 1, 1)]))


def test_basins_are_read_from_the_feature_table(tmp_path):
    path = write_gpkg(tmp_path / "basins.gpkg", [(7, gpkg_blob(wkb_polygon([rect(0, 0, 1, 1)])))])
    assert list(mankoff.read_basins(path)) == [(7, [rect(0, 0, 1, 1)])]


def test_a_two_part_basin_counts_the_area_of_both_parts(tmp_path):
    # Basin 1 in two 3 x 3 parts, each below a 12-cell threshold but together above it, on
    # either side of basin 2; the divides between them are drawn only if 1 is large.
    parts = [[rect(0, 0, 3, 3)], [rect(6, 0, 9, 3)]]
    path = write_gpkg(
        tmp_path / "basins.gpkg",
        [(1, gpkg_blob(wkb_multipolygon(parts))), (2, gpkg_blob(wkb_polygon([rect(3, 0, 6, 3)])))],
    )
    assert [basin_id for basin_id, _ in mankoff.read_basins(path)] == [1, 1, 2]
    payload = build(mankoff.read_basins(path), min_area_km2=0.12)
    assert payload["basin_count"] == 2 and payload["large_basin_count"] == 1
    payload = build(mankoff.read_basins(path), min_area_km2=0.09)
    assert payload["large_basin_count"] == 2
    assert sorted(decode(arc)[0, 0] for arc in payload["divides"]) == [3, 6]


def test_basins_in_another_projection_are_refused(tmp_path):
    path = write_gpkg(tmp_path / "basins.gpkg", [(7, gpkg_blob(wkb_polygon([rect(0, 0, 1, 1)])))], srs_id=4326)
    with pytest.raises(ValueError, match="EPSG:3413"):
        list(mankoff.read_basins(path))


# ------------------------------------------------------------------ lattice edges


def test_a_ring_becomes_one_key_per_unit_edge():
    edges = mankoff.ring_unit_edges(rect(0, 0, 2, 1))
    assert edges.size == 6 and np.unique(edges).size == 6


def test_a_vertex_off_the_lattice_is_refused():
    with pytest.raises(ValueError, match="lattice"):
        mankoff.ring_unit_edges([(0.0, 0.0), (150.0, 0.0), (150.0, 100.0), (0.0, 100.0), (0.0, 0.0)])


def test_a_diagonal_segment_is_refused():
    with pytest.raises(ValueError, match="along the lattice"):
        mankoff.ring_unit_edges([(0.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)])


def test_holes_are_taken_out_of_a_basin_area():
    assert mankoff.polygon_area_m2([rect(0, 0, 4, 4), rect(1, 1, 2, 2)]) == pytest.approx(15 * L * L)


def test_shared_edges_know_both_basins_and_the_margin_knows_one():
    keys, owners, _ = mankoff.collect_edges([(1, [rect(0, 0, 1, 1)]), (2, [rect(1, 0, 2, 1)])])
    unique, left, right = mankoff.classify_edges(keys, owners)
    shared = right != -1
    assert shared.sum() == 1 and {left[shared][0], right[shared][0]} == {1, 2}
    assert (right == -1).sum() == 6


# ------------------------------------------------------------------ divides


def build(basins, **kwargs):
    kwargs.setdefault("min_area_km2", 0.05)  # five lattice cells
    kwargs.setdefault("tolerance_m", 0.0)
    return mankoff.build_payload(basins, **kwargs)


def test_the_divide_between_two_basins_is_stored_once_and_the_margin_not_at_all():
    payload = build([(1, [rect(0, 0, 3, 3)]), (2, [rect(3, 0, 6, 3)])])
    assert payload["divide_count"] == 1
    assert decode(payload["divides"][0]).tolist() == [[3, 0], [3, 3]]
    assert payload["point_count"] == 2


def test_only_divides_between_two_large_basins_are_kept():
    # 1 | 2 side by side, and a one-cell basin 3 on 2's far side.
    payload = build([(1, [rect(0, 0, 3, 3)]), (2, [rect(3, 0, 6, 3)]), (3, [rect(6, 0, 7, 1)])])
    assert payload["large_basin_count"] == 2
    assert [decode(arc).tolist() for arc in payload["divides"]] == [[[3, 0], [3, 3]]]
    assert payload["large_area_fraction"] == pytest.approx(18 / 19, abs=1e-4)


def test_divides_break_at_junctions():
    # Three basins meeting at (3, 3): a T of three divides.
    payload = build([(1, [rect(0, 0, 3, 3)]), (2, [rect(3, 0, 6, 3)]), (3, [rect(0, 3, 6, 6)])])
    ends = sorted(tuple(map(tuple, decode(arc)[[0, -1]].tolist())) for arc in payload["divides"])
    assert ends == [((0, 3), (3, 3)), ((3, 0), (3, 3)), ((3, 3), (6, 3))]


def test_a_basin_enclosed_by_another_is_one_closed_loop():
    payload = build([(1, [rect(0, 0, 9, 9), rect(3, 3, 6, 6)]), (2, [rect(3, 3, 6, 6)])])
    assert payload["divide_count"] == 1
    loop = decode(payload["divides"][0])
    assert loop[0].tolist() == loop[-1].tolist()
    assert {tuple(p) for p in loop.tolist()} == {(3, 3), (6, 3), (6, 6), (3, 6)}


def test_parts_of_one_basin_that_touch_draw_no_divide_between_them():
    payload = build([(1, [rect(0, 0, 3, 3)]), (1, [rect(3, 0, 6, 3)]), (2, [rect(0, 3, 6, 6)])])
    assert [decode(arc)[[0, -1]].tolist() for arc in payload["divides"]] == [[[0, 3], [6, 3]]]


def test_simplification_drops_steps_within_the_tolerance_and_keeps_the_ends():
    # A staircase divide one cell high, between two basins.
    lower = [(0, 0), (8, 0), (8, 1), (4, 1), (4, 2), (0, 2), (0, 0)]
    upper = [(0, 2), (4, 2), (4, 1), (8, 1), (8, 4), (0, 4), (0, 2)]
    basins = [(1, [[(x * L, y * L) for x, y in lower]]), (2, [[(x * L, y * L) for x, y in upper]])]
    exact = decode(build(basins)["divides"][0]).tolist()
    assert exact == [[0, 2], [4, 2], [4, 1], [8, 1]]
    smooth = decode(build(basins, tolerance_m=1.5 * L)["divides"][0]).tolist()
    assert smooth == [[0, 2], [8, 1]]


def test_douglas_peucker_keeps_a_point_beyond_the_tolerance():
    points = np.array([[0, 0], [5, 3], [10, 0]], dtype=float)
    assert mankoff.douglas_peucker(points, 2.0).tolist() == [0, 1, 2]
    assert mankoff.douglas_peucker(points, 4.0).tolist() == [0, 2]


def test_a_rebuild_is_byte_identical():
    basins = [(1, [rect(0, 0, 3, 3)]), (2, [rect(3, 0, 6, 3)]), (3, [rect(0, 3, 6, 6)])]
    assert json.dumps(build(basins)) == json.dumps(build(list(reversed(basins))))


# ------------------------------------------------------------------ the package


@pytest.fixture(scope="module")
def package():
    return json.loads(PACKAGE.read_text(encoding="utf-8"))


def test_the_package_credits_its_source(package):
    assert package["schema"] == "3d-ice/basin-divides" and package["schema_version"] == 1
    assert package["projection"] == "EPSG:3413"
    assert package["source"]["doi"] == "10.22008/FK2/XKQVL7"
    assert package["source"]["publication_doi"] == "10.5194/essd-12-2811-2020"
    assert package["source"]["license"] == "CC0-1.0"


def test_the_package_counts_match_its_arcs(package):
    assert package["divide_count"] == len(package["divides"])
    assert package["point_count"] == sum(len(arc) // 2 for arc in package["divides"])
    assert all(len(arc) >= 4 and len(arc) % 2 == 0 for arc in package["divides"])
    # The Dataverse v8 ice basins (filled) and the 10 km² rule.
    assert package["basin_count"] == 23_906
    assert package["min_area_km2"] == 10.0
    assert package["large_basin_count"] == 3_126
    assert 0.97 < package["large_area_fraction"] < 0.98


def test_the_package_lies_on_greenland(package):
    lattice = package["lattice_m"]
    starts = np.array([arc[:2] for arc in package["divides"]]) * lattice
    assert starts[:, 0].min() > -700_000 and starts[:, 0].max() < 950_000
    assert starts[:, 1].min() > -3_450_000 and starts[:, 1].max() < -550_000


def test_the_package_stays_small_enough_to_load_on_demand():
    assert PACKAGE.stat().st_size < 600_000
