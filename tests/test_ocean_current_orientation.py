"""Orientation of the Antarctic ocean streamlines.

WAOM2 runs on a grid whose axes are EPSG:3031 x and y, so its ROMS `angle` (XI axis to
east) equals the EPSG:3031 longitude. Version 0.3.0 rotated the native components by that
angle, turning them into east/north components, and then stepped the streamlines with them
as if they were x/y components: every traced vector was turned by the local longitude, so
the Antarctic Circumpolar Current ran westward in the Pacific sector. These tests pin the
component conversion and check the shipped package against the one current whose direction
is not in doubt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "static" / "tools" / "data"
COMBINED_BASENAME = "antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean"
SHIPPED_BASENAMES = (
    COMBINED_BASENAME,
    "antarctica_ocean_currents_waom2_yr5_annual_combined_light",
)


def _decode_segments(basename: str) -> dict[str, np.ndarray]:
    meta = json.loads((DATA_DIR / f"{basename}.meta.json").read_text())
    payload = (DATA_DIR / f"{basename}.bin").read_bytes()
    fields = {}
    for field in meta["fields"]:
        if "dtype" not in field:
            continue
        dtype = np.dtype(field["dtype"])
        fields[field["name"]] = np.frombuffer(
            payload, dtype=dtype, count=field["byte_length"] // dtype.itemsize, offset=field["byte_offset"]
        )
    return fields


def _flow_relative_to_east(fields: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    x0, y0 = fields["x0_ps_m"].astype(np.float64), fields["y0_ps_m"].astype(np.float64)
    x1, y1 = fields["x1_ps_m"].astype(np.float64), fields["y1_ps_m"].astype(np.float64)
    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    dx, dy = x1 - x0, y1 - y0
    lam = np.arctan2(xm, ym)
    east = dx * np.cos(lam) - dy * np.sin(lam)
    north = dx * np.sin(lam) + dy * np.cos(lam)
    depth = 0.5 * (fields["depth0_m"] + fields["depth1_m"])
    return xm, ym, np.degrees(lam), east, north, np.hypot(dx, dy), depth


class TestGridToProjectedComponents:
    def test_identity_on_an_epsg3031_aligned_grid(self, antarctica_ocean_currents_module):
        x = np.array([[3.2e6, 1.5e5], [-2.9e6, 1.5e5]])
        y = np.array([[-2.0e5, -2.9e6], [-2.0e5, 2.5e6]])
        angle = np.arctan2(x, y)  # WAOM2: angle equals the EPSG:3031 longitude
        u = np.array([[[0.3, -0.1], [0.05, 0.2]]])
        v = np.array([[[0.1, 0.4], [-0.2, 0.0]]])
        ux, vy = antarctica_ocean_currents_module.grid_to_projected_components(u, v, angle, x, y)
        np.testing.assert_allclose(ux, u, atol=1e-7)
        np.testing.assert_allclose(vy, v, atol=1e-7)

    def test_east_aligned_grid_at_90e_maps_east_to_minus_y(self, antarctica_ocean_currents_module):
        # A grid whose XI axis points east (angle 0); at 90 deg E, east is the -y direction.
        x, y, angle = np.array([[1.0e6]]), np.array([[0.0]]), np.array([[0.0]])
        ux, vy = antarctica_ocean_currents_module.grid_to_projected_components(
            np.array([[1.0]]), np.array([[0.0]]), angle, x, y
        )
        np.testing.assert_allclose([ux.item(), vy.item()], [0.0, -1.0], atol=1e-12)

    def test_east_aligned_grid_at_180_maps_north_to_minus_y(self, antarctica_ocean_currents_module):
        # At 180 deg, north (towards the equator) points along -y in EPSG:3031.
        x, y, angle = np.array([[0.0]]), np.array([[-1.0e6]]), np.array([[0.0]])
        ux, vy = antarctica_ocean_currents_module.grid_to_projected_components(
            np.array([[0.0]]), np.array([[1.0]]), angle, x, y
        )
        np.testing.assert_allclose([ux.item(), vy.item()], [0.0, -1.0], atol=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize("basename", SHIPPED_BASENAMES)
def test_circumpolar_current_flows_east_in_every_sector(basename):
    """North of about 62 S the ACC and the gyres' northern limbs flow eastward at all longitudes.

    The test requires at least 0.6 eastward and a mean flow within 50 deg of east in every
    well-sampled sector. Rebuilt from the corrected field, every such sector is at least 0.63
    eastward (the
    full package's lowest is over the Kerguelen Plateau, where eddies and the deflected ACC
    lower the fraction although the mean flow is 8 deg from east) and its mean flow lies
    within 40 deg of east. The v0.3.0 packages were 0.10-0.34 eastward from 90 E to 135 W,
    with mean flows of -142, -135, +107 and +148 deg there: a rotation by the longitude moves
    the mean flow by up to 180 deg, far outside these bounds. The 0-45 E sector holds only the
    domain's corner north of 62 S, on the poleward-turning eastern limb of the Weddell Gyre,
    so sectors with under 4 % of the offshore segments are not judged.
    """
    if not (DATA_DIR / f"{basename}.bin").exists():
        pytest.skip(f"{basename} is not committed")
    xm, ym, lam_deg, east, north, length, depth = _flow_relative_to_east(_decode_segments(basename))
    offshore = (np.hypot(xm, ym) > 3.0e6) & (depth < 500.0) & (length > 0)
    checked = 0
    for start in range(-180, 180, 45):
        sector = offshore & (lam_deg >= start) & (lam_deg < start + 45)
        if sector.sum() < 0.04 * offshore.sum():
            continue
        weights = length[sector]
        eastward_fraction = float(np.sum(weights * (east[sector] > 0)) / np.sum(weights))
        mean_angle = float(np.degrees(np.angle(np.sum(weights * (east[sector] + 1j * north[sector]) / length[sector]))))
        assert eastward_fraction >= 0.6, f"{start}..{start + 45} deg: eastward fraction {eastward_fraction:.2f}"
        assert abs(mean_angle) <= 50.0, f"{start}..{start + 45} deg: mean flow {mean_angle:.0f} deg from east"
        checked += 1
    assert checked >= 6
