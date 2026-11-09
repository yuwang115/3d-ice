#!/usr/bin/env python3
"""Draw the layers of the home page's data tower from the explorer's own data packages.

The home page stacks one image per dataset into a slowly turning tower (see
static/js/3d-ice-data-tower.js). Each image is drawn here from the package the explorer
loads, with the explorer's own colour scale where it has one, so a layer looks on the home
page as it does once opened:

- ice flow speed, basal friction, ice-shelf basal melt, effective pressure, subglacial
  channels, ocean streamlines and bed elevation use the stops and lookup tables of
  static/tools/js/explorer-app.js;
- the observed elevation change, the 2300 projection and the sea ice use those of
  surface-change.js, ice-projection.js and sea-ice.js, at the state each view opens in:
  the change since 1995 at the last epoch, the high-emission scenario in 2300, and the
  September of the latest year;
- the ice surface, the ice-free rebound and the drainage basins have no colour scale in the
  explorer, which draws them as 3D shapes or outlines; they get one of their own here.

Every image covers the region's terrain grid (BedMachine), so the layers of a region stack
cell on cell. Outputs go to static/tools/media/3d-ice/tower/<region>-<layer>.webp, plus a
blurred footprint, <region>-shadow.webp, for the tower's base. Run it again after a package
changes: python3 scripts/build_home_data_tower.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFilter

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "static" / "tools" / "data"
OUTPUT_DIR = ROOT_DIR / "static" / "tools" / "media" / "3d-ice" / "tower"

LONG_SIDE_PX = 600
# Fields are sampled at twice the output size and filtered down; lines at three times.
FIELD_OVERSAMPLE = 2
LINE_OVERSAMPLE = 3
WEBP_QUALITY = 84
INT16_FILL = -32768

# ------------------------------------------------------------------ colour scales
# Copied from the explorer; tests/test_build_home_data_tower.py holds each copy to its source.

VELOCITY_STOPS = [
    (0.0, (0.06, 0.2, 0.5)),
    (0.35, (0.08, 0.62, 0.86)),
    (0.65, (0.95, 0.9, 0.27)),
    (1.0, (0.9, 0.2, 0.12)),
]
VELOCITY_RANGE = (0.0, 3000.0)
VELOCITY_KNEE = 20.0

BASAL_FRICTION_STOPS = [
    (0.0, (0.047, 0.118, 0.259)),
    (0.26, (0.125, 0.478, 0.678)),
    (0.52, (0.88, 0.91, 0.79)),
    (0.78, (0.851, 0.545, 0.188)),
    (1.0, (0.588, 0.094, 0.114)),
]

BASAL_MELT_STOPS = [
    (0.0, (0.082, 0.216, 0.376)),
    (0.22, (0.31, 0.655, 0.847)),
    (0.5, (0.965, 0.969, 0.984)),
    (0.78, (0.969, 0.639, 0.357)),
    (1.0, (0.541, 0.067, 0.09)),
]
BASAL_MELT_RANGE = (-1.0, 15.0)
BASAL_MELT_LINTHRESH = 0.05

THICKNESS_CHANGE_STOPS = [
    (0.0, (0.404, 0.0, 0.122)),
    (0.18, (0.698, 0.094, 0.169)),
    (0.33, (0.839, 0.376, 0.302)),
    (0.44, (0.957, 0.647, 0.51)),
    (0.5, (0.95, 0.97, 1.0)),
    (0.56, (0.573, 0.773, 0.871)),
    (0.67, (0.263, 0.576, 0.765)),
    (0.82, (0.129, 0.4, 0.675)),
    (1.0, (0.02, 0.188, 0.38)),
]
THICKNESS_CHANGE_SOFT_M = 10.0
THICKNESS_CHANGE_MAX_M = 1000.0
SURFACE_CHANGE_FADE_SIGMA = 2.0
SURFACE_CHANGE_FADED_SHARE = 0.25

SEA_ICE_THRESHOLD_PERCENT = 15.0
SEA_ICE_LOW_COLOR = (0.62, 0.75, 0.88)
SEA_ICE_HIGH_COLOR = (0.97, 0.98, 1.0)
SEA_ICE_LOW_ALPHA = 0.5
SEA_ICE_HIGH_ALPHA = 0.92
SEA_ICE_EDGE_COLOR = (1.0, 0.55, 0.1)
# seaIceRampColor composites the ice over this sea for its legend.
SEA_ICE_SEA_COLOR = (0.06, 0.16, 0.3)

BED_ELEVATION_RANGE = (-7000.0, 3500.0)
EFFECTIVE_PRESSURE_REFERENCE_PA = 5_000_000.0
CHANNEL_DISCHARGE_RANGE = (1e-3, 100.0)
OCEAN_FRESH_COLD = (0.0, 0.87, 0.99)
OCEAN_SALTY_COLD = (0.23, 0.11, 0.88)
OCEAN_FRESH_WARM = (0.0, 0.76, 0.2)
OCEAN_SALTY_WARM = (0.99, 0.28, 0.0)
GROUNDING_LINE_COLOR = (0x10 / 255, 0x23 / 255, 0x3A / 255)

# The tower's own scales, for layers the explorer draws without one.
ICE_SURFACE_STOPS = [
    (0.0, (0.06, 0.25, 0.4)),
    (0.18, (0.11, 0.42, 0.56)),
    (0.42, (0.34, 0.66, 0.75)),
    (0.68, (0.71, 0.88, 0.9)),
    (1.0, (0.97, 0.99, 0.99)),
]
REBOUND_COLORMAP = "magma"
REBOUND_MAX_M = 1000.0
BASIN_TINTS = [
    (0.74, 0.87, 0.93),
    (0.62, 0.8, 0.84),
    (0.8, 0.9, 0.84),
    (0.68, 0.78, 0.9),
    (0.86, 0.9, 0.95),
    (0.71, 0.86, 0.8),
]
FLOATING_ICE_COLOR = (0.39, 0.66, 0.77)
# The faint land under the sea-ice and ocean layers, a blue-grey that reads on both themes.
UNDERLAY_COLOR = (0.56, 0.66, 0.74)
UNDERLAY_ALPHA = 0.2
GROUNDED_ICE_COLOR = (0.84, 0.9, 0.94)
ICE_FREE_LAND_COLOR = (0.72, 0.66, 0.58)


def sample_stops(stops: list[tuple[float, tuple[float, float, float]]], t: np.ndarray) -> np.ndarray:
    """RGB (0-1) at positions `t` of a piecewise-linear ramp, clamped at its ends."""
    positions = np.array([stop[0] for stop in stops])
    colors = np.array([stop[1] for stop in stops])
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return np.stack([np.interp(t, positions, colors[:, channel]) for channel in range(3)], axis=-1)


def sample_lut(lut: np.ndarray, t: np.ndarray) -> np.ndarray:
    """RGB (0-1) at positions `t` of an evenly spaced lookup table, interpolated."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(lut) - 1)
    index = np.arange(len(lut))
    return np.stack([np.interp(t, index, lut[:, channel]) for channel in range(3)], axis=-1)


def read_rgb_table(path: Path) -> np.ndarray:
    """A GMT/cmocean `.rgb` table as an (n, 3) array in 0-1, as parseRgbTableText reads it."""
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("ncolors"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            rows.append([float(value) for value in parts[:3]])
    table = np.array(rows)
    return np.clip(table / 255.0 if table.max() > 1 else table, 0.0, 1.0)


def velocity_t(speed: np.ndarray) -> np.ndarray:
    lo, hi = VELOCITY_RANGE
    clamped = np.clip(speed, lo, hi)
    return np.clip(np.log1p(clamped / VELOCITY_KNEE) / np.log1p(hi / VELOCITY_KNEE), 0.0, 1.0)


def basal_friction_t(value_mpa: np.ndarray, display_range: tuple[float, float], knee: float) -> np.ndarray:
    lo, hi = display_range
    clamped = np.clip(value_mpa, lo, hi)
    scaled = np.log1p(np.maximum(0.0, clamped - lo) / knee) / np.log1p(max(1e-6, hi - lo) / knee)
    return np.clip(scaled, 0.0, 1.0)


def basal_melt_t(melt: np.ndarray) -> np.ndarray:
    lo, hi = BASAL_MELT_RANGE
    melt = np.clip(melt, lo, hi)
    negative_max = max(BASAL_MELT_LINTHRESH, abs(lo))
    positive_max = max(BASAL_MELT_LINTHRESH, hi)
    below = 0.5 * (1 - np.clip(np.log1p(np.abs(melt) / BASAL_MELT_LINTHRESH) / np.log1p(negative_max / BASAL_MELT_LINTHRESH), 0, 1))
    above = 0.5 + 0.5 * np.clip(np.log1p(np.maximum(melt, 0) / BASAL_MELT_LINTHRESH) / np.log1p(positive_max / BASAL_MELT_LINTHRESH), 0, 1)
    return np.where(melt <= 0, below, above)


def thickness_change_t(delta: np.ndarray, soft: float = THICKNESS_CHANGE_SOFT_M, saturation: float = THICKNESS_CHANGE_MAX_M) -> np.ndarray:
    """thicknessChangeScaleT: asinh, linear within `soft`, saturating at `saturation`; 0.5 is no change."""
    delta = np.nan_to_num(np.asarray(delta, dtype=np.float64))
    magnitude = np.minimum(1.0, np.arcsinh(np.abs(delta) / soft) / np.arcsinh(saturation / soft))
    return 0.5 + 0.5 * np.sign(delta) * magnitude


def sea_ice_rgba(percent: np.ndarray) -> np.ndarray:
    """seaIceColor: RGB and opacity of the ice at a concentration; none below 15 %."""
    percent = np.asarray(percent, dtype=np.float64)
    t = np.clip((percent - SEA_ICE_THRESHOLD_PERCENT) / (100 - SEA_ICE_THRESHOLD_PERCENT), 0.0, 1.0)
    shade = t * t * (3 - 2 * t)
    low, high = np.array(SEA_ICE_LOW_COLOR), np.array(SEA_ICE_HIGH_COLOR)
    rgb = low + shade[..., None] * (high - low)
    alpha = SEA_ICE_LOW_ALPHA + t * (SEA_ICE_HIGH_ALPHA - SEA_ICE_LOW_ALPHA)
    alpha = np.where(percent >= SEA_ICE_THRESHOLD_PERCENT, alpha, 0.0)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def ocean_current_rgb(theta: np.ndarray, salinity: np.ndarray, visualization: dict[str, Any]) -> np.ndarray:
    """oceanCurrentColor: cold fresh cyan, cold salty indigo, warm fresh green, warm salty orange."""
    theta_lo, theta_hi = visualization.get("temperature_range_c", (-2.0, 8.0))
    sal_lo, sal_hi = visualization.get("salinity_range_psu", (30.0, 35.0))
    theta_t = np.clip((theta - theta_lo) / max(1e-6, theta_hi - theta_lo), 0.0, 1.0)[..., None]
    sal_t = np.clip((salinity - sal_lo) / max(1e-6, sal_hi - sal_lo), 0.0, 1.0)[..., None]
    fresh = np.array(OCEAN_FRESH_COLD) + theta_t * (np.array(OCEAN_FRESH_WARM) - np.array(OCEAN_FRESH_COLD))
    salty = np.array(OCEAN_SALTY_COLD) + theta_t * (np.array(OCEAN_SALTY_WARM) - np.array(OCEAN_SALTY_COLD))
    rgb = fresh + sal_t * (salty - fresh)
    lift = 0.022 + theta_t * 0.018
    return np.clip(rgb + lift * np.array([1.0, 0.65, 0.18]), 0.0, 1.0)


def bed_rgb(bed: np.ndarray, relief: np.ndarray) -> np.ndarray:
    """bedColor: GMT_relief, its ocean half below 0 m and its land half above."""
    lo, hi = BED_ELEVATION_RANGE
    bed = np.clip(np.nan_to_num(bed), lo, hi)
    ocean = sample_lut(relief[:128], (bed - lo) / (0 - lo))
    land = sample_lut(relief[128:], bed / hi)
    return np.where((bed < 0)[..., None], ocean, land)


def channel_t(discharge: np.ndarray) -> np.ndarray:
    lo, hi = CHANNEL_DISCHARGE_RANGE
    clamped = np.clip(discharge, lo, hi)
    return (np.log10(clamped) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))


def hillshade(elevation: np.ndarray, cell_m: float, exaggeration: float = 6.0) -> np.ndarray:
    """Lambertian shade (about 0.55-1.15) lit from the upper left, flat ground at 1."""
    z = np.nan_to_num(elevation) * exaggeration
    gy, gx = np.gradient(z, cell_m)
    normal = np.stack([-gx, gy, np.ones_like(z)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-0.55, 0.55, 0.63])
    light /= np.linalg.norm(light)
    flat = light[2]
    return np.clip(0.35 + 0.65 * (normal @ light) / flat, 0.55, 1.15)


# ------------------------------------------------------------------ packages


@dataclass(frozen=True)
class Grid:
    """A package grid; (x0, y0) is the centre of the first cell, rows run north to south."""

    nx: int
    ny: int
    x0: float
    y0: float
    dx: float
    dy: float

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "Grid":
        grid = meta["grid"]
        return cls(int(grid["nx"]), int(grid["ny"]), float(grid["x0_m"]), float(grid["y0_m"]), float(grid["dx_m"]), float(grid["dy_m"]))

    def extent(self) -> tuple[float, float, float, float]:
        """(left, right, top, bottom) cell edges in metres."""
        left = self.x0 - self.dx / 2
        top = self.y0 - self.dy / 2
        return left, left + self.nx * self.dx, top, top + self.ny * self.dy


class Package:
    """A `.meta.json` and `.bin` pair, decoded as static/tools/js/data-contract.js does."""

    def __init__(self, data_dir: Path, basename: str) -> None:
        self.meta = json.loads((data_dir / f"{basename}.meta.json").read_text(encoding="utf-8"))
        self.payload = (data_dir / f"{basename}.bin").read_bytes()
        self.grid = Grid.from_meta(self.meta) if self.meta.get("grid") else None

    def field(self, name: str) -> dict[str, Any]:
        for field in self.meta["fields"]:
            if field["name"] == name:
                return field
        raise KeyError(f"{name} is not a field of {self.meta.get('title')}")

    def raw(self, name: str) -> np.ndarray:
        field = self.field(name)
        dtype = np.dtype(field["dtype"]).newbyteorder("<")
        start = int(field["byte_offset"])
        return np.frombuffer(self.payload[start : start + int(field["byte_length"])], dtype=dtype)

    def values(self, name: str) -> np.ndarray:
        """Float values of a field, NaN where it holds its fill code."""
        field = self.field(name)
        raw = self.raw(name)
        if field["dtype"] == "float32":
            values = raw.astype(np.float64)
            fill = field.get("fill_value", self.meta.get("quantization", {}).get("float_fill_value"))
            keep = np.isfinite(values) if fill is None else np.isfinite(values) & (values != float(fill))
            return np.where(keep, values, np.nan)
        if field["dtype"] != "int16":
            values = raw.astype(np.float64)
            return np.where(raw == int(field["fill_value"]), np.nan, values) if "fill_value" in field else values
        package = self.meta.get("quantization", {})
        scale = float(field.get("scale", package.get("scale", 1.0)))
        offset = float(field.get("offset", package.get("offset", 0.0)))
        fill = int(field.get("fill_value", package.get("int16_fill_value", INT16_FILL)))
        return np.where(raw == fill, np.nan, raw * scale + offset)

    @property
    def grid_shape(self) -> tuple[int, int]:
        assert self.grid is not None
        return (self.grid.ny, self.grid.nx)

    def grid_field(self, name: str) -> np.ndarray:
        assert self.grid is not None, "a gridded field needs a package grid"
        return self.values(name).reshape(self.grid.ny, self.grid.nx)

    def frame(self, name: str, index: int) -> np.ndarray:
        """One frame of a frame-major sparse field, placed back on the grid (NaN off the domain)."""
        assert self.grid is not None
        domain = self.raw("domain_mask").reshape(self.grid.ny, self.grid.nx) > 0
        count = int(domain.sum())
        values = self.values(name)
        out = np.full((self.grid.ny, self.grid.nx), np.nan)
        out[domain] = values[index * count : (index + 1) * count]
        return out


# ------------------------------------------------------------------ sampling and layers


@dataclass(frozen=True)
class Canvas:
    """The output image: its pixel size and the area it covers, (left, right, top, bottom)."""

    width: int
    height: int
    extent: tuple[float, float, float, float]

    @classmethod
    def for_grid(cls, grid: Grid, long_side: int = LONG_SIDE_PX) -> "Canvas":
        left, right, top, bottom = grid.extent()
        aspect = (right - left) / (top - bottom)
        if aspect >= 1:
            return cls(long_side, max(1, round(long_side / aspect)), grid.extent())
        return cls(max(1, round(long_side * aspect)), long_side, grid.extent())

    def to_pixels(self, x: np.ndarray, y: np.ndarray, scale: int = 1) -> tuple[np.ndarray, np.ndarray]:
        left, right, top, bottom = self.extent
        return (x - left) / (right - left) * self.width * scale, (top - y) / (top - bottom) * self.height * scale


def sample_rgba(rgba: np.ndarray, grid: Grid, canvas: Canvas, scale: int = FIELD_OVERSAMPLE) -> np.ndarray:
    """Bilinear samples of a (ny, nx, 4) straight-alpha field at the canvas's pixel centres."""
    width, height = canvas.width * scale, canvas.height * scale
    left, right, top, bottom = canvas.extent
    xs = left + (np.arange(width) + 0.5) * (right - left) / width
    ys = top - (np.arange(height) + 0.5) * (top - bottom) / height
    cols = (xs - grid.x0) / grid.dx
    rows = (ys - grid.y0) / grid.dy
    premultiplied = np.concatenate([rgba[..., :3] * rgba[..., 3:4], rgba[..., 3:4]], axis=-1)
    padded = np.pad(np.nan_to_num(premultiplied), ((1, 1), (1, 1), (0, 0)))
    c = np.clip(cols + 1, 0, grid.nx)
    r = np.clip(rows + 1, 0, grid.ny)
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    c1, r1 = np.minimum(c0 + 1, grid.nx + 1), np.minimum(r0 + 1, grid.ny + 1)
    fc, fr = (c - c0)[None, :, None], (r - r0)[:, None, None]
    top_row = padded[r0][:, c0] * (1 - fc) + padded[r0][:, c1] * fc
    bottom_row = padded[r1][:, c0] * (1 - fc) + padded[r1][:, c1] * fc
    return top_row * (1 - fr) + bottom_row * fr


def premultiplied_to_image(premultiplied: np.ndarray, canvas: Canvas) -> Image.Image:
    """An RGBA image of the canvas's size from a premultiplied array at any multiple of it."""
    data = np.clip(np.round(np.nan_to_num(premultiplied) * 255), 0, 255).astype(np.uint8)
    image = Image.frombuffer("RGBa", (data.shape[1], data.shape[0]), data.tobytes(), "raw", "RGBa", 0, 1)
    if image.size != (canvas.width, canvas.height):
        image = image.resize((canvas.width, canvas.height), Image.Resampling.LANCZOS)
    return image.convert("RGBA")


def with_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.concatenate([np.clip(rgb, 0, 1), np.clip(np.nan_to_num(alpha), 0, 1)[..., None]], axis=-1)


def over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Straight-alpha `top` composited over straight-alpha `bottom`."""
    a_top, a_bottom = top[..., 3:4], bottom[..., 3:4]
    alpha = a_top + a_bottom * (1 - a_top)
    rgb = (top[..., :3] * a_top + bottom[..., :3] * a_bottom * (1 - a_top)) / np.maximum(alpha, 1e-6)
    return np.concatenate([rgb, alpha], axis=-1)


def silhouette(mask: np.ndarray, rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
    """A flat underlay of the ice and land, so a layer drawn only in places still reads whole."""
    shape = mask.shape
    return with_alpha(np.broadcast_to(np.array(rgb), shape + (3,)), np.where(mask > 0, alpha, 0.0))


def draw_lines(canvas: Canvas, segments: np.ndarray, colors: np.ndarray, widths: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Antialiased line segments ((x0, y0, x1, y1) in metres) as a premultiplied array."""
    scale = LINE_OVERSAMPLE
    image = Image.new("RGBA", (canvas.width * scale, canvas.height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x0, y0 = canvas.to_pixels(segments[:, 0], segments[:, 1], scale)
    x1, y1 = canvas.to_pixels(segments[:, 2], segments[:, 3], scale)
    rgba = np.clip(np.round(np.concatenate([colors, alphas[:, None]], axis=1) * 255), 0, 255).astype(int)
    for i in np.argsort(alphas, kind="stable"):
        draw.line(
            [(float(x0[i]), float(y0[i])), (float(x1[i]), float(y1[i]))],
            fill=tuple(rgba[i]),
            width=max(1, int(round(widths[i] * scale))),
        )
    data = np.asarray(image, dtype=np.float64) / 255
    return np.concatenate([data[..., :3] * data[..., 3:4], data[..., 3:4]], axis=-1)


def premultiplied_over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return top + bottom * (1 - top[..., 3:4])


def resize_premultiplied(premultiplied: np.ndarray, width: int, height: int) -> np.ndarray:
    channels = [
        np.asarray(Image.fromarray(premultiplied[..., k].astype(np.float32), mode="F").resize((width, height), Image.Resampling.BOX))
        for k in range(4)
    ]
    return np.stack(channels, axis=-1).astype(np.float64)


@dataclass(frozen=True)
class Region:
    key: str
    terrain: str
    velocity: str
    basal_friction: str
    rebound: str
    sea_ice: str
    ocean: str
    basins: str
    surface_max_m: float
    # Every n-th ocean streamline is drawn; the Antarctic package has more than the image needs.
    streamline_stride: int = 1
    surface_change: str | None = None
    surface_change_rate: str | None = None
    rise: str | None = None
    hydrology: str | None = None
    projection: str | None = None


REGIONS = {
    "antarctica": Region(
        key="antarctica",
        terrain="bedmachine_antarctica_v4_480",
        velocity="antarctic_ice_velocity_phase_v01_480",
        basal_friction="antarctica_basal_friction_480",
        rebound="antarctica_isostatic_response_480",
        sea_ice="nsidc_sea_ice_antarctica",
        ocean="antarctica_ocean_currents_waom2_yr5_annual_combined_light",
        basins="imbie_refined_basins_v2.json",
        surface_max_m=4000.0,
        streamline_stride=2,
        surface_change="cpom_antarctica_dh_480",
        rise="rise_antarctica_480",
        hydrology="antarctica_subglacial_hydrology_480",
        projection="ismip6_2300_mean8_ae05_480",
    ),
    "greenland": Region(
        key="greenland",
        terrain="bedmachine_greenland_v6_3km",
        velocity="greenland_ice_velocity_3km",
        basal_friction="greenland_basal_friction_3km",
        rebound="greenland_isostatic_response_3km",
        sea_ice="nsidc_sea_ice_greenland",
        ocean="greenland_ocean_currents_cmems_202508",
        basins="greenland_basins_ps_v1_4_2.json",
        surface_max_m=3300.0,
    ),
}


# ------------------------------------------------------------------ layer drawing


@dataclass
class Context:
    """What every layer of a region draws against: its terrain and the canvas it covers."""

    region: Region
    data_dir: Path
    canvas: Canvas
    terrain: Package
    bed: np.ndarray
    surface: np.ndarray
    mask: np.ndarray
    relief: np.ndarray
    dense: np.ndarray
    matter: np.ndarray

    @classmethod
    def load(cls, region: Region, data_dir: Path = DATA_DIR, long_side: int = LONG_SIDE_PX) -> "Context":
        terrain = Package(data_dir, region.terrain)
        return cls(
            region=region,
            data_dir=data_dir,
            canvas=Canvas.for_grid(terrain.grid, long_side),
            terrain=terrain,
            bed=terrain.grid_field("bed"),
            surface=terrain.grid_field("surface"),
            mask=terrain.grid_field("mask").astype(np.uint8),
            relief=read_rgb_table(data_dir / "GMT_relief.rgb"),
            dense=read_rgb_table(data_dir / "cmocean_dense.rgb"),
            matter=read_rgb_table(data_dir / "cmocean_matter.rgb"),
        )

    @property
    def grid(self) -> Grid:
        return self.terrain.grid

    @property
    def ice(self) -> np.ndarray:
        return np.isin(self.mask, (2, 3, 4))

    @property
    def grounded(self) -> np.ndarray:
        return np.isin(self.mask, (2, 4))

    @property
    def land_or_ice(self) -> np.ndarray:
        return np.isin(self.mask, (1, 2, 3, 4))

    def package(self, basename: str) -> Package:
        return Package(self.data_dir, basename)

    def sample(self, rgba: np.ndarray, grid: Grid | None = None) -> np.ndarray:
        return sample_rgba(rgba, grid or self.grid, self.canvas)

    def sample_scalar(self, values: np.ndarray, grid: Grid | None = None) -> np.ndarray:
        """Bilinear samples of a gridded value at the canvas's pixel centres. NaN cells take no
        part (each sample is the mean of its valid neighbours), and a sample with none is 0."""
        valid = np.isfinite(values)
        filled = np.where(valid, values, 0.0)
        sampled = self.sample(np.stack([filled, filled, filled, valid * 1.0], axis=-1), grid)
        weight = sampled[..., 3]
        return np.where(weight > 1e-3, sampled[..., 0] / np.maximum(weight, 1e-3), 0.0)

    def shade(self, elevation: np.ndarray, exaggeration: float, strength: float = 1.0) -> np.ndarray:
        shade = hillshade(elevation, abs(self.grid.dx), exaggeration)
        return (1 - strength + strength * shade)[..., None]


def draw_ice_surface(ctx: Context) -> np.ndarray:
    t = np.nan_to_num(ctx.surface) / ctx.region.surface_max_m
    rgb = sample_stops(ICE_SURFACE_STOPS, t) * ctx.shade(ctx.surface, 10.0)
    return ctx.sample(with_alpha(rgb, ctx.ice))


def draw_velocity(ctx: Context) -> np.ndarray:
    package = ctx.package(ctx.region.velocity)
    speed = np.hypot(package.grid_field("vx"), package.grid_field("vy"))
    rgb = sample_stops(VELOCITY_STOPS, velocity_t(np.nan_to_num(speed))) * ctx.shade(ctx.surface, 10.0, 0.25)
    return ctx.sample(with_alpha(rgb, np.isfinite(speed) & ctx.ice), package.grid)


def draw_basal_friction(ctx: Context) -> np.ndarray:
    package = ctx.package(ctx.region.basal_friction)
    visualization = package.meta.get("visualization", {})
    display_range = tuple(visualization.get("display_range_mpa", (0.0, 0.3)))
    knee = float(visualization.get("knee_mpa", 0.015))
    friction = package.grid_field("basal_friction")
    rgb = sample_stops(BASAL_FRICTION_STOPS, basal_friction_t(np.nan_to_num(friction), display_range, knee))
    return ctx.sample(with_alpha(rgb, np.isfinite(friction) & ctx.grounded), package.grid)


def draw_bed(ctx: Context) -> np.ndarray:
    # The bed under the ice and the ice-free land; the sea floor around is left out, so the
    # layer keeps the continent's shape in the stack.
    rgb = bed_rgb(ctx.bed, ctx.relief) * ctx.shade(ctx.bed, 4.0, 0.6)
    return ctx.sample(with_alpha(rgb, ctx.land_or_ice))


def draw_rebound(ctx: Context) -> np.ndarray:
    package = ctx.package(ctx.region.rebound)
    uplift = package.grid_field("solid_surface_change")
    ramp = colormaps[REBOUND_COLORMAP]
    t = 0.12 + 0.86 * np.clip(np.nan_to_num(uplift) / REBOUND_MAX_M, 0, 1)
    rgb = ramp(t)[..., :3] * ctx.shade(ctx.bed, 3.0, 0.35)
    return ctx.sample(with_alpha(rgb, np.isfinite(uplift) & ctx.land_or_ice), package.grid)


def grounding_line_cells(mask: np.ndarray) -> np.ndarray:
    """Grounded cells (2 or 4) with a floating neighbour (3) along a row or a column."""
    grounded = np.isin(mask, (2, 4))
    floating = np.pad(mask == 3, 1)
    neighbour = floating[:-2, 1:-1] | floating[2:, 1:-1] | floating[1:-1, :-2] | floating[1:-1, 2:]
    return grounded & neighbour


def draw_footprint(ctx: Context) -> np.ndarray:
    rgb = np.zeros(ctx.mask.shape + (3,))
    rgb[np.isin(ctx.mask, (2, 4))] = GROUNDED_ICE_COLOR
    rgb[ctx.mask == 3] = FLOATING_ICE_COLOR
    rgb[ctx.mask == 1] = ICE_FREE_LAND_COLOR
    rgb[grounding_line_cells(ctx.mask)] = GROUNDING_LINE_COLOR
    return ctx.sample(with_alpha(rgb, ctx.land_or_ice))


def draw_shadow(ctx: Context) -> Image.Image:
    """The footprint as a soft shadow for the tower's base."""
    footprint = premultiplied_to_image(ctx.sample(silhouette(ctx.land_or_ice, (0, 0, 0), 1.0)), ctx.canvas)
    alpha = footprint.getchannel("A").filter(ImageFilter.GaussianBlur(radius=ctx.canvas.width / 70))
    shadow = Image.new("RGBA", footprint.size, (8, 31, 48, 0))
    shadow.putalpha(alpha.point(lambda value: round(value * 0.42)))
    return shadow


def latest_month(months: list[str], month: int) -> int:
    """Frame index of the latest record month that is the given calendar month."""
    suffix = f"-{month:02d}"
    matches = [i for i, label in enumerate(months) if label.endswith(suffix)]
    if not matches:
        raise ValueError(f"the record holds no month {suffix}")
    return matches[-1]


def reference_frequency(package: Package, month: int) -> np.ndarray:
    """Share (percent) of the 1981-2010 years with ice in each cell, for one calendar month."""
    grid = package.grid
    domain = package.raw("domain_mask").reshape(grid.ny, grid.nx) > 0
    count = int(domain.sum())
    values = package.values("reference_ice_frequency")[(month - 1) * count : month * count]
    out = np.full((grid.ny, grid.nx), np.nan)
    out[domain] = np.where(values == 255, np.nan, values)
    return out


def gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """A separable Gaussian blur of a 2D array, edges held at their value."""
    if sigma <= 0:
        return field
    radius = max(1, int(np.ceil(3 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(field, radius, mode="edge")
    rows = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, rows)


def contour_line(field: np.ndarray, level: float, width_px: float) -> np.ndarray:
    """Antialiased opacity of the line where a sampled field crosses `level`."""
    gy, gx = np.gradient(field)
    distance = np.abs(field - level) / np.maximum(np.hypot(gx, gy), 1e-6)
    return np.clip(1 - distance / width_px, 0, 1)


def draw_sea_ice(ctx: Context, month: int = 9) -> np.ndarray:
    """The ice of the September the explorer opens on, over the dark sea its legend uses,
    with NSIDC's 1981-2010 median ice edge for that month in orange."""
    package = ctx.package(ctx.region.sea_ice)
    index = latest_month(package.meta["frames"]["months"], month)
    # Concentration is interpolated before it is coloured, so the 25 km cells do not show.
    concentration = ctx.sample_scalar(package.frame("concentration", index), package.grid)
    rgba = sea_ice_rgba(concentration)
    sea = np.array(SEA_ICE_SEA_COLOR)
    rgb = sea + rgba[..., 3:4] * (rgba[..., :3] - sea)
    on = (rgba[..., 3] > 0)[..., None] * 1.0
    ice = np.concatenate([rgb * on, on], axis=-1)
    # A blur of half a cell rounds off the staircase the cells would give the edge.
    cell_px = abs(package.grid.dx) / (ctx.canvas.extent[1] - ctx.canvas.extent[0]) * ctx.canvas.width * FIELD_OVERSAMPLE
    smooth = gaussian_blur(ctx.sample_scalar(reference_frequency(package, month), package.grid), cell_px / 2)
    edge = contour_line(smooth, 50.0, 1.4 * FIELD_OVERSAMPLE)
    land = ctx.sample(silhouette(ctx.land_or_ice, UNDERLAY_COLOR, UNDERLAY_ALPHA))
    # Outside the package's domain the share reads 0, which would trace the coast as well.
    near_land = Image.fromarray(np.uint8(land[..., 3] > 0) * 255).filter(ImageFilter.MaxFilter(2 * round(cell_px) + 1))
    edge = edge * (np.asarray(near_land) == 0)
    edge_layer = np.concatenate([np.array(SEA_ICE_EDGE_COLOR) * edge[..., None], edge[..., None]], axis=-1)
    return premultiplied_over(edge_layer, premultiplied_over(ice, land))


def draw_surface_change(ctx: Context) -> np.ndarray:
    package = ctx.package(ctx.region.surface_change)
    visualization = package.meta.get("visualization", {})
    soft = float(visualization.get("linear_core_m", 1.0))
    saturation = float(visualization.get("saturation_m", 150.0))
    last = int(package.meta["frames"]["count"]) - 1
    change = package.frame("surface_change", last)
    domain = package.raw("domain_mask").reshape(package.grid.ny, package.grid.nx) > 0
    sigma = np.full(domain.shape, np.nan)
    sigma[domain] = package.values("final_uncertainty")
    rgb = sample_stops(THICKNESS_CHANGE_STOPS, thickness_change_t(change, soft, saturation))
    neutral = sample_stops(THICKNESS_CHANGE_STOPS, np.array(0.5))
    faded = np.abs(np.nan_to_num(change)) < SURFACE_CHANGE_FADE_SIGMA * np.nan_to_num(sigma)
    rgb = np.where(faded[..., None], neutral + SURFACE_CHANGE_FADED_SHARE * (rgb - neutral), rgb)
    rgb = rgb * ctx.shade(ctx.surface, 10.0, 0.2)
    return ctx.sample(with_alpha(rgb, np.isfinite(change)), package.grid)


def draw_projection(ctx: Context, year: int = 2300) -> np.ndarray:
    package = ctx.package(ctx.region.projection)
    years = list(package.meta["frames"]["years"])
    present = package.frame("thickness", 0)
    future = package.frame("thickness", years.index(year))
    change = future - present
    rgb = sample_stops(THICKNESS_CHANGE_STOPS, thickness_change_t(change)) * ctx.shade(ctx.surface, 10.0, 0.2)
    # Where the models lose the ice there is none to colour: the layer has a hole there.
    return ctx.sample(with_alpha(rgb, np.nan_to_num(future) > 0), package.grid)


def draw_basal_melt(ctx: Context) -> np.ndarray:
    package = ctx.package(ctx.region.rise)
    shelf = package.grid_field("mask") == 2
    melt = package.grid_field("ismr")
    rgb = sample_stops(BASAL_MELT_STOPS, basal_melt_t(np.nan_to_num(melt)))
    shelves = ctx.sample(with_alpha(rgb, shelf & np.isfinite(melt)), package.grid)
    grounded = ctx.sample(silhouette(ctx.grounded | (ctx.mask == 1), (0.8, 0.86, 0.9), 0.55))
    return premultiplied_over(shelves, grounded)


def draw_subglacial(ctx: Context) -> np.ndarray:
    package = ctx.package(ctx.region.hydrology)
    pressure = package.grid_field("effective_pressure")
    rgb = sample_lut(ctx.dense, np.clip(np.maximum(np.nan_to_num(pressure), 0) / EFFECTIVE_PRESSURE_REFERENCE_PA, 0, 1))
    field = ctx.sample(with_alpha(rgb, np.isfinite(pressure) & ctx.grounded), package.grid)
    grid = package.grid
    cols = [package.raw(name).astype(np.float64) for name in ("channel_col1", "channel_row1", "channel_col2", "channel_row2")]
    segments = np.stack(
        [grid.x0 + cols[0] * grid.dx, grid.y0 + cols[1] * grid.dy, grid.x0 + cols[2] * grid.dx, grid.y0 + cols[3] * grid.dy],
        axis=1,
    )
    t = channel_t(package.values("channel_discharge"))
    channels = draw_lines(ctx.canvas, segments, sample_lut(ctx.matter, t), 0.5 + 1.7 * t, 0.35 + 0.65 * t)
    return premultiplied_over(resize_like(channels, field), field)


def draw_ocean(ctx: Context) -> np.ndarray:
    package = ctx.package(ctx.region.ocean)
    names = ("x0_ps_m", "y0_ps_m", "x1_ps_m", "y1_ps_m")
    segments = np.stack([package.values(name) for name in names], axis=1)
    theta = 0.5 * (package.values("theta0_c") + package.values("theta1_c"))
    salinity = 0.5 * (package.values("sal0_psu") + package.values("sal1_psu"))
    depth = package.values("depth0_m")
    # Each streamline ends on a segment flagged terminal; a stride keeps every n-th one.
    streamline = np.concatenate([[0], np.cumsum(package.values("terminal_flag")[:-1] > 0)])
    keep = np.isfinite(segments).all(axis=1) & np.isfinite(theta) & np.isfinite(salinity)
    keep &= streamline % ctx.region.streamline_stride == 0
    order = np.argsort(-np.nan_to_num(depth[keep]))
    segments, theta, salinity = segments[keep][order], theta[keep][order], salinity[keep][order]
    colors = ocean_current_rgb(theta, salinity, package.meta.get("visualization", {}))
    count = len(segments)
    lines = draw_lines(ctx.canvas, segments, colors, np.full(count, 0.85), np.linspace(0.55, 0.95, count))
    land = ctx.sample(silhouette(ctx.land_or_ice, UNDERLAY_COLOR, UNDERLAY_ALPHA))
    return premultiplied_over(resize_like(lines, land), land)


def draw_basins(ctx: Context) -> np.ndarray:
    basins = json.loads((ctx.data_dir / ctx.region.basins).read_text(encoding="utf-8"))["basins"]
    scale = FIELD_OVERSAMPLE
    size = (ctx.canvas.width * scale, ctx.canvas.height * scale)
    fills = Image.new("RGBA", size, (0, 0, 0, 0))
    outlines = Image.new("RGBA", size, (0, 0, 0, 0))
    fill_draw, outline_draw = ImageDraw.Draw(fills), ImageDraw.Draw(outlines)
    for index, basin in enumerate(basins):
        tint = tuple(round(channel * 255) for channel in BASIN_TINTS[index % len(BASIN_TINTS)]) + (255,)
        for ring in basin["segments_xy_m"]:
            points = np.asarray(ring, dtype=np.float64)
            if len(points) < 3:
                continue
            px, py = ctx.canvas.to_pixels(points[:, 0], points[:, 1], scale)
            path = list(zip(px.tolist(), py.tolist()))
            fill_draw.polygon(path, fill=tint)
            outline_draw.line(path + [path[0]], fill=(255, 255, 255, 235), width=max(1, round(0.9 * scale)))
    fills.alpha_composite(outlines)
    data = np.asarray(fills, dtype=np.float64) / 255
    premultiplied = np.concatenate([data[..., :3] * data[..., 3:4], data[..., 3:4]], axis=-1)
    # Clipped to the ice, as the explorer draws the outlines on it.
    ice = ctx.sample(silhouette(ctx.ice, (1, 1, 1), 1.0))[..., 3:4]
    return premultiplied * ice


def resize_like(premultiplied: np.ndarray, reference: np.ndarray) -> np.ndarray:
    height, width = reference.shape[:2]
    if premultiplied.shape[:2] == (height, width):
        return premultiplied
    return resize_premultiplied(premultiplied, width, height)


LayerDrawer = Callable[[Context], np.ndarray]

# The layers of each region's tower, top to bottom: the ice surface, what lies on it, the sea,
# the bed and the solid Earth beneath; the footprint sits on the base.
LAYERS: dict[str, dict[str, LayerDrawer]] = {
    "antarctica": {
        "ice-surface": draw_ice_surface,
        "surface-change": draw_surface_change,
        "velocity": draw_velocity,
        "projection": draw_projection,
        "sea-ice": draw_sea_ice,
        "ocean": draw_ocean,
        "basal-melt": draw_basal_melt,
        "basal-friction": draw_basal_friction,
        "subglacial": draw_subglacial,
        "bed": draw_bed,
        "rebound": draw_rebound,
        "basins": draw_basins,
        "grounding-line": draw_footprint,
    },
    "greenland": {
        "ice-surface": draw_ice_surface,
        "velocity": draw_velocity,
        "sea-ice": draw_sea_ice,
        "ocean": draw_ocean,
        "basal-friction": draw_basal_friction,
        "bed": draw_bed,
        "rebound": draw_rebound,
        "basins": draw_basins,
        "grounding-line": draw_footprint,
    },
}


def save_webp(image: Image.Image, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "WEBP", quality=WEBP_QUALITY, alpha_quality=90, method=6)
    return path.stat().st_size


def build_region(region_key: str, output_dir: Path = OUTPUT_DIR, data_dir: Path = DATA_DIR, long_side: int = LONG_SIDE_PX, only: set[str] | None = None) -> dict[str, int]:
    """Draw and save one region's layers; returns the size in bytes of each file written."""
    ctx = Context.load(REGIONS[region_key], data_dir, long_side)
    written = {}
    for layer_key, draw in LAYERS[region_key].items():
        if only and layer_key not in only:
            continue
        image = premultiplied_to_image(draw(ctx), ctx.canvas)
        written[f"{region_key}-{layer_key}.webp"] = save_webp(image, output_dir / f"{region_key}-{layer_key}.webp")
    if not only or "shadow" in only:
        written[f"{region_key}-shadow.webp"] = save_webp(draw_shadow(ctx), output_dir / f"{region_key}-shadow.webp")
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", choices=sorted(REGIONS), action="append", help="draw only this region (repeatable)")
    parser.add_argument("--layer", action="append", help="draw only this layer key (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--long-side", type=int, default=LONG_SIDE_PX, help="pixels along each image's longer side")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    only = set(args.layer) if args.layer else None
    for region_key in args.region or sorted(REGIONS):
        for name, size in build_region(region_key, args.output_dir, DATA_DIR, args.long_side, only).items():
            print(f"{name}: {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
