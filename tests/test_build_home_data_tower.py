"""The home page's data tower images (scripts/build_home_data_tower.py).

The builder copies the explorer's colour scales so a layer looks on the home page as it
does once opened; these tests hold each copy to the JavaScript it came from, check the
drawing helpers on small arrays, and draw a few real layers small.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import build_home_data_tower as tower
from tests.test_home_page import TOWER_LAYERS, parse

ROOT_DIR = Path(__file__).resolve().parent.parent
JS_DIR = ROOT_DIR / "static" / "tools" / "js"
RUNTIME = (JS_DIR / "explorer-app.js").read_text(encoding="utf-8")


def js_array(source: str, name: str):
    """A JavaScript array literal of numbers, as Python lists."""
    match = re.search(rf"const {name} = (\[.*?\n\]);", source, re.S)
    assert match, f"{name} not found"
    return json.loads(re.sub(r",(\s*[\]}])", r"\1", match.group(1)))


def js_number(source: str, name: str) -> float:
    match = re.search(rf"const {name} = (-?[\d_.]+(?:e-?\d+)?);", source)
    assert match, f"{name} not found"
    return float(match.group(1).replace("_", ""))


def as_stops(stops):
    return [(float(t), tuple(float(c) for c in rgb)) for t, rgb in stops]


@pytest.mark.unit
class TestScalesMatchTheExplorer:
    @pytest.mark.parametrize(
        ("name", "copy", "source"),
        [
            ("VELOCITY_COLOR_STOPS", "VELOCITY_STOPS", RUNTIME),
            ("BASAL_FRICTION_COLOR_STOPS", "BASAL_FRICTION_STOPS", RUNTIME),
            ("RISE_BASAL_MELT_COLOR_STOPS", "BASAL_MELT_STOPS", RUNTIME),
            ("THICKNESS_CHANGE_STOPS", "THICKNESS_CHANGE_STOPS", (JS_DIR / "ice-projection.js").read_text(encoding="utf-8")),
        ],
    )
    def test_colour_stops_are_the_explorers(self, name, copy, source):
        assert as_stops(js_array(source, name)) == as_stops(getattr(tower, copy))

    def test_numbers_are_the_explorers(self):
        projection = (JS_DIR / "ice-projection.js").read_text(encoding="utf-8")
        surface = (JS_DIR / "surface-change.js").read_text(encoding="utf-8")
        sea_ice = (JS_DIR / "sea-ice.js").read_text(encoding="utf-8")
        assert js_number(projection, "THICKNESS_CHANGE_SOFT_METERS") == tower.THICKNESS_CHANGE_SOFT_M
        assert js_number(projection, "THICKNESS_CHANGE_MAX_METERS") == tower.THICKNESS_CHANGE_MAX_M
        assert js_number(surface, "FADE_SIGMA") == tower.SURFACE_CHANGE_FADE_SIGMA
        assert js_number(surface, "FADED_COLOUR_SHARE") == tower.SURFACE_CHANGE_FADED_SHARE
        assert js_number(sea_ice, "EXTENT_THRESHOLD_PERCENT") == tower.SEA_ICE_THRESHOLD_PERCENT
        assert (js_number(RUNTIME, "BED_ELEVATION_MIN"), js_number(RUNTIME, "BED_ELEVATION_MAX")) == tower.BED_ELEVATION_RANGE
        assert js_number(RUNTIME, "EFFECTIVE_PRESSURE_REFERENCE_PA") == tower.EFFECTIVE_PRESSURE_REFERENCE_PA
        assert (js_number(RUNTIME, "CHANNEL_DISCHARGE_MIN"), js_number(RUNTIME, "CHANNEL_DISCHARGE_MAX")) == tower.CHANNEL_DISCHARGE_RANGE
        assert int(re.search(r"GROUNDING_LINE_COLOR = 0x([0-9a-f]{6});", RUNTIME).group(1), 16) == int(
            "".join(f"{round(c * 255):02x}" for c in tower.GROUNDING_LINE_COLOR), 16
        )

    def test_visualization_settings_are_the_explorers(self):
        velocity = re.search(r"VELOCITY_VISUALIZATION = Object\.freeze\(\{\s*range: \[(\d+), (\d+)\],\s*knee: (\d+),", RUNTIME)
        assert tuple(float(v) for v in velocity.groups()[:2]) == tower.VELOCITY_RANGE
        assert float(velocity.group(3)) == tower.VELOCITY_KNEE
        melt = re.search(r"RISE_BASAL_MELT_VISUALIZATION = Object\.freeze\(\{\s*range: \[(-?\d+), (\d+)\],\s*linthresh: ([\d.]+),", RUNTIME)
        assert (float(melt.group(1)), float(melt.group(2))) == tower.BASAL_MELT_RANGE
        assert float(melt.group(3)) == tower.BASAL_MELT_LINTHRESH

    def test_the_sea_ice_style_and_the_ocean_colours_are_the_explorers(self):
        sea_ice = (JS_DIR / "sea-ice.js").read_text(encoding="utf-8")
        style = re.search(r"SEA_ICE_STYLE = Object\.freeze\(\{(.*?)\}\);", sea_ice, re.S).group(1)
        colour = lambda key: tuple(float(v) for v in re.search(rf"{key}: Object\.freeze\(\[([^\]]+)\]\)", style).group(1).split(","))
        number = lambda key: float(re.search(rf"{key}: ([\d.]+)", style).group(1))
        assert colour("lowColor") == tower.SEA_ICE_LOW_COLOR and colour("highColor") == tower.SEA_ICE_HIGH_COLOR
        assert colour("edgeColor") == tower.SEA_ICE_EDGE_COLOR
        assert (number("lowAlpha"), number("highAlpha")) == (tower.SEA_ICE_LOW_ALPHA, tower.SEA_ICE_HIGH_ALPHA)
        sea = re.search(r"export function seaIceRampColor\(t, sea = \[([^\]]+)\]\)", sea_ice).group(1)
        assert tuple(float(v) for v in sea.split(",")) == tower.SEA_ICE_SEA_COLOR
        body = re.search(r"function oceanCurrentColor\(.*?\n\}", RUNTIME, re.S).group(0)
        for name, copy in (("coldFresh", "OCEAN_FRESH_COLD"), ("coldSalty", "OCEAN_SALTY_COLD"), ("warmFresh", "OCEAN_FRESH_WARM"), ("warmSalty", "OCEAN_SALTY_WARM")):
            values = re.search(rf"const {name} = \[([^\]]+)\];", body).group(1)
            assert tuple(float(v) for v in values.split(",")) == getattr(tower, copy), name


@pytest.mark.unit
class TestScaleFunctions:
    def test_ramps_interpolate_and_clamp(self):
        stops = [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.5, 0.0))]
        assert np.allclose(tower.sample_stops(stops, np.array([-1, 0.5, 2])), [[0, 0, 0], [0.5, 0.25, 0], [1, 0.5, 0]])
        lut = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=float)
        assert np.allclose(tower.sample_lut(lut, np.array([0.25, 0.75])), [[0.5, 0.5, 0.5], [0.5, 0.5, 1]])

    def test_scales_follow_the_explorers_formulas(self):
        assert math.isclose(tower.velocity_t(np.array(20.0)), math.log1p(1) / math.log1p(150))
        assert tower.velocity_t(np.array(1e6)) == 1.0
        assert math.isclose(tower.basal_friction_t(np.array(0.015), (0, 0.3), 0.015), math.log1p(1) / math.log1p(20))
        assert tower.basal_melt_t(np.array(0.0)) == 0.5
        assert tower.basal_melt_t(np.array(15.0)) == 1.0 and tower.basal_melt_t(np.array(-1.0)) == 0.0
        assert tower.thickness_change_t(np.array(0.0)) == 0.5
        assert math.isclose(tower.thickness_change_t(np.array(-50.0)) + tower.thickness_change_t(np.array(50.0)), 1.0)
        assert tower.thickness_change_t(np.array(5000.0)) == 1.0
        assert math.isclose(tower.channel_t(np.array(1.0)), 0.6)

    def test_sea_ice_below_the_threshold_is_open_water(self):
        rgba = tower.sea_ice_rgba(np.array([0.0, 14.9, 15.0, 100.0]))
        assert list(rgba[:, 3]) == [0, 0, tower.SEA_ICE_LOW_ALPHA, tower.SEA_ICE_HIGH_ALPHA]
        assert np.allclose(rgba[3, :3], tower.SEA_ICE_HIGH_COLOR)

    def test_the_ocean_colour_mixes_temperature_and_salinity(self):
        vis = {"temperature_range_c": [0, 1], "salinity_range_psu": [0, 1]}
        cold_fresh = tower.ocean_current_rgb(np.array([0.0]), np.array([0.0]), vis)[0]
        warm_salty = tower.ocean_current_rgb(np.array([1.0]), np.array([1.0]), vis)[0]
        assert np.allclose(cold_fresh, np.clip(np.array(tower.OCEAN_FRESH_COLD) + 0.022 * np.array([1, 0.65, 0.18]), 0, 1))
        assert warm_salty[0] > 0.95 and warm_salty[2] < 0.1

    def test_the_bed_takes_the_ocean_half_below_sea_level(self):
        relief = np.vstack([np.zeros((128, 3)), np.ones((128, 3))])
        rgb = tower.bed_rgb(np.array([-100.0, 100.0]), relief)
        assert np.allclose(rgb[0], 0) and np.allclose(rgb[1], 1)

    def test_hillshade_lights_flat_ground_evenly(self):
        shade = tower.hillshade(np.zeros((5, 5)), 1000.0)
        assert np.allclose(shade, 1.0)


@pytest.mark.unit
class TestDrawingHelpers:
    def test_a_grid_extent_runs_from_cell_edge_to_cell_edge(self):
        grid = tower.Grid(nx=4, ny=2, x0=5.0, y0=-5.0, dx=10.0, dy=-10.0)
        assert grid.extent() == (0.0, 40.0, 0.0, -20.0)
        canvas = tower.Canvas.for_grid(grid, long_side=80)
        assert (canvas.width, canvas.height) == (80, 40)
        x, y = canvas.to_pixels(np.array([0.0, 40.0]), np.array([0.0, -20.0]))
        assert list(x) == [0, 80] and list(y) == [0, 40]

    def test_sampling_keeps_a_constant_field_and_fades_at_its_edge(self):
        grid = tower.Grid(nx=8, ny=8, x0=0.5, y0=7.5, dx=1.0, dy=-1.0)
        canvas = tower.Canvas.for_grid(grid, long_side=8)
        field = np.zeros((8, 8, 4))
        field[...] = (0.2, 0.4, 0.6, 1.0)
        sampled = tower.sample_rgba(field, grid, canvas, scale=1)
        assert sampled.shape == (8, 8, 4)
        assert np.allclose(sampled[4, 4], (0.2, 0.4, 0.6, 1.0))

    def test_scalar_samples_skip_cells_without_a_value(self):
        grid = tower.Grid(nx=4, ny=4, x0=0.5, y0=3.5, dx=1.0, dy=-1.0)
        terrain = type("Terrain", (), {"grid": grid})()
        ctx = tower.Context.__new__(tower.Context)
        ctx.terrain = terrain
        ctx.canvas = tower.Canvas.for_grid(grid, long_side=4)
        values = np.full((4, 4), 50.0)
        values[:, 0] = np.nan
        sampled = ctx.sample_scalar(values)
        assert np.allclose(sampled[:, 1:], 50.0), "a missing neighbour does not pull a sample down"

    def test_compositing_premultiplied_layers(self):
        top = np.array([[[0.5, 0.0, 0.0, 0.5]]])
        bottom = np.array([[[0.0, 0.0, 1.0, 1.0]]])
        assert np.allclose(tower.premultiplied_over(top, bottom), [[[0.5, 0.0, 0.5, 1.0]]])
        straight = tower.over(np.array([[[1.0, 0, 0, 0.5]]]), np.array([[[0, 0, 1.0, 1.0]]]))
        assert np.allclose(straight, [[[0.5, 0, 0.5, 1.0]]])

    def test_the_grounding_line_is_grounded_ice_beside_floating_ice(self):
        mask = np.array(
            [
                [0, 2, 2, 2],
                [0, 2, 3, 3],
                [0, 4, 2, 3],
            ]
        )
        assert tower.grounding_line_cells(mask).astype(int).tolist() == [
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]

    def test_the_latest_month_is_found_or_refused(self):
        assert tower.latest_month(["2024-09", "2025-03", "2025-09", "2026-01"], 9) == 2
        with pytest.raises(ValueError):
            tower.latest_month(["2025-01"], 9)

    def test_a_contour_line_sits_where_the_field_crosses_its_level(self):
        field = np.tile(np.arange(10.0), (3, 1)) * 10
        line = tower.contour_line(field, 50.0, 1.0)
        assert line[1, 5] == 1.0 and line[1, 0] == 0.0

    def test_a_blur_keeps_a_flat_field_flat(self):
        assert np.allclose(tower.gaussian_blur(np.full((6, 6), 3.0), 1.5), 3.0)
        assert np.array_equal(tower.gaussian_blur(np.eye(3), 0), np.eye(3))


@pytest.mark.integration
class TestPackagesAndImages:
    def test_packages_decode_their_fill_codes_as_missing(self):
        package = tower.Package(tower.DATA_DIR, "nsidc_sea_ice_greenland")
        frame = package.frame("concentration", len(package.meta["frames"]["months"]) - 1)
        assert frame.shape == (package.grid.ny, package.grid.nx)
        assert np.isnan(frame).any() and np.nanmax(frame) <= 100
        terrain = tower.Package(tower.DATA_DIR, "bedmachine_greenland_v6_3km")
        assert np.isfinite(terrain.grid_field("bed")).any()

    def test_every_region_draws_the_layers_its_page_lists(self):
        for locale in ("en", "zh"):
            tower_section = parse(locale).find("section", id="data-tower")
            for node in tower_section.find_all("ol", cls="explorer-tower-layers"):
                region = node.attrs["data-tower-region"]
                assert list(tower.LAYERS[region]) == TOWER_LAYERS[region], f"{locale} {region}"

    def test_the_committed_images_are_the_builders_size(self):
        for region_key, layers in tower.LAYERS.items():
            grid = tower.Package(tower.DATA_DIR, tower.REGIONS[region_key].terrain).grid
            canvas = tower.Canvas.for_grid(grid)
            for name in [*layers, "shadow"]:
                path = tower.OUTPUT_DIR / f"{region_key}-{name}.webp"
                with Image.open(path) as image:
                    assert image.size == (canvas.width, canvas.height), path.name
                    assert image.mode == "RGBA", path.name

    def test_drawing_a_region_small_writes_its_layers(self, tmp_path):
        written = tower.build_region("greenland", tmp_path, long_side=48, only={"ice-surface", "grounding-line", "sea-ice", "shadow"})
        assert sorted(written) == [
            "greenland-grounding-line.webp",
            "greenland-ice-surface.webp",
            "greenland-sea-ice.webp",
            "greenland-shadow.webp",
        ]
        with Image.open(tmp_path / "greenland-ice-surface.webp") as image:
            assert max(image.size) == 48
            alpha = np.asarray(image.getchannel("A"))
            assert alpha.max() > 200 and alpha.min() == 0, "the ice is drawn and the sea around it is clear"

    def test_every_antarctic_layer_draws_something(self, tmp_path):
        written = tower.build_region("antarctica", tmp_path, long_side=40)
        assert sorted(written) == sorted([f"antarctica-{name}.webp" for name in [*tower.LAYERS["antarctica"], "shadow"]])
        for name in written:
            with Image.open(tmp_path / name) as image:
                assert np.asarray(image.getchannel("A")).max() > 60, f"{name} is empty"

    def test_the_command_line_draws_what_it_is_asked_for(self, tmp_path, capsys):
        tower.main(["--region", "greenland", "--layer", "bed", "--output-dir", str(tmp_path), "--long-side", "32"])
        assert (tmp_path / "greenland-bed.webp").is_file()
        assert not (tmp_path / "greenland-velocity.webp").exists()
        assert "greenland-bed.webp" in capsys.readouterr().out
