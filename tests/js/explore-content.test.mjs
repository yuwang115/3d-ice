/**
 * The public edition's copy lives in one module, in both locales. Every tour stop and every
 * layer explainer must exist in both, every stop must only drive controls the public pages
 * actually show, and every explainer a page asks for must exist. The numbers the
 * high-emission stop quotes must be the ones its packages carry.
 */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { EXPLORE_CONTENT, TOUR_CHAPTERS, getExploreContent } from "../../static/tools/js/explore-content.js";
import { projectLatLon } from "../../static/tools/js/polar-projection.js";
import { TOUR_CONTROL_DEFAULTS, resolveChapterControls } from "../../static/tools/js/explore-tour.js";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const LOCALES = ["en-US", "zh-CN"];
const PUBLIC_PAGES = {
  "en-US": readFileSync(resolve(repoRoot, "static/explore/index.html"), "utf8"),
  "zh-CN": readFileSync(resolve(repoRoot, "static/zh/explore/index.html"), "utf8"),
};

// The extents of the two terrain grids the public edition loads, in projected metres
// (bedmachine_antarctica_v4_480 and bedmachine_greenland_v6_3km).
const GRID_BOUNDS = {
  antarctica: { xMin: -3333000, xMax: 3327000, yMin: -3327000, yMax: 3333000 },
  greenland: { xMin: -652925, xMax: 877075, yMin: -3383675, yMax: -632675 },
};

function projectionMeta(scenario) {
  return JSON.parse(readFileSync(resolve(repoRoot, `static/tools/data/ismip6_2300_mean8_${scenario}_480.meta.json`), "utf8"));
}

/**
 * Each domain cell of a projection package with its change in ice thickness from the first
 * frame to the last (metres), its longitude and its distance from the pole (km).
 */
function thicknessChangeByCell(scenario) {
  const meta = projectionMeta(scenario);
  const bin = readFileSync(resolve(repoRoot, `static/tools/data/ismip6_2300_mean8_${scenario}_480.bin`));
  const fields = Object.fromEntries(meta.fields.map((field) => [field.name, field]));
  assert.equal(fields.thickness.dtype, "int16");
  assert.deepEqual([meta.quantization.scale, meta.quantization.offset], [1, 0]);
  const cells = meta.domain.cell_count;
  const lastFrame = meta.frames.count - 1;
  const { nx, ny, x0_m: x0, y0_m: y0, dx_m: dx, dy_m: dy } = meta.grid;
  const thicknessAt = (frame, cell) => bin.readInt16LE(fields.thickness.byte_offset + 2 * (frame * cells + cell));
  const changes = [];
  for (let row = 0; row < ny; row += 1) {
    for (let col = 0; col < nx; col += 1) {
      if (!bin[fields.domain_mask.byte_offset + row * nx + col]) continue;
      const cell = changes.length;
      const [x, y] = [x0 + col * dx, y0 + row * dy];
      changes.push({
        metres: thicknessAt(lastFrame, cell) - thicknessAt(0, cell),
        lonDeg: (Math.atan2(x, y) * 180) / Math.PI,
        poleKm: Math.hypot(x, y) / 1000,
      });
    }
  }
  assert.equal(changes.length, cells);
  return changes;
}

/** The models' mean sea-level contribution of a projection package in a year, in metres. */
function meanSeaLevelIn(meta, year) {
  const { years, sea_level_contribution_m: values } = meta.series;
  return values[years.indexOf(year)];
}

function nonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function assertSources(sources, label) {
  assert.ok(Array.isArray(sources), `${label}: sources is a list`);
  for (const source of sources) {
    assert.ok(nonEmptyString(source.text), `${label}: source text`);
    if (source.url !== undefined) assert.match(source.url, /^https:\/\//, `${label}: source url`);
  }
}

test("both locales have the same user-interface strings", () => {
  const english = Object.keys(EXPLORE_CONTENT["en-US"].ui).sort();
  assert.deepEqual(Object.keys(EXPLORE_CONTENT["zh-CN"].ui).sort(), english);
  for (const locale of LOCALES) {
    for (const [key, value] of Object.entries(EXPLORE_CONTENT[locale].ui)) {
      assert.ok(nonEmptyString(value), `${locale} ui.${key}`);
    }
  }
});

test("every tour stop is written in both locales, with sources", () => {
  assert.ok(TOUR_CHAPTERS.length >= 6 && TOUR_CHAPTERS.length <= 9, "six to nine stops");
  const ids = TOUR_CHAPTERS.map((chapter) => chapter.id);
  assert.equal(new Set(ids).size, ids.length, "stop ids are unique");
  for (const locale of LOCALES) {
    const chapters = EXPLORE_CONTENT[locale].chapters;
    assert.deepEqual(Object.keys(chapters).sort(), [...ids].sort(), `${locale} stops`);
    for (const id of ids) {
      const chapter = chapters[id];
      assert.ok(nonEmptyString(chapter.title), `${locale} ${id} title`);
      assert.ok(Array.isArray(chapter.body) && chapter.body.length > 0, `${locale} ${id} body`);
      chapter.body.forEach((paragraph, index) => assert.ok(nonEmptyString(paragraph), `${locale} ${id} body[${index}]`));
      assertSources(chapter.sources, `${locale} ${id}`);
    }
  }
});

test("both locales cite the same sources for each stop and explainer", () => {
  const urls = (entry) => entry.sources.map((source) => source.url || "").join("|");
  for (const id of TOUR_CHAPTERS.map((chapter) => chapter.id)) {
    assert.equal(urls(EXPLORE_CONTENT["zh-CN"].chapters[id]), urls(EXPLORE_CONTENT["en-US"].chapters[id]), id);
  }
  for (const id of Object.keys(EXPLORE_CONTENT["en-US"].info)) {
    assert.equal(urls(EXPLORE_CONTENT["zh-CN"].info[id]), urls(EXPLORE_CONTENT["en-US"].info[id]), id);
  }
});

test("every tour stop drives only the controls the public pages show", () => {
  for (const chapter of TOUR_CHAPTERS) {
    assert.doesNotThrow(() => resolveChapterControls(chapter.view.controls), chapter.id);
    for (const [locale, page] of Object.entries(PUBLIC_PAGES)) {
      const ids = [
        ...Object.keys(TOUR_CONTROL_DEFAULTS),
        ...Object.keys(chapter.view.sliders || {}),
        ...Object.keys(chapter.view.menus || {}),
      ];
      for (const id of ids) {
        assert.match(page, new RegExp(`id="${id}"`), `${locale} page lacks #${id} used by stop ${chapter.id}`);
      }
    }
  }
});

test("every tour stop frames a point on its region's grid, from a sensible angle", () => {
  let region = "antarctica";
  for (const chapter of TOUR_CHAPTERS) {
    region = chapter.view.region || region;
    assert.ok(region in GRID_BOUNDS, `${chapter.id}: region ${region}`);
    const { sliders = {}, camera, animate } = chapter.view;
    if (sliders.exaggeration !== undefined) assert.ok(sliders.exaggeration >= 0.5 && sliders.exaggeration <= 8, chapter.id);
    if (sliders.iceOpacity !== undefined) assert.ok(sliders.iceOpacity >= 0 && sliders.iceOpacity <= 1, chapter.id);
    if (camera !== "default") {
      const { x, y } = projectLatLon(camera.lat, camera.lon, region);
      const bounds = GRID_BOUNDS[region];
      assert.ok(x >= bounds.xMin && x <= bounds.xMax && y >= bounds.yMin && y <= bounds.yMax, `${chapter.id} is on the grid`);
      const span = camera.fitKm ?? camera.distanceKm;
      assert.ok(span > 50 && span < 8000, `${chapter.id} framing`);
      assert.ok(camera.elevationDeg >= 10 && camera.elevationDeg <= 85, `${chapter.id} elevation`);
    }
    if (animate) {
      assert.ok(
        ["reboundProgress", "projectionYear", "iceOpacity", "exaggeration"].includes(animate.control),
        `${chapter.id} animates a slider`
      );
      assert.ok([undefined, "linear"].includes(animate.easing), `${chapter.id} easing`);
      if (animate.control === "reboundProgress") {
        assert.ok(chapter.view.controls.showIsostaticRebound, `${chapter.id} animates a layer it shows`);
      } else if (animate.control === "projectionYear") {
        assert.ok(chapter.view.controls.showIceProjection, `${chapter.id} animates a layer it shows`);
      } else {
        assert.equal(animate.to, sliders[animate.control], `${chapter.id} ends its animation where its sliders say`);
      }
      for (const page of Object.values(PUBLIC_PAGES)) assert.match(page, new RegExp(`id="${animate.control}"`));
      assert.ok(animate.durationMs >= 1000 && animate.durationMs <= 15000, chapter.id);
    }
  }
});

test("the tour visits both regions", () => {
  const regions = new Set(TOUR_CHAPTERS.map((chapter) => chapter.view.region).filter(Boolean));
  assert.deepEqual([...regions].sort(), ["antarctica", "greenland"]);
  assert.equal(TOUR_CHAPTERS[0].view.region, "antarctica", "the tour starts in a known region");
});

test("the tour's melt runs as fast as the first-switch melt, 0 to 100 % in 3 s", () => {
  const stop = TOUR_CHAPTERS.find((chapter) => chapter.id === "without-ice");
  assert.deepEqual(stop.view.animate, { control: "reboundProgress", from: 0, to: 100, durationMs: 3000 });
  const guide = readFileSync(resolve(repoRoot, "static/tools/js/explore-guide.js"), "utf8");
  assert.match(guide, /REBOUND_DEMO = Object\.freeze\(\{ control: "reboundProgress", from: 0, to: 100, durationMs: 3000 \}\)/);
});

test("a stop that shows the projection plays a packaged scenario's whole timeline, without the layers it replaces", () => {
  const stops = TOUR_CHAPTERS.filter((chapter) => chapter.view.controls.showIceProjection);
  assert.ok(stops.length > 0, "the tour shows the projection");
  for (const chapter of stops) {
    const { region, controls, menus = {}, animate } = chapter.view;
    assert.equal(region, "antarctica", chapter.id);
    const scenario = menus.projectionScenario;
    const metaPath = resolve(repoRoot, `static/tools/data/ismip6_2300_mean8_${scenario}_480.meta.json`);
    assert.ok(existsSync(metaPath), `${chapter.id} asks for scenario ${scenario}, which has no package`);
    const { years } = projectionMeta(scenario).series;
    assert.equal(animate?.control, "projectionYear", `${chapter.id} plays the years`);
    assert.deepEqual([animate.from, animate.to], [years[0], years.at(-1)], `${chapter.id} plays the whole timeline`);
    assert.equal(animate.easing, "linear", `${chapter.id} gives every year the same time`);
    // Switching either on would switch the projection off again.
    assert.ok(!controls.showOceanCurrents && !controls.showIsostaticRebound, `${chapter.id} layers`);
    // The projection's flowlines and the sea its floating ice rides on.
    assert.ok(controls.showFlowline && controls.showSea, `${chapter.id} keeps the flowlines and the sea`);
  }
});

test("the high-emission stop quotes the numbers its packages carry", () => {
  const stop = TOUR_CHAPTERS.find((chapter) => chapter.id === "high-emissions");
  assert.equal(stop.view.menus.projectionScenario, "ae05");
  const high = projectionMeta("ae05");
  assert.equal(high.experiment.scenario, "SSP5-8.5");
  assert.equal(high.experiment.shelf_collapse, false);
  assert.equal(high.models.length, 8, "eight models");
  const low = projectionMeta("ae10");
  assert.equal(low.experiment.scenario, "SSP1-2.6");
  const highest = Math.max(...Object.values(high.series.per_model_end_m));
  const lowest = Math.min(...Object.values(high.series.per_model_end_m));
  // About 1.5 m on average, the models from no rise at all to 4 m; about 5 cm with low emissions.
  assert.equal(Math.round(meanSeaLevelIn(high, 2300) * 10) / 10, 1.5);
  assert.equal(Math.round(highest), 4);
  assert.ok(lowest <= 0, `the lowest model ${lowest} m`);
  assert.equal(Math.round(meanSeaLevelIn(low, 2300) * 100), 5);
  // Little change for about a century.
  assert.ok(Math.abs(meanSeaLevelIn(high, 2100)) < 0.05, "little change by 2100");
  // More than a kilometre of thinning in places, mostly in West Antarctica (between the
  // Ross Sea and the Weddell Sea) but also on the East Antarctic coast (north of about 74 S).
  const thinned = thicknessChangeByCell("ae05").filter((cell) => cell.metres < -1000);
  const west = thinned.filter((cell) => cell.lonDeg > -180 && cell.lonDeg < -20);
  const eastCoast = thinned.filter((cell) => cell.lonDeg > 60 && cell.lonDeg < 170 && cell.poleKm > 1800);
  assert.ok(west.length > thinned.length / 2, `${west.length} of ${thinned.length} cells in West Antarctica`);
  assert.ok(eastCoast.length > 0, "and some on the East Antarctic coast");
  const [english, chinese] = ["en-US", "zh-CN"].map((locale) => EXPLORE_CONTENT[locale].chapters["high-emissions"].body.join(" "));
  const englishQuotes = [
    "SSP5-8.5",
    "eight computer models",
    "about 1.5 m",
    "to 4 m",
    "about 5 cm",
    "2015 to 2300",
    "more than a kilometre",
    "mostly in West Antarctica",
    "East Antarctic coast",
  ];
  const chineseQuotes = ["SSP5-8.5", "八个冰盖计算机模型", "约 1.5 米", "上升 4 米", "约 5 厘米", "2300 年", "超过 1 公里", "大多在西南极", "东南极沿岸"];
  for (const quoted of englishQuotes) assert.ok(english.includes(quoted), quoted);
  for (const quoted of chineseQuotes) assert.ok(chinese.includes(quoted), quoted);
});

test("the last stop ends the tour with a choice of the regions the tour visits", () => {
  const last = TOUR_CHAPTERS.at(-1);
  const visited = [...new Set(TOUR_CHAPTERS.map((chapter) => chapter.view.region).filter(Boolean))].sort();
  assert.deepEqual([...last.startRegions].sort(), visited);
  assert.ok(TOUR_CHAPTERS.slice(0, -1).every((chapter) => !chapter.startRegions), "only the last stop ends the tour");
  for (const locale of LOCALES) {
    const copy = EXPLORE_CONTENT[locale];
    for (const region of last.startRegions) {
      assert.ok(nonEmptyString(copy.chapters[last.id].startRegions?.[region]), `${locale} ${region} label`);
    }
    assert.match(copy.ui.startIn, /\{region\}/, `${locale} ui.startIn names the region`);
    assert.match(copy.ui.readoutYear, /\{year\}/, `${locale} ui.readoutYear`);
    assert.match(copy.ui.readoutSeaLevel, /\{value\}/, `${locale} ui.readoutSeaLevel`);
  }
});

test("every explainer a public page asks for is written in both locales", () => {
  const infoIds = Object.keys(EXPLORE_CONTENT["en-US"].info).sort();
  assert.deepEqual(Object.keys(EXPLORE_CONTENT["zh-CN"].info).sort(), infoIds);
  for (const [locale, page] of Object.entries(PUBLIC_PAGES)) {
    const requested = [...page.matchAll(/data-info="([A-Za-z]+)"/g)].map((match) => match[1]);
    assert.ok(requested.length >= 8, `${locale} page has explainers`);
    for (const id of requested) assert.ok(infoIds.includes(id), `${locale} page asks for unknown explainer ${id}`);
    for (const id of infoIds) assert.ok(requested.includes(id), `${locale} page never shows explainer ${id}`);
  }
  for (const locale of LOCALES) {
    for (const [id, info] of Object.entries(EXPLORE_CONTENT[locale].info)) {
      assert.ok(nonEmptyString(info.title), `${locale} info.${id} title`);
      assert.ok(Array.isArray(info.body) && info.body.every(nonEmptyString), `${locale} info.${id} body`);
      assertSources(info.sources, `${locale} info.${id}`);
    }
  }
});

test("every explainer button on a public page has exactly one card", () => {
  for (const [locale, page] of Object.entries(PUBLIC_PAGES)) {
    const buttons = [...page.matchAll(/<button[^>]*data-info="([A-Za-z]+)"/g)].map((match) => match[1]);
    const cards = [...page.matchAll(/data-info-card="([A-Za-z]+)"/g)].map((match) => match[1]);
    assert.deepEqual([...cards].sort(), [...buttons].sort(), `${locale} buttons and cards pair up`);
    assert.equal(new Set(buttons).size, buttons.length, `${locale} has no duplicate explainers`);
  }
});

test("the tour launcher's copy names the number of stops the tour has", () => {
  const english = { 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine" }[TOUR_CHAPTERS.length];
  const chinese = { 6: "六", 7: "七", 8: "八", 9: "九" }[TOUR_CHAPTERS.length];
  assert.match(PUBLIC_PAGES["en-US"], new RegExp(`${english} short stops`));
  assert.match(PUBLIC_PAGES["zh-CN"], new RegExp(`${chinese}个简短的站点`));
});

test("content falls back to English for an unknown locale", () => {
  assert.equal(getExploreContent("fr-FR"), EXPLORE_CONTENT["en-US"]);
  assert.equal(getExploreContent("zh-CN"), EXPLORE_CONTENT["zh-CN"]);
});

test("copy stays plain text, with no markup to escape", () => {
  const walk = (value, path) => {
    if (typeof value === "string") {
      assert.doesNotMatch(value, /[<>]/, `${path} contains markup`);
      assert.doesNotMatch(value, /TODO|TBD|lorem/i, `${path} is unfinished`);
      return;
    }
    if (value && typeof value === "object") {
      for (const [key, child] of Object.entries(value)) walk(child, `${path}.${key}`);
    }
  };
  walk(EXPLORE_CONTENT, "content");
});
