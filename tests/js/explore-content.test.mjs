/**
 * The public edition's copy lives in one module, in both locales. Every tour stop and every
 * layer explainer must exist in both, every stop must only drive controls the public pages
 * actually show, and every explainer a page asks for must exist.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
  assert.ok(TOUR_CHAPTERS.length >= 6 && TOUR_CHAPTERS.length <= 8, "six to eight stops");
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
      for (const id of [...Object.keys(TOUR_CONTROL_DEFAULTS), ...Object.keys(chapter.view.sliders || {})]) {
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
      assert.ok(["reboundProgress", "iceOpacity", "exaggeration"].includes(animate.control), `${chapter.id} animates a slider`);
      if (animate.control === "reboundProgress") {
        assert.ok(chapter.view.controls.showIsostaticRebound, `${chapter.id} animates a layer it shows`);
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
  const english = { 6: "Six", 7: "Seven", 8: "Eight" }[TOUR_CHAPTERS.length];
  const chinese = { 6: "六", 7: "七", 8: "八" }[TOUR_CHAPTERS.length];
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
