import assert from "node:assert/strict";
import test from "node:test";

import {
  getPolarFeatureLabel,
  normalizeSearchText,
  searchPolarFeatures,
} from "../../static/tools/js/polar-feature-search.js";

const FEATURES = [
  {
    id: "antarctica-station-amundsen-scott",
    region: "antarctica",
    layer: "research_stations",
    kind: "research_station",
    name: "Amundsen–Scott South Pole Station",
    name_zh: "阿蒙森-斯科特南极站",
    aliases: ["South Pole Station"],
    operator: "United States Antarctic Program",
    country: "United States",
    display_priority: 1,
  },
  {
    id: "greenland-station-summit",
    region: "greenland",
    layer: "research_stations",
    kind: "research_station",
    name: "Summit Station",
    name_zh: "格陵兰峰顶站",
    aliases: ["GEOSummit"],
    operator: "National Science Foundation",
    country: "United States",
    display_priority: 1,
  },
  {
    id: "antarctica-place-transantarctic-mountains",
    region: "antarctica",
    layer: "geographic_names",
    kind: "mountain_range",
    name: "Transantarctic Mountains",
    name_zh: "横贯南极山脉",
    aliases: [],
    display_priority: 1,
  },
];

test("normalizeSearchText handles punctuation, accents, and Chinese text", () => {
  assert.equal(normalizeSearchText("  Amundsen–Scott  "), "amundsen scott");
  assert.equal(normalizeSearchText("Sør Rondane"), "sor rondane");
  assert.equal(normalizeSearchText("横贯南极山脉"), "横贯南极山脉");
  assert.equal(normalizeSearchText("[Summit]*"), "summit");
});

test("searchPolarFeatures searches names, aliases, operators, countries, and kinds", () => {
  assert.equal(searchPolarFeatures(FEATURES, "GEOSummit")[0]?.id, "greenland-station-summit");
  assert.equal(searchPolarFeatures(FEATURES, "National Science Foundation")[0]?.id, "greenland-station-summit");
  assert.equal(searchPolarFeatures(FEATURES, "横贯南极山脉")[0]?.id, "antarctica-place-transantarctic-mountains");
  assert.equal(searchPolarFeatures(FEATURES, "mountain range")[0]?.id, "antarctica-place-transantarctic-mountains");
});

test("searchPolarFeatures ranks exact and prefix matches ahead of substring matches", () => {
  const ranked = searchPolarFeatures(
    [
      ...FEATURES,
      {
        ...FEATURES[1],
        id: "greenland-place-summit-lake",
        layer: "geographic_names",
        kind: "lake",
        name: "Summit Lake",
      },
    ],
    "summit",
  );

  assert.deepEqual(
    ranked.map((feature) => feature.id),
    ["greenland-station-summit", "greenland-place-summit-lake"],
  );
});

test("searchPolarFeatures supports filters, limits, and empty queries", () => {
  assert.deepEqual(searchPolarFeatures(FEATURES, ""), []);
  assert.deepEqual(
    searchPolarFeatures(FEATURES, "station", { region: "greenland", layer: "research_stations", limit: 1 }).map(
      (feature) => feature.id,
    ),
    ["greenland-station-summit"],
  );
});

test("getPolarFeatureLabel prefers Chinese and falls back to the primary name", () => {
  assert.equal(getPolarFeatureLabel(FEATURES[2], "zh-CN"), "横贯南极山脉");
  assert.equal(getPolarFeatureLabel({ ...FEATURES[2], name_zh: "" }, "zh-CN"), "Transantarctic Mountains");
  assert.equal(getPolarFeatureLabel(FEATURES[2], "en-US"), "Transantarctic Mountains");
});
