import assert from "node:assert/strict";
import test from "node:test";

import {
  createRefinedBasinSearchFeatures,
  getPolarFeatureLabel,
  normalizeSearchText,
  searchPolarFeatures,
} from "../../static/tools/js/polar-feature-search.js";
import {
  createPolarMarkerCanvas,
  createPolarLabelCanvas,
  getPolarFeatureLabelStyle,
  getPolarFeatureMarkerMaterialOptions,
  getPolarFeatureMarkerStyle,
  POLAR_LABEL_STYLES,
} from "../../static/tools/js/polar-feature-label-style.js";
import {
  fetchRefinedBasinJson,
  MAX_REFINED_BASIN_RESPONSE_BYTES,
  validateRefinedBasinDataset,
} from "../../static/tools/js/polar-refined-basins.js";

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

test("createRefinedBasinSearchFeatures maps basin label points into searchable features", () => {
  const payload = {
    basins: [
      { id: 1, name: "Abbot", label_xy_m: [-1786728, -112853], area_km2: 26340.5 },
      { id: 2, name: "", label_xy_m: [0, 0] },
      { id: 3, name: "Missing point" },
      { id: {}, name: "Object ID", label_xy_m: [0, 0] },
      { id: 4, name: {}, label_xy_m: [0, 0] },
      { id: 5, name: "Coerced coordinates", label_xy_m: [null, "0"] },
    ],
  };
  const originalPayload = structuredClone(payload);

  const features = createRefinedBasinSearchFeatures(payload, "antarctica");

  assert.equal(features.length, 1);
  assert.deepEqual(
    {
      id: features[0].id,
      layer: features[0].layer,
      kind: features[0].kind,
      region: features[0].region,
      name: features[0].name,
      x_m: features[0].x_m,
      y_m: features[0].y_m,
    },
    {
      id: "antarctica-refined-basin-1",
      layer: "refined_basins",
      kind: "refined_basin",
      region: "antarctica",
      name: "Abbot",
      x_m: -1786728,
      y_m: -112853,
    },
  );
  assert.equal(searchPolarFeatures(features, "Abbot refined basin")[0]?.id, features[0].id);
  assert.equal(searchPolarFeatures(features, "refined basins")[0]?.id, features[0].id);
  assert.deepEqual(payload, originalPayload);
  assert.throws(() => createRefinedBasinSearchFeatures(payload, "arctic"), /region/i);
  assert.throws(
    () => createRefinedBasinSearchFeatures({ basins: Array.from({ length: 1_001 }, () => ({})) }, "antarctica"),
    /exceeds/i,
  );
});

function createCanvasHarness() {
  const calls = [];
  const context = {
    measureText: (text) => ({ width: String(text).length * 18 }),
    strokeText: (...args) => calls.push(["strokeText", ...args]),
    fillText: (...args) => calls.push(["fillText", ...args]),
  };
  const canvas = {
    width: 0,
    height: 0,
    getContext: () => context,
  };
  return {
    calls,
    canvas,
    documentRef: {
      createElement: (tagName) => {
        assert.equal(tagName, "canvas");
        return canvas;
      },
    },
  };
}

test("polar labels use refined-basin-style outlined text with distinct layer colors", () => {
  const stationHarness = createCanvasHarness();
  const namesHarness = createCanvasHarness();

  createPolarLabelCanvas(stationHarness.documentRef, "South Pole Station", { layer: "research_stations" });
  createPolarLabelCanvas(namesHarness.documentRef, "Transantarctic Mountains", { layer: "geographic_names" });

  assert.deepEqual(stationHarness.calls.map(([method]) => method), ["strokeText", "fillText"]);
  assert.deepEqual(namesHarness.calls.map(([method]) => method), ["strokeText", "fillText"]);
  assert.notEqual(
    POLAR_LABEL_STYLES.research_stations.textColor,
    POLAR_LABEL_STYLES.geographic_names.textColor,
  );
  assert.match(POLAR_LABEL_STYLES.research_stations.textColor, /^#[0-9a-f]{6}$/i);
  assert.match(POLAR_LABEL_STYLES.geographic_names.textColor, /^#[0-9a-f]{6}$/i);
  assert.deepEqual(
    {
      fontSize: getPolarFeatureLabelStyle("research_stations", "antarctica").fontSize,
      strokeWidth: getPolarFeatureLabelStyle("research_stations", "antarctica").strokeWidth,
      worldScale: getPolarFeatureLabelStyle("research_stations", "antarctica").worldScale,
    },
    { fontSize: 44, strokeWidth: 7, worldScale: 0.0076 },
  );
  assert.deepEqual(
    {
      fontSize: getPolarFeatureLabelStyle("geographic_names", "greenland").fontSize,
      strokeWidth: getPolarFeatureLabelStyle("geographic_names", "greenland").strokeWidth,
      worldScale: getPolarFeatureLabelStyle("geographic_names", "greenland").worldScale,
    },
    { fontSize: 56, strokeWidth: 8, worldScale: 0.0104 },
  );
  assert.equal(
    getPolarFeatureLabelStyle("research_stations", "antarctica", { selected: true }).textColor,
    POLAR_LABEL_STYLES.research_stations.textColor,
  );
  assert.equal(
    getPolarFeatureLabelStyle("unknown", "unknown", { selected: true }).textColor,
    POLAR_LABEL_STYLES.geographic_names.textColor,
  );
  assert.equal(
    createPolarLabelCanvas({ createElement: () => ({ getContext: () => null }) }, "Unavailable"),
    null,
  );
});

function hexSaturationAndLightness(hexColor) {
  const channels = hexColor.slice(1).match(/.{2}/g).map((value) => Number.parseInt(value, 16) / 255);
  const maximum = Math.max(...channels);
  const minimum = Math.min(...channels);
  const lightness = (maximum + minimum) / 2;
  const saturation = maximum === minimum
    ? 0
    : (maximum - minimum) / (1 - Math.abs(2 * lightness - 1));
  let hue = 0;
  if (maximum !== minimum) {
    const delta = maximum - minimum;
    if (maximum === channels[0]) hue = ((channels[1] - channels[2]) / delta) % 6;
    else if (maximum === channels[1]) hue = (channels[2] - channels[0]) / delta + 2;
    else hue = (channels[0] - channels[1]) / delta + 4;
    hue = (hue * 60 + 360) % 360;
  }
  return { hue, saturation, lightness };
}

function contrastRatio(hexLeft, hexRight) {
  const luminance = (hex) => {
    const channels = hex.slice(1).match(/.{2}/g).map((value) => Number.parseInt(value, 16) / 255);
    const linear = channels.map((value) => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
    return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
  };
  const left = luminance(hexLeft);
  const right = luminance(hexRight);
  return (Math.max(left, right) + 0.05) / (Math.min(left, right) + 0.05);
}

test("station and geographic markers use muted pale colors, circular dots, and half-size radii", () => {
  const stationStyle = getPolarFeatureMarkerStyle("research_stations");
  const geographicStyle = getPolarFeatureMarkerStyle("geographic_names");
  const stationColor = hexSaturationAndLightness(stationStyle.color);
  const geographicColor = hexSaturationAndLightness(geographicStyle.color);

  assert.equal(stationStyle.color, POLAR_LABEL_STYLES.research_stations.textColor);
  assert.equal(geographicStyle.color, POLAR_LABEL_STYLES.geographic_names.textColor);
  assert.ok(stationColor.saturation <= 0.45 && stationColor.lightness >= 0.68);
  assert.ok(geographicColor.saturation <= 0.45 && geographicColor.lightness >= 0.68);
  assert.ok(stationColor.hue >= 35 && stationColor.hue <= 60);
  assert.ok(geographicColor.hue >= 190 && geographicColor.hue <= 220);
  const hueDistance = Math.min(
    Math.abs(stationColor.hue - geographicColor.hue),
    360 - Math.abs(stationColor.hue - geographicColor.hue),
  );
  assert.ok(hueDistance >= 80);
  assert.ok(contrastRatio(stationStyle.color, "#07111a") >= 3);
  assert.ok(contrastRatio(geographicStyle.color, "#07111a") >= 3);
  assert.equal(stationStyle.size, 3.75);
  assert.equal(geographicStyle.size, 2.75);
  assert.equal(
    getPolarFeatureLabelStyle("research_stations", "antarctica", { selected: true }).textColor,
    stationStyle.color,
  );
  assert.equal(POLAR_LABEL_STYLES.refined_basins.textColor, "#ffe8a3");

  const markerTexture = { texture: true };
  assert.deepEqual(getPolarFeatureMarkerMaterialOptions("research_stations", markerTexture), {
    color: stationStyle.color,
    size: 3.75,
    map: markerTexture,
    alphaTest: 0.05,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.96,
    depthTest: false,
    depthWrite: false,
  });

  const calls = [];
  const canvas = createPolarMarkerCanvas({
    createElement: () => ({
      width: 0,
      height: 0,
      getContext: () => ({
        clearRect: (...args) => calls.push(["clearRect", ...args]),
        beginPath: () => calls.push("beginPath"),
        arc: (...args) => calls.push(["arc", ...args]),
        fill: () => calls.push("fill"),
      }),
    }),
  });
  assert.equal(canvas.width, canvas.height);
  assert.deepEqual(calls.map((call) => Array.isArray(call) ? call[0] : call), ["clearRect", "beginPath", "arc", "fill"]);
  const arcCall = calls.find((call) => Array.isArray(call) && call[0] === "arc");
  assert.deepEqual(arcCall, ["arc", 32, 32, 30, 0, Math.PI * 2]);
  assert.ok(arcCall[3] < canvas.width / 2);
  assert.equal(createPolarMarkerCanvas({ createElement: () => ({ getContext: () => null }) }), null);
});

const VALID_REFINED_BASIN_PAYLOAD = {
  basin_count: 1,
  basins: [
    {
      id: "CE",
      name: "Central East",
      label_xy_m: [10, 20],
      area_km2: 123,
      segments_xy_m: [[[0, 0], [10, 20]]],
    },
  ],
};

test("validateRefinedBasinDataset accepts bounded geometry and rejects unsafe shapes", () => {
  assert.equal(validateRefinedBasinDataset(VALID_REFINED_BASIN_PAYLOAD), VALID_REFINED_BASIN_PAYLOAD);
  assert.throws(() => validateRefinedBasinDataset(null), /invalid/i);
  assert.throws(
    () => validateRefinedBasinDataset({ basin_count: 1_001, basins: Array.from({ length: 1_001 }, () => ({})) }),
    /exceeds/i,
  );
  assert.throws(
    () => validateRefinedBasinDataset({ ...VALID_REFINED_BASIN_PAYLOAD, basins: [] }),
    /basin_count/i,
  );
  assert.throws(
    () =>
      validateRefinedBasinDataset({
        basin_count: 2,
        basins: [VALID_REFINED_BASIN_PAYLOAD.basins[0], VALID_REFINED_BASIN_PAYLOAD.basins[0]],
      }),
    /duplicate/i,
  );
  assert.throws(
    () =>
      validateRefinedBasinDataset({
        basin_count: 1,
        basins: [{ ...VALID_REFINED_BASIN_PAYLOAD.basins[0], segments_xy_m: [[[0, Number.NaN]]] }],
      }),
    /coordinate/i,
  );
  assert.throws(
    () =>
      validateRefinedBasinDataset({
        basin_count: 1,
        basins: [{ ...VALID_REFINED_BASIN_PAYLOAD.basins[0], area_km2: -1 }],
      }),
    /area/i,
  );
  assert.throws(
    () =>
      validateRefinedBasinDataset({
        basin_count: 1,
        basins: [{ ...VALID_REFINED_BASIN_PAYLOAD.basins[0], segments_xy_m: null }],
      }),
    /segment count/i,
  );
  assert.throws(
    () =>
      validateRefinedBasinDataset({
        basin_count: 1,
        basins: [{ ...VALID_REFINED_BASIN_PAYLOAD.basins[0], id: {} }],
      }),
    /ID/i,
  );
  assert.throws(
    () =>
      validateRefinedBasinDataset({
        basin_count: 1,
        basins: [{ ...VALID_REFINED_BASIN_PAYLOAD.basins[0], label_xy_m: [null, "20"] }],
      }),
    /coordinate/i,
  );
});

test("fetchRefinedBasinJson enforces response byte limits before parsing", async () => {
  const response = new Response(JSON.stringify(VALID_REFINED_BASIN_PAYLOAD), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
  const loaded = await fetchRefinedBasinJson("/basins.json", "Basin load failed", {
    fetchImpl: async () => response,
  });
  assert.equal(loaded.basin_count, 1);

  const declaredOversize = new Response("{}", {
    status: 200,
    headers: { "content-length": String(MAX_REFINED_BASIN_RESPONSE_BYTES + 1) },
  });
  await assert.rejects(
    fetchRefinedBasinJson("/oversize.json", "Basin load failed", { fetchImpl: async () => declaredOversize }),
    /size limit/i,
  );
  await assert.rejects(
    fetchRefinedBasinJson("/stream-oversize.json", "Basin load failed", {
      fetchImpl: async () => new Response("12345", { status: 200 }),
      maximumBytes: 3,
    }),
    /size limit/i,
  );
  await assert.rejects(
    fetchRefinedBasinJson("/server-error.json", "Basin load failed", {
      fetchImpl: async () => new Response("", { status: 503 }),
    }),
    /503/,
  );
  await assert.rejects(
    fetchRefinedBasinJson("/invalid.json", "Basin load failed", {
      fetchImpl: async () => new Response("not-json", { status: 200 }),
    }),
    /invalid JSON/i,
  );

  const fallbackBytes = new TextEncoder().encode(JSON.stringify(VALID_REFINED_BASIN_PAYLOAD));
  await assert.rejects(
    fetchRefinedBasinJson("/unbounded-response.json", "Basin load failed", {
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        headers: { get: () => String(fallbackBytes.byteLength) },
        body: null,
        arrayBuffer: async () => fallbackBytes.buffer,
      }),
    }),
    /bounded streaming/i,
  );
});
