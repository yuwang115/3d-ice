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
  getPolarFeatureMarkerShape,
  getPolarFeatureMarkerStyle,
  POLAR_LABEL_STYLES,
} from "../../static/tools/js/polar-feature-label-style.js";
import {
  formatCoreDepth,
  formatElevation,
  formatOldestIce,
  POLAR_FEATURE_TEXT,
  polarFeatureDetailRows,
  polarFeatureTypeLabel,
  renderPolarFeatureDetails,
  safeSourceUrl,
} from "../../static/tools/js/polar-feature-details.js";
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

const ICE_CORE = {
  id: "antarctica-ice-core-wais-divide",
  region: "antarctica",
  layer: "ice_cores",
  kind: "ice_core",
  core_type: "deep",
  feature_type: "Deep ice core",
  name: "WAIS Divide",
  name_zh: "",
  aliases: ["WDC06A", "冰芯"],
  project: "WAIS Divide Ice Core Project",
  operator: "United States",
  drilled: "2006–2011",
  depth_m: 3405,
  reached_bed: false,
  oldest_ka: 68,
  elevation_m: 1766,
  lat: -79.468,
  lon: -112.086,
  citation: "WAIS Divide Project Members 2013, Nature 500, 440–444",
  source_url: "https://doi.org/10.1038/nature12376",
  display_priority: 1,
};

test("ice cores are found by name, core ID, kind and the Chinese term", () => {
  const features = [...FEATURES, ICE_CORE];
  assert.equal(searchPolarFeatures(features, "WDC06A")[0]?.id, ICE_CORE.id);
  assert.equal(searchPolarFeatures(features, "ice core")[0]?.id, ICE_CORE.id);
  assert.equal(searchPolarFeatures(features, "deep ice core")[0]?.id, ICE_CORE.id);
  assert.equal(searchPolarFeatures(features, "冰芯")[0]?.id, ICE_CORE.id);
});

test("ice-core details give depth, age and elevation in each locale", () => {
  const en = POLAR_FEATURE_TEXT.en;
  const zh = POLAR_FEATURE_TEXT.zh;
  assert.equal(formatCoreDepth(ICE_CORE, en, "en-US"), "3,405 m");
  assert.equal(formatCoreDepth({ ...ICE_CORE, reached_bed: true }, en, "en-US"), "3,405 m, to the bed");
  assert.equal(formatCoreDepth({ ...ICE_CORE, reached_bed: true }, zh, "zh-CN"), "3,405 米（钻至冰床）");
  assert.equal(formatCoreDepth({ ...ICE_CORE, depth_m: null }, en, "en-US"), "");
  assert.equal(formatCoreDepth({ ...ICE_CORE, depth_m: 0 }, en, "en-US"), "");
  assert.equal(formatOldestIce(ICE_CORE, en, "en-US"), "≈68,000 years");
  assert.equal(formatOldestIce({ ...ICE_CORE, oldest_ka: 800 }, en, "en-US"), "≈800,000 years");
  assert.equal(formatOldestIce({ ...ICE_CORE, oldest_ka: 2700 }, en, "en-US"), "≈2.7 million years");
  assert.equal(formatOldestIce({ ...ICE_CORE, oldest_ka: 800 }, zh, "zh-CN"), "约 80 万年");
  assert.equal(formatOldestIce({ ...ICE_CORE, oldest_ka: 68 }, zh, "zh-CN"), "约 6.8 万年");
  assert.equal(formatOldestIce({ ...ICE_CORE, oldest_ka: 2.5 }, zh, "zh-CN"), "约 2,500 年");
  assert.equal(formatOldestIce({ ...ICE_CORE, oldest_ka: null }, en, "en-US"), "");
  assert.equal(formatElevation(ICE_CORE, en, "en-US"), "1,766 m");
  assert.equal(formatElevation({ ...ICE_CORE, elevation_m: "" }, en, "en-US"), "");
});

test("detail rows list an ice core's project, drilling and record before its position", () => {
  const rows = polarFeatureDetailRows(ICE_CORE, POLAR_FEATURE_TEXT.en, "en-US");
  assert.deepEqual(rows.map(([term]) => term), [
    "Project",
    "Operator",
    "Drilled",
    "Core depth",
    "Continuous record",
    "Surface elevation",
    "Coordinates",
  ]);
  assert.deepEqual(rows.at(-1), ["Coordinates", "-79.468°, -112.086°"]);
  assert.deepEqual(
    polarFeatureDetailRows({ ...ICE_CORE, core_type: "blue_ice", oldest_ka: 134 }, POLAR_FEATURE_TEXT.zh, "zh-CN")[4],
    ["最老定年冰", "约 13.4 万年"],
  );

  const station = polarFeatureDetailRows(
    { ...FEATURES[0], status: "temporarily_closed", lat: -90, lon: 0, area_km2: 12.4 },
    POLAR_FEATURE_TEXT.en,
    "en-US",
  );
  assert.deepEqual(station.map(([term]) => term), ["Operator", "Country", "Status", "Coordinates", "Area"]);
  assert.deepEqual(station[2], ["Status", "Temporarily closed"]);
  assert.deepEqual(station[4], ["Area", "12 km²"]);
});

test("an ice core is typed by its kind of core, other features by their kind", () => {
  assert.equal(polarFeatureTypeLabel(ICE_CORE, POLAR_FEATURE_TEXT.en), "Deep ice core");
  assert.equal(polarFeatureTypeLabel({ ...ICE_CORE, core_type: "blue_ice" }, POLAR_FEATURE_TEXT.zh), "蓝冰区钻探点");
  assert.equal(polarFeatureTypeLabel({ ...ICE_CORE, core_type: "unknown" }, POLAR_FEATURE_TEXT.en), "Ice core");
  assert.equal(polarFeatureTypeLabel(FEATURES[2], POLAR_FEATURE_TEXT.en), "Mountain range");
  assert.equal(polarFeatureTypeLabel({ kind: "glacier" }, POLAR_FEATURE_TEXT.en), "glacier");
});

test("source links accept DOIs and the catalogue hosts only", () => {
  assert.equal(safeSourceUrl("https://doi.org/10.1038/nature12376"), "https://doi.org/10.1038/nature12376");
  assert.equal(safeSourceUrl("https://placenames.aq/information"), "https://placenames.aq/information");
  assert.equal(safeSourceUrl("http://doi.org/10.1038/nature12376"), "");
  assert.equal(safeSourceUrl("https://example.org/"), "");
  assert.equal(safeSourceUrl("javascript:alert(1)"), "");
  assert.equal(safeSourceUrl(""), "");
});

function createDetailsHarness() {
  const createElement = (tagName) => {
    const element = {
      tagName,
      children: [],
      textContent: "",
      append(...nodes) {
        this.children.push(...nodes);
      },
      replaceChildren(...nodes) {
        this.children = nodes;
      },
    };
    return element;
  };
  const container = createElement("div");
  container.ownerDocument = { createElement };
  container.hidden = true;
  return container;
}

test("the details card names the core's publication and hides when nothing is selected", () => {
  const container = createDetailsHarness();
  const options = { labels: POLAR_FEATURE_TEXT.en, locale: "en-US", getLabel: getPolarFeatureLabel };
  renderPolarFeatureDetails(container, ICE_CORE, options);
  assert.equal(container.hidden, false);
  const [heading, eyebrow, list, link] = container.children;
  assert.equal(heading.textContent, "WAIS Divide");
  assert.equal(eyebrow.textContent, "Deep ice core · Antarctica");
  assert.equal(list.children.length, 14);
  assert.equal(link.href, "https://doi.org/10.1038/nature12376");
  assert.equal(link.rel, "noopener noreferrer");
  assert.equal(link.textContent, "Reference: WAIS Divide Project Members 2013, Nature 500, 440–444");

  renderPolarFeatureDetails(container, ICE_CORE, { ...options, labels: POLAR_FEATURE_TEXT.zh, locale: "zh-CN" });
  assert.equal(container.children.at(-1).textContent, "参考文献：WAIS Divide Project Members 2013, Nature 500, 440–444");
  assert.equal(container.children[1].textContent, "深冰芯 · 南极洲");

  renderPolarFeatureDetails(container, { ...FEATURES[0], source_url: "https://www.comnap.aq/x" }, options);
  assert.equal(container.children.at(-1).textContent, "Source");

  renderPolarFeatureDetails(container, { ...ICE_CORE, source_url: "https://example.org/" }, options);
  assert.equal(container.children.length, 3);

  renderPolarFeatureDetails(container, null, options);
  assert.equal(container.hidden, true);
  assert.equal(container.children.length, 0);
});

test("ice-core markers are pale lilac rings, distinct from station and place dots", () => {
  const coreStyle = getPolarFeatureMarkerStyle("ice_cores");
  const coreColor = hexSaturationAndLightness(coreStyle.color);
  assert.equal(coreStyle.color, POLAR_LABEL_STYLES.ice_cores.textColor);
  assert.ok(coreColor.saturation <= 0.45 && coreColor.lightness >= 0.68);
  assert.ok(coreColor.hue >= 250 && coreColor.hue <= 290);
  for (const other of ["research_stations", "geographic_names"]) {
    const otherHue = hexSaturationAndLightness(getPolarFeatureMarkerStyle(other).color).hue;
    const distance = Math.min(Math.abs(coreColor.hue - otherHue), 360 - Math.abs(coreColor.hue - otherHue));
    assert.ok(distance >= 50, `${other} is too close in hue`);
  }
  assert.ok(contrastRatio(coreStyle.color, "#07111a") >= 3);
  assert.ok(coreStyle.size > getPolarFeatureMarkerStyle("research_stations").size);
  assert.equal(getPolarFeatureMarkerShape("ice_cores"), "ring");
  assert.equal(getPolarFeatureMarkerShape("research_stations"), "dot");
  assert.equal(getPolarFeatureMarkerShape("unknown"), "dot");

  const calls = [];
  const context = {
    clearRect: () => calls.push("clearRect"),
    beginPath: () => calls.push("beginPath"),
    arc: (...args) => calls.push(["arc", ...args]),
    fill: () => calls.push("fill"),
    stroke: () => calls.push("stroke"),
  };
  const canvas = createPolarMarkerCanvas({ createElement: () => ({ getContext: () => context }) }, { shape: "ring" });
  assert.equal(canvas.width, 64);
  assert.deepEqual(calls.map((call) => (Array.isArray(call) ? call[0] : call)), ["clearRect", "beginPath", "arc", "stroke"]);
  // The ring's outer edge stays inside the canvas and leaves a hole at least a third wide.
  const [, , , radius] = calls.find((call) => Array.isArray(call));
  assert.ok(radius + context.lineWidth / 2 <= 31);
  assert.ok((radius - context.lineWidth / 2) * 2 >= 64 / 3);
});
