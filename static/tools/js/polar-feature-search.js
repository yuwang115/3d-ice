const DEFAULT_LIMIT = 12;
const SEARCH_FIELDS = ["name", "name_zh", "operator", "additional_operator", "country", "feature_type", "kind"];
const MAX_REFINED_BASINS = 1_000;
const VALID_REGIONS = new Set(["antarctica", "greenland"]);

export function polarFeatureLanguageKey(locale) {
  return String(locale || "").toLowerCase().startsWith("zh") ? "zh" : "en";
}

export function normalizeSearchText(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[øØ]/g, "o")
    .replace(/[łŁ]/g, "l")
    .replace(/[ðÐ]/g, "d")
    .replace(/[þÞ]/g, "th")
    .replace(/[æÆ]/g, "ae")
    .replace(/[œŒ]/g, "oe")
    .toLocaleLowerCase("en-US")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function getPolarFeatureLabel(feature, locale = "en-US") {
  if (polarFeatureLanguageKey(locale) === "zh" && String(feature?.name_zh || "").trim()) {
    return String(feature.name_zh).trim();
  }
  return String(feature?.name || "").trim();
}

export function createRefinedBasinSearchFeatures(payload, region) {
  if (!VALID_REGIONS.has(region)) throw new Error(`Invalid refined basin region: ${region}`);
  const basins = Array.isArray(payload?.basins) ? payload.basins : [];
  if (basins.length > MAX_REFINED_BASINS) {
    throw new Error(`Refined basin catalogue exceeds ${MAX_REFINED_BASINS} items`);
  }

  return basins.flatMap((basin) => {
    const name = typeof basin?.name === "string" ? basin.name.trim() : "";
    const labelPoint = basin?.label_xy_m;
    const idValue = basin?.id;
    const hasScalarId = typeof idValue === "string" || (typeof idValue === "number" && Number.isFinite(idValue));
    const sourceId = hasScalarId ? String(idValue).trim() : "";
    const safeId = sourceId.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-+|-+$/g, "");
    const xMeters = labelPoint?.[0];
    const yMeters = labelPoint?.[1];
    if (!name || name.length > 180 || !sourceId || !safeId || sourceId.length > 80) return [];
    if (!Array.isArray(labelPoint) || labelPoint.length < 2 || !Number.isFinite(xMeters) || !Number.isFinite(yMeters)) {
      return [];
    }
    const areaKm2 = basin?.area_km2;
    const subregion = typeof basin?.subregion === "string" ? basin.subregion.trim().slice(0, 80) : "";
    return [{
      id: `${region}-refined-basin-${safeId}`,
      region,
      layer: "refined_basins",
      kind: "refined_basin",
      name,
      name_zh: "",
      aliases: [
        `${name} refined basin`,
        `${name} basin`,
        "refined basin",
        "refined basins",
        subregion,
      ].filter(Boolean),
      feature_type: "Refined basin",
      display_priority: 1,
      x_m: xMeters,
      y_m: yMeters,
      area_km2: typeof areaKm2 === "number" && Number.isFinite(areaKm2) && areaKm2 >= 0 ? areaKm2 : null,
      subregion,
    }];
  });
}

function searchableValues(feature) {
  const values = SEARCH_FIELDS.map((field) => feature?.[field]);
  values.push(...(Array.isArray(feature?.aliases) ? feature.aliases : []));
  return values.map(normalizeSearchText).filter(Boolean);
}

function matchScore(feature, query) {
  const values = searchableValues(feature);
  let score = Number.POSITIVE_INFINITY;
  for (const value of values) {
    if (value === query) score = Math.min(score, 0);
    else if (value.startsWith(query)) score = Math.min(score, 1);
    else if (value.split(" ").some((token) => token.startsWith(query))) score = Math.min(score, 2);
    else if (value.includes(query)) score = Math.min(score, 3);
  }
  return score;
}

export function searchPolarFeatures(features, query, options = {}) {
  const normalizedQuery = normalizeSearchText(query);
  if (!normalizedQuery) return [];
  const region = options.region || "";
  const layer = options.layer || "";
  const limit = Math.max(1, Math.floor(Number(options.limit) || DEFAULT_LIMIT));

  return (Array.isArray(features) ? features : [])
    .filter((feature) => (!region || feature.region === region) && (!layer || feature.layer === layer))
    .map((feature) => ({ feature, score: matchScore(feature, normalizedQuery) }))
    .filter((entry) => Number.isFinite(entry.score))
    .sort((left, right) => {
      if (left.score !== right.score) return left.score - right.score;
      const layerRank = (value) => ({ research_stations: 0, geographic_names: 1, refined_basins: 2 })[value] ?? 3;
      const layerDifference = layerRank(left.feature.layer) - layerRank(right.feature.layer);
      if (layerDifference) return layerDifference;
      const priorityDifference = Number(left.feature.display_priority || 99) - Number(right.feature.display_priority || 99);
      if (priorityDifference) return priorityDifference;
      return String(left.feature.name).localeCompare(String(right.feature.name), "en");
    })
    .slice(0, limit)
    .map((entry) => entry.feature);
}
