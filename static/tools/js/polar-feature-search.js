const DEFAULT_LIMIT = 12;
const SEARCH_FIELDS = ["name", "name_zh", "operator", "additional_operator", "country", "feature_type", "kind"];

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
      const layerRank = (value) => (value === "research_stations" ? 0 : 1);
      const layerDifference = layerRank(left.feature.layer) - layerRank(right.feature.layer);
      if (layerDifference) return layerDifference;
      const priorityDifference = Number(left.feature.display_priority || 99) - Number(right.feature.display_priority || 99);
      if (priorityDifference) return priorityDifference;
      return String(left.feature.name).localeCompare(String(right.feature.name), "en");
    })
    .slice(0, limit)
    .map((entry) => entry.feature);
}
