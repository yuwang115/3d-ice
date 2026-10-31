/**
 * Text and the details card for places and features: research stations, geographic
 * names, refined basins and ice-core sites.
 *
 * No scene dependencies: this module runs unchanged under Node's test runner.
 */

export const TRUSTED_SOURCE_HOSTS = new Set([
  "www.comnap.aq",
  "www.interact-gis.org",
  "placenames.aq",
  "www.placenames.aq",
  "kort.nunagis.gl",
  "services2.arcgis.com",
  "www.gebco.net",
  "www.naturalearthdata.com",
  // Each ice-core site links the publication of its core.
  "doi.org",
]);

export const POLAR_FEATURE_TEXT = Object.freeze({
  en: Object.freeze({
    antarctica: "Antarctica",
    greenland: "Greenland",
    loading: "Loading polar places…",
    loadError: "Polar place data could not be loaded.",
    noMatches: "No matches.",
    resultCount: (count) => `Showing ${count} match${count === 1 ? "" : "es"}.`,
    details: "Feature details",
    operator: "Operator",
    country: "Country",
    status: "Status",
    coordinates: "Coordinates",
    source: "Source",
    referenceLink: (citation) => `Reference: ${citation}`,
    area: "Area",
    project: "Project",
    drilled: "Drilled",
    coreDepth: "Core depth",
    // A drilled core gives the age its continuous dated record reaches; a blue-ice site,
    // whose layers are not in order, the age of its oldest dated ice.
    continuousRecord: "Continuous record",
    oldestIce: "Oldest dated ice",
    elevation: "Surface elevation",
    metres: (value) => `${value} m`,
    metresToBed: (value) => `${value} m, to the bed`,
    years: (ka, locale) =>
      ka >= 1000
        ? `≈${formatNumber(ka / 1000, locale, 1)} million years`
        : `≈${formatNumber(ka * 1000, locale, 0)} years`,
    research_station: "Research station",
    ocean: "Ocean",
    sea: "Sea",
    basin: "Basin",
    refined_basin: "Refined basin",
    mountain_range: "Mountain range",
    mountain: "Mountain",
    plateau: "Plateau",
    ice_shelf: "Ice shelf",
    ice_sheet: "Ice sheet",
    fjord: "Fjord",
    strait: "Strait",
    nunatak: "Nunatak",
    ice_core: "Ice core",
    ice_core_deep: "Deep ice core",
    ice_core_intermediate: "Intermediate-depth ice core",
    ice_core_shallow: "Shallow ice core",
    ice_core_blue_ice: "Blue-ice drilling site",
    open: "Open",
    temporarily_closed: "Temporarily closed",
    relocated: "Relocated",
  }),
  zh: Object.freeze({
    antarctica: "南极洲",
    greenland: "格陵兰岛",
    loading: "正在加载极地地理信息…",
    loadError: "无法加载极地地理信息。",
    noMatches: "未找到匹配结果。",
    resultCount: (count) => `显示 ${count} 条匹配结果。`,
    details: "地理信息详情",
    operator: "运营机构",
    country: "国家/地区",
    status: "状态",
    coordinates: "坐标",
    source: "数据来源",
    referenceLink: (citation) => `参考文献：${citation}`,
    area: "面积",
    project: "项目",
    drilled: "钻取年份",
    coreDepth: "冰芯深度",
    continuousRecord: "连续记录",
    oldestIce: "最老定年冰",
    elevation: "冰面海拔",
    metres: (value) => `${value} 米`,
    metresToBed: (value) => `${value} 米（钻至冰床）`,
    years: (ka, locale) =>
      ka >= 10
        ? `约 ${formatNumber(ka / 10, locale, 1)} 万年`
        : `约 ${formatNumber(ka * 1000, locale, 0)} 年`,
    research_station: "科考站",
    ocean: "大洋",
    sea: "海域",
    basin: "盆地",
    refined_basin: "细化流域",
    mountain_range: "山脉",
    mountain: "山峰",
    plateau: "高原",
    ice_shelf: "冰架",
    ice_sheet: "冰盖",
    fjord: "峡湾",
    strait: "海峡",
    nunatak: "冰原岛峰",
    ice_core: "冰芯",
    ice_core_deep: "深冰芯",
    ice_core_intermediate: "中等深度冰芯",
    ice_core_shallow: "浅冰芯",
    ice_core_blue_ice: "蓝冰区钻探点",
    open: "开放",
    temporarily_closed: "暂时关闭",
    relocated: "已迁移",
  }),
});

function formatNumber(value, locale, maximumFractionDigits) {
  return Number(value).toLocaleString(locale, { maximumFractionDigits });
}

function finiteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function safeSourceUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" && TRUSTED_SOURCE_HOSTS.has(url.hostname) ? url.href : "";
  } catch (_error) {
    return "";
  }
}

/** The type shown under a feature's name: an ice core by its kind of core. */
export function polarFeatureTypeLabel(feature, labels) {
  const coreTypeLabel = feature?.kind === "ice_core" ? labels[`ice_core_${feature.core_type}`] : "";
  return coreTypeLabel || labels[feature?.kind] || String(feature?.kind || "");
}

export function formatCoreDepth(feature, labels, locale) {
  const depth = finiteOrNull(feature?.depth_m);
  if (depth === null || depth <= 0) return "";
  const value = formatNumber(depth, locale, 0);
  return feature.reached_bed === true ? labels.metresToBed(value) : labels.metres(value);
}

export function formatOldestIce(feature, labels, locale) {
  const ka = finiteOrNull(feature?.oldest_ka);
  return ka === null || ka <= 0 ? "" : labels.years(ka, locale);
}

export function formatElevation(feature, labels, locale) {
  const elevation = finiteOrNull(feature?.elevation_m);
  return elevation === null ? "" : labels.metres(formatNumber(elevation, locale, 0));
}

/** The detail rows of a feature, in display order, as [term, value] pairs with a value. */
export function polarFeatureDetailRows(feature, labels, locale) {
  const rows = [
    [labels.project, feature.project],
    [labels.operator, feature.operator],
    [labels.country, feature.country],
    [labels.status, labels[feature.status] || String(feature.status || "").replace(/_/g, " ")],
    [labels.drilled, feature.drilled],
    [labels.coreDepth, formatCoreDepth(feature, labels, locale)],
    [feature.core_type === "blue_ice" ? labels.oldestIce : labels.continuousRecord, formatOldestIce(feature, labels, locale)],
    [labels.elevation, formatElevation(feature, labels, locale)],
  ];
  const latitude = finiteOrNull(feature.lat);
  const longitude = finiteOrNull(feature.lon);
  if (latitude !== null && longitude !== null) {
    rows.push([labels.coordinates, `${latitude.toFixed(3)}°, ${longitude.toFixed(3)}°`]);
  }
  const areaKm2 = finiteOrNull(feature.area_km2);
  if (areaKm2 !== null) {
    rows.push([labels.area, `${formatNumber(areaKm2, locale, 0)} km²`]);
  }
  return rows.filter(([, value]) => String(value || "").trim());
}

/** Fill the details card with a feature, or hide it when there is none. */
export function renderPolarFeatureDetails(container, feature, { labels, locale, getLabel }) {
  const documentRef = container.ownerDocument;
  container.replaceChildren();
  if (!feature) {
    container.hidden = true;
    return;
  }
  const heading = documentRef.createElement("h3");
  heading.textContent = getLabel(feature, locale);
  const eyebrow = documentRef.createElement("p");
  eyebrow.className = "polar-feature-details__type";
  eyebrow.textContent = `${polarFeatureTypeLabel(feature, labels)} · ${labels[feature.region] || feature.region}`;
  const list = documentRef.createElement("dl");
  for (const [term, value] of polarFeatureDetailRows(feature, labels, locale)) {
    const dt = documentRef.createElement("dt");
    dt.textContent = term;
    const dd = documentRef.createElement("dd");
    dd.textContent = String(value);
    list.append(dt, dd);
  }
  container.append(heading, eyebrow, list);
  const sourceUrl = safeSourceUrl(feature.source_url);
  if (sourceUrl) {
    const source = documentRef.createElement("a");
    source.href = sourceUrl;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    const citation = String(feature.citation || "").trim();
    source.textContent = citation ? labels.referenceLink(citation) : labels.source;
    container.append(source);
  }
  container.hidden = false;
}
