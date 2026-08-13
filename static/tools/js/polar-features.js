import {
  getPolarFeatureLabel,
  normalizeSearchText,
  polarFeatureLanguageKey,
  searchPolarFeatures,
} from "./polar-feature-search.js";

export { getPolarFeatureLabel, normalizeSearchText, searchPolarFeatures } from "./polar-feature-search.js";

const DEFAULT_LIMIT = 12;

const TEXT = {
  en: {
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
    research_station: "Research station",
    ocean: "Ocean",
    sea: "Sea",
    basin: "Basin",
    mountain_range: "Mountain range",
    mountain: "Mountain",
    plateau: "Plateau",
    ice_shelf: "Ice shelf",
    ice_sheet: "Ice sheet",
    fjord: "Fjord",
    strait: "Strait",
    nunatak: "Nunatak",
    open: "Open",
    temporarily_closed: "Temporarily closed",
    relocated: "Relocated",
  },
  zh: {
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
    research_station: "科考站",
    ocean: "大洋",
    sea: "海域",
    basin: "盆地",
    mountain_range: "山脉",
    mountain: "山峰",
    plateau: "高原",
    ice_shelf: "冰架",
    ice_sheet: "冰盖",
    fjord: "峡湾",
    strait: "海峡",
    nunatak: "冰原岛峰",
    open: "开放",
    temporarily_closed: "暂时关闭",
    relocated: "已迁移",
  },
};

const LAYER_KEYS = {
  research_stations: "researchStations",
  geographic_names: "geographicNames",
};
const MAX_CATALOGUE_ITEMS = 20_000;
const MAX_NAME_LENGTH = 180;
const MAX_FIELD_LENGTH = 500;
const MAX_ALIASES = 30;
const TRUSTED_SOURCE_HOSTS = new Set([
  "www.comnap.aq",
  "www.interact-gis.org",
  "placenames.aq",
  "www.placenames.aq",
  "kort.nunagis.gl",
  "services2.arcgis.com",
  "www.gebco.net",
  "www.naturalearthdata.com",
]);

function boundedText(value, maximum = MAX_FIELD_LENGTH) {
  const text = String(value || "").trim();
  return text.length <= maximum ? text : text.slice(0, maximum);
}

function validateCatalogue(payload, expectedRegion, expectedLayer) {
  if (!payload || payload.schema_version !== 1 || !Array.isArray(payload.items)) {
    throw new Error(`Invalid polar feature catalogue: ${expectedRegion}/${expectedLayer}`);
  }
  if (payload.items.length > MAX_CATALOGUE_ITEMS) {
    throw new Error(`Polar feature catalogue exceeds ${MAX_CATALOGUE_ITEMS} items`);
  }
  const items = payload.items
    .filter(
      (item) =>
      item &&
      item.region === expectedRegion &&
      item.layer === expectedLayer &&
      typeof item.id === "string" &&
      typeof item.name === "string" &&
      item.id.length <= MAX_NAME_LENGTH &&
      item.name.trim().length > 0 &&
      item.name.length <= MAX_NAME_LENGTH &&
      Number.isFinite(Number(item.x_m)) &&
      Number.isFinite(Number(item.y_m)),
    )
    .map((item) => ({
      ...item,
      name: boundedText(item.name, MAX_NAME_LENGTH),
      name_zh: boundedText(item.name_zh, MAX_NAME_LENGTH),
      aliases: (Array.isArray(item.aliases) ? item.aliases : [])
        .slice(0, MAX_ALIASES)
        .map((alias) => boundedText(alias, MAX_NAME_LENGTH))
        .filter(Boolean),
      operator: boundedText(item.operator),
      additional_operator: boundedText(item.additional_operator),
      country: boundedText(item.country),
      feature_type: boundedText(item.feature_type),
      status: boundedText(item.status),
      source_url: safeSourceUrl(item.source_url),
    }));
  return { ...payload, items };
}

function safeSourceUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return url.protocol === "https:" && TRUSTED_SOURCE_HOSTS.has(url.hostname) ? url.href : "";
  } catch (_error) {
    return "";
  }
}

function appendDetailRow(documentRef, list, term, value) {
  if (!String(value || "").trim()) return;
  const dt = documentRef.createElement("dt");
  dt.textContent = term;
  const dd = documentRef.createElement("dd");
  dd.textContent = String(value);
  list.append(dt, dd);
}

export function createPolarFeaturesController(options) {
  const {
    THREE,
    scene,
    locale = "en-US",
    elements,
    dataUrls,
    getRegion,
    changeRegion,
    waitForRegionReady,
    getScenePoint,
    getExaggeration = () => 1,
    focusScenePoint,
  } = options;
  const language = polarFeatureLanguageKey(locale);
  const labels = TEXT[language];
  const documentRef = elements.searchInput.ownerDocument;
  const catalogueCache = new Map();
  const cataloguePromises = new Map();
  const groups = new Map();
  const listeners = [];
  const state = {
    researchStations: { enabled: Boolean(elements.stationToggle.checked), loaded: false, totalCount: 0, visibleCount: 0 },
    geographicNames: { enabled: Boolean(elements.namesToggle.checked), loaded: false, totalCount: 0, visibleCount: 0 },
    search: { query: "", resultCount: 0, activeIndex: -1 },
    selectedFeature: null,
  };
  let allFeaturesPromise = null;
  let searchResults = [];
  let selectedMarker = null;
  let destroyed = false;
  let selectionGeneration = 0;
  let blurTimer = null;
  const requestAbortController = new AbortController();

  function on(target, type, handler) {
    target.addEventListener(type, handler);
    listeners.push(() => target.removeEventListener(type, handler));
  }

  function layerToggle(layer) {
    return layer === "research_stations" ? elements.stationToggle : elements.namesToggle;
  }

  async function loadCatalogue(region, layer) {
    const cacheKey = `${region}/${layer}`;
    if (catalogueCache.has(cacheKey)) return catalogueCache.get(cacheKey);
    if (cataloguePromises.has(cacheKey)) return cataloguePromises.get(cacheKey);
    const url = dataUrls?.[region]?.[layer];
    if (!url) throw new Error(`Missing polar feature URL: ${cacheKey}`);
    const promise = fetch(url, { signal: requestAbortController.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load polar features (${response.status})`);
        return response.json();
      })
      .then((payload) => {
        const catalogue = validateCatalogue(payload, region, layer);
        catalogueCache.set(cacheKey, catalogue);
        return catalogue;
      })
      .finally(() => cataloguePromises.delete(cacheKey));
    cataloguePromises.set(cacheKey, promise);
    return promise;
  }

  function loadAllFeatures() {
    if (allFeaturesPromise) return allFeaturesPromise;
    allFeaturesPromise = Promise.all(
      ["antarctica", "greenland"].flatMap((region) =>
        ["research_stations", "geographic_names"].map((layer) => loadCatalogue(region, layer)),
      ),
    )
      .then((catalogues) => catalogues.flatMap((catalogue) => catalogue.items))
      .catch((error) => {
        allFeaturesPromise = null;
        throw error;
      });
    return allFeaturesPromise;
  }

  function disposeObject(object) {
    if (!object) return;
    scene.remove(object);
    object.traverse((child) => {
      child.geometry?.dispose?.();
      const materials = Array.isArray(child.material) ? child.material : child.material ? [child.material] : [];
      for (const material of materials) {
        material.map?.dispose?.();
        material.dispose?.();
      }
    });
  }

  function createLabelSprite(text, { selected = false, station = false } = {}) {
    const fontSize = selected ? 38 : 30;
    const paddingX = 22;
    const canvas = documentRef.createElement("canvas");
    const context = canvas.getContext("2d");
    context.font = `600 ${fontSize}px system-ui, sans-serif`;
    canvas.width = Math.min(900, Math.ceil(context.measureText(text).width + paddingX * 2));
    canvas.height = fontSize + 24;
    context.font = `600 ${fontSize}px system-ui, sans-serif`;
    context.textBaseline = "middle";
    context.fillStyle = selected ? "rgba(5, 25, 38, 0.96)" : "rgba(5, 25, 38, 0.82)";
    context.strokeStyle = selected ? "#fff5c7" : station ? "#ffbd70" : "#83e8ff";
    context.lineWidth = selected ? 5 : 3;
    context.fillRect(2, 2, canvas.width - 4, canvas.height - 4);
    context.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
    context.fillStyle = "#ffffff";
    context.fillText(text, paddingX, canvas.height / 2 + 1);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }),
    );
    const scale = selected ? 0.021 : 0.017;
    sprite.scale.set(canvas.width * scale, canvas.height * scale, 1);
    sprite.renderOrder = selected ? 31 : 30;
    return sprite;
  }

  function scenePointFor(feature) {
    const point = getScenePoint(feature);
    if (!point || ![point.x, point.z, point.baseY].every(Number.isFinite)) return null;
    return { ...point, y: point.baseY * Number(getExaggeration() || 1) };
  }

  function buildLayer(catalogue, layer) {
    const previous = groups.get(layer);
    disposeObject(previous);
    groups.delete(layer);
    const group = new THREE.Group();
    group.name = `polar-${layer}`;
    const rendered = [];
    for (const feature of catalogue.items) {
      const point = scenePointFor(feature);
      if (point) rendered.push({ feature, point });
    }

    if (rendered.length) {
      const positions = new Float32Array(rendered.length * 3);
      const baseY = new Float32Array(rendered.length);
      rendered.forEach(({ point }, index) => {
        positions[index * 3] = point.x;
        positions[index * 3 + 1] = point.y;
        positions[index * 3 + 2] = point.z;
        baseY[index] = point.baseY;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const station = layer === "research_stations";
      const points = new THREE.Points(
        geometry,
        new THREE.PointsMaterial({
          color: station ? 0xff9b42 : 0x5ce1ff,
          size: station ? 7.5 : 5.5,
          sizeAttenuation: false,
          transparent: true,
          opacity: 0.96,
          depthTest: false,
          depthWrite: false,
        }),
      );
      points.renderOrder = 29;
      points.userData.baseY = baseY;
      group.add(points);

      const labelLimit = station ? 90 : 100;
      rendered
        .slice()
        .sort((left, right) =>
          Number(left.feature.display_priority || 99) - Number(right.feature.display_priority || 99) ||
          left.feature.name.localeCompare(right.feature.name, "en"),
        )
        .slice(0, labelLimit)
        .forEach(({ feature, point }) => {
          const sprite = createLabelSprite(getPolarFeatureLabel(feature, locale), { station });
          sprite.position.set(point.x, point.y + 0.5, point.z);
          sprite.userData.baseY = point.baseY;
          sprite.userData.heightOffset = 0.5;
          group.add(sprite);
        });
    }
    scene.add(group);
    groups.set(layer, group);
    const stateKey = LAYER_KEYS[layer];
    state[stateKey] = {
      ...state[stateKey],
      loaded: true,
      totalCount: catalogue.items.length,
      visibleCount: rendered.length,
    };
  }

  async function ensureLayer(layer) {
    const stateKey = LAYER_KEYS[layer];
    if (!state[stateKey].enabled || destroyed) return;
    const region = getRegion();
    const catalogue = await loadCatalogue(region, layer);
    if (destroyed || getRegion() !== region || !state[stateKey].enabled) return;
    buildLayer(catalogue, layer);
  }

  function clearSelectedMarker() {
    disposeObject(selectedMarker);
    selectedMarker = null;
  }

  function renderSelectedMarker(feature) {
    clearSelectedMarker();
    if (!feature || feature.region !== getRegion()) return null;
    const point = scenePointFor(feature);
    if (!point) return null;
    selectedMarker = new THREE.Group();
    const sprite = createLabelSprite(getPolarFeatureLabel(feature, locale), { selected: true });
    sprite.position.set(point.x, point.y + 1.2, point.z);
    sprite.userData.baseY = point.baseY;
    sprite.userData.heightOffset = 1.2;
    selectedMarker.add(sprite);
    scene.add(selectedMarker);
    return point;
  }

  function showDetails(feature) {
    const container = elements.details;
    container.replaceChildren();
    if (!feature) {
      container.hidden = true;
      return;
    }
    const heading = documentRef.createElement("h3");
    heading.textContent = getPolarFeatureLabel(feature, locale);
    const eyebrow = documentRef.createElement("p");
    eyebrow.className = "polar-feature-details__type";
    eyebrow.textContent = `${labels[feature.kind] || feature.kind} · ${labels[feature.region] || feature.region}`;
    const list = documentRef.createElement("dl");
    appendDetailRow(documentRef, list, labels.operator, feature.operator);
    appendDetailRow(documentRef, list, labels.country, feature.country);
    appendDetailRow(
      documentRef,
      list,
      labels.status,
      labels[feature.status] || String(feature.status || "").replace(/_/g, " "),
    );
    appendDetailRow(
      documentRef,
      list,
      labels.coordinates,
      `${Number(feature.lat).toFixed(3)}°, ${Number(feature.lon).toFixed(3)}°`,
    );
    container.append(heading, eyebrow, list);
    const sourceUrl = safeSourceUrl(feature.source_url);
    if (sourceUrl) {
      const source = documentRef.createElement("a");
      source.href = sourceUrl;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      source.textContent = labels.source;
      container.append(source);
    }
    container.hidden = false;
  }

  function updateActiveOption() {
    const optionsEls = Array.from(elements.results.querySelectorAll('[role="option"]'));
    optionsEls.forEach((optionEl, index) => optionEl.setAttribute("aria-selected", String(index === state.search.activeIndex)));
    const active = optionsEls[state.search.activeIndex];
    if (active) {
      elements.searchInput.setAttribute("aria-activedescendant", active.id);
      active.scrollIntoView({ block: "nearest" });
    } else {
      elements.searchInput.removeAttribute("aria-activedescendant");
    }
  }

  async function chooseFeature(feature) {
    if (!feature || destroyed) return;
    const generation = ++selectionGeneration;
    const isCancelled = () => destroyed || generation !== selectionGeneration;
    try {
      elements.results.hidden = true;
      elements.searchInput.setAttribute("aria-expanded", "false");
      const stateKey = LAYER_KEYS[feature.layer];
      state[stateKey] = { ...state[stateKey], enabled: true };
      layerToggle(feature.layer).checked = true;
      if (feature.region !== getRegion()) {
        await changeRegion(feature.region);
        await waitForRegionReady(feature.region, isCancelled);
      }
      if (isCancelled()) return;
      await ensureLayer(feature.layer);
      if (isCancelled() || feature.region !== getRegion()) return;
      state.selectedFeature = feature;
      showDetails(feature);
      const point = renderSelectedMarker(feature);
      if (point) focusScenePoint(feature, point);
    } catch (error) {
      if (isCancelled()) return;
      console.warn("Polar feature selection failed:", error);
      elements.searchStatus.textContent = labels.loadError;
    }
  }

  function renderSearchResults(results) {
    elements.results.replaceChildren();
    results.forEach((feature, index) => {
      const option = documentRef.createElement("li");
      option.id = `polar-search-option-${index}`;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.tabIndex = -1;
      const name = documentRef.createElement("span");
      name.className = "polar-search-result__name";
      name.textContent = getPolarFeatureLabel(feature, locale);
      const meta = documentRef.createElement("span");
      meta.className = "polar-search-result__meta";
      meta.textContent = `${labels[feature.kind] || feature.kind} · ${labels[feature.region] || feature.region}`;
      option.append(name, meta);
      option.addEventListener("pointerdown", (event) => event.preventDefault());
      option.addEventListener("click", () => void chooseFeature(feature));
      elements.results.append(option);
    });
    const hasResults = results.length > 0;
    elements.results.hidden = !hasResults;
    elements.searchInput.setAttribute("aria-expanded", String(hasResults));
    updateActiveOption();
  }

  async function updateSearch() {
    const query = elements.searchInput.value;
    state.search = { query, resultCount: 0, activeIndex: -1 };
    if (!normalizeSearchText(query)) {
      searchResults = [];
      elements.searchStatus.textContent = "";
      renderSearchResults([]);
      return;
    }
    elements.searchStatus.textContent = labels.loading;
    try {
      const features = await loadAllFeatures();
      if (query !== elements.searchInput.value || destroyed) return;
      searchResults = searchPolarFeatures(features, query, { limit: DEFAULT_LIMIT });
      state.search = { query, resultCount: searchResults.length, activeIndex: -1 };
      elements.searchStatus.textContent = searchResults.length ? labels.resultCount(searchResults.length) : labels.noMatches;
      renderSearchResults(searchResults);
    } catch (error) {
      console.warn("Polar feature search failed:", error);
      searchResults = [];
      elements.searchStatus.textContent = labels.noMatches;
      renderSearchResults([]);
    }
  }

  function handleSearchKeydown(event) {
    if (!searchResults.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const start = state.search.activeIndex < 0 ? (direction > 0 ? -1 : 0) : state.search.activeIndex;
      state.search = {
        ...state.search,
        activeIndex: (start + direction + searchResults.length) % searchResults.length,
      };
      updateActiveOption();
      return;
    }
    if (event.key === "Enter" && state.search.activeIndex >= 0) {
      event.preventDefault();
      void chooseFeature(searchResults[state.search.activeIndex]);
      return;
    }
    if (event.key === "Escape") {
      elements.results.hidden = true;
      elements.searchInput.setAttribute("aria-expanded", "false");
    }
  }

  async function handleLayerToggle(layer, checked) {
    const stateKey = LAYER_KEYS[layer];
    state[stateKey] = { ...state[stateKey], enabled: checked };
    if (checked) {
      try {
        await ensureLayer(layer);
      } catch (error) {
        console.warn(`Polar feature layer failed (${layer}):`, error);
        elements.searchStatus.textContent = labels.loadError;
      }
      return;
    }
    disposeObject(groups.get(layer));
    groups.delete(layer);
    state[stateKey] = { ...state[stateKey], visibleCount: 0 };
    if (state.selectedFeature?.layer === layer) {
      clearSelectedMarker();
    }
  }

  function initialize() {
    elements.searchInput.setAttribute("aria-expanded", "false");
    on(elements.searchInput, "input", () => void updateSearch());
    on(elements.searchInput, "keydown", handleSearchKeydown);
    on(elements.searchInput, "focus", () => {
      if (blurTimer !== null) {
        window.clearTimeout(blurTimer);
        blurTimer = null;
      }
      if (searchResults.length) {
        elements.results.hidden = false;
        elements.searchInput.setAttribute("aria-expanded", "true");
      }
    });
    on(elements.searchInput, "blur", () => {
      blurTimer = window.setTimeout(() => {
        blurTimer = null;
        elements.results.hidden = true;
        elements.searchInput.setAttribute("aria-expanded", "false");
      }, 120);
    });
    on(elements.stationToggle, "change", () => void handleLayerToggle("research_stations", elements.stationToggle.checked));
    on(elements.namesToggle, "change", () => void handleLayerToggle("geographic_names", elements.namesToggle.checked));
    void loadAllFeatures().catch((error) => {
      console.warn("Polar feature preload failed:", error);
    });
  }

  function clearScene() {
    for (const [layer, group] of groups) {
      disposeObject(group);
      groups.delete(layer);
      const stateKey = LAYER_KEYS[layer];
      state[stateKey] = { ...state[stateKey], visibleCount: 0 };
    }
    clearSelectedMarker();
  }

  async function onTerrainReady() {
    clearScene();
    await Promise.all(
      ["research_stations", "geographic_names"].map((layer) =>
        ensureLayer(layer).catch((error) => {
          console.warn(`Polar feature layer failed (${layer}):`, error);
        }),
      ),
    );
    if (state.selectedFeature?.region === getRegion()) renderSelectedMarker(state.selectedFeature);
  }

  function updateExaggeration() {
    const exaggeration = Number(getExaggeration() || 1);
    for (const object of [...groups.values(), selectedMarker].filter(Boolean)) {
      object.traverse((child) => {
        const positions = child.geometry?.getAttribute?.("position");
        const baseY = child.userData?.baseY;
        if (positions && baseY instanceof Float32Array) {
          for (let index = 0; index < baseY.length; index += 1) positions.setY(index, baseY[index] * exaggeration);
          positions.needsUpdate = true;
        } else if (Number.isFinite(baseY)) {
          child.position.y = baseY * exaggeration + Number(child.userData.heightOffset || 0);
        }
      });
    }
  }

  function getState() {
    const selectedFeature = state.selectedFeature
      ? {
          id: state.selectedFeature.id,
          name: state.selectedFeature.name,
          region: state.selectedFeature.region,
          layer: state.selectedFeature.layer,
          kind: state.selectedFeature.kind,
        }
      : null;
    return {
      featureLayers: {
        researchStations: { ...state.researchStations },
        geographicNames: { ...state.geographicNames },
      },
      search: { ...state.search },
      selectedFeature,
      loading: cataloguePromises.size > 0,
    };
  }

  function destroy() {
    destroyed = true;
    selectionGeneration += 1;
    requestAbortController.abort();
    if (blurTimer !== null) window.clearTimeout(blurTimer);
    blurTimer = null;
    clearScene();
    listeners.splice(0).forEach((remove) => remove());
  }

  return { initialize, clearScene, onTerrainReady, updateExaggeration, getState, destroy };
}
