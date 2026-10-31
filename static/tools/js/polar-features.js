import {
  getPolarFeatureLabel,
  normalizeSearchText,
  polarFeatureLanguageKey,
  searchPolarFeatures,
} from "./polar-feature-search.js";
import {
  createPolarLabelCanvas,
  createPolarMarkerCanvas,
  getPolarFeatureLabelStyle,
  getPolarFeatureMarkerMaterialOptions,
  getPolarFeatureMarkerShape,
} from "./polar-feature-label-style.js";
import {
  POLAR_FEATURE_TEXT,
  polarFeatureTypeLabel,
  renderPolarFeatureDetails,
  safeSourceUrl,
} from "./polar-feature-details.js";

export {
  createRefinedBasinSearchFeatures,
  getPolarFeatureLabel,
  normalizeSearchText,
  searchPolarFeatures,
} from "./polar-feature-search.js";

const DEFAULT_LIMIT = 12;
const SEARCH_REGIONS = ["antarctica", "greenland"];
const BASE_SEARCH_LAYERS = ["research_stations", "geographic_names"];
// Point layers a page may leave out: a page offers one by passing its catalogue URLs.
const OPTIONAL_POINT_LAYERS = ["ice_cores"];
const LABEL_HEIGHT_OFFSET = 0.5;

const LAYER_KEYS = {
  research_stations: "researchStations",
  geographic_names: "geographicNames",
  ice_cores: "iceCores",
};
const MAX_CATALOGUE_ITEMS = 20_000;
const MAX_NAME_LENGTH = 180;
const MAX_FIELD_LENGTH = 500;
const MAX_ALIASES = 30;
const MAX_CATALOGUE_RESPONSE_BYTES = 2 * 1024 * 1024;
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
      typeof item.kind === "string" &&
      typeof item.name === "string" &&
      item.id.length <= MAX_NAME_LENGTH &&
      item.kind.trim().length > 0 &&
      item.kind.length <= MAX_NAME_LENGTH &&
      item.name.trim().length > 0 &&
      item.name.length <= MAX_NAME_LENGTH &&
      typeof item.x_m === "number" &&
      typeof item.y_m === "number" &&
      Number.isFinite(item.x_m) &&
      Number.isFinite(item.y_m),
    )
    .map((item) => ({
      id: item.id,
      region: item.region,
      layer: item.layer,
      kind: boundedText(item.kind, MAX_NAME_LENGTH),
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
      display_priority: Number.isFinite(item.display_priority) ? item.display_priority : 99,
      x_m: item.x_m,
      y_m: item.y_m,
      lat: Number.isFinite(item.lat) ? item.lat : null,
      lon: Number.isFinite(item.lon) ? item.lon : null,
      area_km2: Number.isFinite(item.area_km2) && item.area_km2 >= 0 ? item.area_km2 : null,
      subregion: boundedText(item.subregion),
      project: boundedText(item.project),
      drilled: boundedText(item.drilled, MAX_NAME_LENGTH),
      core_type: boundedText(item.core_type, MAX_NAME_LENGTH),
      depth_m: Number.isFinite(item.depth_m) && item.depth_m > 0 ? item.depth_m : null,
      reached_bed: item.reached_bed === true,
      oldest_ka: Number.isFinite(item.oldest_ka) && item.oldest_ka > 0 ? item.oldest_ka : null,
      elevation_m: Number.isFinite(item.elevation_m) ? item.elevation_m : null,
      citation: boundedText(item.citation),
    }));
  return {
    schema_version: 1,
    region: expectedRegion,
    layer: expectedLayer,
    feature_count: items.length,
    items,
  };
}

async function readBoundedJsonResponse(response, maximumBytes = MAX_CATALOGUE_RESPONSE_BYTES) {
  const declaredSize = Number(response.headers?.get?.("content-length") || 0);
  if (declaredSize > maximumBytes) throw new Error("Polar feature catalogue exceeds the response size limit");
  if (!response.body?.getReader) throw new Error("Polar feature catalogue requires bounded streaming");
  const reader = response.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > maximumBytes) {
      await reader.cancel();
      throw new Error("Polar feature catalogue exceeds the response size limit");
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return JSON.parse(new TextDecoder().decode(bytes));
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
    activateRefinedBasinFeature = async () => false,
  } = options;
  const language = polarFeatureLanguageKey(locale);
  const labels = POLAR_FEATURE_TEXT[language];
  const layerToggles = {
    research_stations: elements.stationToggle,
    geographic_names: elements.namesToggle,
    ice_cores: elements.iceCoreToggle || null,
  };
  // A layer is offered only where the page passes its catalogues and its toggle.
  const pointLayers = [
    ...BASE_SEARCH_LAYERS,
    ...OPTIONAL_POINT_LAYERS.filter(
      (layer) => layerToggles[layer] && SEARCH_REGIONS.every((region) => dataUrls?.[region]?.[layer]),
    ),
  ];
  const documentRef = elements.searchInput.ownerDocument;
  const catalogueCache = new Map();
  const cataloguePromises = new Map();
  const groups = new Map();
  const labelSprites = new Map();
  const listeners = [];
  const state = {
    researchStations: { enabled: Boolean(elements.stationToggle.checked), loaded: false, totalCount: 0, visibleCount: 0 },
    geographicNames: { enabled: Boolean(elements.namesToggle.checked), loaded: false, totalCount: 0, visibleCount: 0 },
    iceCores: { enabled: Boolean(layerToggles.ice_cores?.checked), loaded: false, totalCount: 0, visibleCount: 0 },
    search: { query: "", resultCount: 0, activeIndex: -1 },
    selectedFeature: null,
  };
  let searchResults = [];
  let focusedExistingLabelId = null;
  let destroyed = false;
  let selectionGeneration = 0;
  let blurTimer = null;
  const requestAbortController = new AbortController();

  function on(target, type, handler) {
    target.addEventListener(type, handler);
    listeners.push(() => target.removeEventListener(type, handler));
  }

  function layerToggle(layer) {
    return layerToggles[layer];
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
        return readBoundedJsonResponse(response);
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

  function loadSearchCatalogues(layers = pointLayers) {
    return Promise.all(
      SEARCH_REGIONS.flatMap((region) =>
        layers.map((layer) => loadCatalogue(region, layer)),
      ),
    );
  }

  async function loadAllFeatures() {
    const catalogues = await loadSearchCatalogues(BASE_SEARCH_LAYERS);
    // Optional layers search when they load and never take the base layers down with them.
    // A page that offers no basin layer passes no basin catalogue, and searches without it.
    const optional = [
      ...pointLayers.filter((layer) => !BASE_SEARCH_LAYERS.includes(layer)).flatMap((layer) =>
        SEARCH_REGIONS.map((region) => [region, layer]),
      ),
      ...SEARCH_REGIONS.filter((region) => dataUrls?.[region]?.refined_basins).map((region) => [region, "refined_basins"]),
    ];
    const optionalResults = await Promise.allSettled(optional.map(([region, layer]) => loadCatalogue(region, layer)));
    const optionalCatalogues = optionalResults.flatMap((result, index) => {
      if (result.status === "fulfilled") return [result.value];
      console.warn(`Search catalogue ${optional[index].join("/")} could not be loaded:`, result.reason);
      return [];
    });
    return [...catalogues, ...optionalCatalogues].flatMap((catalogue) => catalogue.items);
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

  function createLabelSprite(text, { layer = "geographic_names", region = getRegion() } = {}) {
    const canvas = createPolarLabelCanvas(documentRef, text, { layer, region });
    if (!canvas) return null;
    const style = getPolarFeatureLabelStyle(layer, region);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }),
    );
    sprite.scale.set(canvas.width * style.worldScale, canvas.height * style.worldScale, 1);
    sprite.renderOrder = 30;
    return sprite;
  }

  function clearLayerLabels(layer) {
    for (const [featureId, entry] of labelSprites) {
      if (entry.layer === layer) labelSprites.delete(featureId);
    }
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
    clearLayerLabels(layer);
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
      const markerCanvas = createPolarMarkerCanvas(documentRef, { shape: getPolarFeatureMarkerShape(layer) });
      const markerTexture = markerCanvas ? new THREE.CanvasTexture(markerCanvas) : null;
      if (markerTexture) {
        markerTexture.colorSpace = THREE.SRGBColorSpace;
        markerTexture.minFilter = THREE.LinearFilter;
        markerTexture.magFilter = THREE.LinearFilter;
        markerTexture.generateMipmaps = false;
      }
      const points = new THREE.Points(
        geometry,
        new THREE.PointsMaterial(getPolarFeatureMarkerMaterialOptions(layer, markerTexture)),
      );
      points.renderOrder = 29;
      points.userData.baseY = baseY;
      group.add(points);

      rendered
        .slice()
        .sort((left, right) =>
          Number(left.feature.display_priority || 99) - Number(right.feature.display_priority || 99) ||
          left.feature.name.localeCompare(right.feature.name, "en"),
        )
        .forEach(({ feature, point }) => {
          const sprite = createLabelSprite(getPolarFeatureLabel(feature, locale), { layer, region: feature.region });
          if (!sprite) return;
          // Ice cores are drilled beside stations (Vostok, Concordia, Kohnen, Summit), so
          // their labels stack above the station's rather than over it.
          const heightOffset = layer === "ice_cores" ? LABEL_HEIGHT_OFFSET + sprite.scale.y : LABEL_HEIGHT_OFFSET;
          sprite.position.set(point.x, point.y + heightOffset, point.z);
          sprite.userData.baseY = point.baseY;
          sprite.userData.heightOffset = heightOffset;
          sprite.userData.featureId = feature.id;
          labelSprites.set(feature.id, { sprite, layer, region: feature.region });
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
    if (groups.has(layer)) return;
    const region = getRegion();
    const catalogue = await loadCatalogue(region, layer);
    if (destroyed || getRegion() !== region || !state[stateKey].enabled || groups.has(layer)) return;
    buildLayer(catalogue, layer);
  }

  function focusPointFor(feature) {
    const existingLabel = labelSprites.get(feature?.id);
    if (existingLabel?.region === getRegion()) {
      focusedExistingLabelId = feature.id;
      return {
        x: existingLabel.sprite.position.x,
        baseY: existingLabel.sprite.userData.baseY,
        z: existingLabel.sprite.position.z,
      };
    }
    focusedExistingLabelId = null;
    return scenePointFor(feature);
  }

  function showDetails(feature) {
    renderPolarFeatureDetails(elements.details, feature, { labels, locale, getLabel: getPolarFeatureLabel });
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
      if (feature.region !== getRegion()) {
        await changeRegion(feature.region);
        await waitForRegionReady(feature.region, isCancelled);
      }
      if (isCancelled()) return;
      if (feature.layer === "refined_basins") {
        const activated = await activateRefinedBasinFeature(feature, isCancelled);
        if (!activated || isCancelled()) return;
      } else {
        const stateKey = LAYER_KEYS[feature.layer];
        if (!stateKey || !pointLayers.includes(feature.layer)) {
          throw new Error(`Unsupported polar feature layer: ${feature.layer}`);
        }
        state[stateKey] = { ...state[stateKey], enabled: true };
        layerToggle(feature.layer).checked = true;
        await ensureLayer(feature.layer);
      }
      if (isCancelled() || feature.region !== getRegion()) return;
      state.selectedFeature = feature;
      showDetails(feature);
      const point = focusPointFor(feature);
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
      meta.textContent = `${polarFeatureTypeLabel(feature, labels)} · ${labels[feature.region] || feature.region}`;
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
      if (query !== elements.searchInput.value || destroyed) return;
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
    clearLayerLabels(layer);
    state[stateKey] = { ...state[stateKey], visibleCount: 0 };
    if (state.selectedFeature?.layer === layer) {
      state.selectedFeature = null;
      focusedExistingLabelId = null;
      showDetails(null);
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
    if (pointLayers.includes("ice_cores")) {
      on(layerToggles.ice_cores, "change", () => void handleLayerToggle("ice_cores", layerToggles.ice_cores.checked));
    }
    if (elements.basinToggle) {
      on(elements.basinToggle, "change", () => {
        if (!elements.basinToggle.checked && state.selectedFeature?.layer === "refined_basins") {
          state.selectedFeature = null;
          focusedExistingLabelId = null;
          showDetails(null);
        }
      });
    }
    void loadSearchCatalogues().catch((error) => {
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
    labelSprites.clear();
    focusedExistingLabelId = null;
  }

  async function onTerrainReady() {
    clearScene();
    await Promise.all(
      pointLayers.map((layer) =>
        ensureLayer(layer).catch((error) => {
          console.warn(`Polar feature layer failed (${layer}):`, error);
        }),
      ),
    );
    if (state.selectedFeature?.region === getRegion() && labelSprites.has(state.selectedFeature.id)) {
      focusedExistingLabelId = state.selectedFeature.id;
    }
  }

  function updateExaggeration() {
    const exaggeration = Number(getExaggeration() || 1);
    for (const object of groups.values()) {
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
    let selectionOverlayCount = 0;
    scene.traverse((object) => {
      if (object.userData?.isPolarSelectionOverlay) selectionOverlayCount += 1;
    });
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
        researchStations: {
          ...state.researchStations,
          renderedLabelCount: [...labelSprites.values()].filter((entry) => entry.layer === "research_stations").length,
        },
        geographicNames: {
          ...state.geographicNames,
          renderedLabelCount: [...labelSprites.values()].filter((entry) => entry.layer === "geographic_names").length,
        },
        iceCores: {
          ...state.iceCores,
          offered: pointLayers.includes("ice_cores"),
          renderedLabelCount: [...labelSprites.values()].filter((entry) => entry.layer === "ice_cores").length,
        },
      },
      search: { ...state.search },
      selectedFeature,
      focusedExistingLabelId,
      selectionOverlayCount,
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
