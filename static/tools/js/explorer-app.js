/**
 * Explorer runtime shared by the English and Chinese explorer pages.
 *
 * The page supplies the markup and <html lang>; its head scripts publish
 * window.__iceAssetBase, window.__iceRuntimeTheme and window.__3dIceLocale, and every
 * user-visible string goes through t(). Data packages are decoded by ./data-contract.js.
 */

const queryParams = new URLSearchParams(window.location.search);
const assetBaseApi = window.__iceAssetBase || null;
const runtimeThemeApi = window.__iceRuntimeTheme || null;
const localeApi = window.__3dIceLocale || null;
const assetUrl = (relativePath) =>
  assetBaseApi && typeof assetBaseApi.assetUrl === "function"
    ? assetBaseApi.assetUrl(relativePath)
    : new URL(String(relativePath || "").replace(/^\/+/, ""), new URL("/tools/", window.location.href)).toString();
const { createPolarFeaturesController } = await import(assetUrl("js/polar-features.js"));
const { fetchRefinedBasinJson } = await import(assetUrl("js/polar-refined-basins.js"));
const { decodeFieldToFloat32, parseField } = await import(assetUrl("js/data-contract.js"));
const polarFeatureDataUrls = Object.freeze({
  antarctica: Object.freeze({
    research_stations: assetUrl("data/antarctica_research_stations.json"),
    geographic_names: assetUrl("data/antarctica_geographic_names.json"),
    refined_basins: assetUrl("data/antarctica_refined_basins_search.json"),
  }),
  greenland: Object.freeze({
    research_stations: assetUrl("data/greenland_research_stations.json"),
    geographic_names: assetUrl("data/greenland_geographic_names.json"),
    refined_basins: assetUrl("data/greenland_refined_basins_search.json"),
  }),
});
const lightLogoUrl = assetUrl("3d-ice-logo-light.jpg");
const darkLogoUrl = assetUrl("3d-ice-logo.jpg");
const pageLocale = localeApi
  ? localeApi.normalizeLocale(document.documentElement.dataset.locale || document.documentElement.lang)
  : "en-US";
const numberLocale = localeApi ? localeApi.getIntlLocale(pageLocale) : "en-US";
const isChineseLocale = pageLocale === "zh-CN";
const t = (key, vars = {}) => (localeApi ? localeApi.t(pageLocale, key, vars) : key);
const errorLabel = (key) => {
  const statusMarker = "__STATUS__";
  return t(key, { status: statusMarker }).replace(new RegExp(`\\s*[（(]${statusMarker}[）)]\\s*$`), "");
};
const localizeWorkerStage = (stageKey, stage) =>
  localeApi ? localeApi.localizeWorkerStage(pageLocale, stageKey, stage) : stage || "";
const localizeErrorMessage = (message) =>
  localeApi ? localeApi.localizeErrorMessage(pageLocale, message) : message;
const getLocalizedRegionInfo = (regionKey) =>
  localeApi ? localeApi.getLocalizedRegionInfo(pageLocale, regionKey) : {};
const getLocalizedDatasetInfo = (regionKey, datasetKey) =>
  localeApi ? localeApi.getLocalizedDatasetInfo(pageLocale, regionKey, datasetKey) : {};
if (localeApi) {
  document.documentElement.lang = pageLocale;
  localeApi.initPage({ locale: pageLocale, switchers: ["#panelLocaleSwitcher"] });
}
const SHOWCASE_SCROLL_MESSAGE_TYPE = "3d-ice:showcase-scroll";
const mode = (queryParams.get("mode") || "").toLowerCase();
const regionParam = (queryParams.get("region") || "").toLowerCase();
const presetParam = (queryParams.get("preset") || "").toLowerCase();
const isShowcaseMode = mode === "showcase";
const isPreviewMode = mode === "preview";
const showcaseMobileLinkoutMode = queryParams.get("mobileLinkout") === "1";
const showcaseDesktopInteractiveMode = queryParams.get("desktopInteractive") === "1";
const recordingModeRequested = queryParams.get("recording") === "1";
const recordingModeEnabled = recordingModeRequested && !isShowcaseMode && !isPreviewMode;
const coarsePointerQuery = window.matchMedia("(pointer: coarse)");
const compactViewportQuery = window.matchMedia("(max-width: 980px)");
const narrowViewportQuery = window.matchMedia("(max-width: 820px)");
const runtimeThemeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
const flowMotionMediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
let runtimeThemeObserver = null;
let runtimeThemeStorageHandler = null;
let runtimeThemeMediaHandler = null;

function isCoarsePointerInput() {
  return coarsePointerQuery.matches;
}

function syncPointerModeClass() {
  document.body.classList.toggle("coarse-pointer", isCoarsePointerInput());
}

function shouldUseMobileDrawer() {
  return !isShowcaseMode && !isPreviewMode && isCoarsePointerInput() && compactViewportQuery.matches;
}

function shouldUseShowcaseMobileLinkout() {
  return isShowcaseMode && showcaseMobileLinkoutMode && narrowViewportQuery.matches;
}

function isDesktopInteractiveShowcase() {
  return isShowcaseMode && showcaseDesktopInteractiveMode && !isCoarsePointerInput();
}

function shouldRunShowcaseAutoOrbit() {
  return isShowcaseMode || isPreviewMode;
}

function shouldUseInteractionGate() {
  if (isPreviewMode) return false;
  if (isDesktopInteractiveShowcase()) return false;
  return isShowcaseMode;
}

if (isShowcaseMode) {
  document.body.classList.add("showcase");
  if (presetParam) {
    document.body.classList.add(`showcase-preset-${presetParam.replace(/[^a-z0-9_-]/g, "")}`);
  }
  if (showcaseDesktopInteractiveMode) {
    document.body.classList.add("showcase-desktop-interactive");
  }
  if (showcaseMobileLinkoutMode) {
    document.body.classList.add("showcase-mobile-linkout");
  }
}
if (isPreviewMode) {
  document.body.classList.add("preview-embed");
}
if (recordingModeEnabled) {
  document.body.classList.add("recording-mode");
}
syncPointerModeClass();
if (shouldUseMobileDrawer()) {
  document.body.classList.add("mobile-drawer");
}

const REGIONS = {
  antarctica: {
    key: "antarctica",
    label: "Antarctica",
    intro:
      "Explore Antarctica from every angle. Rotate, zoom, and peel back the ice to uncover a hidden world in interactive 3D.",
    refinedBasinsUrl: assetUrl("data/imbie_refined_basins_v2.json"),
    basinToggleLabel: "Show refined basins",
    basinStatusLabel: "refined basins",
    defaultDatasetKey: "balanced",
    capabilities: {
      velocity: true,
      basalFriction: true,
      flowline: true,
      rise: true,
      oceanCurrents: true,
      refinedBasins: true,
      hydrology: true,
      isostaticRebound: true,
    },
    sources: {
      geometry: {
        text: "MEaSUREs BedMachine Antarctica, Version 4",
        url: "https://nsidc.org/data/NSIDC-0756/versions/4",
      },
      velocity: {
        text: "MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1",
        url: "https://nsidc.org/data/NSIDC-0754/versions/1",
      },
      basalFriction: {
        text: "Jager et al. (2026) Insights from an ensemble inverse method to quantify basal friction uncertainties for ice-sheet models - Part 2: Antarctic ice sheet.",
        url: "https://essopenarchive.org/doi/full/10.22541/essoar.177099457.70593031",
      },
      rise: {
        text: "Galton-Fenzi et al. (2025) Multi-model estimate of Antarctic ice-shelf basal melting and ocean drivers, V1",
        url: "https://data.aad.gov.au/metadata/RISE",
      },
      oceanCurrents: {
        text: "Whole Antarctica Ocean Model 2-km (WAOM2) simulated annual-mean ocean circulation, temperature and salinity",
        url: "https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1027704/full",
      },
      basins: {
        text: "MEaSUREs Antarctic Boundaries for IPY 2007-2009 (Version 2, IMBIE refined basins)",
        url: "https://nsidc.org/data/NSIDC-0709/versions/2",
      },
      hydrology: {
        text: "Ehrenfeucht et al. (2025) Antarctic Wide Subglacial Hydrology Modeling",
        url: "https://zenodo.org/records/12738170",
      },
    },
    datasets: {
      balanced: {
        id: "balanced",
        label: "BedMachine v4 — Balanced",
        summary: "10 km grid; ~3.0 MB",
        metaUrl: assetUrl("data/bedmachine_antarctica_v4_480.meta.json"),
        binUrl: assetUrl("data/bedmachine_antarctica_v4_480.bin"),
        velocityMetaUrl: assetUrl("data/antarctic_ice_velocity_phase_v01_480.meta.json"),
        velocityBinUrl: assetUrl("data/antarctic_ice_velocity_phase_v01_480.bin"),
        basalFrictionMetaUrl: assetUrl("data/antarctica_basal_friction_480.meta.json?v=20260314-taub-v1"),
        basalFrictionBinUrl: assetUrl("data/antarctica_basal_friction_480.bin?v=20260314-taub-v1"),
        riseMetaUrl: assetUrl("data/rise_antarctica_480.meta.json"),
        riseBinUrl: assetUrl("data/rise_antarctica_480.bin"),
        oceanCurrentsMetaUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.meta.json?v=20260312-waom2-combined-v3"),
        oceanCurrentsBinUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.bin?v=20260312-waom2-combined-v3"),
        hydrologyMetaUrl: assetUrl("data/antarctica_subglacial_hydrology_480.meta.json"),
        hydrologyBinUrl: assetUrl("data/antarctica_subglacial_hydrology_480.bin"),
      },
      hd: {
        id: "hd",
        label: "BedMachine v4 — HD",
        summary: "4 km grid; ~18.6 MB",
        metaUrl: assetUrl("data/bedmachine_antarctica_v4_741.meta.json"),
        binUrl: assetUrl("data/bedmachine_antarctica_v4_741.bin"),
        velocityMetaUrl: assetUrl("data/antarctic_ice_velocity_phase_v01_741.meta.json"),
        velocityBinUrl: assetUrl("data/antarctic_ice_velocity_phase_v01_741.bin"),
        basalFrictionMetaUrl: assetUrl("data/antarctica_basal_friction_741.meta.json?v=20260314-taub-v1"),
        basalFrictionBinUrl: assetUrl("data/antarctica_basal_friction_741.bin?v=20260314-taub-v1"),
        riseMetaUrl: assetUrl("data/rise_antarctica_741.meta.json"),
        riseBinUrl: assetUrl("data/rise_antarctica_741.bin"),
        oceanCurrentsMetaUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.meta.json?v=20260312-waom2-combined-v3"),
        oceanCurrentsBinUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.bin?v=20260312-waom2-combined-v3"),
        hydrologyMetaUrl: assetUrl("data/antarctica_subglacial_hydrology_741.meta.json"),
        hydrologyBinUrl: assetUrl("data/antarctica_subglacial_hydrology_741.bin"),
      },
      bedmap3: {
        id: "bedmap3",
        label: "Bedmap3 — Balanced",
        summary: "10 km grid; ~3.0 MB",
        metaUrl: assetUrl("data/bedmap3_antarctica_10km.meta.json"),
        binUrl: assetUrl("data/bedmap3_antarctica_10km.bin"),
        velocityMetaUrl: assetUrl("data/bedmap3_antarctica_velocity_10km.meta.json"),
        velocityBinUrl: assetUrl("data/bedmap3_antarctica_velocity_10km.bin"),
        basalFrictionMetaUrl: assetUrl("data/bedmap3_antarctica_basal_friction_10km.meta.json"),
        basalFrictionBinUrl: assetUrl("data/bedmap3_antarctica_basal_friction_10km.bin"),
        oceanCurrentsMetaUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.meta.json?v=20260312-waom2-combined-v3"),
        oceanCurrentsBinUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.bin?v=20260312-waom2-combined-v3"),
        hydrologyMetaUrl: assetUrl("data/bedmap3_antarctica_subglacial_hydrology_10km.meta.json"),
        hydrologyBinUrl: assetUrl("data/bedmap3_antarctica_subglacial_hydrology_10km.bin"),
        capabilities: {
          velocity: true,
          basalFriction: true,
          flowline: true,
          rise: false,
          oceanCurrents: true,
          refinedBasins: true,
          hydrology: true,
        },
        sources: {
          geometry: {
            text: "Bedmap3 Antarctica, Version 1.0",
            url: "https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2",
          },
        },
        disableBackgroundWarmup: true,
      },
      "bedmap3-hd": {
        id: "bedmap3-hd",
        label: "Bedmap3 — HD",
        summary: "4 km grid; ~18.6 MB, desktop recommended",
        metaUrl: assetUrl("data/bedmap3_antarctica_4km.meta.json"),
        binUrl: assetUrl("data/bedmap3_antarctica_4km.bin"),
        velocityMetaUrl: assetUrl("data/bedmap3_antarctica_velocity_4km.meta.json"),
        velocityBinUrl: assetUrl("data/bedmap3_antarctica_velocity_4km.bin"),
        basalFrictionMetaUrl: assetUrl("data/bedmap3_antarctica_basal_friction_4km.meta.json"),
        basalFrictionBinUrl: assetUrl("data/bedmap3_antarctica_basal_friction_4km.bin"),
        oceanCurrentsMetaUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.meta.json?v=20260312-waom2-combined-v3"),
        oceanCurrentsBinUrl: assetUrl("data/antarctica_ocean_currents_waom2_yr5_annual_combined_cavity80km_remote_open_ocean.bin?v=20260312-waom2-combined-v3"),
        hydrologyMetaUrl: assetUrl("data/bedmap3_antarctica_subglacial_hydrology_4km.meta.json"),
        hydrologyBinUrl: assetUrl("data/bedmap3_antarctica_subglacial_hydrology_4km.bin"),
        capabilities: {
          velocity: true,
          basalFriction: true,
          flowline: true,
          rise: false,
          oceanCurrents: true,
          refinedBasins: true,
          hydrology: true,
        },
        sources: {
          geometry: {
            text: "Bedmap3 Antarctica, Version 1.0",
            url: "https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2",
          },
        },
        disableBackgroundWarmup: true,
      },
    },
  },
  greenland: {
    key: "greenland",
    label: "Greenland",
    intro:
      "Explore Greenland from every angle. Rotate, zoom, and peel back the ice to uncover a hidden world in interactive 3D.",
    refinedBasinsUrl: assetUrl("data/greenland_basins_ps_v1_4_2.json"),
    basinToggleLabel: "Show basins",
    basinStatusLabel: "basins",
    defaultDatasetKey: "3km",
    capabilities: {
      velocity: true,
      basalFriction: true,
      flowline: true,
      rise: false,
      oceanCurrents: true,
      refinedBasins: true,
      hydrology: false,
      isostaticRebound: true,
    },
    sources: {
      geometry: {
        text: "IceBridge BedMachine Greenland, Version 6",
        url: "https://nsidc.org/data/idbmg4/versions/6",
      },
      velocity: {
        text: "MEaSUREs ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities, Version 2",
        url: "https://nsidc.org/data/NSIDC-0776/versions/2",
      },
      basalFriction: {
        text: "Jager et al. (2026) Insights from an ensemble inverse method to quantify basal friction uncertainties for ice-sheet models - Part 1: Greenland ice sheet",
        url: "https://essopenarchive.org/doi/full/10.22541/essoar.177099472.28419248",
      },
      oceanCurrents: {
        text: "Copernicus Marine Arctic Ocean Physics Analysis and Forecast (monthly mean ocean velocity, temperature, and salinity)",
        url: "https://data.marine.copernicus.eu/product/ARCTIC_ANALYSISFORECAST_PHY_002_001/description",
      },
      basins: "Greenland Basins PS v1.4.2 catchment boundaries",
    },
    datasets: {
      "3km": {
        id: "3km",
        label: "Balanced",
        summary: "3 km grid; ~3.1 MB",
        metaUrl: assetUrl("data/bedmachine_greenland_v6_3km.meta.json"),
        binUrl: assetUrl("data/bedmachine_greenland_v6_3km.bin"),
        velocityMetaUrl: assetUrl("data/greenland_ice_velocity_3km.meta.json"),
        velocityBinUrl: assetUrl("data/greenland_ice_velocity_3km.bin"),
        basalFrictionMetaUrl: assetUrl("data/greenland_basal_friction_3km.meta.json?v=20260314-taub-v1"),
        basalFrictionBinUrl: assetUrl("data/greenland_basal_friction_3km.bin?v=20260314-taub-v1"),
        oceanCurrentsMetaUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.meta.json?v=20260312-greenland-layered-v1"),
        oceanCurrentsBinUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.bin?v=20260312-greenland-layered-v1"),
      },
      "1km": {
        id: "1km",
        label: "HD",
        summary: "1 km grid; ~28.2 MB",
        metaUrl: assetUrl("data/bedmachine_greenland_v6_1km.meta.json"),
        binUrl: assetUrl("data/bedmachine_greenland_v6_1km.bin"),
        velocityMetaUrl: assetUrl("data/greenland_ice_velocity_1km.meta.json"),
        velocityBinUrl: assetUrl("data/greenland_ice_velocity_1km.bin"),
        basalFrictionMetaUrl: assetUrl("data/greenland_basal_friction_1km.meta.json?v=20260314-taub-v1"),
        basalFrictionBinUrl: assetUrl("data/greenland_basal_friction_1km.bin?v=20260314-taub-v1"),
        oceanCurrentsMetaUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.meta.json?v=20260312-greenland-layered-v1"),
        oceanCurrentsBinUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.bin?v=20260312-greenland-layered-v1"),
        velocityMeshStride: 3,
        disableBackgroundWarmup: true,
      },
      qrf: {
        id: "qrf",
        label: "QRF 2025 — Balanced",
        summary: "3 km grid; ~3.1 MB",
        metaUrl: assetUrl("data/greenland_qrf_2025_3km.meta.json"),
        binUrl: assetUrl("data/greenland_qrf_2025_3km.bin"),
        velocityMetaUrl: assetUrl("data/greenland_ice_velocity_3km.meta.json"),
        velocityBinUrl: assetUrl("data/greenland_ice_velocity_3km.bin"),
        basalFrictionMetaUrl: assetUrl("data/greenland_basal_friction_3km.meta.json?v=20260314-taub-v1"),
        basalFrictionBinUrl: assetUrl("data/greenland_basal_friction_3km.bin?v=20260314-taub-v1"),
        oceanCurrentsMetaUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.meta.json?v=20260312-greenland-layered-v1"),
        oceanCurrentsBinUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.bin?v=20260312-greenland-layered-v1"),
        sources: {
          geometry: {
            text: "Palmer et al. (2025) Quantile Regression Forest Greenland Subglacial Topography",
            url: "https://doi.org/10.1017/jog.2025.10071",
          },
        },
        disableBackgroundWarmup: true,
      },
      "qrf-hd": {
        id: "qrf-hd",
        label: "QRF 2025 — HD",
        summary: "1 km grid; ~28.2 MB, desktop recommended",
        metaUrl: assetUrl("data/greenland_qrf_2025_1km.meta.json"),
        binUrl: assetUrl("data/greenland_qrf_2025_1km.bin"),
        velocityMetaUrl: assetUrl("data/greenland_ice_velocity_1km.meta.json"),
        velocityBinUrl: assetUrl("data/greenland_ice_velocity_1km.bin"),
        basalFrictionMetaUrl: assetUrl("data/greenland_basal_friction_1km.meta.json?v=20260314-taub-v1"),
        basalFrictionBinUrl: assetUrl("data/greenland_basal_friction_1km.bin?v=20260314-taub-v1"),
        oceanCurrentsMetaUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.meta.json?v=20260312-greenland-layered-v1"),
        oceanCurrentsBinUrl: assetUrl("data/greenland_ocean_currents_cmems_202508.bin?v=20260312-greenland-layered-v1"),
        sources: {
          geometry: {
            text: "Palmer et al. (2025) Quantile Regression Forest Greenland Subglacial Topography",
            url: "https://doi.org/10.1017/jog.2025.10071",
          },
        },
        velocityMeshStride: 3,
        disableBackgroundWarmup: true,
      },
    },
  },
};

Object.entries(REGIONS).forEach(([regionKey, region]) => {
  const localizedRegion = getLocalizedRegionInfo(regionKey);
  if (localizedRegion.label) {
    region.label = localizedRegion.label;
  }
  if (localizedRegion.intro) {
    region.intro = localizedRegion.intro;
  }
  if (localizedRegion.basinToggleLabel) {
    region.basinToggleLabel = localizedRegion.basinToggleLabel;
  }
  if (localizedRegion.basinStatusLabel) {
    region.basinStatusLabel = localizedRegion.basinStatusLabel;
  }
  Object.entries(region.datasets || {}).forEach(([datasetKey, dataset]) => {
    const localizedDataset = getLocalizedDatasetInfo(regionKey, datasetKey);
    if (localizedDataset.label) {
      dataset.label = localizedDataset.label;
    }
    if (localizedDataset.summary) {
      dataset.summary = localizedDataset.summary;
    }
  });
});

function getRegionConfig(regionKey) {
  return REGIONS[regionKey] || REGIONS.antarctica;
}

function getDefaultDatasetKey(regionKey) {
  const region = getRegionConfig(regionKey);
  return region.defaultDatasetKey || Object.keys(region.datasets)[0] || "balanced";
}

function getDatasetConfig(regionKey, datasetKey) {
  const region = getRegionConfig(regionKey);
  const dataset = region.datasets[datasetKey] || region.datasets[getDefaultDatasetKey(regionKey)];
  if (!dataset) return null;
  return {
    ...dataset,
    regionKey: region.key,
    regionLabel: region.label,
    capabilities: { ...region.capabilities, ...(dataset.capabilities || {}) },
    sources: { ...region.sources, ...(dataset.sources || {}) },
    refinedBasinsUrl: dataset.refinedBasinsUrl || region.refinedBasinsUrl || null,
    basinToggleLabel: dataset.basinToggleLabel || region.basinToggleLabel || t("explorer.regions.antarctica.basinToggleLabel"),
    basinStatusLabel: dataset.basinStatusLabel || region.basinStatusLabel || t("explorer.regions.antarctica.basinStatusLabel"),
  };
}

const SHOWCASE_PRESETS = {
  "home-hero": {
    regionKey: "antarctica",
    datasetKey: "balanced",
    exaggeration: 4.8,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: true,
      showVelocity: false,
      showFlowline: false,
      showBasalFriction: false,
      showBasalMelt: false,
      showThermalDriving: false,
      showOceanCurrents: false,
      showRefinedBasins: false,
      showEffectivePressure: false,
      showSubglacialChannels: false,
      showSea: false,
    },
    camera: {
      position: [11.01, 62.36, 79.95],
      target: [4.52, -6.48, 1.43],
      fov: 40,
    },
  },
  "tools-hero": {
    regionKey: "antarctica",
    datasetKey: "balanced",
    exaggeration: 4.8,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: false,
      showVelocity: false,
      showFlowline: false,
      showBasalFriction: false,
      showBasalMelt: false,
      showThermalDriving: false,
      showOceanCurrents: false,
      showRefinedBasins: false,
      showEffectivePressure: false,
      showSubglacialChannels: false,
      showSea: false,
    },
    camera: {
      position: [7.2, 102.3, 99.53],
      target: [6.12, -7.82, 3.69],
      fov: 45,
    },
  },
};

function getShowcasePreset(presetKey) {
  return SHOWCASE_PRESETS[presetKey] || SHOWCASE_PRESETS["home-hero"];
}

const EXPLORER_PRESETS = {
  "antarctica-velocity-flowlines": {
    regionKey: "antarctica",
    datasetKey: "balanced",
    exaggeration: 4.8,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: true,
      showVelocity: true,
      showFlowline: true,
    },
  },
  "antarctica-basin-boundary": {
    regionKey: "antarctica",
    datasetKey: "balanced",
    exaggeration: 4.8,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: true,
      showRefinedBasins: true,
    },
  },
  "antarctica-subglacial-features": {
    regionKey: "antarctica",
    datasetKey: "balanced",
    exaggeration: 4.8,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: false,
      showIceBottom: false,
      showEffectivePressure: true,
      showSubglacialChannels: true,
    },
  },
  "antarctica-ocean-circulations": {
    regionKey: "antarctica",
    datasetKey: "balanced",
    exaggeration: 4.8,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: true,
      showOceanCurrents: true,
      showSea: true,
    },
    oceanLayers: {
      surface: true,
      upper: true,
      mid: true,
      lower: true,
    },
  },
  "greenland-velocity-flowlines": {
    regionKey: "greenland",
    datasetKey: "3km",
    exaggeration: 4.2,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: true,
      showVelocity: true,
      showFlowline: true,
    },
  },
  "greenland-ocean-circulations": {
    regionKey: "greenland",
    datasetKey: "3km",
    exaggeration: 4.2,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: true,
      showOceanCurrents: true,
      showSea: true,
    },
    oceanLayers: {
      surface: true,
      upper: true,
      mid: true,
      lower: true,
    },
  },
};

function getExplorerPreset(presetKey) {
  return EXPLORER_PRESETS[presetKey] || null;
}

const initialExplorerPreset = isShowcaseMode ? getShowcasePreset(presetParam) : getExplorerPreset(presetParam);
const previewRegionKey = initialExplorerPreset?.regionKey || (REGIONS[regionParam] ? regionParam : "antarctica");
const lockedRegionKey = isShowcaseMode ? initialExplorerPreset?.regionKey || "antarctica" : isPreviewMode ? previewRegionKey : null;
const lockedDatasetKey = isShowcaseMode
  ? initialExplorerPreset?.datasetKey || "balanced"
  : isPreviewMode
  ? initialExplorerPreset?.datasetKey || getDefaultDatasetKey(previewRegionKey)
  : null;
const initialRegionKey = lockedRegionKey || initialExplorerPreset?.regionKey || (REGIONS[regionParam] ? regionParam : "antarctica");
const datasetSelectionByRegion = {};

const viewerEl = document.getElementById("viewer");
const viewerShellEl = document.getElementById("viewerShell");
const panelEl = document.querySelector(".panel");
const panelBrandLogoEl = document.querySelector(".panel-brand-logo");
const panelSubtitleEl = document.getElementById("panelSubtitle");
const panelHeaderEl = document.querySelector(".panel-header");
const statusEl = document.getElementById("status");
const metaListEl = document.getElementById("metaList");
const fieldStatsEl = document.getElementById("fieldStats");
const metaSectionOpenState = new Map();
const interactionHintEl = document.getElementById("interactionHint");
const loadingOverlayEl = document.getElementById("loadingOverlay");
const loadingStageEl = document.getElementById("loadingStage");
const loadingFillEl = document.getElementById("loadingFill");
const loadingPercentEl = document.getElementById("loadingPercent");
const loadingHintEl = document.getElementById("loadingHint");
const DEFAULT_INTERACTION_HINT = interactionHintEl ? interactionHintEl.innerHTML : "";

function getEmbeddedTopRoot() {
  try {
    if (window.top && window.top !== window) {
      return window.top.document?.documentElement || null;
    }
  } catch (_error) {
    return null;
  }
  return null;
}

function resolveRuntimeTheme() {
  if (runtimeThemeApi && typeof runtimeThemeApi.resolveTheme === "function") {
    return runtimeThemeApi.resolveTheme();
  }
  return runtimeThemeMediaQuery.matches ? "dark" : "light";
}

function applyRuntimeTheme(theme) {
  const nextTheme =
    runtimeThemeApi && typeof runtimeThemeApi.applyTheme === "function"
      ? runtimeThemeApi.applyTheme(theme)
      : theme === "dark"
      ? "dark"
      : "light";
  if (!runtimeThemeApi) {
    document.documentElement.dataset.theme = nextTheme;
    document.documentElement.style.colorScheme = nextTheme;
  }
  /* Second writer of the brand logo's src. Skipped wherever the panel is
     hidden, or a theme change would fetch the 244 KB logo that the markup
     above deliberately withheld — the embed syncs its theme from the host
     page, so this path does fire there. */
  if (panelBrandLogoEl && !isShowcaseMode && !isPreviewMode) {
    panelBrandLogoEl.src =
      nextTheme === "dark" ? runtimeThemeApi?.darkLogoSrc || darkLogoUrl : runtimeThemeApi?.lightLogoSrc || lightLogoUrl;
  }
  return nextTheme;
}

function syncRuntimeTheme() {
  applyRuntimeTheme(resolveRuntimeTheme());
}

function bindRuntimeTheme() {
  syncRuntimeTheme();

  const topRoot = getEmbeddedTopRoot();
  if (topRoot && "MutationObserver" in window) {
    runtimeThemeObserver = new MutationObserver(() => {
      syncRuntimeTheme();
    });
    runtimeThemeObserver.observe(topRoot, { attributes: true, attributeFilter: ["class"] });
  }

  runtimeThemeStorageHandler = (event) => {
    if (!runtimeThemeApi?.storageKey || event.key === runtimeThemeApi.storageKey || event.key === null) {
      syncRuntimeTheme();
    }
  };
  window.addEventListener("storage", runtimeThemeStorageHandler);

  runtimeThemeMediaHandler = () => {
    syncRuntimeTheme();
  };
  runtimeThemeMediaQuery.addEventListener("change", runtimeThemeMediaHandler);
}

const controlsUI = {
  panelCloseButton: document.getElementById("panelCloseButton"),
  polarSearchInput: document.getElementById("polarSearchInput"),
  polarSearchResults: document.getElementById("polarSearchResults"),
  polarSearchStatus: document.getElementById("polarSearchStatus"),
  polarFeatureDetails: document.getElementById("polarFeatureDetails"),
  showResearchStations: document.getElementById("showResearchStations"),
  showGeographicNames: document.getElementById("showGeographicNames"),
  regionPreset: document.getElementById("regionPreset"),
  resolutionPreset: document.getElementById("resolutionPreset"),
  exaggeration: document.getElementById("exaggeration"),
  exaggerationValue: document.getElementById("exaggerationValue"),
  iceOpacity: document.getElementById("iceOpacity"),
  iceOpacityValue: document.getElementById("iceOpacityValue"),
  showBed: document.getElementById("showBed"),
  showIce: document.getElementById("showIce"),
  showIceBottom: document.getElementById("showIceBottom"),
  showVelocity: document.getElementById("showVelocity"),
  showFlowline: document.getElementById("showFlowline"),
  animateFlow: document.getElementById("animateFlow"),
  flowlineProfileCardMount: document.getElementById("flowlineProfileCardMount"),
  showBasalFriction: document.getElementById("showBasalFriction"),
  showBasalMeltRow: document.getElementById("showBasalMeltRow"),
  showBasalMelt: document.getElementById("showBasalMelt"),
  showThermalDrivingRow: document.getElementById("showThermalDrivingRow"),
  showThermalDriving: document.getElementById("showThermalDriving"),
  showOceanCurrents: document.getElementById("showOceanCurrents"),
  oceanCurrentLayerControls: document.getElementById("oceanCurrentLayerControls"),
  showOceanLayerSurface: document.getElementById("showOceanLayerSurface"),
  showOceanLayerUpper: document.getElementById("showOceanLayerUpper"),
  showOceanLayerMid: document.getElementById("showOceanLayerMid"),
  showOceanLayerLower: document.getElementById("showOceanLayerLower"),
  showRefinedBasins: document.getElementById("showRefinedBasins"),
  showRefinedBasinsLabel: document.getElementById("showRefinedBasinsLabel"),
  showEffectivePressure: document.getElementById("showEffectivePressure"),
  showSubglacialChannels: document.getElementById("showSubglacialChannels"),
  showSea: document.getElementById("showSea"),
  reboundLegendNote: document.getElementById("reboundLegendNote"),
  showIsostaticRebound: document.getElementById("showIsostaticRebound"),
  isostaticReboundControls: document.getElementById("isostaticReboundControls"),
  reboundProgress: document.getElementById("reboundProgress"),
  reboundProgressValue: document.getElementById("reboundProgressValue"),
  reboundProgressNote: document.getElementById("reboundProgressNote"),
  reboundModel: document.getElementById("reboundModel"),
  reboundModelNote: document.getElementById("reboundModelNote"),
  reboundSeaLevel: document.getElementById("reboundSeaLevel"),
  reboundSeaLevelValue: document.getElementById("reboundSeaLevelValue"),
  reboundSeaLevelNote: document.getElementById("reboundSeaLevelNote"),
  highlightEmergentLand: document.getElementById("highlightEmergentLand"),
  wireframe: document.getElementById("wireframe"),
  resetView: document.getElementById("resetView"),
  fullscreenToggle: document.getElementById("fullscreenToggle"),
  interactionToggle: document.getElementById("interactionToggle"),
  viewerFullscreenToggle: document.getElementById("viewerFullscreenToggle"),
  bedLegendSection: document.getElementById("bedLegendSection"),
  legendBar: document.getElementById("legendBar"),
  velocityLegendSection: document.getElementById("velocityLegendSection"),
  velocityLegendBar: document.getElementById("velocityLegendBar"),
  velocityLegendLabels: document.getElementById("velocityLegendLabels"),
  velocityLegendNote: document.getElementById("velocityLegendNote"),
  basalFrictionLegendSection: document.getElementById("basalFrictionLegendSection"),
  basalFrictionLegendBar: document.getElementById("basalFrictionLegendBar"),
  basalFrictionLegendLabels: document.getElementById("basalFrictionLegendLabels"),
  basalFrictionLegendNote: document.getElementById("basalFrictionLegendNote"),
  basalMeltLegendSection: document.getElementById("basalMeltLegendSection"),
  basalMeltLegendBar: document.getElementById("basalMeltLegendBar"),
  basalMeltLegendLabels: document.getElementById("basalMeltLegendLabels"),
  basalMeltLegendNote: document.getElementById("basalMeltLegendNote"),
  thermalDrivingLegendSection: document.getElementById("thermalDrivingLegendSection"),
  thermalDrivingLegendBar: document.getElementById("thermalDrivingLegendBar"),
  thermalDrivingLegendLabels: document.getElementById("thermalDrivingLegendLabels"),
  oceanLegendSection: document.getElementById("oceanLegendSection"),
  oceanLegendCanvas: document.getElementById("oceanLegendCanvas"),
  oceanLegendWarmLabel: document.getElementById("oceanLegendWarmLabel"),
  oceanLegendColdLabel: document.getElementById("oceanLegendColdLabel"),
  oceanLegendFreshLabel: document.getElementById("oceanLegendFreshLabel"),
  oceanLegendSaltyLabel: document.getElementById("oceanLegendSaltyLabel"),
  effectivePressureLegendSection: document.getElementById("effectivePressureLegendSection"),
  effectivePressureLegendBar: document.getElementById("effectivePressureLegendBar"),
  channelLegendSection: document.getElementById("channelLegendSection"),
  channelLegendBar: document.getElementById("channelLegendBar"),
  capturePanel: document.getElementById("capturePanel"),
  captureHideButton: document.getElementById("captureHideButton"),
  capturePlayButton: document.getElementById("capturePlayButton"),
  captureResetButton: document.getElementById("captureResetButton"),
  captureOrbitToggle: document.getElementById("captureOrbitToggle"),
  captureZoomToggle: document.getElementById("captureZoomToggle"),
  captureDirectionCw: document.getElementById("captureDirectionCw"),
  captureDirectionCcw: document.getElementById("captureDirectionCcw"),
  captureSpeed: document.getElementById("captureSpeed"),
  captureSpeedValue: document.getElementById("captureSpeedValue"),
  captureZoomAmount: document.getElementById("captureZoomAmount"),
  captureZoomAmountValue: document.getElementById("captureZoomAmountValue"),
  captureManualButtons: Array.from(document.querySelectorAll("[data-capture-manual]")),
};

const RECORDING_AUTO_ORBIT_RAD_PER_SEC = 0.18;
const RECORDING_AUTO_ZOOM_HZ = 0.12;
const RECORDING_MANUAL_RESPONSE_SEC = 0.12;
const RECORDING_MANUAL_THETA_RAD_PER_SEC = 0.75;
const RECORDING_MANUAL_PHI_RAD_PER_SEC = 0.55;
const RECORDING_PLAYBACK_TILT_SPEED_RATIO = 1.05;
const RECORDING_PLAYBACK_TILT_SPEED_MIN = 0.06;
const RECORDING_PLAYBACK_TILT_SPEED_MAX = 0.32;
const RECORDING_MIN_POLAR_EPS = 0.015;
const RECORDING_FRAME_DT_LIMIT_MS = 48;
const SHOWCASE_AUTO_ORBIT_RAD_PER_SEC = 0.055;
const SHOWCASE_INITIAL_HOLD_SEC = 1.6;
const SHOWCASE_RECORDING_DEFAULT_SPEED = 0.6;
const SHOWCASE_RECORDING_DEFAULT_ZOOM_PERCENT = 15;
const SHOWCASE_INTERACTION_IDLE_RESUME_MS = 800;
const OCEAN_CURRENT_RENDER_ORDER = 9;
const ICE_SURFACE_RENDER_ORDER = 10;
const ICE_SIDE_STRICT_OCCLUSION_RENDER_ORDER = 10.5;
const ICE_BOTTOM_RENDER_ORDER = 11;

let THREE;
let OrbitControlsCtor;
let renderer;
let scene;
let camera;
let orbit;
let bedMesh;
let iceMesh;
let iceBottomMesh;
let iceSideMesh;
let velocitySurfaceMesh;
let basalFrictionMesh;
let basalMeltMesh;
let thermalDrivingMesh;
let effectivePressureMesh;
let subglacialChannelMesh;
let flowlineMesh;
let selectedFlowlineHighlight;
let oceanCurrentMesh;
let seaLevelMesh;
let refinedBasinBedLines;
let refinedBasinSurfaceLines;
let refinedBasinBedLabels;
let refinedBasinSurfaceLabels;
let polarFeaturesController = null;
let currentRefinedBasinData = null;
let refinedBasinLoadPromise = null;
let refinedBasinLoadingUrl = null;
let velocityField;
let reboundModule = null;
let reboundModulePromise = null;
let reboundWorker = null;
let reboundWorkerUnavailable = false;
const reboundWorkerPending = new Map();
let reboundWorkerRequestSeq = 0;
let reboundLoadPromise = null;
let reboundGeometryFrame = null;
let reboundGeometryWantsNormals = false;
let reboundSeaLevelDebounce = null;
let velocityDataTexture = null;
let velocityContinuousSamplingWarningIssued = false;
let basalMeltDataTexture = null;
let thermalDrivingDataTexture = null;
let riseContinuousSamplingWarningIssued = false;
let animationHandle;
let currentRegionKey = initialRegionKey;
let currentDatasetKey = lockedDatasetKey || initialExplorerPreset?.datasetKey || getDefaultDatasetKey(currentRegionKey);
let loadGeneration = 0;
let interactionGateEnabled = false;
let interactionGateActive = true;
let interactionGateTapState = null;
let showcaseInteractionResumeTimer = null;
let showcaseLastActivityAtMs = 0;
let mobileDrawerEnabled = false;
let mobilePanelOpen = false;
let mobilePanelOffsetPx = 0;
let mobilePanelClosedOffsetPx = 0;
let mobilePanelDragState = null;
let mobilePanelIgnoreTap = false;
let mobilePanelResizeObserver = null;
let currentCoreContext = null;
let currentVelocityMeta = null;
let currentBasalFrictionMeta = null;
let currentRiseMeta = null;
let currentHydrologyMeta = null;
let currentOceanCurrentMeta = null;
let currentVelocityMedianSpeed = Number.NaN;
let velocityLoadPromise = null;
let basalFrictionLoadPromise = null;
let riseLoadPromise = null;
let hydrologyLoadPromise = null;
let oceanCurrentLoadPromise = null;
let viewerInteracted = false;
let backgroundWarmupScheduled = false;
let backgroundWarmupStarted = false;
let backgroundWarmupTimer = null;
let viewerResizeObserver = null;
let viewerResizeAnimationFrame = 0;
let viewerResizeStabilizeTimer = null;
let lastRenderFrameTimeMs = 0;
let pendingViewResetOnLoad = false;
let geometryWorker = null;
let geometryWorkerRequestSeq = 0;
let flowlineRaycaster = null;
let flowlinePointerNdc = null;
let flowlinePickGesture = null;
let selectedFlowlineState = null;
let flowlineGuidanceStatusShown = false;
const geometryWorkerPending = new Map();
const refinedBasinLabelTextureCache = new Map();
let refinedBasinLabelTexturePixels = 0;
const refinedBasinDataCache = new Map();
const refinedBasinDataLoadPromises = new Map();
const refinedBasinUnavailableUrls = new Set();
const recordingMotionState = {
  enabled: recordingModeEnabled,
  panelVisible: recordingModeEnabled,
  playing: false,
  orbitEnabled: true,
  zoomEnabled: true,
  direction: "cw",
  speed: 0.7,
  zoomAmount: 0.08,
  autoElapsedSec: 0,
  baseTarget: null,
  baseTheta: Number.NaN,
  basePhi: Number.NaN,
  baseRadius: Number.NaN,
  manualIntent: { theta: 0, phi: 0, radius: 0 },
  manualVelocity: { theta: 0, phi: 0, radius: 0 },
  manualKeyState: {
    orbitLeft: false,
    orbitRight: false,
    tiltUp: false,
    tiltDown: false,
    zoomIn: false,
    zoomOut: false,
  },
  manualButtonState: {
    orbitLeft: false,
    orbitRight: false,
    tiltUp: false,
    tiltDown: false,
    zoomIn: false,
    zoomOut: false,
  },
};
const showcaseMotionState = {
  enabled: isShowcaseMode || isPreviewMode,
  active: false,
  userInteracting: false,
  baseTarget: null,
  baseTheta: Number.NaN,
  basePhi: Number.NaN,
  baseRadius: Number.NaN,
  autoElapsedSec: 0,
};
let recordingScratchOffset = null;
let recordingScratchSpherical = null;
let recordingScratchTarget = null;

const SHOWCASE_HINT_IDLE = t("explorer.interaction.showcaseIdleHtml");
const SHOWCASE_HINT_ACTIVE = t("explorer.interaction.showcaseActiveHtml");
const TOUCH_HINT_IDLE = t("explorer.interaction.touchIdleHtml");
const TOUCH_HINT_ACTIVE = t("explorer.interaction.touchActiveHtml");
const INTERACTION_TAP_MOVE_THRESHOLD_PX = 8;
const MOBILE_PANEL_PEEK_PX = 56;
const MOBILE_PANEL_DRAG_MOVE_THRESHOLD_PX = 3;
const MOBILE_PANEL_DRAG_SLOW_GAIN = 0.24;
const MOBILE_PANEL_DRAG_FAST_GAIN = 0.88;
const MOBILE_PANEL_DRAG_FAST_SPEED_PX_PER_MS = 1.2;
const MOBILE_PANEL_FLING_SPEED_PX_PER_MS = 0.55;
const MOBILE_PANEL_FLING_MIN_TRAVEL_PX = 24;
const OCEAN_CURRENT_LAYER_ORDER = ["surface", "upper", "mid", "lower"];
const OCEAN_CURRENT_LAYER_CONTROL_IDS = {
  surface: "showOceanLayerSurface",
  upper: "showOceanLayerUpper",
  mid: "showOceanLayerMid",
  lower: "showOceanLayerLower",
};
function getDefaultOceanCurrentLayerSelection(regionKey = currentRegionKey) {
  if (regionKey === "antarctica") {
    return {
      surface: true,
      upper: false,
      mid: false,
      lower: false,
    };
  }
  return {
    surface: true,
    upper: true,
    mid: true,
    lower: true,
  };
}

function cloneOceanCurrentLayerSelection(selection) {
  return {
    surface: Boolean(selection?.surface),
    upper: Boolean(selection?.upper),
    mid: Boolean(selection?.mid),
    lower: Boolean(selection?.lower),
  };
}

const rememberedOceanCurrentLayerSelectionByRegion = {
  antarctica: getDefaultOceanCurrentLayerSelection("antarctica"),
  greenland: getDefaultOceanCurrentLayerSelection("greenland"),
};

function getReadyStatusText(context) {
  if (isShowcaseMode) return "";
  if (isPreviewMode) return t("explorer.status.previewReady");
  return context
    ? t("explorer.status.readyWithContext", {
        region: context.dataset.regionLabel,
        dataset: context.dataset.label,
      })
    : t("explorer.status.ready");
}

function getLoadingStatusText(region, dataset) {
  if (isShowcaseMode) return "";
  if (isPreviewMode) {
    return t("explorer.status.loadingPreview", { region: region.label });
  }
  return t("explorer.status.loadingCoreTerrain", { region: region.label, dataset: dataset.label });
}

function setToggleAvailability(control, enabled) {
  if (!control) return;
  control.disabled = !enabled;
  if (!enabled) {
    control.checked = false;
  }
}

function getOceanCurrentSeedBucketKeys(oceanMeta) {
  const samplingBuckets = Array.isArray(oceanMeta?.sampling?.seed_buckets) ? oceanMeta.sampling.seed_buckets : [];
  const bucketKeys = samplingBuckets
    .map((bucket) => (bucket && typeof bucket.key === "string" ? bucket.key : ""))
    .filter((key) => key.length > 0);
  if (bucketKeys.length) return bucketKeys;
  const bucketCounts = oceanMeta?.coverage?.streamlines_by_seed_bucket;
  return bucketCounts && typeof bucketCounts === "object" ? Object.keys(bucketCounts) : [];
}

function getOceanCurrentLayerFromBucketKey(bucketKey) {
  if (typeof bucketKey !== "string") return "";
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    if (bucketKey.endsWith(`_${layer}`) || bucketKey === layer) return layer;
  }
  return "";
}

function getOceanCurrentLayerSplitInfo(oceanMeta) {
  const availableByLayer = { surface: false, upper: false, mid: false, lower: false };
  const bucketCounts = oceanMeta?.coverage?.streamlines_by_seed_bucket;
  if (!bucketCounts || typeof bucketCounts !== "object") {
    return { enabled: false, availableByLayer, orderedBuckets: [] };
  }
  const orderedBuckets = [];
  for (const bucketKey of getOceanCurrentSeedBucketKeys(oceanMeta)) {
    const bucketCount = Number(bucketCounts[bucketKey] || 0);
    if (!(bucketCount > 0)) continue;
    const layer = getOceanCurrentLayerFromBucketKey(bucketKey);
    if (!layer) {
      return { enabled: false, availableByLayer: { surface: false, upper: false, mid: false, lower: false }, orderedBuckets: [] };
    }
    availableByLayer[layer] = true;
    orderedBuckets.push({ key: bucketKey, layer, count: bucketCount });
  }
  return {
    enabled: orderedBuckets.length > 0,
    availableByLayer,
    orderedBuckets,
  };
}

function getOceanCurrentLayerControl(layer) {
  return controlsUI[OCEAN_CURRENT_LAYER_CONTROL_IDS[layer]] || null;
}

function getRememberedOceanCurrentLayerSelection(regionKey = currentRegionKey) {
  if (!rememberedOceanCurrentLayerSelectionByRegion[regionKey]) {
    rememberedOceanCurrentLayerSelectionByRegion[regionKey] = getDefaultOceanCurrentLayerSelection(regionKey);
  }
  return rememberedOceanCurrentLayerSelectionByRegion[regionKey];
}

function rememberOceanCurrentLayerSelection(regionKey = currentRegionKey) {
  const remembered = getRememberedOceanCurrentLayerSelection(regionKey);
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    const control = getOceanCurrentLayerControl(layer);
    if (!control) continue;
    remembered[layer] = Boolean(control.checked);
  }
}

function restoreOceanCurrentLayerSelection(regionKey = currentRegionKey) {
  const remembered = cloneOceanCurrentLayerSelection(getRememberedOceanCurrentLayerSelection(regionKey));
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    const control = getOceanCurrentLayerControl(layer);
    if (!control) continue;
    control.checked = Boolean(remembered[layer]);
  }
}

function hasActiveOceanCurrentLayerSelection(oceanMeta = currentOceanCurrentMeta) {
  const splitInfo = getOceanCurrentLayerSplitInfo(oceanMeta);
  if (!splitInfo.enabled) return true;
  return OCEAN_CURRENT_LAYER_ORDER.some((layer) => splitInfo.availableByLayer[layer] && getOceanCurrentLayerControl(layer)?.checked);
}

function updateOceanCurrentLayerControls() {
  const region = getRegionConfig(currentRegionKey);
  const datasetKey = region.datasets[currentDatasetKey]
    ? currentDatasetKey
    : datasetSelectionByRegion[region.key] || getDefaultDatasetKey(region.key);
  const dataset = getDatasetConfig(region.key, datasetKey);
  const capabilities = dataset?.capabilities || region.capabilities;
  const oceanCurrentAvailable = Boolean(
    !isShowcaseMode && capabilities.oceanCurrents && dataset?.oceanCurrentsMetaUrl && dataset?.oceanCurrentsBinUrl
  );
  const splitInfo = getOceanCurrentLayerSplitInfo(currentOceanCurrentMeta);
  const showControls = Boolean(oceanCurrentAvailable && splitInfo.enabled);
  if (controlsUI.oceanCurrentLayerControls) {
    controlsUI.oceanCurrentLayerControls.hidden = !showControls;
  }
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    const control = getOceanCurrentLayerControl(layer);
    if (!control) continue;
    control.disabled = !(showControls && controlsUI.showOceanCurrents.checked && splitInfo.availableByLayer[layer]);
  }
}

function updateOceanCurrentLayerVisibility(mesh = oceanCurrentMesh, oceanMeta = currentOceanCurrentMeta) {
  if (!mesh) return;
  const splitInfo = getOceanCurrentLayerSplitInfo(oceanMeta);
  mesh.visible = Boolean(controlsUI.showOceanCurrents.checked && hasActiveOceanCurrentLayerSelection(oceanMeta));
  const layerMeshes = mesh.userData?.layerMeshes;
  if (!layerMeshes || !splitInfo.enabled) return;
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    const layerMesh = layerMeshes[layer];
    if (!layerMesh) continue;
    layerMesh.visible = Boolean(splitInfo.availableByLayer[layer] && getOceanCurrentLayerControl(layer)?.checked);
  }
}

function canUseMainThreadOceanCurrentFallback(oceanMeta, oceanBuffer) {
  const segmentCount = Number(oceanMeta?.segment_count || oceanMeta?.flowline_segment_count || 0);
  const streamlineCount = Number(oceanMeta?.streamline_count || oceanMeta?.flowline_count || 0);
  const bufferBytes = oceanBuffer instanceof ArrayBuffer ? oceanBuffer.byteLength : 0;
  const segmentSafe = !(segmentCount > 0) || segmentCount <= OCEAN_CURRENT_MAIN_THREAD_FALLBACK_SEGMENT_LIMIT;
  const streamlineSafe = !(streamlineCount > 0) || streamlineCount <= OCEAN_CURRENT_MAIN_THREAD_FALLBACK_SEGMENT_LIMIT;
  const bufferSafe = !(bufferBytes > 0) || bufferBytes <= OCEAN_CURRENT_MAIN_THREAD_FALLBACK_BIN_BYTES_LIMIT;
  return segmentSafe && streamlineSafe && bufferSafe;
}

function getDatasetBasinUnavailable(dataset) {
  const basinsUrl = typeof dataset?.refinedBasinsUrl === "string" ? dataset.refinedBasinsUrl : "";
  return basinsUrl ? refinedBasinUnavailableUrls.has(basinsUrl) : false;
}

function updateRefinedBasinToggleLabel(regionKey) {
  if (!controlsUI.showRefinedBasinsLabel) return;
  const region = getRegionConfig(regionKey);
  controlsUI.showRefinedBasinsLabel.textContent =
    region.basinToggleLabel || t("explorer.regions.antarctica.basinToggleLabel");
}

function updateResolutionControlAvailability({ isLoading = false } = {}) {
  if (!controlsUI.resolutionPreset) return;
  const region = getRegionConfig(currentRegionKey);
  const datasetCount = Object.keys(region.datasets).length;
  controlsUI.resolutionPreset.disabled = Boolean(isLoading || lockedDatasetKey || datasetCount < 2);
}

function updateRegionControlAvailability({ isLoading = false } = {}) {
  if (!controlsUI.regionPreset) return;
  controlsUI.regionPreset.disabled = Boolean(isLoading || lockedRegionKey);
}

function syncRiseToggleRowVisibility(capabilities) {
  const showRiseRows = Boolean(capabilities?.rise);
  if (!showRiseRows) {
    controlsUI.showBasalMelt.checked = false;
    controlsUI.showThermalDriving.checked = false;
  }
  if (controlsUI.showBasalMeltRow) {
    controlsUI.showBasalMeltRow.hidden = !showRiseRows;
  }
  if (controlsUI.showThermalDrivingRow) {
    controlsUI.showThermalDrivingRow.hidden = !showRiseRows;
  }
}

function updateRegionLayerAvailability(regionKey) {
  const region = getRegionConfig(regionKey);
  const datasetKey = datasetSelectionByRegion[region.key] || getDefaultDatasetKey(region.key);
  const dataset = getDatasetConfig(region.key, datasetKey);
  const capabilities = dataset?.capabilities || region.capabilities;
  const basinAvailable = Boolean(
    !isShowcaseMode && capabilities.refinedBasins && dataset?.refinedBasinsUrl && !getDatasetBasinUnavailable(dataset)
  );
  const basalFrictionAvailable = Boolean(
    !isShowcaseMode && capabilities.basalFriction && dataset?.basalFrictionMetaUrl && dataset?.basalFrictionBinUrl
  );
  const riseAvailable = Boolean(!isShowcaseMode && capabilities.rise && dataset?.riseMetaUrl && dataset?.riseBinUrl);
  const oceanCurrentAvailable = Boolean(
    !isShowcaseMode && capabilities.oceanCurrents && dataset?.oceanCurrentsMetaUrl && dataset?.oceanCurrentsBinUrl
  );
  syncRiseToggleRowVisibility(capabilities);
  setToggleAvailability(controlsUI.showVelocity, capabilities.velocity);
  setToggleAvailability(controlsUI.showFlowline, capabilities.velocity && capabilities.flowline);
  setToggleAvailability(controlsUI.showBasalFriction, basalFrictionAvailable);
  setToggleAvailability(controlsUI.showBasalMelt, riseAvailable);
  setToggleAvailability(controlsUI.showThermalDriving, riseAvailable);
  setToggleAvailability(controlsUI.showOceanCurrents, oceanCurrentAvailable);
  setToggleAvailability(controlsUI.showRefinedBasins, basinAvailable);
  setToggleAvailability(controlsUI.showEffectivePressure, capabilities.hydrology);
  setToggleAvailability(controlsUI.showSubglacialChannels, capabilities.hydrology);
  setToggleAvailability(
    controlsUI.showIsostaticRebound,
    Boolean(!isShowcaseMode && capabilities.isostaticRebound)
  );
  updateOceanCurrentLayerControls();
  updateLegendVisibility();
}

function updateLegendVisibility() {
  const region = getRegionConfig(currentRegionKey);
  const datasetKey = region.datasets[currentDatasetKey]
    ? currentDatasetKey
    : datasetSelectionByRegion[region.key] || getDefaultDatasetKey(region.key);
  const dataset = getDatasetConfig(region.key, datasetKey);
  const capabilities = dataset?.capabilities || region.capabilities;
  const basalFrictionAvailable = Boolean(
    !isShowcaseMode && capabilities.basalFriction && dataset?.basalFrictionMetaUrl && dataset?.basalFrictionBinUrl
  );
  const riseAvailable = Boolean(!isShowcaseMode && capabilities.rise && dataset?.riseMetaUrl && dataset?.riseBinUrl);
  const oceanCurrentAvailable = Boolean(
    !isShowcaseMode && capabilities.oceanCurrents && dataset?.oceanCurrentsMetaUrl && dataset?.oceanCurrentsBinUrl
  );
  const hydrologyAvailable = Boolean(!isShowcaseMode && capabilities.hydrology);
  const oceanCurrentVisible = Boolean(
    oceanCurrentAvailable && controlsUI.showOceanCurrents.checked && hasActiveOceanCurrentLayerSelection(currentOceanCurrentMeta)
  );

  if (controlsUI.bedLegendSection) {
    controlsUI.bedLegendSection.hidden = !controlsUI.showBed.checked;
  }
  if (controlsUI.reboundLegendNote) {
    controlsUI.reboundLegendNote.hidden = !controlsUI.showIsostaticRebound?.checked;
    if (controlsUI.showIsostaticRebound?.checked) {
      controlsUI.reboundLegendNote.textContent = t("explorer.rebound.legendNote");
    }
  }
  if (controlsUI.velocityLegendSection) {
    controlsUI.velocityLegendSection.hidden = !(capabilities.velocity && (controlsUI.showVelocity.checked || controlsUI.showFlowline.checked));
  }
  if (controlsUI.basalFrictionLegendSection) {
    controlsUI.basalFrictionLegendSection.hidden = !(basalFrictionAvailable && controlsUI.showBasalFriction.checked);
  }
  if (controlsUI.basalMeltLegendSection) {
    controlsUI.basalMeltLegendSection.hidden = !(riseAvailable && controlsUI.showBasalMelt.checked);
  }
  if (controlsUI.thermalDrivingLegendSection) {
    controlsUI.thermalDrivingLegendSection.hidden = !(riseAvailable && controlsUI.showThermalDriving.checked);
  }
  if (controlsUI.oceanLegendSection) {
    controlsUI.oceanLegendSection.hidden = !oceanCurrentVisible;
  }
  if (controlsUI.effectivePressureLegendSection) {
    controlsUI.effectivePressureLegendSection.hidden = !(hydrologyAvailable && controlsUI.showEffectivePressure.checked);
  }
  if (controlsUI.channelLegendSection) {
    controlsUI.channelLegendSection.hidden = !(hydrologyAvailable && controlsUI.showSubglacialChannels.checked);
  }
}

function populateResolutionPreset(regionKey, preferredDatasetKey = null) {
  const region = getRegionConfig(regionKey);
  const candidateKey = preferredDatasetKey || datasetSelectionByRegion[region.key] || getDefaultDatasetKey(region.key);
  const selectedKey = region.datasets[candidateKey] ? candidateKey : getDefaultDatasetKey(region.key);

  controlsUI.resolutionPreset.innerHTML = "";
  Object.values(region.datasets).forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.label} (${dataset.summary})`;
    controlsUI.resolutionPreset.appendChild(option);
  });

  controlsUI.resolutionPreset.value = selectedKey;
  datasetSelectionByRegion[region.key] = selectedKey;
  return selectedKey;
}

function refreshRegionUi(regionKey, preferredDatasetKey = null) {
  const region = getRegionConfig(regionKey);
  if (panelSubtitleEl) {
    panelSubtitleEl.textContent = region.intro;
  }
  updateRefinedBasinToggleLabel(region.key);
  if (controlsUI.regionPreset) {
    controlsUI.regionPreset.value = region.key;
  }
  const resolvedDatasetKey = populateResolutionPreset(region.key, preferredDatasetKey);
  updateRegionLayerAvailability(region.key);
  updateRegionControlAvailability();
  updateResolutionControlAvailability();
  updateOceanCurrentLegend();
  return resolvedDatasetKey;
}

function applyExplorerPresetControls(preset) {
  if (!preset) return;

  const toggleDefaults = {
    showBed: true,
    showIce: true,
    showIceBottom: true,
    showVelocity: false,
    showFlowline: false,
    showBasalFriction: false,
    showBasalMelt: false,
    showThermalDriving: false,
    showOceanCurrents: false,
    showRefinedBasins: false,
    showEffectivePressure: false,
    showSubglacialChannels: false,
    showSea: false,
    showIsostaticRebound: false,
  };
  const presetToggles = preset.toggles || {};
  for (const [key, defaultValue] of Object.entries(toggleDefaults)) {
    if (!controlsUI[key]) continue;
    controlsUI[key].checked = Object.prototype.hasOwnProperty.call(presetToggles, key)
      ? Boolean(presetToggles[key])
      : defaultValue;
  }

  if (controlsUI.wireframe) {
    controlsUI.wireframe.checked = false;
  }

  if (controlsUI.exaggeration) {
    const exaggeration = Number.isFinite(preset.exaggeration) ? preset.exaggeration : Number(controlsUI.exaggeration.value);
    controlsUI.exaggeration.value = String(exaggeration);
    controlsUI.exaggerationValue.textContent = `${exaggeration.toFixed(1).replace(/\.0$/, "")}x`;
  }

  if (controlsUI.iceOpacity) {
    const iceOpacity = Number.isFinite(preset.iceOpacity) ? preset.iceOpacity : Number(controlsUI.iceOpacity.value);
    controlsUI.iceOpacity.value = String(iceOpacity);
    controlsUI.iceOpacityValue.textContent = iceOpacity.toFixed(2);
  }

  if (controlsUI.reboundProgress) controlsUI.reboundProgress.value = "100";
  if (controlsUI.reboundSeaLevel) controlsUI.reboundSeaLevel.value = "0";
  if (controlsUI.reboundModel) controlsUI.reboundModel.value = "flexural";
  if (controlsUI.highlightEmergentLand) controlsUI.highlightEmergentLand.checked = true;

  const oceanLayers = preset.oceanLayers;
  if (oceanLayers) {
    if (controlsUI.showOceanLayerSurface) controlsUI.showOceanLayerSurface.checked = Boolean(oceanLayers.surface);
    if (controlsUI.showOceanLayerUpper) controlsUI.showOceanLayerUpper.checked = Boolean(oceanLayers.upper);
    if (controlsUI.showOceanLayerMid) controlsUI.showOceanLayerMid.checked = Boolean(oceanLayers.mid);
    if (controlsUI.showOceanLayerLower) controlsUI.showOceanLayerLower.checked = Boolean(oceanLayers.lower);
  }
}

datasetSelectionByRegion[currentRegionKey] = currentDatasetKey;
currentDatasetKey = refreshRegionUi(currentRegionKey, currentDatasetKey);

if (isShowcaseMode) {
  controlsUI.regionPreset.value = initialExplorerPreset?.regionKey || "antarctica";
  controlsUI.resolutionPreset.value = initialExplorerPreset?.datasetKey || "balanced";
  applyExplorerPresetControls(initialExplorerPreset);
} else {
  applyExplorerPresetControls(initialExplorerPreset);
  if (shouldUseMobileDrawer()) {
    controlsUI.iceOpacity.value = "1";
    controlsUI.iceOpacityValue.textContent = "1.00";
  }
  if (isPreviewMode && interactionHintEl) {
    interactionHintEl.innerHTML = "";
  }
}
updateLegendVisibility();

const DEFAULT_CAMERA_POSES = {
  antarctica: {
    position: [3, 53, 77],
    target: [3.5, -7, 3],
    fov: 45,
  },
  greenland: {
    position: [3.36, 81.62, 82.79],
    target: [5.14, -11.77, 6.48],
    fov: 45,
  },
};

function getDefaultCameraPose(regionKey = currentRegionKey) {
  if (isShowcaseMode) {
    return initialExplorerPreset?.camera || SHOWCASE_PRESETS["home-hero"].camera;
  }
  return DEFAULT_CAMERA_POSES[regionKey] || DEFAULT_CAMERA_POSES.antarctica;
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function getShowcaseMotionConfig() {
  if (isShowcaseMode && recordingModeRequested) {
    const requestedSpeed = Number(queryParams.get("recordingSpeed"));
    const requestedZoomPercent = Number(queryParams.get("recordingZoomAmount"));
    return {
      holdSec: 0,
      orbitRadPerSec: RECORDING_AUTO_ORBIT_RAD_PER_SEC,
      zoomHz: RECORDING_AUTO_ZOOM_HZ,
      speed: clamp(
        Number.isFinite(requestedSpeed) ? requestedSpeed : SHOWCASE_RECORDING_DEFAULT_SPEED,
        0.25,
        2
      ),
      zoomAmount: clamp(
        (Number.isFinite(requestedZoomPercent) ? requestedZoomPercent : SHOWCASE_RECORDING_DEFAULT_ZOOM_PERCENT) / 100,
        0,
        0.3
      ),
    };
  }

  return {
    holdSec: SHOWCASE_INITIAL_HOLD_SEC,
    orbitRadPerSec: SHOWCASE_AUTO_ORBIT_RAD_PER_SEC,
    zoomHz: 0,
    speed: 1,
    zoomAmount: 0,
  };
}

function isEditableElementActive() {
  const activeEl = document.activeElement;
  if (!activeEl) return false;
  const activeTag = activeEl.tagName || "";
  return activeTag === "INPUT" || activeTag === "SELECT" || activeTag === "TEXTAREA" || activeEl.isContentEditable;
}

function setTransientStatus(message, durationMs = 1800) {
  if (!statusEl || !message) return;
  const previous = statusEl.textContent;
  statusEl.textContent = message;
  window.setTimeout(() => {
    if (statusEl.textContent === message) {
      statusEl.textContent = previous;
    }
  }, durationMs);
}

function isFlowlineGuidanceVisible() {
  return Boolean(controlsUI.showFlowline?.checked && flowlineMesh && flowlineMesh.visible);
}

function maybeShowFlowlineGuidanceStatus() {
  if (flowlineGuidanceStatusShown || isPreviewMode || isShowcaseMode) return;
  if (!statusEl || isCoarsePointerInput() || selectedFlowlineState) return;
  if (!isFlowlineGuidanceVisible()) return;
  flowlineGuidanceStatusShown = true;
  setTransientStatus(t("explorer.status.flowlineProfileHint"), 2400);
}

function refreshFlowlineGuidanceUi() {
  updateInteractionHint(getShowcaseInteractionHintActiveState());
  maybeShowFlowlineGuidanceStatus();
}

function formatRecordingSpeedLabel(value) {
  return `${Number(value).toFixed(2)}x`;
}

function formatRecordingZoomLabel(value) {
  const percent = Number(value) * 100;
  const digits = Number.isInteger(percent) ? 0 : 1;
  return `${percent.toFixed(digits)}%`;
}

function ensureRecordingScratchObjects() {
  if (!THREE) return false;
  if (!recordingScratchOffset) {
    recordingScratchOffset = new THREE.Vector3();
  }
  if (!recordingScratchSpherical) {
    recordingScratchSpherical = new THREE.Spherical();
  }
  if (!recordingScratchTarget) {
    recordingScratchTarget = new THREE.Vector3();
  }
  return true;
}

function getRecordingStateSnapshot() {
  return {
    enabled: Boolean(recordingModeEnabled),
    panelVisible: Boolean(recordingModeEnabled && recordingMotionState.panelVisible),
    playing: Boolean(recordingModeEnabled && recordingMotionState.playing),
    orbitEnabled: Boolean(recordingModeEnabled && recordingMotionState.orbitEnabled),
    zoomEnabled: Boolean(recordingModeEnabled && recordingMotionState.zoomEnabled),
    direction: recordingMotionState.direction,
    speed: Number(recordingMotionState.speed.toFixed(2)),
    zoomAmount: Number(recordingMotionState.zoomAmount.toFixed(3)),
  };
}

function updateRecordingControlsUi() {
  if (!controlsUI.capturePanel) return;
  const enabled = Boolean(recordingModeEnabled);
  controlsUI.capturePanel.hidden = !enabled || !recordingMotionState.panelVisible;
  if (!enabled) return;

  const hasAutoMotion = recordingMotionState.orbitEnabled || recordingMotionState.zoomEnabled;
  if (controlsUI.capturePlayButton) {
    controlsUI.capturePlayButton.textContent = recordingMotionState.playing
      ? t("explorer.capture.stop")
      : t("explorer.capture.play");
    controlsUI.capturePlayButton.disabled = !recordingMotionState.playing && !hasAutoMotion;
    controlsUI.capturePlayButton.setAttribute("aria-pressed", recordingMotionState.playing ? "true" : "false");
  }
  if (controlsUI.captureOrbitToggle) {
    controlsUI.captureOrbitToggle.checked = recordingMotionState.orbitEnabled;
  }
  if (controlsUI.captureZoomToggle) {
    controlsUI.captureZoomToggle.checked = recordingMotionState.zoomEnabled;
  }
  if (controlsUI.captureSpeed) {
    controlsUI.captureSpeed.value = String(recordingMotionState.speed);
  }
  if (controlsUI.captureSpeedValue) {
    controlsUI.captureSpeedValue.textContent = formatRecordingSpeedLabel(recordingMotionState.speed);
  }
  if (controlsUI.captureZoomAmount) {
    controlsUI.captureZoomAmount.value = String(recordingMotionState.zoomAmount * 100);
  }
  if (controlsUI.captureZoomAmountValue) {
    controlsUI.captureZoomAmountValue.textContent = formatRecordingZoomLabel(recordingMotionState.zoomAmount);
  }
  if (controlsUI.captureDirectionCw) {
    controlsUI.captureDirectionCw.classList.toggle("is-active", recordingMotionState.direction === "cw");
    controlsUI.captureDirectionCw.setAttribute("aria-pressed", recordingMotionState.direction === "cw" ? "true" : "false");
  }
  if (controlsUI.captureDirectionCcw) {
    controlsUI.captureDirectionCcw.classList.toggle("is-active", recordingMotionState.direction === "ccw");
    controlsUI.captureDirectionCcw.setAttribute("aria-pressed", recordingMotionState.direction === "ccw" ? "true" : "false");
  }
}

function setRecordingPanelVisible(visible, { announce = true } = {}) {
  if (!recordingModeEnabled) return;
  recordingMotionState.panelVisible = Boolean(visible);
  updateRecordingControlsUi();
  if (!announce) return;
  if (recordingMotionState.panelVisible) {
    setTransientStatus(t("explorer.capture.shown"));
  } else {
    setTransientStatus(t("explorer.capture.hiddenHint"));
  }
}

function getRecordingManualAxisValue(negativeKey, positiveKey) {
  const keyState = recordingMotionState.manualKeyState;
  const buttonState = recordingMotionState.manualButtonState;
  const negative = Number(Boolean(keyState[negativeKey])) + Number(Boolean(buttonState[negativeKey]));
  const positive = Number(Boolean(keyState[positiveKey])) + Number(Boolean(buttonState[positiveKey]));
  return clamp(positive - negative, -1, 1);
}

function syncRecordingManualIntent() {
  recordingMotionState.manualIntent.theta = getRecordingManualAxisValue("orbitLeft", "orbitRight");
  recordingMotionState.manualIntent.phi = getRecordingManualAxisValue("tiltUp", "tiltDown");
  recordingMotionState.manualIntent.radius = getRecordingManualAxisValue("zoomIn", "zoomOut");
}

function zeroRecordingManualVelocity() {
  recordingMotionState.manualVelocity.theta = 0;
  recordingMotionState.manualVelocity.phi = 0;
  recordingMotionState.manualVelocity.radius = 0;
}

function clearRecordingManualState({ preserveVelocity = false } = {}) {
  for (const key of Object.keys(recordingMotionState.manualKeyState)) {
    recordingMotionState.manualKeyState[key] = false;
  }
  for (const key of Object.keys(recordingMotionState.manualButtonState)) {
    recordingMotionState.manualButtonState[key] = false;
  }
  syncRecordingManualIntent();
  if (!preserveVelocity) {
    zeroRecordingManualVelocity();
  }
}

function syncOrbitImmediately() {
  if (!orbit) return;
  const previousDamping = orbit.enableDamping;
  orbit.enableDamping = false;
  orbit.update();
  orbit.enableDamping = previousDamping;
}

function captureOrbitBasePose(state) {
  if (!state || !camera || !orbit || !ensureRecordingScratchObjects()) return false;
  recordingScratchOffset.copy(camera.position).sub(orbit.target);
  recordingScratchSpherical.setFromVector3(recordingScratchOffset);
  if (!(recordingScratchSpherical.radius > 0)) return false;
  recordingScratchSpherical.makeSafe();
  state.baseTarget = {
    x: Number(orbit.target.x),
    y: Number(orbit.target.y),
    z: Number(orbit.target.z),
  };
  state.baseTheta = Number(recordingScratchSpherical.theta);
  state.basePhi = Number(recordingScratchSpherical.phi);
  state.baseRadius = Number(recordingScratchSpherical.radius);
  return true;
}

function captureRecordingBasePoseFromCurrentView() {
  if (!recordingModeEnabled) return false;
  return captureOrbitBasePose(recordingMotionState);
}

function captureShowcaseBasePoseFromCurrentView() {
  if (!isShowcaseMode && !isPreviewMode) return false;
  return captureOrbitBasePose(showcaseMotionState);
}

function stopRecordingPlayback({ syncBasePose = true } = {}) {
  if (!recordingModeEnabled) return;
  recordingMotionState.playing = false;
  recordingMotionState.autoElapsedSec = 0;
  if (syncBasePose) {
    syncOrbitImmediately();
    captureRecordingBasePoseFromCurrentView();
  }
  updateRecordingControlsUi();
}

function resetRecordingMotionState({ syncBasePose = true } = {}) {
  if (!recordingModeEnabled) return;
  stopRecordingPlayback({ syncBasePose: false });
  clearRecordingManualState();
  if (syncBasePose) {
    syncOrbitImmediately();
    captureRecordingBasePoseFromCurrentView();
  }
  updateRecordingControlsUi();
}

function setRecordingDirection(direction) {
  if (!recordingModeEnabled) return;
  recordingMotionState.direction = direction === "ccw" ? "ccw" : "cw";
  updateRecordingControlsUi();
}

function getRecordingManualRadiusSpeed(radius) {
  return clamp(radius * 0.18, 1.5, 14);
}

function getRecordingManualPhiSpeed() {
  if (!recordingMotionState.playing) {
    return RECORDING_MANUAL_PHI_RAD_PER_SEC;
  }
  const speedMultiplier = clamp(recordingMotionState.speed, 0.25, 2);
  const playbackAngularSpeed = RECORDING_AUTO_ORBIT_RAD_PER_SEC * speedMultiplier;
  return clamp(
    playbackAngularSpeed * RECORDING_PLAYBACK_TILT_SPEED_RATIO,
    RECORDING_PLAYBACK_TILT_SPEED_MIN,
    RECORDING_PLAYBACK_TILT_SPEED_MAX
  );
}

function canBlendRecordingPlaybackWithManualAction(action) {
  return action === "tiltUp" || action === "tiltDown";
}

function easeVelocityToward(current, target, deltaSeconds) {
  if (!(deltaSeconds > 0)) return current;
  const alpha = 1 - Math.exp(-deltaSeconds / RECORDING_MANUAL_RESPONSE_SEC);
  return current + (target - current) * alpha;
}

function getRecordingOrbitBounds() {
  return {
    minDistance: Math.max(0.001, Number.isFinite(orbit?.minDistance) ? orbit.minDistance : 0.001),
    maxDistance: Number.isFinite(orbit?.maxDistance) ? orbit.maxDistance : Infinity,
    minPolarAngle: Math.max(RECORDING_MIN_POLAR_EPS, Number.isFinite(orbit?.minPolarAngle) ? orbit.minPolarAngle : 0),
    maxPolarAngle: Math.min(
      Math.PI - RECORDING_MIN_POLAR_EPS,
      Number.isFinite(orbit?.maxPolarAngle) ? orbit.maxPolarAngle : Math.PI
    ),
  };
}

function applyRecordingOrbitFrame(targetLike, theta, phi, radius) {
  if (!camera || !orbit || !ensureRecordingScratchObjects()) return false;
  const { minDistance, maxDistance, minPolarAngle, maxPolarAngle } = getRecordingOrbitBounds();

  recordingScratchTarget.set(Number(targetLike?.x) || 0, Number(targetLike?.y) || 0, Number(targetLike?.z) || 0);
  recordingScratchSpherical.radius = clamp(radius, minDistance, maxDistance);
  recordingScratchSpherical.theta = Number(theta) || 0;
  recordingScratchSpherical.phi = clamp(phi, minPolarAngle, maxPolarAngle);
  recordingScratchSpherical.makeSafe();
  recordingScratchOffset.setFromSpherical(recordingScratchSpherical);

  orbit.target.copy(recordingScratchTarget);
  camera.position.copy(recordingScratchTarget).add(recordingScratchOffset);
  camera.lookAt(orbit.target);
  return true;
}

function startRecordingPlayback() {
  if (!recordingModeEnabled) return;
  if (!camera || !orbit) return;
  if (!recordingMotionState.orbitEnabled && !recordingMotionState.zoomEnabled) {
    setTransientStatus(t("explorer.capture.enableOrbitOrZoomFirst"));
    updateRecordingControlsUi();
    return;
  }
  clearRecordingManualState();
  syncOrbitImmediately();
  if (!captureRecordingBasePoseFromCurrentView()) return;
  recordingMotionState.autoElapsedSec = 0;
  recordingMotionState.playing = true;
  updateRecordingControlsUi();
}

function toggleRecordingPlayback() {
  if (!recordingModeEnabled) return;
  if (recordingMotionState.playing) {
    stopRecordingPlayback({ syncBasePose: true });
  } else {
    startRecordingPlayback();
  }
}

function stopShowcaseAutoOrbit({ syncBasePose = true } = {}) {
  if (!isShowcaseMode && !isPreviewMode) return;
  showcaseMotionState.active = false;
  showcaseMotionState.autoElapsedSec = 0;
  if (syncBasePose) {
    syncOrbitImmediately();
    captureShowcaseBasePoseFromCurrentView();
  }
}

function startShowcaseAutoOrbit() {
  if (!shouldRunShowcaseAutoOrbit() || !camera || !orbit) return;
  if (!currentCoreContext || isLoadingOverlayVisible()) return;
  syncOrbitImmediately();
  if (!captureShowcaseBasePoseFromCurrentView()) return;
  showcaseMotionState.autoElapsedSec = 0;
  showcaseMotionState.active = true;
}

function clearShowcaseInteractionResumeTimer() {
  if (showcaseInteractionResumeTimer === null) return;
  window.clearTimeout(showcaseInteractionResumeTimer);
  showcaseInteractionResumeTimer = null;
}

function getShowcaseInteractionHintActiveState() {
  if (isDesktopInteractiveShowcase()) {
    return showcaseMotionState.userInteracting;
  }
  return interactionGateActive;
}

function shouldForwardShowcaseWheelToPage() {
  if (!isShowcaseMode) return false;
  if (interactionGateEnabled) {
    return !interactionGateActive;
  }
  return isDesktopInteractiveShowcase() && !showcaseMotionState.userInteracting;
}

function syncDesktopInteractiveShowcasePrompt() {
  if (!viewerShellEl) return;
  viewerShellEl.classList.toggle(
    "showcase-desktop-idle",
    isDesktopInteractiveShowcase() && !showcaseMotionState.userInteracting
  );
}

function setShowcaseUserInteracting(active) {
  showcaseMotionState.userInteracting = Boolean(active);
  if (isDesktopInteractiveShowcase() && orbit && renderer) {
    orbit.enabled = showcaseMotionState.userInteracting;
    renderer.domElement.style.touchAction = showcaseMotionState.userInteracting ? "none" : "auto";
    updateInteractionHint(showcaseMotionState.userInteracting);
  }
  syncDesktopInteractiveShowcasePrompt();
}

function noteDesktopInteractiveShowcaseActivity({ scheduleResume = true } = {}) {
  if (!isDesktopInteractiveShowcase()) return;
  showcaseLastActivityAtMs = performance.now();
  if (!showcaseMotionState.userInteracting) {
    setShowcaseUserInteracting(true);
    stopShowcaseAutoOrbit({ syncBasePose: true });
  }
  if (scheduleResume) {
    scheduleShowcaseInteractionResume();
  } else {
    clearShowcaseInteractionResumeTimer();
  }
}

function shouldAutoResumeShowcaseInteraction() {
  return (
    isShowcaseMode &&
    !isCoarsePointerInput() &&
    !shouldUseShowcaseMobileLinkout() &&
    (interactionGateEnabled || isDesktopInteractiveShowcase())
  );
}

function scheduleShowcaseInteractionResume() {
  clearShowcaseInteractionResumeTimer();
  if (!shouldAutoResumeShowcaseInteraction()) return;
  if (interactionGateEnabled && !interactionGateActive) return;
  if (isDesktopInteractiveShowcase() && !showcaseMotionState.userInteracting) return;
  showcaseInteractionResumeTimer = window.setTimeout(() => {
    showcaseInteractionResumeTimer = null;
    if (interactionGateEnabled) {
      if (!interactionGateActive) return;
      setInteractionGateActive(false);
      return;
    }
    if (isDesktopInteractiveShowcase()) {
      setShowcaseUserInteracting(false);
      startShowcaseAutoOrbit();
    }
  }, SHOWCASE_INTERACTION_IDLE_RESUME_MS);
}

function setRecordingManualSource(sourceKey, action, active) {
  if (!recordingModeEnabled) return;
  const source = recordingMotionState[sourceKey];
  if (!source || !Object.prototype.hasOwnProperty.call(source, action)) return;
  const nextValue = Boolean(active);
  if (source[action] === nextValue) return;
  source[action] = nextValue;
  if (nextValue && (!recordingMotionState.playing || !canBlendRecordingPlaybackWithManualAction(action))) {
    stopRecordingPlayback({ syncBasePose: true });
  }
  syncRecordingManualIntent();
}

function stepRecordingMotion(deltaSeconds) {
  if (!recordingModeEnabled || !camera || !orbit || !ensureRecordingScratchObjects()) return;
  const dt = clamp(Number(deltaSeconds) || 0, 0, RECORDING_FRAME_DT_LIMIT_MS / 1000);
  if (!(dt > 0)) return;
  const { minPolarAngle, maxPolarAngle } = getRecordingOrbitBounds();
  recordingScratchOffset.copy(camera.position).sub(orbit.target);
  recordingScratchSpherical.setFromVector3(recordingScratchOffset);
  if (!(recordingScratchSpherical.radius > 0)) return;
  recordingScratchSpherical.makeSafe();

  const targetThetaVelocity = recordingMotionState.manualIntent.theta * RECORDING_MANUAL_THETA_RAD_PER_SEC;
  const targetPhiVelocity = recordingMotionState.manualIntent.phi * getRecordingManualPhiSpeed();
  const targetRadiusVelocity = recordingMotionState.manualIntent.radius * getRecordingManualRadiusSpeed(recordingScratchSpherical.radius);
  recordingMotionState.manualVelocity.theta = easeVelocityToward(
    recordingMotionState.manualVelocity.theta,
    targetThetaVelocity,
    dt
  );
  recordingMotionState.manualVelocity.phi = easeVelocityToward(recordingMotionState.manualVelocity.phi, targetPhiVelocity, dt);
  recordingMotionState.manualVelocity.radius = easeVelocityToward(
    recordingMotionState.manualVelocity.radius,
    targetRadiusVelocity,
    dt
  );

  const hasManualMovement =
    Math.abs(recordingMotionState.manualVelocity.theta) > 1e-4 ||
    Math.abs(recordingMotionState.manualVelocity.phi) > 1e-4 ||
    Math.abs(recordingMotionState.manualVelocity.radius) > 1e-3;

  if (recordingMotionState.playing) {
    if (!recordingMotionState.baseTarget) {
      captureRecordingBasePoseFromCurrentView();
    }
    const directionSign = recordingMotionState.direction === "ccw" ? 1 : -1;
    const speedMultiplier = clamp(recordingMotionState.speed, 0.25, 2);
    const baseTarget = recordingMotionState.baseTarget || {
      x: Number(orbit.target.x),
      y: Number(orbit.target.y),
      z: Number(orbit.target.z),
    };
    const baseTheta = Number.isFinite(recordingMotionState.baseTheta) ? recordingMotionState.baseTheta : 0;
    const baseRadius =
      Number.isFinite(recordingMotionState.baseRadius) && recordingMotionState.baseRadius > 0
        ? recordingMotionState.baseRadius
        : camera.position.distanceTo(orbit.target);
    recordingMotionState.basePhi = clamp(
      Number.isFinite(recordingMotionState.basePhi) ? recordingMotionState.basePhi + recordingMotionState.manualVelocity.phi * dt : Math.PI / 2,
      minPolarAngle,
      maxPolarAngle
    );
    recordingMotionState.autoElapsedSec += dt;

    const theta = recordingMotionState.orbitEnabled
      ? baseTheta + directionSign * RECORDING_AUTO_ORBIT_RAD_PER_SEC * speedMultiplier * recordingMotionState.autoElapsedSec
      : baseTheta;
    const zoomOffset =
      recordingMotionState.zoomEnabled
        ? -Math.sin(recordingMotionState.autoElapsedSec * Math.PI * 2 * RECORDING_AUTO_ZOOM_HZ * speedMultiplier) *
          baseRadius *
          recordingMotionState.zoomAmount
        : 0;
    applyRecordingOrbitFrame(baseTarget, theta, recordingMotionState.basePhi, baseRadius + zoomOffset);
    if (!hasManualMovement) {
      zeroRecordingManualVelocity();
    }
    return;
  }

  if (!hasManualMovement) {
    zeroRecordingManualVelocity();
    return;
  }

  const currentTarget = {
    x: Number(orbit.target.x),
    y: Number(orbit.target.y),
    z: Number(orbit.target.z),
  };
  const theta = recordingScratchSpherical.theta + recordingMotionState.manualVelocity.theta * dt;
  const phi = recordingScratchSpherical.phi + recordingMotionState.manualVelocity.phi * dt;
  const radius = recordingScratchSpherical.radius + recordingMotionState.manualVelocity.radius * dt;
  applyRecordingOrbitFrame(currentTarget, theta, phi, radius);
}

function stepShowcaseMotion(deltaSeconds) {
  if ((!isShowcaseMode && !isPreviewMode) || !camera || !orbit) return;
  if (!shouldRunShowcaseAutoOrbit()) {
    stopShowcaseAutoOrbit({ syncBasePose: false });
    return;
  }
  if (!currentCoreContext || isLoadingOverlayVisible()) return;
  if (isDesktopInteractiveShowcase() && showcaseMotionState.userInteracting) {
    if (showcaseLastActivityAtMs > 0 && performance.now() - showcaseLastActivityAtMs >= SHOWCASE_INTERACTION_IDLE_RESUME_MS) {
      setShowcaseUserInteracting(false);
      startShowcaseAutoOrbit();
    }
  }
  if (isDesktopInteractiveShowcase() && showcaseMotionState.userInteracting) {
    stopShowcaseAutoOrbit({ syncBasePose: false });
    return;
  }
  if (isShowcaseMode && interactionGateEnabled && interactionGateActive) {
    stopShowcaseAutoOrbit({ syncBasePose: false });
    return;
  }
  if (!showcaseMotionState.active) {
    startShowcaseAutoOrbit();
  }
  if (!showcaseMotionState.active) return;

  const dt = clamp(Number(deltaSeconds) || 0, 0, RECORDING_FRAME_DT_LIMIT_MS / 1000);
  if (!(dt > 0)) return;

  const baseTarget = showcaseMotionState.baseTarget;
  const baseTheta = showcaseMotionState.baseTheta;
  const basePhi = showcaseMotionState.basePhi;
  const baseRadius = showcaseMotionState.baseRadius;
  if (!baseTarget || !Number.isFinite(baseTheta) || !Number.isFinite(basePhi) || !(baseRadius > 0)) {
    startShowcaseAutoOrbit();
    return;
  }

  const motionConfig = getShowcaseMotionConfig();
  showcaseMotionState.autoElapsedSec += dt;
  const orbitElapsedSec = Math.max(0, showcaseMotionState.autoElapsedSec - motionConfig.holdSec);
  const theta = baseTheta - motionConfig.orbitRadPerSec * motionConfig.speed * orbitElapsedSec;
  const zoomOffset =
    motionConfig.zoomAmount > 0
      ? -Math.sin(orbitElapsedSec * Math.PI * 2 * motionConfig.zoomHz * motionConfig.speed) *
        baseRadius *
        motionConfig.zoomAmount
      : 0;
  applyRecordingOrbitFrame(baseTarget, theta, basePhi, baseRadius + zoomOffset);
}

function isFlowLightAnimationEnabled() {
  return Boolean(controlsUI.animateFlow?.checked && !flowMotionMediaQuery.matches);
}

function revealFlowlineForAnimation() {
  if (
    !isFlowLightAnimationEnabled() ||
    controlsUI.showFlowline.checked ||
    controlsUI.showOceanCurrents.checked
  ) {
    return;
  }
  controlsUI.showFlowline.checked = true;
  controlsUI.showFlowline.dispatchEvent(new Event("change"));
}

function syncFlowLightAnimationPreference() {
  if (flowMotionMediaQuery.matches && controlsUI.animateFlow) {
    controlsUI.animateFlow.checked = false;
  }
  const enabled = isFlowLightAnimationEnabled();
  flowLightUniforms.enabled.value = enabled ? 1 : 0;
  setFlowLightParticleOverlaysVisible(flowlineMesh, enabled);
  setFlowLightParticleOverlaysVisible(oceanCurrentMesh, enabled);
}

function stepFlowLightAnimation(deltaSeconds) {
  syncFlowLightAnimationPreference();
  if (!isFlowLightAnimationEnabled() || document.hidden) return;
  flowLightElapsedSeconds += clamp(Number(deltaSeconds) || 0, 0, RECORDING_FRAME_DT_LIMIT_MS / 1000);
  flowLightUniforms.time.value = flowLightElapsedSeconds;
}

function stepRuntime(deltaMs = 16) {
  const safeDeltaMs = clamp(Number(deltaMs) || 0, 0, RECORDING_FRAME_DT_LIMIT_MS);
  if (isShowcaseMode || isPreviewMode) {
    stepShowcaseMotion(safeDeltaMs / 1000);
  }
  if (recordingModeEnabled) {
    stepRecordingMotion(safeDeltaMs / 1000);
  }
  stepFlowLightAnimation(safeDeltaMs / 1000);
  if (orbit) orbit.update();
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(from, to, t) {
  return [lerp(from[0], to[0], t), lerp(from[1], to[1], t), lerp(from[2], to[2], t)];
}

const BED_ELEVATION_MIN = -7000;
const BED_ELEVATION_MAX = 3500;
const TARGET_WORLD_EXTENT_UNITS = 128;
const BASE_HORIZONTAL_VERTICAL_SCALE_RATIO = 52000 / 3800;
const FLOWLINE_PICK_MOVE_THRESHOLD_PX = 7;
const FLOWLINE_PROFILE_SAMPLE_TARGET = 88;
const FLOWLINE_HIGHLIGHT_SAMPLE_TARGET = 144;
const FLOWLINE_SURFACE_OFFSET_M = 26;
const FLOWLINE_SEED_TARGET = 2400;
const FLOWLINE_SEED_PASSES = 2;
const FLOWLINE_MIN_SEED_SPEED = 15;
const FLOWLINE_MIN_TRACE_SPEED = 4;
const FLOWLINE_MAX_STEPS = 180;
const FLOWLINE_STEP_CELLS = 0.6;
const FLOWLINE_REVERSE_DIRECTION_DOT = -0.12;
const FLOW_LIGHT_PATTERN_SCALE = 0.16;
const FLOW_LIGHT_ICE_RATE = 1.25;
const FLOW_LIGHT_OCEAN_RATE = 0.85;
const FLOW_LIGHT_STATIC_OPACITY = 0.9;
const FLOW_LIGHT_ICE_ACTIVE_BASE_OPACITY = 0.24;
const FLOW_LIGHT_OCEAN_ACTIVE_BASE_OPACITY = 0.18;
const FLOW_LIGHT_PULSE_OPACITY = 0.76;
const FLOW_LIGHT_ICE_PARTICLE_SAMPLE_STRIDE = 4;
const FLOW_LIGHT_OCEAN_PARTICLE_SAMPLE_STRIDE = 12;
const FLOW_LIGHT_ICE_PARTICLE_SIZE = 7;
const FLOW_LIGHT_OCEAN_PARTICLE_SIZE = 5;
const flowLightUniforms = {
  time: { value: 0 },
  enabled: { value: 1 },
};
let flowLightElapsedSeconds = 0;
const VELOCITY_VISUALIZATION = Object.freeze({
  range: [0, 3000],
  knee: 20,
  ticks: [0, 50, 200, 500, 1000, 3000],
  note: "",
});
const BASAL_FRICTION_COLOR_STOPS = [
  [0.0, [0.047, 0.118, 0.259]],
  [0.26, [0.125, 0.478, 0.678]],
  [0.52, [0.88, 0.91, 0.79]],
  [0.78, [0.851, 0.545, 0.188]],
  [1.0, [0.588, 0.094, 0.114]],
];
const BASAL_FRICTION_VISUALIZATION = Object.freeze({
  range: [0, 0.3],
  knee: 0.015,
  ticks: [0, 0.02, 0.04, 0.08, 0.15, 0.3],
  note: "",
});
const VELOCITY_BLOCKING_OVERLAY_ENABLED = false;
const OCEAN_CURRENT_BLOCKING_OVERLAY_ENABLED = true;
const OCEAN_CURRENT_MAIN_THREAD_FALLBACK_SEGMENT_LIMIT = 250000;
const OCEAN_CURRENT_MAIN_THREAD_FALLBACK_BIN_BYTES_LIMIT = 16 * 1024 * 1024;
const VELOCITY_SURFACE_OPACITY = 0.82;
const OCEAN_CURRENT_BED_CLEARANCE_M = 20;
const EFFECTIVE_PRESSURE_REFERENCE_PA = 5_000_000;
const CHANNEL_DISCHARGE_MIN = 1e-3;
const CHANNEL_DISCHARGE_MAX = 100;
const CHANNEL_STRIP_WIDTH_M_MIN = 1500;
const CHANNEL_STRIP_WIDTH_M_MAX = 5000;
const EFFECTIVE_PRESSURE_SURFACE_OFFSET_M = 10;
const SUBGLACIAL_CHANNEL_SURFACE_OFFSET_M = 14;
const RISE_SURFACE_OFFSET_M = 14;
const REFINED_BASIN_BED_OFFSET_M = 90;
const REFINED_BASIN_SURFACE_OFFSET_M = 24;
const REFINED_BASIN_BED_LINE_WIDTH_M = 5000;
const REFINED_BASIN_SURFACE_LINE_WIDTH_M = 5000;
const REFINED_BASIN_BED_LABEL_OFFSET_M = 36;
const REFINED_BASIN_SURFACE_LABEL_OFFSET_M = 42;
const REFINED_BASIN_LABEL_WORLD_SCALE = 0.0076;
const GREENLAND_REFINED_BASIN_LABEL_WORLD_SCALE = 0.0104;
const MAX_REFINED_BASIN_LABELS = 220;
const MAX_REFINED_BASIN_LABEL_CANVAS_WIDTH = 1600;
const MAX_REFINED_BASIN_LABEL_TEXTURE_PIXELS = 12_000_000;
const OCEAN_CURRENT_ARROW_HEAD_RATIO = 0.4;
const OCEAN_CURRENT_ARROW_HEAD_MIN_UNITS = 0.16;
const OCEAN_CURRENT_ARROW_HEAD_MAX_UNITS = 1.0;
const OCEAN_CURRENT_ARROW_HEAD_WIDTH_RATIO = 0.65;
const RISE_BASAL_MELT_COLOR_STOPS = [
  [0.0, [0.082, 0.216, 0.376]],
  [0.22, [0.31, 0.655, 0.847]],
  [0.5, [0.965, 0.969, 0.984]],
  [0.78, [0.969, 0.639, 0.357]],
  [1.0, [0.541, 0.067, 0.09]],
];
const RISE_BASAL_MELT_VISUALIZATION = Object.freeze({
  range: [-1, 15],
  linthresh: 0.05,
  ticks: [-1, -0.3, -0.1, 0, 0.3, 1, 3, 15],
  note: "",
});
const RISE_THERMAL_DRIVING_COLOR_STOPS = [
  [0.0, [0.031, 0.114, 0.345]],
  [0.22, [0.133, 0.369, 0.659]],
  [0.52, [0.114, 0.569, 0.753]],
  [0.78, [0.498, 0.804, 0.733]],
  [1.0, [1.0, 0.953, 0.639]],
];
// Keep HD velocity responsive with a decimated mesh.
const HD_VELOCITY_MESH_STRIDE = 2;
// Effective-pressure overlay should stay conformal to bed to avoid local intersections.
const HD_HYDROLOGY_MESH_STRIDE = 1;
const GMT_RELIEF_RGB_URL = assetUrl("data/GMT_relief.rgb");
const CMOCEAN_DENSE_RGB_URL = assetUrl("data/cmocean_dense.rgb");
const CMOCEAN_MATTER_RGB_URL = assetUrl("data/cmocean_matter.rgb");
const GMT_RELIEF_OCEAN_ZERO = [0.975613, 0.999387, 0.994608];
const GMT_RELIEF_LAND_ZERO = [0.284908, 0.466429, 0.196078];

let gmtReliefOceanLut = [];
let gmtReliefLandLut = [];
let effectivePressureLut = [];
let channelDischargeLut = [];

function sampleColorLUT(lut, t) {
  if (!lut.length) return [0, 0, 0];
  const scaled = clamp01(t) * (lut.length - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(lut.length - 1, i0 + 1);
  return lerpColor(lut[i0], lut[i1], scaled - i0);
}

function parseRgbTableText(text, expectedCount = null) {
  const lines = text.split(/\r?\n/);
  const rawColors = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || line.startsWith("ncolors")) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 3) continue;
    const r = Number(parts[0]);
    const g = Number(parts[1]);
    const b = Number(parts[2]);
    if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) continue;
    rawColors.push([r, g, b]);
  }

  if (expectedCount !== null && rawColors.length !== expectedCount) {
    throw new Error(`Unexpected RGB color count: ${rawColors.length} (expected ${expectedCount})`);
  }

  const has255Range = rawColors.some(([r, g, b]) => r > 1 || g > 1 || b > 1);
  const scale = has255Range ? 1 / 255 : 1;
  return rawColors.map(([r, g, b]) => [clamp01(r * scale), clamp01(g * scale), clamp01(b * scale)]);
}

function setLoadingOverlayVisible(visible) {
  if (!loadingOverlayEl) return;
  loadingOverlayEl.classList.toggle("hidden", !visible);
  loadingOverlayEl.setAttribute("aria-busy", visible ? "true" : "false");
}

function updateLoadingProgress(progress, stageText, hintText = null) {
  const pct = clamp01(progress);
  if (loadingFillEl) {
    loadingFillEl.style.transform = `scaleX(${pct})`;
  }
  if (loadingPercentEl) {
    loadingPercentEl.textContent = `${Math.round(pct * 100)}%`;
  }
  if (stageText && loadingStageEl) {
    loadingStageEl.textContent = stageText;
  }
  if (hintText && loadingHintEl) {
    loadingHintEl.textContent = hintText;
  }
}

function detectLowEndDevice() {
  const memory = Number(navigator.deviceMemory || 0);
  const cores = Number(navigator.hardwareConcurrency || 0);
  const coarsePointer = isCoarsePointerInput();
  const veryLowMemory = memory > 0 && memory <= 3;
  const constrainedCpu = cores > 0 && cores <= 4;
  if (veryLowMemory) return true;
  if (coarsePointer && memory > 0 && memory <= 4 && constrainedCpu) return true;
  if (coarsePointer && memory === 0 && constrainedCpu) return true;
  return false;
}

function hasConstrainedNetwork() {
  const connection = navigator.connection;
  if (connection?.saveData) return true;
  const networkType = String(connection?.effectiveType || "");
  if (networkType.includes("2g") || networkType.includes("3g")) return true;
  return false;
}

function detectMobileHighQualityDevice(rendererInstance = null) {
  if (isShowcaseMode) return false;
  if (!shouldUseMobileDrawer()) return false;
  if (detectLowEndDevice()) return false;
  if (hasConstrainedNetwork()) return false;

  const memory = Number(navigator.deviceMemory || 0);
  const cores = Number(navigator.hardwareConcurrency || 0);
  const dpr = Number(window.devicePixelRatio || 1);
  const strongHardware =
    (memory >= 6 && cores >= 6) ||
    (memory >= 4 && cores >= 8) ||
    (memory === 0 && cores >= 6 && dpr >= 2.6);
  if (!strongHardware) return false;

  const maxTextureSize = Number(rendererInstance?.capabilities?.maxTextureSize || 0);
  if (maxTextureSize > 0 && maxTextureSize < 4096) return false;
  return true;
}

async function loadThreeRuntime() {
  if (THREE && OrbitControlsCtor) return;
  updateLoadingProgress(0.06, t("explorer.loading.loadingThreeRuntime"));
  const [threeModule, orbitModule] = await Promise.all([
    import(assetUrl("vendor/three/three.module.min.js")),
    import(assetUrl("vendor/three/OrbitControls.js")),
  ]);
  THREE = threeModule;
  OrbitControlsCtor = orbitModule.OrbitControls;
}

async function fetchJsonStrict(url, errorLabel) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`${errorLabel} (${response.status})`);
  }
  return response.json();
}

async function fetchArrayBufferStrict(url, errorLabel) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`${errorLabel} (${response.status})`);
  }
  return response.arrayBuffer();
}

async function fetchArrayBufferWithProgress(url, progressStart, progressEnd, stageText, responseErrorLabel = "") {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`${responseErrorLabel || `Failed to load ${url}`} (${response.status})`);
  }

  const totalBytes = Number(response.headers.get("content-length")) || 0;
  if (!response.body || totalBytes <= 0) {
    const buffer = await response.arrayBuffer();
    updateLoadingProgress(progressEnd, stageText);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    chunks.push(value);
    received += value.byteLength;
    const byteRatio = totalBytes ? received / totalBytes : 1;
    const progress = progressStart + (progressEnd - progressStart) * clamp01(byteRatio);
    updateLoadingProgress(progress, `${stageText} ${Math.round(clamp01(byteRatio) * 100)}%`);
  }

  const joined = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  updateLoadingProgress(progressEnd, stageText);
  return joined.buffer;
}

function nextAnimationFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
}

function ensureGeometryWorker() {
  if (geometryWorker) return geometryWorker;
  geometryWorker = new Worker(assetUrl("antarctica-geometry-worker.js"), { type: "module" });
  geometryWorker.addEventListener("message", (event) => {
    const data = event.data || {};
    const pending = geometryWorkerPending.get(data.id);
    if (!pending) return;

    if (data.kind === "progress") {
      if (typeof pending.onProgress === "function") {
        pending.onProgress(data.progress, localizeWorkerStage(data.stageKey || "", data.stage || ""));
      }
      return;
    }

    geometryWorkerPending.delete(data.id);
    if (data.kind === "result" && data.ok) {
      pending.resolve(data.result);
      return;
    }

    const message = localizeErrorMessage(data?.error?.message || t("explorer.errors.workerTaskFailed"));
    pending.reject(new Error(message));
  });
  geometryWorker.addEventListener("error", (event) => {
    const message = localizeErrorMessage(event?.message || t("explorer.errors.workerCrashed"));
    geometryWorkerPending.forEach((pending) => pending.reject(new Error(message)));
    geometryWorkerPending.clear();
  });
  return geometryWorker;
}

function runGeometryWorkerTask(task, payload, { transfer = [], onProgress = null } = {}) {
  const worker = ensureGeometryWorker();
  const id = ++geometryWorkerRequestSeq;
  return new Promise((resolve, reject) => {
    geometryWorkerPending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, task, payload }, transfer);
  });
}

function terminateGeometryWorker() {
  if (!geometryWorker) return;
  geometryWorker.terminate();
  geometryWorker = null;
  geometryWorkerPending.forEach((pending) => pending.reject(new Error(t("explorer.errors.workerTerminated"))));
  geometryWorkerPending.clear();
}

async function loadBedColorTable() {
  const response = await fetch(GMT_RELIEF_RGB_URL);
  if (!response.ok) {
    throw new Error(`${errorLabel("explorer.errors.failedToLoadBedColorTable")} (${response.status})`);
  }
  const colors = parseRgbTableText(await response.text(), 256);

  if (colors.length !== 256) {
    throw new Error(`Unexpected GMT_relief color count: ${colors.length} (expected 256)`);
  }

  gmtReliefOceanLut = colors.slice(0, 128);
  gmtReliefLandLut = colors.slice(128);
}

async function loadEffectivePressureColorTable() {
  const response = await fetch(CMOCEAN_DENSE_RGB_URL);
  if (!response.ok) {
    throw new Error(`${errorLabel("explorer.errors.failedToLoadEffectivePressureColorTable")} (${response.status})`);
  }
  effectivePressureLut = parseRgbTableText(await response.text(), 256);
}

async function loadChannelDischargeColorTable() {
  const response = await fetch(CMOCEAN_MATTER_RGB_URL);
  if (!response.ok) {
    throw new Error(`${errorLabel("explorer.errors.failedToLoadChannelColorTable")} (${response.status})`);
  }
  channelDischargeLut = parseRgbTableText(await response.text(), 256);
}

function bedColor(heightMeters) {
  const clamped = Math.max(BED_ELEVATION_MIN, Math.min(BED_ELEVATION_MAX, heightMeters));
  const oceanLut = gmtReliefOceanLut.length ? gmtReliefOceanLut : [GMT_RELIEF_OCEAN_ZERO];
  const landLut = gmtReliefLandLut.length ? gmtReliefLandLut : [GMT_RELIEF_LAND_ZERO];

  if (clamped < 0) {
    const t = clamp01((clamped - BED_ELEVATION_MIN) / (0 - BED_ELEVATION_MIN));
    return sampleColorLUT(oceanLut, t);
  }

  if (clamped > 0) {
    const t = clamp01(clamped / BED_ELEVATION_MAX);
    return sampleColorLUT(landLut, t);
  }

  // Keep 0 m exactly between the two endpoint colors, with high contrast above/below.
  const oceanZero = oceanLut[oceanLut.length - 1] || GMT_RELIEF_OCEAN_ZERO;
  const landZero = landLut[0] || GMT_RELIEF_LAND_ZERO;
  return lerpColor(oceanZero, landZero, 0.5);
}

function rgbToHex(rgb) {
  const toHex = (value) => Math.round(clamp01(value) * 255).toString(16).padStart(2, "0");
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`;
}

function updateContinuousLegend(targetEl, sampleFn, stopCount = 64) {
  if (!targetEl) return;
  const stops = [];
  for (let i = 0; i <= stopCount; i += 1) {
    const t = i / stopCount;
    stops.push(`${rgbToHex(sampleFn(t))} ${(t * 100).toFixed(3)}%`);
  }
  targetEl.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
}

function getVelocityVisualizationConfig() {
  return VELOCITY_VISUALIZATION;
}

function getBasalFrictionVisualizationConfig() {
  const metaRange = currentBasalFrictionMeta?.visualization?.display_range_mpa;
  const metaKnee = Number(currentBasalFrictionMeta?.visualization?.knee_mpa);
  const metaTicks = Array.isArray(currentBasalFrictionMeta?.visualization?.ticks_mpa)
    ? currentBasalFrictionMeta.visualization.ticks_mpa.map((value) => Number(value))
    : null;
  const fallback = BASAL_FRICTION_VISUALIZATION;
  const rawTicks =
    Array.isArray(metaTicks) && metaTicks.length >= 2 && metaTicks.every((value) => Number.isFinite(value))
      ? metaTicks
      : fallback.ticks;
  const ticks = Array.from(new Set(rawTicks.map((value) => Number(value.toFixed(6))))).sort((a, b) => a - b);
  return {
    range:
      Array.isArray(metaRange) && metaRange.length >= 2 && Number.isFinite(Number(metaRange[0])) && Number.isFinite(Number(metaRange[1]))
        ? [Number(metaRange[0]), Number(metaRange[1])]
        : fallback.range,
    knee: Number.isFinite(metaKnee) && metaKnee > 0 ? metaKnee : fallback.knee,
    ticks,
    note: currentBasalFrictionMeta?.visualization?.note || fallback.note,
  };
}

function velocityScaleT(speedMetersPerYear) {
  const { range, knee } = getVelocityVisualizationConfig();
  const minValue = Number(range[0]);
  const maxValue = Number(range[1]);
  const clamped = Math.max(minValue, Math.min(maxValue, Number(speedMetersPerYear)));
  const safeKnee = Math.max(1e-6, Number(knee) || 1);
  const scaled = Math.log1p(clamped / safeKnee) / Math.log1p(maxValue / safeKnee);
  return clamp01(scaled);
}

function basalFrictionScaleT(valueMpa) {
  const { range, knee } = getBasalFrictionVisualizationConfig();
  const minValue = Number(range[0]);
  const maxValue = Number(range[1]);
  const clamped = Math.max(minValue, Math.min(maxValue, Number(valueMpa)));
  const safeKnee = Math.max(1e-6, Number(knee) || 1);
  const scaled =
    Math.log1p(Math.max(0, clamped - minValue) / safeKnee) /
    Math.log1p(Math.max(1e-6, maxValue - minValue) / safeKnee);
  return clamp01(scaled);
}

function formatVelocityTick(value) {
  if (value === 0) return "0 m/yr";
  return formatCoord(value, 0);
}

function formatBasalFrictionTick(value) {
  if (value === 0) return "0 Mpa";
  if (Math.abs(value) < 0.1) return formatCoord(value, 2);
  if (Math.abs(value) < 1) {
    const roundedToTenth = Number(value.toFixed(1));
    return Math.abs(value - roundedToTenth) > 1e-6 ? formatCoord(value, 2) : formatCoord(value, 1);
  }
  return formatCoord(value, 1);
}

function layoutTickLegendLabels(targetEl, { minGapPx = 6 } = {}) {
  if (!targetEl) return false;
  const labels = Array.from(targetEl.querySelectorAll("span[data-left-percent]"));
  if (!labels.length) {
    targetEl.style.removeProperty("height");
    return true;
  }

  const containerWidth = targetEl.clientWidth;
  if (!containerWidth) return false;

  const fontSizePx = parseFloat(window.getComputedStyle(targetEl).fontSize) || 12;
  const rowHeightPx = Math.max(14, Math.ceil(fontSizePx * 1.15));
  const rowStridePx = rowHeightPx + 2;
  const rowRightEdges = [];

  for (const label of labels) {
    const leftPercent = clamp01((Number(label.dataset.leftPercent) || 0) / 100) * 100;
    const leftPx = (leftPercent / 100) * containerWidth;
    const widthPx = Math.ceil(label.getBoundingClientRect().width || label.offsetWidth || 0);
    const anchor = label.dataset.anchor || "center";
    let labelLeftPx = leftPx;
    if (anchor === "center") {
      labelLeftPx -= widthPx * 0.5;
    } else if (anchor === "right") {
      labelLeftPx -= widthPx;
    }
    labelLeftPx = clamp(labelLeftPx, 0, Math.max(0, containerWidth - widthPx));

    let rowIndex = 0;
    while (rowIndex < rowRightEdges.length && labelLeftPx < rowRightEdges[rowIndex] + minGapPx) {
      rowIndex += 1;
    }
    rowRightEdges[rowIndex] = labelLeftPx + widthPx;

    label.style.left = `${labelLeftPx.toFixed(2)}px`;
    label.style.top = `${rowIndex * rowStridePx}px`;
    label.style.transform = "none";
  }

  const rowCount = Math.max(1, rowRightEdges.length);
  targetEl.style.height = `${(rowCount - 1) * rowStridePx + rowHeightPx}px`;
  return true;
}

function scheduleTickLegendLayout(targetEl, options) {
  if (!targetEl) return;
  window.requestAnimationFrame(() => {
    if (layoutTickLegendLabels(targetEl, options)) return;
    window.requestAnimationFrame(() => {
      layoutTickLegendLabels(targetEl, options);
    });
  });
}

function renderTickLegendLabels(targetEl, ticks, scaleFn, formatFn, layoutOptions = {}) {
  if (!targetEl) return;
  targetEl.classList.add("legend-labels--ticks");
  targetEl.innerHTML = ticks
    .map((value) => {
      const left = clamp01(scaleFn(value)) * 100;
      const anchor = left <= 4 ? "left" : left >= 94 ? "right" : "center";
      const translate = anchor === "left" ? "translateX(0)" : anchor === "right" ? "translateX(-100%)" : "translateX(-50%)";
      return `<span data-left-percent="${left.toFixed(4)}" data-anchor="${anchor}" style="left:${left.toFixed(
        2
      )}%;transform:${translate};">${formatFn(value)}</span>`;
    })
    .join("");
  scheduleTickLegendLayout(targetEl, layoutOptions);
}

function renderVelocityLegendTicks() {
  if (!controlsUI.velocityLegendLabels) return;
  const { ticks } = getVelocityVisualizationConfig();
  renderTickLegendLabels(controlsUI.velocityLegendLabels, ticks, velocityScaleT, formatVelocityTick);
}

function renderBasalFrictionLegendTicks() {
  if (!controlsUI.basalFrictionLegendLabels) return;
  const { ticks } = getBasalFrictionVisualizationConfig();
  renderTickLegendLabels(
    controlsUI.basalFrictionLegendLabels,
    ticks,
    basalFrictionScaleT,
    formatBasalFrictionTick,
    { minGapPx: 8 }
  );
}

function formatBasalMeltTick(value) {
  if (value === getBasalMeltVisualizationConfig().ticks[0]) return `${formatCoord(value, 0)} m/yr`;
  if (value === 0) return "0";
  if (Math.abs(value) >= 10) return formatCoord(value, 0);
  if (Math.abs(value) >= 1) return formatCoord(value, 0);
  return formatCoord(value, 1);
}

function getBasalMeltVisualizationConfig() {
  return RISE_BASAL_MELT_VISUALIZATION;
}

function basalMeltScaleT(meltMetersPerYear) {
  const { range, linthresh } = getBasalMeltVisualizationConfig();
  const minValue = Number(range[0]);
  const maxValue = Number(range[1]);
  const melt = Math.max(minValue, Math.min(maxValue, Number(meltMetersPerYear)));
  if (melt <= 0) {
    const magnitude = Math.abs(melt);
    const negativeMax = Math.max(linthresh, Math.abs(minValue));
    const scaled = Math.log1p(magnitude / linthresh) / Math.log1p(negativeMax / linthresh);
    return 0.5 * (1 - clamp01(scaled));
  }
  const positiveMax = Math.max(linthresh, maxValue);
  const scaled = Math.log1p(melt / linthresh) / Math.log1p(positiveMax / linthresh);
  return 0.5 + 0.5 * clamp01(scaled);
}

function renderBasalMeltLegendTicks() {
  if (!controlsUI.basalMeltLegendLabels) return;
  const { ticks } = getBasalMeltVisualizationConfig();
  renderTickLegendLabels(controlsUI.basalMeltLegendLabels, ticks, basalMeltScaleT, formatBasalMeltTick);
}

function updateBedLegend() {
  if (!controlsUI.legendBar) return;
  if (!gmtReliefOceanLut.length || !gmtReliefLandLut.length) return;

  const stops = [];

  for (let i = 0; i < gmtReliefOceanLut.length; i += 1) {
    const p = (i / (gmtReliefOceanLut.length - 1)) * 50;
    stops.push(`${rgbToHex(gmtReliefOceanLut[i])} ${p.toFixed(3)}%`);
  }

  // Duplicate the 50% position to enforce a hard contrast jump at sea level.
  stops.push(`${rgbToHex(gmtReliefOceanLut[gmtReliefOceanLut.length - 1])} 50.000%`);
  stops.push(`${rgbToHex(gmtReliefLandLut[0])} 50.000%`);

  for (let i = 0; i < gmtReliefLandLut.length; i += 1) {
    const p = 50 + (i / (gmtReliefLandLut.length - 1)) * 50;
    stops.push(`${rgbToHex(gmtReliefLandLut[i])} ${p.toFixed(3)}%`);
  }

  controlsUI.legendBar.style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
}

function updateVelocityLegend() {
  updateContinuousLegend(controlsUI.velocityLegendBar, (t) => {
    const { range, knee } = getVelocityVisualizationConfig();
    const maxValue = Number(range[1]);
    const safeKnee = Math.max(1e-6, Number(knee) || 1);
    const speed = safeKnee * Math.expm1(t * Math.log1p(maxValue / safeKnee));
    return velocityColor(speed);
  });
  renderVelocityLegendTicks();
  if (controlsUI.velocityLegendNote) {
    const note = getVelocityVisualizationConfig().note || "";
    controlsUI.velocityLegendNote.textContent = note;
    controlsUI.velocityLegendNote.hidden = !note;
  }
}

function updateBasalFrictionLegend() {
  updateContinuousLegend(controlsUI.basalFrictionLegendBar, (t) => {
    const { range, knee } = getBasalFrictionVisualizationConfig();
    const minValue = Number(range[0]);
    const maxValue = Number(range[1]);
    const safeKnee = Math.max(1e-6, Number(knee) || 1);
    const value = minValue + safeKnee * Math.expm1(t * Math.log1p(Math.max(1e-6, maxValue - minValue) / safeKnee));
    return basalFrictionColor(value);
  });
  renderBasalFrictionLegendTicks();
  if (controlsUI.basalFrictionLegendNote) {
    const note = getBasalFrictionVisualizationConfig().note || "";
    controlsUI.basalFrictionLegendNote.textContent = note;
    controlsUI.basalFrictionLegendNote.hidden = !note;
  }
}

function updateBasalMeltLegend() {
  if (!controlsUI.basalMeltLegendBar || !controlsUI.basalMeltLegendLabels) return;
  updateContinuousLegend(controlsUI.basalMeltLegendBar, (t) => sampleColorStops(RISE_BASAL_MELT_COLOR_STOPS, t));
  renderBasalMeltLegendTicks();
  if (controlsUI.basalMeltLegendNote) {
    const note = getBasalMeltVisualizationConfig().note || "";
    controlsUI.basalMeltLegendNote.textContent = note;
    controlsUI.basalMeltLegendNote.hidden = !note;
  }
}

function updateThermalDrivingLegend() {
  if (!controlsUI.thermalDrivingLegendBar || !controlsUI.thermalDrivingLegendLabels) return;
  const range = getRiseVisualizationRange("thermal_driving");
  updateContinuousLegend(controlsUI.thermalDrivingLegendBar, (t) => thermalDrivingColor(lerp(range[0], range[1], t)));
  const mid = (range[0] + range[1]) * 0.5;
  controlsUI.thermalDrivingLegendLabels.innerHTML = `
    <span>${formatCoord(range[0], 2)} °C</span>
    <span>${formatCoord(mid, 2)}</span>
    <span>${formatCoord(range[1], 2)} °C</span>
  `;
}

function updateEffectivePressureLegend() {
  updateContinuousLegend(controlsUI.effectivePressureLegendBar, (t) => {
    const pressure = t * EFFECTIVE_PRESSURE_REFERENCE_PA;
    return effectivePressureColor(pressure);
  });
}

function updateChannelLegend() {
  updateContinuousLegend(controlsUI.channelLegendBar, (t) => {
    const discharge = CHANNEL_DISCHARGE_MIN * (CHANNEL_DISCHARGE_MAX / CHANNEL_DISCHARGE_MIN) ** clamp01(t);
    return subglacialChannelColor(discharge);
  });
}

function getOceanCurrentVisualizationRanges(oceanMeta) {
  const visualization = oceanMeta?.visualization || {};
  const thetaRange = Array.isArray(visualization.temperature_range_c) ? visualization.temperature_range_c : [-2, 8];
  const salinityRange = Array.isArray(visualization.salinity_range_psu) ? visualization.salinity_range_psu : [30, 35];
  return {
    thetaMin: Number(thetaRange[0]),
    thetaMax: Number(thetaRange[1]),
    salinityMin: Number(salinityRange[0]),
    salinityMax: Number(salinityRange[1]),
  };
}

function formatOceanLegendValue(value, unit, digits = 1) {
  if (!Number.isFinite(value)) return `n/a ${unit}`;
  return `${Number(value).toLocaleString(numberLocale, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })} ${unit}`;
}

function updateOceanCurrentLegend() {
  if (
    !controlsUI.oceanLegendSection ||
    !controlsUI.oceanLegendCanvas ||
    !controlsUI.oceanLegendWarmLabel ||
    !controlsUI.oceanLegendColdLabel ||
    !controlsUI.oceanLegendFreshLabel ||
    !controlsUI.oceanLegendSaltyLabel
  ) {
    return;
  }

  const region = getRegionConfig(currentRegionKey);
  const datasetKey = currentDatasetKey || datasetSelectionByRegion[region.key] || getDefaultDatasetKey(region.key);
  const dataset = getDatasetConfig(region.key, datasetKey);
  const oceanLegendVisible = Boolean(
    !isShowcaseMode && region.capabilities?.oceanCurrents && dataset?.oceanCurrentsMetaUrl && controlsUI.showOceanCurrents?.checked
  );
  controlsUI.oceanLegendSection.hidden = !oceanLegendVisible;
  if (!oceanLegendVisible) return;

  const ranges = getOceanCurrentVisualizationRanges(currentOceanCurrentMeta);
  const { thetaMin, thetaMax, salinityMin, salinityMax } = ranges;
  controlsUI.oceanLegendWarmLabel.innerHTML = `<strong>${t("explorer.legends.warm")}</strong>${formatOceanLegendValue(
    thetaMax,
    "°C"
  )}`;
  controlsUI.oceanLegendColdLabel.innerHTML = `<strong>${t("explorer.legends.cold")}</strong>${formatOceanLegendValue(
    thetaMin,
    "°C"
  )}`;
  controlsUI.oceanLegendFreshLabel.innerHTML = `<strong>${t("explorer.legends.fresh")}</strong>${formatOceanLegendValue(
    salinityMin,
    "PSU"
  )}`;
  controlsUI.oceanLegendSaltyLabel.innerHTML = `<strong>${t("explorer.legends.salty")}</strong>${formatOceanLegendValue(
    salinityMax,
    "PSU"
  )}`;

  const canvas = controlsUI.oceanLegendCanvas;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const width = canvas.width;
  const height = canvas.height;
  const image = ctx.createImageData(width, height);
  const data = image.data;
  const thetaSpan = Math.max(1e-6, thetaMax - thetaMin);
  const salinitySpan = Math.max(1e-6, salinityMax - salinityMin);

  for (let y = 0; y < height; y += 1) {
    const theta = thetaMin + ((height - 1 - y) / Math.max(1, height - 1)) * thetaSpan;
    for (let x = 0; x < width; x += 1) {
      const salinity = salinityMin + (x / Math.max(1, width - 1)) * salinitySpan;
      const color = oceanCurrentColor(theta, salinity, currentOceanCurrentMeta);
      const idx = (y * width + x) * 4;
      data[idx] = Math.round(clamp01(color[0]) * 255);
      data[idx + 1] = Math.round(clamp01(color[1]) * 255);
      data[idx + 2] = Math.round(clamp01(color[2]) * 255);
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function isIceCoveredMask(maskValue) {
  // BedMachine mask=4 is Lake Vostok: subglacial water beneath valid ice thickness/surface.
  return maskValue === 2 || maskValue === 3 || maskValue === 4;
}

function iceColor(thicknessMeters, maskValue) {
  const t = clamp01(thicknessMeters / 4200);
  const grounded = lerpColor([0.72, 0.84, 0.95], [0.97, 0.99, 1.0], t);
  const floating = lerpColor([0.58, 0.78, 0.93], [0.87, 0.95, 1.0], t);
  return maskValue === 3 ? floating : grounded;
}

function iceBottomColor(thicknessMeters, maskValue) {
  const t = clamp01(thicknessMeters / 4200);
  const grounded = lerpColor([0.22, 0.42, 0.58], [0.48, 0.66, 0.78], t);
  const floating = lerpColor([0.14, 0.34, 0.56], [0.4, 0.61, 0.79], t);
  return maskValue === 3 ? floating : grounded;
}

function sampleColorStops(stops, t) {
  if (!stops.length) return [1, 1, 1];
  const clamped = clamp01(t);
  if (clamped <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i += 1) {
    if (clamped <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const localT = (clamped - t0) / Math.max(1e-6, t1 - t0);
      return lerpColor(c0, c1, localT);
    }
  }
  return stops[stops.length - 1][1];
}

function velocityColor(speedMetersPerYear) {
  const scaled = velocityScaleT(speedMetersPerYear);
  return sampleColorStops(
    [
      [0.0, [0.06, 0.2, 0.5]],
      [0.35, [0.08, 0.62, 0.86]],
      [0.65, [0.95, 0.9, 0.27]],
      [1.0, [0.9, 0.2, 0.12]],
    ],
    scaled
  );
}

function basalFrictionColor(valueMpa) {
  return sampleColorStops(BASAL_FRICTION_COLOR_STOPS, basalFrictionScaleT(valueMpa));
}

function getVelocityTextureType(rendererInstance, nx = 0, ny = 0) {
  if (!THREE || !rendererInstance) return null;
  const maxDimension = Math.max(0, Number(nx) || 0, Number(ny) || 0);
  const maxTextureSize = Number(rendererInstance.capabilities?.maxTextureSize || 0);
  if (maxDimension > 0 && maxTextureSize > 0 && maxDimension > maxTextureSize) {
    return null;
  }

  const isWebGL2 = Boolean(rendererInstance.capabilities?.isWebGL2);
  const extensions = rendererInstance.extensions;
  if (isWebGL2 || extensions?.get("OES_texture_float")) {
    return THREE.FloatType;
  }
  if (isWebGL2 || extensions?.get("OES_texture_half_float")) {
    return THREE.HalfFloatType;
  }
  return null;
}

function supportsContinuousVelocitySampling(rendererInstance, nx = 0, ny = 0) {
  return Boolean(getVelocityTextureType(rendererInstance, nx, ny));
}

function createVelocityDataTexture(workerResult, nx, ny, rendererInstance) {
  const textureType = getVelocityTextureType(rendererInstance, nx, ny);
  if (!textureType) return null;

  const cellCount = nx * ny;
  if (
    workerResult.velocityX.length !== cellCount ||
    workerResult.velocityY.length !== cellCount ||
    workerResult.velocityValid.length !== cellCount
  ) {
    throw new Error("Velocity texture field length mismatch.");
  }

  let textureData;
  if (textureType === THREE.FloatType) {
    textureData = new Float32Array(cellCount * 4);
    for (let i = 0; i < cellCount; i += 1) {
      const base = i * 4;
      const isValid = workerResult.velocityValid[i] > 0;
      textureData[base] = isValid ? workerResult.velocityX[i] : 0;
      textureData[base + 1] = isValid ? workerResult.velocityY[i] : 0;
      textureData[base + 2] = isValid ? 1 : 0;
      textureData[base + 3] = 0;
    }
  } else {
    const toHalfFloat = THREE.DataUtils?.toHalfFloat;
    if (typeof toHalfFloat !== "function") {
      return null;
    }
    textureData = new Uint16Array(cellCount * 4);
    for (let i = 0; i < cellCount; i += 1) {
      const base = i * 4;
      const isValid = workerResult.velocityValid[i] > 0;
      textureData[base] = toHalfFloat(isValid ? workerResult.velocityX[i] : 0);
      textureData[base + 1] = toHalfFloat(isValid ? workerResult.velocityY[i] : 0);
      textureData[base + 2] = toHalfFloat(isValid ? 1 : 0);
      textureData[base + 3] = toHalfFloat(0);
    }
  }

  const texture = new THREE.DataTexture(textureData, nx, ny, THREE.RGBAFormat, textureType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.unpackAlignment = 1;
  if ("colorSpace" in texture && "NoColorSpace" in THREE) {
    texture.colorSpace = THREE.NoColorSpace;
  }
  texture.needsUpdate = true;
  return texture;
}

function createScalarFieldDataTexture(values, validMask, nx, ny, rendererInstance) {
  const textureType = getVelocityTextureType(rendererInstance, nx, ny);
  if (!textureType) return null;

  const cellCount = nx * ny;
  if (values.length !== cellCount || validMask.length !== cellCount) {
    throw new Error("Scalar field texture length mismatch.");
  }

  let textureData;
  if (textureType === THREE.FloatType) {
    textureData = new Float32Array(cellCount * 4);
    for (let i = 0; i < cellCount; i += 1) {
      const base = i * 4;
      const isValid = validMask[i] > 0 && Number.isFinite(values[i]);
      textureData[base] = isValid ? values[i] : 0;
      textureData[base + 1] = isValid ? 1 : 0;
      textureData[base + 2] = 0;
      textureData[base + 3] = 0;
    }
  } else {
    const toHalfFloat = THREE.DataUtils?.toHalfFloat;
    if (typeof toHalfFloat !== "function") {
      return null;
    }
    textureData = new Uint16Array(cellCount * 4);
    for (let i = 0; i < cellCount; i += 1) {
      const base = i * 4;
      const isValid = validMask[i] > 0 && Number.isFinite(values[i]);
      textureData[base] = toHalfFloat(isValid ? values[i] : 0);
      textureData[base + 1] = toHalfFloat(isValid ? 1 : 0);
      textureData[base + 2] = toHalfFloat(0);
      textureData[base + 3] = toHalfFloat(0);
    }
  }

  const texture = new THREE.DataTexture(textureData, nx, ny, THREE.RGBAFormat, textureType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.unpackAlignment = 1;
  if ("colorSpace" in texture && "NoColorSpace" in THREE) {
    texture.colorSpace = THREE.NoColorSpace;
  }
  texture.needsUpdate = true;
  return texture;
}

function createVelocitySurfaceShaderMaterial(texture, nx, ny) {
  const { range, knee } = getVelocityVisualizationConfig();
  const velocityMax = Number(range[1]);
  const safeKnee = Math.max(1e-6, Number(knee) || 1);
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL1,
    uniforms: {
      uVelocityTex: { value: texture },
      uVelocityTexSize: { value: new THREE.Vector2(nx, ny) },
      uVelocityMax: { value: velocityMax },
      uVelocityKnee: { value: safeKnee },
      uOpacity: { value: VELOCITY_SURFACE_OPACITY },
    },
    vertexShader: `
      precision highp float;

      varying vec2 vUv;

      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      uniform sampler2D uVelocityTex;
      uniform vec2 uVelocityTexSize;
      uniform float uVelocityMax;
      uniform float uVelocityKnee;
      uniform float uOpacity;

      varying vec2 vUv;

      vec3 velocityPalette(float t) {
        const vec3 c0 = vec3(0.06, 0.20, 0.50);
        const vec3 c1 = vec3(0.08, 0.62, 0.86);
        const vec3 c2 = vec3(0.95, 0.90, 0.27);
        const vec3 c3 = vec3(0.90, 0.20, 0.12);
        float clamped = clamp(t, 0.0, 1.0);
        if (clamped <= 0.35) {
          return mix(c0, c1, clamped / 0.35);
        }
        if (clamped <= 0.65) {
          return mix(c1, c2, (clamped - 0.35) / 0.30);
        }
        return mix(c2, c3, (clamped - 0.65) / 0.35);
      }

      float velocityScaleT(float speed) {
        float clamped = clamp(speed, 0.0, uVelocityMax);
        return clamp(log(1.0 + clamped / uVelocityKnee) / log(1.0 + uVelocityMax / uVelocityKnee), 0.0, 1.0);
      }

      void main() {
        vec2 texSize = max(uVelocityTexSize, vec2(1.0));
        vec2 texelPos = clamp(vUv, vec2(0.0), vec2(1.0)) * (texSize - 1.0);
        vec2 maxBase = max(texSize - 2.0, vec2(0.0));
        vec2 base = clamp(floor(texelPos), vec2(0.0), maxBase);
        vec2 frac = clamp(texelPos - base, vec2(0.0), vec2(1.0));
        vec2 texelSize = 1.0 / texSize;

        vec4 s00 = texture2D(uVelocityTex, (base + vec2(0.5, 0.5)) * texelSize);
        vec4 s10 = texture2D(uVelocityTex, (base + vec2(1.5, 0.5)) * texelSize);
        vec4 s01 = texture2D(uVelocityTex, (base + vec2(0.5, 1.5)) * texelSize);
        vec4 s11 = texture2D(uVelocityTex, (base + vec2(1.5, 1.5)) * texelSize);

        if (s00.b < 0.5 || s10.b < 0.5 || s01.b < 0.5 || s11.b < 0.5) {
          discard;
        }

        vec2 v0 = mix(s00.rg, s10.rg, frac.x);
        vec2 v1 = mix(s01.rg, s11.rg, frac.x);
        vec2 velocity = mix(v0, v1, frac.y);
        float speed = length(velocity);
        vec3 color = velocityPalette(velocityScaleT(speed));
        gl_FragColor = vec4(color, uOpacity);
      }
    `,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
}

function getRiseVisualizationRangeForMeta(riseMeta, key) {
  const visualization = riseMeta?.visualization?.[key];
  if (key === "basal_melt") {
    const { range } = getBasalMeltVisualizationConfig();
    return [Number(range[0]), Number(range[1])];
  }
  if (key === "thermal_driving") {
    const range = Array.isArray(visualization?.display_range_c) ? visualization.display_range_c : [0, 0.8];
    return [Number(range[0]), Number(range[1])];
  }
  return [0, 1];
}

function getRiseVisualizationRange(key) {
  return getRiseVisualizationRangeForMeta(currentRiseMeta, key);
}

function basalMeltColor(meltMetersPerYear) {
  const t = basalMeltScaleT(meltMetersPerYear);
  return sampleColorStops(RISE_BASAL_MELT_COLOR_STOPS, t);
}

function thermalDrivingColor(thermalDrivingC) {
  const [minValue, maxValue] = getRiseVisualizationRange("thermal_driving");
  const t = clamp01((Number(thermalDrivingC) - minValue) / Math.max(1e-6, maxValue - minValue));
  return sampleColorStops(RISE_THERMAL_DRIVING_COLOR_STOPS, t);
}

function createRiseFieldShaderMaterial(texture, nx, ny, { kind, riseMeta }) {
  const thermalRange = getRiseVisualizationRangeForMeta(riseMeta, "thermal_driving");
  const basalVisualization = getBasalMeltVisualizationConfig();
  const mode = kind === "basal_melt" ? 0 : 1;
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL1,
    uniforms: {
      uFieldTex: { value: texture },
      uFieldTexSize: { value: new THREE.Vector2(nx, ny) },
      uMode: { value: mode },
      uThermalRange: { value: new THREE.Vector2(Number(thermalRange[0]), Number(thermalRange[1])) },
      uBasalRange: { value: new THREE.Vector2(Number(basalVisualization.range[0]), Number(basalVisualization.range[1])) },
      uBasalLinthresh: { value: Math.max(1e-6, Number(basalVisualization.linthresh) || 1) },
      uOpacity: { value: 0.98 },
    },
    vertexShader: `
      precision highp float;

      varying vec2 vUv;

      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      uniform sampler2D uFieldTex;
      uniform vec2 uFieldTexSize;
      uniform float uMode;
      uniform vec2 uThermalRange;
      uniform vec2 uBasalRange;
      uniform float uBasalLinthresh;
      uniform float uOpacity;

      varying vec2 vUv;

      vec3 basalPalette(float t) {
        const vec3 c0 = vec3(0.082, 0.216, 0.376);
        const vec3 c1 = vec3(0.310, 0.655, 0.847);
        const vec3 c2 = vec3(0.965, 0.969, 0.984);
        const vec3 c3 = vec3(0.969, 0.639, 0.357);
        const vec3 c4 = vec3(0.541, 0.067, 0.090);
        float clamped = clamp(t, 0.0, 1.0);
        if (clamped <= 0.22) {
          return mix(c0, c1, clamped / 0.22);
        }
        if (clamped <= 0.50) {
          return mix(c1, c2, (clamped - 0.22) / 0.28);
        }
        if (clamped <= 0.78) {
          return mix(c2, c3, (clamped - 0.50) / 0.28);
        }
        return mix(c3, c4, (clamped - 0.78) / 0.22);
      }

      vec3 thermalPalette(float t) {
        const vec3 c0 = vec3(0.031, 0.114, 0.345);
        const vec3 c1 = vec3(0.133, 0.369, 0.659);
        const vec3 c2 = vec3(0.114, 0.569, 0.753);
        const vec3 c3 = vec3(0.498, 0.804, 0.733);
        const vec3 c4 = vec3(1.000, 0.953, 0.639);
        float clamped = clamp(t, 0.0, 1.0);
        if (clamped <= 0.22) {
          return mix(c0, c1, clamped / 0.22);
        }
        if (clamped <= 0.52) {
          return mix(c1, c2, (clamped - 0.22) / 0.30);
        }
        if (clamped <= 0.78) {
          return mix(c2, c3, (clamped - 0.52) / 0.26);
        }
        return mix(c3, c4, (clamped - 0.78) / 0.22);
      }

      float basalScaleT(float value) {
        float clampedValue = clamp(value, uBasalRange.x, uBasalRange.y);
        if (clampedValue <= 0.0) {
          float magnitude = abs(clampedValue);
          float negativeMax = max(uBasalLinthresh, abs(uBasalRange.x));
          float scaled = log(1.0 + magnitude / uBasalLinthresh) / log(1.0 + negativeMax / uBasalLinthresh);
          return 0.5 * (1.0 - clamp(scaled, 0.0, 1.0));
        }
        float positiveMax = max(uBasalLinthresh, uBasalRange.y);
        float scaled = log(1.0 + clampedValue / uBasalLinthresh) / log(1.0 + positiveMax / uBasalLinthresh);
        return 0.5 + 0.5 * clamp(scaled, 0.0, 1.0);
      }

      void main() {
        vec2 texSize = max(uFieldTexSize, vec2(1.0));
        vec2 texelPos = clamp(vUv, vec2(0.0), vec2(1.0)) * (texSize - 1.0);
        vec2 maxBase = max(texSize - 2.0, vec2(0.0));
        vec2 base = clamp(floor(texelPos), vec2(0.0), maxBase);
        vec2 frac = clamp(texelPos - base, vec2(0.0), vec2(1.0));
        vec2 texelSize = 1.0 / texSize;

        vec4 s00 = texture2D(uFieldTex, (base + vec2(0.5, 0.5)) * texelSize);
        vec4 s10 = texture2D(uFieldTex, (base + vec2(1.5, 0.5)) * texelSize);
        vec4 s01 = texture2D(uFieldTex, (base + vec2(0.5, 1.5)) * texelSize);
        vec4 s11 = texture2D(uFieldTex, (base + vec2(1.5, 1.5)) * texelSize);

        if (s00.g < 0.5 || s10.g < 0.5 || s01.g < 0.5 || s11.g < 0.5) {
          discard;
        }

        float v0 = mix(s00.r, s10.r, frac.x);
        float v1 = mix(s01.r, s11.r, frac.x);
        float value = mix(v0, v1, frac.y);

        vec3 color;
        if (uMode < 0.5) {
          color = basalPalette(basalScaleT(value));
        } else {
          float t = clamp((value - uThermalRange.x) / max(1e-6, uThermalRange.y - uThermalRange.x), 0.0, 1.0);
          color = thermalPalette(t);
        }

        gl_FragColor = vec4(color, uOpacity);
      }
    `,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
}

function oceanCurrentColor(thetaC, salinityPsu, oceanMeta) {
  const visualization = oceanMeta?.visualization || {};
  const thetaRange = Array.isArray(visualization.temperature_range_c) ? visualization.temperature_range_c : [-2, 8];
  const salinityRange = Array.isArray(visualization.salinity_range_psu) ? visualization.salinity_range_psu : [30, 35];
  const thetaSpan = Math.max(1e-6, Number(thetaRange[1]) - Number(thetaRange[0]));
  const salinitySpan = Math.max(1e-6, Number(salinityRange[1]) - Number(salinityRange[0]));
  const thetaT = clamp01((Number(thetaC) - Number(thetaRange[0])) / thetaSpan);
  const salinityT = clamp01((Number(salinityPsu) - Number(salinityRange[0])) / salinitySpan);

  const coldFresh = [0.0, 0.87, 0.99];
  const coldSalty = [0.23, 0.11, 0.88];
  const warmFresh = [0.0, 0.76, 0.2];
  const warmSalty = [0.99, 0.28, 0.0];

  const freshBlend = lerpColor(coldFresh, warmFresh, thetaT);
  const saltyBlend = lerpColor(coldSalty, warmSalty, thetaT);
  const color = lerpColor(freshBlend, saltyBlend, salinityT);
  const brightnessLift = 0.022 + thetaT * 0.018;
  return [
    clamp01(color[0] + brightnessLift),
    clamp01(color[1] + brightnessLift * 0.65),
    clamp01(color[2] + brightnessLift * 0.18),
  ];
}

function effectivePressureColor(pressurePa) {
  const scaled = clamp01(Math.max(0, pressurePa) / EFFECTIVE_PRESSURE_REFERENCE_PA);
  if (effectivePressureLut.length) {
    return sampleColorLUT(effectivePressureLut, scaled);
  }
  return sampleColorStops(
    [
      [0.0, [0.9, 0.95, 0.95]],
      [1.0, [0.21, 0.05, 0.14]],
    ],
    scaled
  );
}

function channelDischargeNormalized(dischargeM3PerS) {
  if (!Number.isFinite(dischargeM3PerS) || dischargeM3PerS <= 0) return 0;
  const clamped = Math.min(CHANNEL_DISCHARGE_MAX, Math.max(CHANNEL_DISCHARGE_MIN, dischargeM3PerS));
  return (
    (Math.log10(clamped) - Math.log10(CHANNEL_DISCHARGE_MIN)) /
    (Math.log10(CHANNEL_DISCHARGE_MAX) - Math.log10(CHANNEL_DISCHARGE_MIN))
  );
}

function subglacialChannelColor(dischargeM3PerS) {
  const scaled = channelDischargeNormalized(dischargeM3PerS);
  if (channelDischargeLut.length) {
    return sampleColorLUT(channelDischargeLut, scaled);
  }
  return sampleColorStops(
    [
      [0.0, [0.06, 0.16, 0.38]],
      [0.3, [0.05, 0.5, 0.74]],
      [0.62, [0.73, 0.85, 0.25]],
      [1.0, [0.86, 0.24, 0.1]],
    ],
    scaled
  );
}

function buildSurfaceGeometry({
  nx,
  ny,
  dxMeters,
  dyMeters,
  horizontalMetersPerUnit,
  verticalMetersPerUnit,
  heights,
  valid,
  colorFn,
  extraField,
  mask,
  computeNormals = true,
}) {
  const vertexCount = nx * ny;
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Uint8Array(vertexCount * 3);
  const indices = [];
  const halfX = (nx - 1) / 2;
  const halfY = (ny - 1) / 2;
  const absDy = Math.abs(dyMeters);

  for (let row = 0; row < ny; row += 1) {
    for (let col = 0; col < nx; col += 1) {
      const i = row * nx + col;
      const px = ((col - halfX) * dxMeters) / horizontalMetersPerUnit;
      const pz = ((row - halfY) * absDy) / horizontalMetersPerUnit;
      const h = heights[i];
      const isValid = valid[i];
      positions[3 * i] = px;
      positions[3 * i + 1] = isValid ? h / verticalMetersPerUnit : 0;
      positions[3 * i + 2] = pz;

      const rgb = colorFn(h, extraField ? extraField[i] : 0, mask ? mask[i] : 0);
      colors[3 * i] = Math.round(rgb[0] * 255);
      colors[3 * i + 1] = Math.round(rgb[1] * 255);
      colors[3 * i + 2] = Math.round(rgb[2] * 255);
    }
  }

  for (let row = 0; row < ny - 1; row += 1) {
    for (let col = 0; col < nx - 1; col += 1) {
      const i0 = row * nx + col;
      const i1 = i0 + 1;
      const i2 = i0 + nx;
      const i3 = i2 + 1;

      if (valid[i0] && valid[i2] && valid[i1]) {
        indices.push(i0, i2, i1);
      }
      if (valid[i1] && valid[i2] && valid[i3]) {
        indices.push(i1, i2, i3);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
  geometry.setIndex(indices);
  if (computeNormals) {
    geometry.computeVertexNormals();
  }
  return geometry;
}

function buildIceSideGeometry({
  nx,
  ny,
  dxMeters,
  dyMeters,
  horizontalMetersPerUnit,
  verticalMetersPerUnit,
  topHeights,
  bottomHeights,
  valid,
  colorFn,
  extraField,
  mask,
}) {
  const positions = [];
  const colors = [];
  const halfX = (nx - 1) / 2;
  const halfY = (ny - 1) / 2;
  const absDy = Math.abs(dyMeters);

  const vertexCache = new Array(nx * ny);
  const colorCache = new Array(nx * ny);

  const getVertex = (index) => {
    if (vertexCache[index]) return vertexCache[index];
    const row = Math.floor(index / nx);
    const col = index % nx;
    const x = ((col - halfX) * dxMeters) / horizontalMetersPerUnit;
    const z = ((row - halfY) * absDy) / horizontalMetersPerUnit;
    const topY = topHeights[index] / verticalMetersPerUnit;
    const botY = bottomHeights[index] / verticalMetersPerUnit;
    const value = {
      top: [x, topY, z],
      bottom: [x, botY, z],
    };
    vertexCache[index] = value;
    return value;
  };

  const getColor = (index) => {
    if (colorCache[index]) return colorCache[index];
    const rgb = colorFn(topHeights[index], extraField ? extraField[index] : 0, mask ? mask[index] : 0);
    const value = [Math.round(rgb[0] * 255), Math.round(rgb[1] * 255), Math.round(rgb[2] * 255)];
    colorCache[index] = value;
    return value;
  };

  const pushVertex = (point, color) => {
    positions.push(point[0], point[1], point[2]);
    colors.push(color[0], color[1], color[2]);
  };

  const pushBoundaryQuad = (a, b) => {
    if (
      !Number.isFinite(topHeights[a]) ||
      !Number.isFinite(topHeights[b]) ||
      !Number.isFinite(bottomHeights[a]) ||
      !Number.isFinite(bottomHeights[b])
    ) {
      return;
    }

    const vA = getVertex(a);
    const vB = getVertex(b);
    const cA = getColor(a);
    const cB = getColor(b);

    pushVertex(vA.top, cA);
    pushVertex(vB.top, cB);
    pushVertex(vA.bottom, cA);

    pushVertex(vB.top, cB);
    pushVertex(vB.bottom, cB);
    pushVertex(vA.bottom, cA);
  };

  const edgeMap = new Map();
  const addTriangleEdge = (a, b) => {
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    const existing = edgeMap.get(key);
    if (existing) {
      existing.count += 1;
      return;
    }
    edgeMap.set(key, { a, b, count: 1 });
  };

  for (let row = 0; row < ny - 1; row += 1) {
    for (let col = 0; col < nx - 1; col += 1) {
      const i0 = row * nx + col;
      const i1 = i0 + 1;
      const i2 = i0 + nx;
      const i3 = i2 + 1;

      if (valid[i0] && valid[i2] && valid[i1]) {
        addTriangleEdge(i0, i2);
        addTriangleEdge(i2, i1);
        addTriangleEdge(i1, i0);
      }
      if (valid[i1] && valid[i2] && valid[i3]) {
        addTriangleEdge(i1, i2);
        addTriangleEdge(i2, i3);
        addTriangleEdge(i3, i1);
      }
    }
  }

  edgeMap.forEach((edge) => {
    if (edge.count === 1) {
      pushBoundaryQuad(edge.a, edge.b);
    }
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Uint8Array(colors), 3, true));
  if (positions.length > 0) {
    geometry.computeVertexNormals();
  }
  return geometry;
}

function quickselectInPlace(values, k) {
  let left = 0;
  let right = values.length - 1;

  while (left < right) {
    const pivotIndex = left + Math.floor(Math.random() * (right - left + 1));
    const pivotValue = values[pivotIndex];
    [values[pivotIndex], values[right]] = [values[right], values[pivotIndex]];

    let store = left;
    for (let i = left; i < right; i += 1) {
      if (values[i] < pivotValue) {
        [values[store], values[i]] = [values[i], values[store]];
        store += 1;
      }
    }
    [values[right], values[store]] = [values[store], values[right]];

    if (k === store) {
      return values[store];
    }
    if (k < store) {
      right = store - 1;
    } else {
      left = store + 1;
    }
  }

  return values[left];
}

function finiteMedian(values) {
  let finiteCount = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (Number.isFinite(values[i])) finiteCount += 1;
  }
  if (finiteCount === 0) return Number.NaN;

  const sample = new Float64Array(finiteCount);
  let cursor = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (Number.isFinite(value)) {
      sample[cursor] = value;
      cursor += 1;
    }
  }

  const mid = Math.floor(finiteCount / 2);
  const upper = quickselectInPlace(sample, mid);
  if (finiteCount % 2 === 1) return upper;
  const lower = quickselectInPlace(sample, mid - 1);
  return (lower + upper) / 2;
}

function sampleVelocityBilinear(field, col, row) {
  if (col < 1 || row < 1 || col > field.nx - 2 || row > field.ny - 2) return null;

  const c0 = Math.floor(col);
  const c1 = c0 + 1;
  const r0 = Math.floor(row);
  const r1 = r0 + 1;
  const i00 = r0 * field.nx + c0;
  const i10 = r0 * field.nx + c1;
  const i01 = r1 * field.nx + c0;
  const i11 = r1 * field.nx + c1;

  if (!field.velocityValid[i00] || !field.velocityValid[i10] || !field.velocityValid[i01] || !field.velocityValid[i11]) {
    return null;
  }

  const tx = col - c0;
  const ty = row - r0;
  const w00 = (1 - tx) * (1 - ty);
  const w10 = tx * (1 - ty);
  const w01 = (1 - tx) * ty;
  const w11 = tx * ty;

  const vx =
    field.velocityX[i00] * w00 + field.velocityX[i10] * w10 + field.velocityX[i01] * w01 + field.velocityX[i11] * w11;
  const vy =
    field.velocityY[i00] * w00 + field.velocityY[i10] * w10 + field.velocityY[i01] * w01 + field.velocityY[i11] * w11;
  const speed = Math.hypot(vx, vy);
  if (!Number.isFinite(speed)) return null;
  return { vx, vy, speed };
}

function sampleSurfaceBilinear(field, col, row) {
  if (col < 1 || row < 1 || col > field.nx - 2 || row > field.ny - 2) return Number.NaN;

  const c0 = Math.floor(col);
  const c1 = c0 + 1;
  const r0 = Math.floor(row);
  const r1 = r0 + 1;
  const i00 = r0 * field.nx + c0;
  const i10 = r0 * field.nx + c1;
  const i01 = r1 * field.nx + c0;
  const i11 = r1 * field.nx + c1;
  if (!field.iceValid[i00] || !field.iceValid[i10] || !field.iceValid[i01] || !field.iceValid[i11]) {
    return sampleSurfaceNearest(field, col, row);
  }

  const h00 = field.surfaceHeights[i00];
  const h10 = field.surfaceHeights[i10];
  const h01 = field.surfaceHeights[i01];
  const h11 = field.surfaceHeights[i11];
  if (!Number.isFinite(h00) || !Number.isFinite(h10) || !Number.isFinite(h01) || !Number.isFinite(h11)) {
    return sampleSurfaceNearest(field, col, row);
  }

  const tx = col - c0;
  const ty = row - r0;
  const w00 = (1 - tx) * (1 - ty);
  const w10 = tx * (1 - ty);
  const w01 = (1 - tx) * ty;
  const w11 = tx * ty;
  return h00 * w00 + h10 * w10 + h01 * w01 + h11 * w11;
}

function velocityGridDirection(field, col, row) {
  const velocity = sampleVelocityBilinear(field, col, row);
  if (!velocity || velocity.speed < FLOWLINE_MIN_TRACE_SPEED) return null;

  const absDx = Math.abs(field.dxMeters);
  const absDy = Math.abs(field.dyMeters);
  const dCol = velocity.vx / absDx;
  const dRow = -velocity.vy / absDy;
  const norm = Math.hypot(dCol, dRow);
  if (norm < 1e-8) return null;
  return {
    speed: velocity.speed,
    dCol: dCol / norm,
    dRow: dRow / norm,
  };
}

function sampleSurfaceNearest(field, col, row) {
  const c = Math.round(col);
  const r = Math.round(row);
  if (c < 0 || r < 0 || c >= field.nx || r >= field.ny) return Number.NaN;
  const idx = r * field.nx + c;
  if (!field.iceValid[idx]) return Number.NaN;
  const value = field.surfaceHeights[idx];
  return Number.isFinite(value) ? value : Number.NaN;
}

function sampleIceBottomNearest(field, col, row) {
  const c = Math.round(col);
  const r = Math.round(row);
  if (c < 0 || r < 0 || c >= field.nx || r >= field.ny) return Number.NaN;
  const idx = r * field.nx + c;
  if (field.iceBottomValid?.[idx]) {
    const value = field.iceBottomHeights?.[idx];
    if (Number.isFinite(value)) return value;
  }
  if (field.iceValid?.[idx] && Number.isFinite(field.surfaceHeights?.[idx]) && Number.isFinite(field.thickness?.[idx])) {
    return field.surfaceHeights[idx] - field.thickness[idx];
  }
  return Number.NaN;
}

function sampleIceBottomBilinear(field, col, row) {
  if (col < 1 || row < 1 || col > field.nx - 2 || row > field.ny - 2) return sampleIceBottomNearest(field, col, row);

  const c0 = Math.floor(col);
  const c1 = c0 + 1;
  const r0 = Math.floor(row);
  const r1 = r0 + 1;
  const i00 = r0 * field.nx + c0;
  const i10 = r0 * field.nx + c1;
  const i01 = r1 * field.nx + c0;
  const i11 = r1 * field.nx + c1;
  if (!field.iceBottomValid?.[i00] || !field.iceBottomValid?.[i10] || !field.iceBottomValid?.[i01] || !field.iceBottomValid?.[i11]) {
    return sampleIceBottomNearest(field, col, row);
  }

  const h00 = field.iceBottomHeights[i00];
  const h10 = field.iceBottomHeights[i10];
  const h01 = field.iceBottomHeights[i01];
  const h11 = field.iceBottomHeights[i11];
  if (!Number.isFinite(h00) || !Number.isFinite(h10) || !Number.isFinite(h01) || !Number.isFinite(h11)) {
    return sampleIceBottomNearest(field, col, row);
  }

  const tx = col - c0;
  const ty = row - r0;
  const w00 = (1 - tx) * (1 - ty);
  const w10 = tx * (1 - ty);
  const w01 = (1 - tx) * ty;
  const w11 = tx * ty;
  return h00 * w00 + h10 * w10 + h01 * w01 + h11 * w11;
}

function sampleBedNearest(field, col, row) {
  const c = Math.round(col);
  const r = Math.round(row);
  if (c < 0 || r < 0 || c >= field.nx || r >= field.ny) return Number.NaN;
  const idx = r * field.nx + c;
  if (!field.bedValid?.[idx]) return Number.NaN;
  const value = field.bedHeights?.[idx];
  return Number.isFinite(value) ? value : Number.NaN;
}

function sampleBedBilinear(field, col, row) {
  if (col < 1 || row < 1 || col > field.nx - 2 || row > field.ny - 2) return sampleBedNearest(field, col, row);

  const c0 = Math.floor(col);
  const c1 = c0 + 1;
  const r0 = Math.floor(row);
  const r1 = r0 + 1;
  const i00 = r0 * field.nx + c0;
  const i10 = r0 * field.nx + c1;
  const i01 = r1 * field.nx + c0;
  const i11 = r1 * field.nx + c1;
  if (!field.bedValid?.[i00] || !field.bedValid?.[i10] || !field.bedValid?.[i01] || !field.bedValid?.[i11]) {
    return sampleBedNearest(field, col, row);
  }

  const h00 = field.bedHeights[i00];
  const h10 = field.bedHeights[i10];
  const h01 = field.bedHeights[i01];
  const h11 = field.bedHeights[i11];
  if (!Number.isFinite(h00) || !Number.isFinite(h10) || !Number.isFinite(h01) || !Number.isFinite(h11)) {
    return sampleBedNearest(field, col, row);
  }

  const tx = col - c0;
  const ty = row - r0;
  const w00 = (1 - tx) * (1 - ty);
  const w10 = tx * (1 - ty);
  const w01 = (1 - tx) * ty;
  const w11 = tx * ty;
  return h00 * w00 + h10 * w10 + h01 * w01 + h11 * w11;
}

function traceFlowlineDirection(field, seedCol, seedRow, direction) {
  const out = [];
  let col = seedCol;
  let row = seedRow;
  let previousDirection = null;

  for (let step = 0; step < FLOWLINE_MAX_STEPS; step += 1) {
    const guide = velocityGridDirection(field, col, row);
    if (!guide) break;

    const surface = sampleSurfaceBilinear(field, col, row);
    if (!Number.isFinite(surface)) break;

    if (
      previousDirection &&
      guide.dCol * previousDirection.dCol + guide.dRow * previousDirection.dRow < FLOWLINE_REVERSE_DIRECTION_DOT
    ) {
      break;
    }

    out.push({ col, row, speed: guide.speed, surface });

    const halfStep = FLOWLINE_STEP_CELLS * 0.5;
    const midCol = col + direction * guide.dCol * halfStep;
    const midRow = row + direction * guide.dRow * halfStep;
    const midGuide = velocityGridDirection(field, midCol, midRow);
    const nextGuide = midGuide || guide;

    if (
      previousDirection &&
      nextGuide.dCol * previousDirection.dCol + nextGuide.dRow * previousDirection.dRow < FLOWLINE_REVERSE_DIRECTION_DOT
    ) {
      break;
    }

    col += direction * nextGuide.dCol * FLOWLINE_STEP_CELLS;
    row += direction * nextGuide.dRow * FLOWLINE_STEP_CELLS;
    previousDirection = nextGuide;
  }

  return out;
}

function collectFlowlineSeeds(field, seedSpacing) {
  const seeds = [];
  const seen = new Set();
  const halfOffset = Math.max(1, Math.floor(seedSpacing / 2));

  for (let pass = 0; pass < FLOWLINE_SEED_PASSES; pass += 1) {
    const offset = pass === 0 ? 0 : halfOffset;
    const rowStart = 2 + offset;
    const colStart = 2 + offset;

    for (let rowBlock = rowStart; rowBlock < field.ny - 2; rowBlock += seedSpacing) {
      const rowEnd = Math.min(field.ny - 2, rowBlock + seedSpacing);
      for (let colBlock = colStart; colBlock < field.nx - 2; colBlock += seedSpacing) {
        const colEnd = Math.min(field.nx - 2, colBlock + seedSpacing);

        let bestIdx = -1;
        let bestSpeed = FLOWLINE_MIN_SEED_SPEED;

        for (let row = rowBlock; row < rowEnd; row += 1) {
          for (let col = colBlock; col < colEnd; col += 1) {
            const idx = row * field.nx + col;
            if (!field.iceValid[idx] || !field.velocityValid[idx]) continue;
            const speed = field.velocitySpeed[idx];
            if (!Number.isFinite(speed) || speed < bestSpeed) continue;
            bestSpeed = speed;
            bestIdx = idx;
          }
        }

        if (bestIdx < 0) continue;
        const seedRow = Math.floor(bestIdx / field.nx);
        const seedCol = bestIdx % field.nx;
        const key = `${seedCol}:${seedRow}`;
        if (seen.has(key)) continue;
        seen.add(key);
        seeds.push({ col: seedCol, row: seedRow });
      }
    }
  }

  return seeds;
}

function getEvenlySpacedSampleIndices(pointCount, targetCount) {
  if (!(pointCount > 0)) return [];
  const safeTarget = Math.max(2, Math.round(targetCount || 0));
  if (pointCount <= safeTarget) {
    return Array.from({ length: pointCount }, (_unused, index) => index);
  }

  const indices = [0];
  const stride = (pointCount - 1) / (safeTarget - 1);
  let previous = 0;
  for (let i = 1; i < safeTarget - 1; i += 1) {
    const next = Math.round(i * stride);
    if (next <= previous || next >= pointCount - 1) continue;
    indices.push(next);
    previous = next;
  }
  if (indices[indices.length - 1] !== pointCount - 1) {
    indices.push(pointCount - 1);
  }
  return indices;
}

function buildFlowlineProfileSummary(field, flowlineIndex, merged) {
  if (!Array.isArray(merged) || merged.length < 3) return null;

  const absDx = Math.abs(field.dxMeters);
  const absDy = Math.abs(field.dyMeters);
  const halfX = (field.nx - 1) / 2;
  const halfY = (field.ny - 1) / 2;
  const cumulativeDistancesKm = new Float32Array(merged.length);
  const bottomHeights = new Float32Array(merged.length);
  const bedHeights = new Float32Array(merged.length);

  let lengthMeters = 0;
  let speedMin = Number.POSITIVE_INFINITY;
  let speedMax = Number.NEGATIVE_INFINITY;
  let surfaceMin = Number.POSITIVE_INFINITY;
  let surfaceMax = Number.NEGATIVE_INFINITY;
  let thicknessMin = Number.POSITIVE_INFINITY;
  let thicknessMax = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < merged.length; i += 1) {
    const point = merged[i];
    if (i > 0) {
      const prev = merged[i - 1];
      const dxMeters = (point.col - prev.col) * absDx;
      const dyMeters = (point.row - prev.row) * absDy;
      lengthMeters += Math.hypot(dxMeters, dyMeters);
    }
    cumulativeDistancesKm[i] = lengthMeters / 1000;

    const surface = Number(point.surface);
    let bottom = sampleIceBottomBilinear(field, point.col, point.row);
    if (!Number.isFinite(bottom)) bottom = surface;
    if (bottom > surface) bottom = surface;
    bottomHeights[i] = bottom;
    let bed = sampleBedBilinear(field, point.col, point.row);
    if (!Number.isFinite(bed)) bed = bottom;
    if (bed > bottom) bed = bottom;
    bedHeights[i] = bed;

    const thickness = Math.max(0, surface - bottom);
    if (Number.isFinite(point.speed)) {
      speedMin = Math.min(speedMin, point.speed);
      speedMax = Math.max(speedMax, point.speed);
    }
    surfaceMin = Math.min(surfaceMin, surface);
    surfaceMax = Math.max(surfaceMax, surface);
    thicknessMin = Math.min(thicknessMin, thickness);
    thicknessMax = Math.max(thicknessMax, thickness);
  }

  const profileSampleIndices = getEvenlySpacedSampleIndices(merged.length, FLOWLINE_PROFILE_SAMPLE_TARGET);
  const samples = profileSampleIndices.map((sampleIndex) => ({
    distanceKm: cumulativeDistancesKm[sampleIndex],
    surface: merged[sampleIndex].surface,
    bottom: bottomHeights[sampleIndex],
    bed: bedHeights[sampleIndex],
  }));

  const highlightSampleIndices = getEvenlySpacedSampleIndices(merged.length, FLOWLINE_HIGHLIGHT_SAMPLE_TARGET);
  const highlightPositions = [];
  for (const sampleIndex of highlightSampleIndices) {
    const point = merged[sampleIndex];
    const x = ((point.col - halfX) * field.dxMeters) / field.horizontalMetersPerUnit;
    const z = ((point.row - halfY) * absDy) / field.horizontalMetersPerUnit;
    const y = (point.surface + FLOWLINE_SURFACE_OFFSET_M) / field.verticalMetersPerUnit;
    highlightPositions.push(x, y, z);
  }

  return {
    index: flowlineIndex,
    lengthKm: lengthMeters / 1000,
    speedMin: Number.isFinite(speedMin) ? speedMin : Number.NaN,
    speedMax: Number.isFinite(speedMax) ? speedMax : Number.NaN,
    surfaceMin: Number.isFinite(surfaceMin) ? surfaceMin : Number.NaN,
    surfaceMax: Number.isFinite(surfaceMax) ? surfaceMax : Number.NaN,
    thicknessMin: Number.isFinite(thicknessMin) ? thicknessMin : Number.NaN,
    thicknessMax: Number.isFinite(thicknessMax) ? thicknessMax : Number.NaN,
    samples,
    highlightPositions,
  };
}

function createFlowLightMaterial({ staticOpacity, activeBaseOpacity, pulseOpacity, flowRate, useSegmentRates = false, depthTest = true }) {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL1,
    defines: useSegmentRates ? { USE_SEGMENT_FLOW_RATE: "" } : {},
    uniforms: {
      uFlowTime: flowLightUniforms.time,
      uFlowEnabled: flowLightUniforms.enabled,
      uStaticOpacity: { value: staticOpacity },
      uActiveBaseOpacity: { value: activeBaseOpacity },
      uPulseOpacity: { value: pulseOpacity },
      uPatternScale: { value: FLOW_LIGHT_PATTERN_SCALE },
      uFlowRate: { value: flowRate },
    },
    vertexShader: `
      precision highp float;

      attribute vec3 color;
      attribute float flowDistance;
      attribute float flowGlow;

      #ifdef USE_SEGMENT_FLOW_RATE
        attribute float flowRate;
      #endif

      varying vec3 vColor;
      varying float vFlowDistance;
      varying float vFlowRate;
      varying float vFlowGlow;

      void main() {
        vColor = color;
        vFlowDistance = flowDistance;
        #ifdef USE_SEGMENT_FLOW_RATE
          vFlowRate = flowRate;
        #else
          vFlowRate = 1.0;
        #endif
        vFlowGlow = flowGlow;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      uniform float uFlowTime;
      uniform float uFlowEnabled;
      uniform float uStaticOpacity;
      uniform float uActiveBaseOpacity;
      uniform float uPulseOpacity;
      uniform float uPatternScale;
      uniform float uFlowRate;

      varying vec3 vColor;
      varying float vFlowDistance;
      varying float vFlowRate;
      varying float vFlowGlow;

      void main() {
        if (uFlowEnabled < 0.5 || vFlowGlow < 0.5) {
          gl_FragColor = vec4(vColor, uStaticOpacity);
          return;
        }
        float phase = fract(vFlowDistance * uPatternScale - uFlowTime * uFlowRate * vFlowRate);
        float leading = smoothstep(0.03, 0.09, phase);
        float trailing = 1.0 - smoothstep(0.20, 0.34, phase);
        float pulse = leading * trailing;
        vec3 color = mix(vColor, vec3(1.0), pulse * 0.9);
        gl_FragColor = vec4(color, uActiveBaseOpacity + pulse * uPulseOpacity);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest,
  });
}

function createStaticFlowLineMaterial({ baseOpacity, depthTest = true }) {
  return new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: baseOpacity,
    depthWrite: false,
    depthTest,
  });
}

function createFlowLightParticleMaterial({ flowRate, pointSize, useSegmentRates = false }) {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL1,
    defines: useSegmentRates ? { USE_SEGMENT_FLOW_RATE: "" } : {},
    uniforms: {
      uFlowTime: flowLightUniforms.time,
      uFlowEnabled: flowLightUniforms.enabled,
      uPatternScale: { value: FLOW_LIGHT_PATTERN_SCALE },
      uFlowRate: { value: flowRate },
      uPointSize: { value: pointSize },
    },
    vertexShader: `
      precision highp float;

      attribute vec3 color;
      attribute float flowDistance;

      #ifdef USE_SEGMENT_FLOW_RATE
        attribute float flowRate;
      #endif

      uniform float uPointSize;

      varying vec3 vColor;
      varying float vFlowDistance;
      varying float vFlowRate;

      void main() {
        vColor = color;
        vFlowDistance = flowDistance;
        #ifdef USE_SEGMENT_FLOW_RATE
          vFlowRate = flowRate;
        #else
          vFlowRate = 1.0;
        #endif
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uPointSize * clamp(90.0 / max(1.0, -viewPosition.z), 0.55, 1.65);
        gl_Position = projectionMatrix * viewPosition;
      }
    `,
    fragmentShader: `
      precision highp float;

      uniform float uFlowTime;
      uniform float uFlowEnabled;
      uniform float uPatternScale;
      uniform float uFlowRate;

      varying vec3 vColor;
      varying float vFlowDistance;
      varying float vFlowRate;

      void main() {
        if (uFlowEnabled < 0.5) discard;
        float phase = fract(vFlowDistance * uPatternScale - uFlowTime * uFlowRate * vFlowRate);
        float leading = smoothstep(0.03, 0.09, phase);
        float trailing = 1.0 - smoothstep(0.20, 0.34, phase);
        float pulse = leading * trailing;
        if (pulse < 0.02) discard;
        float radius = length(gl_PointCoord - vec2(0.5));
        float disc = 1.0 - smoothstep(0.16, 0.5, radius);
        if (disc < 0.02) discard;
        vec3 color = mix(vColor, vec3(1.0), 0.95);
        gl_FragColor = vec4(color, pulse * disc);
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: true,
  });
}

function addFlowLightParticleOverlay(lines, { sampleStride, pointSize, flowRate, useSegmentRates = false }) {
  const geometry = lines?.geometry;
  const position = geometry?.getAttribute("position");
  const color = geometry?.getAttribute("color");
  const flowDistance = geometry?.getAttribute("flowDistance");
  const flowGlow = geometry?.getAttribute("flowGlow");
  const segmentRate = geometry?.getAttribute("flowRate");
  if (!position || !color || !flowDistance || !flowGlow || (useSegmentRates && !segmentRate)) return null;

  const particlePositions = [];
  const particleColors = [];
  const particleDistances = [];
  const particleRates = [];
  const safeStride = Math.max(1, Math.floor(sampleStride));
  for (let segmentIndex = 0; segmentIndex < Math.floor(position.count / 2); segmentIndex += safeStride) {
    const vertexIndex = segmentIndex * 2 + 1;
    if (flowGlow.getX(vertexIndex) < 0.5) continue;
    particlePositions.push(position.getX(vertexIndex), position.getY(vertexIndex), position.getZ(vertexIndex));
    particleColors.push(color.getX(vertexIndex), color.getY(vertexIndex), color.getZ(vertexIndex));
    particleDistances.push(flowDistance.getX(vertexIndex));
    if (useSegmentRates) particleRates.push(segmentRate.getX(vertexIndex));
  }
  if (!particlePositions.length) return null;

  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(particlePositions), 3));
  particleGeometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(particleColors), 3));
  particleGeometry.setAttribute("flowDistance", new THREE.BufferAttribute(new Float32Array(particleDistances), 1));
  if (useSegmentRates) {
    particleGeometry.setAttribute("flowRate", new THREE.BufferAttribute(new Float32Array(particleRates), 1));
  }

  const particles = new THREE.Points(
    particleGeometry,
    createFlowLightParticleMaterial({ flowRate, pointSize, useSegmentRates })
  );
  particles.renderOrder = lines.renderOrder + 0.1;
  particles.userData.isFlowLightParticleOverlay = true;
  lines.add(particles);
  return particles;
}

function addFlowLightAttributes(geometry, { distances, rates, glows }) {
  const vertexCount = geometry.getAttribute("position")?.count || 0;
  if (
    !vertexCount ||
    distances.length !== vertexCount ||
    (rates && rates.length !== vertexCount) ||
    glows.length !== vertexCount
  ) {
    throw new Error("Flow-light attributes do not match line geometry.");
  }
  const flowDistances = distances instanceof Float32Array ? distances : new Float32Array(distances);
  const flowGlows = glows instanceof Uint8Array ? glows : new Uint8Array(glows);
  geometry.setAttribute("flowDistance", new THREE.BufferAttribute(flowDistances, 1));
  if (rates) {
    const flowRates = rates instanceof Float32Array ? rates : new Float32Array(rates);
    geometry.setAttribute("flowRate", new THREE.BufferAttribute(flowRates, 1));
  }
  // Not normalized: flowGlow is a 0/1 flag, not a 0-255 value scaled into 0..1.
  // Marking it normalized makes both readers see 1/255, so `getX() < 0.5` skips every
  // vertex in the particle builder and `vFlowGlow < 0.5` sends every fragment down the
  // static branch of the shader -- silently disabling the whole flow-light effect.
  geometry.setAttribute("flowGlow", new THREE.BufferAttribute(flowGlows, 1, false));
}

function getIceFlowLightRate(speedMetersPerYear) {
  const referenceSpeed = Math.max(20, Number.isFinite(currentVelocityMedianSpeed) ? currentVelocityMedianSpeed : 100);
  const relativeSpeed = Math.sqrt(Math.max(0, Number(speedMetersPerYear) || 0) / referenceSpeed);
  return clamp(0.62 + relativeSpeed * 0.34, 0.62, 2.1);
}

function buildFlowlineMesh(field) {
  if (!field) return null;
  const buildFlowLights = isFlowLightAnimationEnabled();

  const totalCells = field.nx * field.ny;
  const seedSpacing = Math.max(8, Math.round(Math.sqrt(totalCells / FLOWLINE_SEED_TARGET)));
  const seeds = collectFlowlineSeeds(field, seedSpacing);
  const halfX = (field.nx - 1) / 2;
  const halfY = (field.ny - 1) / 2;
  const absDy = Math.abs(field.dyMeters);

  const positions = [];
  const colors = [];
  const flowDistances = [];
  const flowRates = [];
  const flowGlows = [];
  const segmentFlowlineIndices = [];
  const flowlines = [];

  for (const seed of seeds) {
    const forward = traceFlowlineDirection(field, seed.col, seed.row, 1);
    const backward = traceFlowlineDirection(field, seed.col, seed.row, -1);
    if (!forward.length && !backward.length) continue;

    const merged = backward.reverse().concat(forward.slice(1));
    if (merged.length < 3) continue;

    const flowlineIndex = flowlines.length;
    const summary = buildFlowlineProfileSummary(field, flowlineIndex, merged);
    if (!summary) continue;
    flowlines.push(summary);
    let flowDistance = 0;

    for (let i = 1; i < merged.length; i += 1) {
      const p0 = merged[i - 1];
      const p1 = merged[i];

      const x0 = ((p0.col - halfX) * field.dxMeters) / field.horizontalMetersPerUnit;
      const z0 = ((p0.row - halfY) * absDy) / field.horizontalMetersPerUnit;
      const y0 = (p0.surface + FLOWLINE_SURFACE_OFFSET_M) / field.verticalMetersPerUnit;

      const x1 = ((p1.col - halfX) * field.dxMeters) / field.horizontalMetersPerUnit;
      const z1 = ((p1.row - halfY) * absDy) / field.horizontalMetersPerUnit;
      const y1 = (p1.surface + FLOWLINE_SURFACE_OFFSET_M) / field.verticalMetersPerUnit;

      const c0 = velocityColor(p0.speed);
      const c1 = velocityColor(p1.speed);
      const segmentLength = Math.hypot(x1 - x0, y1 - y0, z1 - z0);

      positions.push(x0, y0, z0, x1, y1, z1);
      colors.push(c0[0], c0[1], c0[2], c1[0], c1[1], c1[2]);
      flowDistances.push(flowDistance, flowDistance + segmentLength);
      flowRates.push(getIceFlowLightRate(p0.speed), getIceFlowLightRate(p1.speed));
      flowGlows.push(1, 1);
      segmentFlowlineIndices.push(flowlineIndex);
      flowDistance += segmentLength;
    }
  }

  if (!positions.length) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(colors), 3));
  if (buildFlowLights) {
    addFlowLightAttributes(geometry, {
      distances: flowDistances,
      rates: flowRates,
      glows: flowGlows,
    });
  }
  const material = buildFlowLights
    ? createFlowLightMaterial({
        staticOpacity: FLOW_LIGHT_STATIC_OPACITY,
        activeBaseOpacity: FLOW_LIGHT_ICE_ACTIVE_BASE_OPACITY,
        pulseOpacity: FLOW_LIGHT_PULSE_OPACITY,
        flowRate: FLOW_LIGHT_ICE_RATE,
        useSegmentRates: true,
      })
    : createStaticFlowLineMaterial({ baseOpacity: FLOW_LIGHT_STATIC_OPACITY });
  const lines = new THREE.LineSegments(geometry, material);
  lines.scale.y = Number(controlsUI.exaggeration.value);
  lines.visible = controlsUI.showFlowline.checked;
  lines.renderOrder = 16;
  if (buildFlowLights) {
    addFlowLightParticleOverlay(lines, {
      sampleStride: FLOW_LIGHT_ICE_PARTICLE_SAMPLE_STRIDE,
      pointSize: FLOW_LIGHT_ICE_PARTICLE_SIZE,
      flowRate: FLOW_LIGHT_ICE_RATE,
      useSegmentRates: true,
    });
  }
  lines.userData.flowlineCount = flowlines.length;
  lines.userData.flowlines = flowlines;
  lines.userData.segmentFlowlineIndices = Uint32Array.from(segmentFlowlineIndices);
  return lines;
}

function buildOceanCurrentMesh(context, oceanMeta, oceanBuffer) {
  if (!context || !oceanMeta || !(oceanBuffer instanceof ArrayBuffer)) return null;
  const buildFlowLights = isFlowLightAnimationEnabled();

  const x0Ps = parseField(oceanMeta, oceanBuffer, "x0_ps_m");
  const y0Ps = parseField(oceanMeta, oceanBuffer, "y0_ps_m");
  const depth0M = parseField(oceanMeta, oceanBuffer, "depth0_m");
  const x1Ps = parseField(oceanMeta, oceanBuffer, "x1_ps_m");
  const y1Ps = parseField(oceanMeta, oceanBuffer, "y1_ps_m");
  const depth1M = parseField(oceanMeta, oceanBuffer, "depth1_m");
  const theta0C = parseField(oceanMeta, oceanBuffer, "theta0_c");
  const sal0Psu = parseField(oceanMeta, oceanBuffer, "sal0_psu");
  const theta1C = parseField(oceanMeta, oceanBuffer, "theta1_c");
  const sal1Psu = parseField(oceanMeta, oceanBuffer, "sal1_psu");
  const terminalFlag = parseField(oceanMeta, oceanBuffer, "terminal_flag");
  const count = x0Ps.length;
  const layerSplitInfo = getOceanCurrentLayerSplitInfo(oceanMeta);
  const useLayerSplit = layerSplitInfo.enabled;

  if (
    y0Ps.length !== count ||
    depth0M.length !== count ||
    x1Ps.length !== count ||
    y1Ps.length !== count ||
    depth1M.length !== count ||
    theta0C.length !== count ||
    sal0Psu.length !== count ||
    theta1C.length !== count ||
    sal1Psu.length !== count ||
    terminalFlag.length !== count
  ) {
    throw new Error("Ocean-current package fields are misaligned.");
  }

  const layerBuffers = useLayerSplit
    ? Object.fromEntries(
      OCEAN_CURRENT_LAYER_ORDER.map((layer) => [layer, { positions: [], colors: [], segmentCount: 0 }])
    )
    : { all: { positions: [], colors: [], segmentCount: 0 } };
  for (const buffer of Object.values(layerBuffers)) {
    buffer.flowDistances = buildFlowLights ? [] : null;
    buffer.flowGlows = buildFlowLights ? [] : null;
    buffer.flowDistance = 0;
  }
  const bucketBoundaries = [];
  if (useLayerSplit) {
    let cumulative = 0;
    for (const bucket of layerSplitInfo.orderedBuckets) {
      cumulative += bucket.count;
      bucketBoundaries.push({ layer: bucket.layer, endExclusive: cumulative });
    }
  }
  let bucketBoundaryIndex = 0;
  let streamlineIndex = 0;

  function getCurrentLayerBuffer() {
    if (!useLayerSplit) return layerBuffers.all;
    while (
      bucketBoundaryIndex < bucketBoundaries.length - 1 &&
      streamlineIndex >= bucketBoundaries[bucketBoundaryIndex].endExclusive
    ) {
      bucketBoundaryIndex += 1;
    }
    const currentLayer = bucketBoundaries[bucketBoundaryIndex]?.layer || "mid";
    return layerBuffers[currentLayer] || layerBuffers.mid;
  }

  function finalizeStreamlineIfNeeded(i, buffer) {
    if (terminalFlag[i]) {
      streamlineIndex += 1;
      if (buffer) buffer.flowDistance = 0;
    }
  }

  function appendOceanCurrentArrow(buffer, scenePoint1, y1, dx, dy, dz, color1) {
    const length = Math.hypot(dx, dy, dz);
    if (!(Number.isFinite(length) && length > 1e-5)) return;
    const headLength = clamp(
      length * OCEAN_CURRENT_ARROW_HEAD_RATIO,
      OCEAN_CURRENT_ARROW_HEAD_MIN_UNITS,
      OCEAN_CURRENT_ARROW_HEAD_MAX_UNITS
    );
    const headWidth = headLength * OCEAN_CURRENT_ARROW_HEAD_WIDTH_RATIO;
    const direction = new THREE.Vector3(dx, dy, dz).normalize();
    let side = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(0, 1, 0));
    if (side.lengthSq() < 1e-6) {
      side = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(1, 0, 0));
    }
    side.normalize().multiplyScalar(headWidth);
    const tip = new THREE.Vector3(scenePoint1.x, y1, scenePoint1.z);
    const headBase = tip.clone().addScaledVector(direction, -headLength);
    const headA = headBase.clone().add(side);
    const headB = headBase.clone().sub(side);
    const arrowColor = lerpColor(color1, [1, 1, 1], 0.35);
    buffer.positions.push(
      tip.x,
      tip.y,
      tip.z,
      headA.x,
      headA.y,
      headA.z,
      tip.x,
      tip.y,
      tip.z,
      headB.x,
      headB.y,
      headB.z
    );
    buffer.colors.push(
      arrowColor[0],
      arrowColor[1],
      arrowColor[2],
      arrowColor[0],
      arrowColor[1],
      arrowColor[2],
      arrowColor[0],
      arrowColor[1],
      arrowColor[2],
      arrowColor[0],
      arrowColor[1],
      arrowColor[2]
    );
    if (buildFlowLights) {
      buffer.flowDistances.push(0, 0, 0, 0);
      buffer.flowGlows.push(0, 0, 0, 0);
    }
  }

  function createOceanCurrentLines(buffer) {
    if (!buffer.positions.length) return null;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buffer.positions), 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(buffer.colors), 3));
    if (buildFlowLights) {
      addFlowLightAttributes(geometry, {
        distances: buffer.flowDistances,
        glows: buffer.flowGlows,
      });
    }
    geometry.computeBoundingSphere();
    const material = buildFlowLights
      ? createFlowLightMaterial({
          staticOpacity: 0.98,
          activeBaseOpacity: FLOW_LIGHT_OCEAN_ACTIVE_BASE_OPACITY,
          pulseOpacity: FLOW_LIGHT_PULSE_OPACITY,
          flowRate: FLOW_LIGHT_OCEAN_RATE,
        })
      : createStaticFlowLineMaterial({ baseOpacity: 0.98 });
    const lines = new THREE.LineSegments(geometry, material);
    lines.renderOrder = OCEAN_CURRENT_RENDER_ORDER;
    if (buildFlowLights) {
      addFlowLightParticleOverlay(lines, {
        sampleStride: FLOW_LIGHT_OCEAN_PARTICLE_SAMPLE_STRIDE,
        pointSize: FLOW_LIGHT_OCEAN_PARTICLE_SIZE,
        flowRate: FLOW_LIGHT_OCEAN_RATE,
      });
    }
    return lines;
  }

  for (let i = 0; i < count; i += 1) {
    const buffer = getCurrentLayerBuffer();
    const projected0 = projectPs71PointToGrid(context, x0Ps[i], y0Ps[i]);
    const projected1 = projectPs71PointToGrid(context, x1Ps[i], y1Ps[i]);
    if (!projected0 || !projected1) {
      finalizeStreamlineIfNeeded(i, buffer);
      continue;
    }

    const depth0 = Math.max(0, Number(depth0M[i]));
    const depth1 = Math.max(0, Number(depth1M[i]));
    const col0 = projected0.col;
    const row0 = projected0.row;
    const col1 = projected1.col;
    const row1 = projected1.row;
    if (col0 < 0 || row0 < 0 || col0 >= context.nx || row0 >= context.ny) {
      finalizeStreamlineIfNeeded(i, buffer);
      continue;
    }
    if (col1 < 0 || row1 < 0 || col1 >= context.nx || row1 >= context.ny) {
      finalizeStreamlineIfNeeded(i, buffer);
      continue;
    }

    const col0Nearest = Math.min(context.nx - 1, Math.max(0, Math.round(col0)));
    const row0Nearest = Math.min(context.ny - 1, Math.max(0, Math.round(row0)));
    const col1Nearest = Math.min(context.nx - 1, Math.max(0, Math.round(col1)));
    const row1Nearest = Math.min(context.ny - 1, Math.max(0, Math.round(row1)));
    const idx0 = row0Nearest * context.nx + col0Nearest;
    const idx1 = row1Nearest * context.nx + col1Nearest;
    const mask0 = Number(context.mask[idx0]);
    const mask1 = Number(context.mask[idx1]);
    const validOceanMask0 = mask0 === 0 || mask0 === 3;
    const validOceanMask1 = mask1 === 0 || mask1 === 3;
    if (!validOceanMask0 || !validOceanMask1) {
      finalizeStreamlineIfNeeded(i, buffer);
      continue;
    }

    const bedHeight0 = sampleGridHeightNearest(context, col0, row0, "bed");
    const bedHeight1 = sampleGridHeightNearest(context, col1, row1, "bed");
    if (!Number.isFinite(bedHeight0) || !Number.isFinite(bedHeight1)) {
      finalizeStreamlineIfNeeded(i, buffer);
      continue;
    }
    if (-bedHeight0 < depth0 + OCEAN_CURRENT_BED_CLEARANCE_M) {
      finalizeStreamlineIfNeeded(i, buffer);
      continue;
    }
    if (-bedHeight1 < depth1 + OCEAN_CURRENT_BED_CLEARANCE_M) {
      finalizeStreamlineIfNeeded(i, buffer);
      continue;
    }

    const scenePoint0 = gridToSceneXZ(context, col0, row0);
    const scenePoint1 = gridToSceneXZ(context, col1, row1);
    const y0 = -depth0 / context.baseConfig.verticalMetersPerUnit;
    const y1 = -depth1 / context.baseConfig.verticalMetersPerUnit;
    const color0 = oceanCurrentColor(theta0C[i], sal0Psu[i], oceanMeta);
    const color1 = oceanCurrentColor(theta1C[i], sal1Psu[i], oceanMeta);

    buffer.positions.push(scenePoint0.x, y0, scenePoint0.z, scenePoint1.x, y1, scenePoint1.z);
    buffer.colors.push(color0[0], color0[1], color0[2], color1[0], color1[1], color1[2]);
    const segmentLength = Math.hypot(scenePoint1.x - scenePoint0.x, y1 - y0, scenePoint1.z - scenePoint0.z);
    if (buildFlowLights) {
      buffer.flowDistances.push(buffer.flowDistance, buffer.flowDistance + segmentLength);
      buffer.flowGlows.push(1, 1);
    }
    buffer.flowDistance += segmentLength;

    if (terminalFlag[i]) {
      const dx = scenePoint1.x - scenePoint0.x;
      const dy = y1 - y0;
      const dz = scenePoint1.z - scenePoint0.z;
      appendOceanCurrentArrow(buffer, scenePoint1, y1, dx, dy, dz, color1);
      streamlineIndex += 1;
      buffer.flowDistance = 0;
    }
    buffer.segmentCount += 1;
  }

  if (!useLayerSplit) {
    const lines = createOceanCurrentLines(layerBuffers.all);
    if (!lines) return null;
    lines.scale.y = Number(controlsUI.exaggeration.value);
    lines.visible = controlsUI.showOceanCurrents.checked;
    lines.userData.flowlineCount = Number(oceanMeta.streamline_count || oceanMeta.flowline_count || 0);
    lines.userData.segmentCount = layerBuffers.all.segmentCount;
    return lines;
  }

  const group = new THREE.Group();
  const layerMeshes = {};
  let totalVisibleSegments = 0;
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    const lines = createOceanCurrentLines(layerBuffers[layer]);
    if (!lines) continue;
    group.add(lines);
    layerMeshes[layer] = lines;
    totalVisibleSegments += layerBuffers[layer].segmentCount;
  }
  if (!Object.keys(layerMeshes).length) return null;
  group.scale.y = Number(controlsUI.exaggeration.value);
  group.renderOrder = OCEAN_CURRENT_RENDER_ORDER;
  group.userData.layerMeshes = layerMeshes;
  group.userData.flowlineCount = Number(oceanMeta.streamline_count || oceanMeta.flowline_count || 0);
  group.userData.segmentCount = totalVisibleSegments;
  updateOceanCurrentLayerVisibility(group, oceanMeta);
  return group;
}

function buildOceanCurrentMeshFromWorkerResult(oceanMeta, workerResult) {
  if (!workerResult || typeof workerResult !== "object") return null;
  const buildFlowLights = Boolean(workerResult.flowLightsBuilt);

  function createOceanCurrentLinesFromArrays(positionsArray, colorsArray, flowDistancesArray, flowGlowsArray) {
    if (
      !(positionsArray instanceof Float32Array) ||
      !(colorsArray instanceof Float32Array) ||
      !positionsArray.length
    ) {
      return null;
    }
    if (buildFlowLights && (!(flowDistancesArray instanceof Float32Array) || !(flowGlowsArray instanceof Uint8Array))) {
      return null;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positionsArray, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colorsArray, 3));
    if (buildFlowLights) {
      addFlowLightAttributes(geometry, {
        distances: flowDistancesArray,
        glows: flowGlowsArray,
      });
    }
    geometry.computeBoundingSphere();
    const material = buildFlowLights
      ? createFlowLightMaterial({
          staticOpacity: 0.98,
          activeBaseOpacity: FLOW_LIGHT_OCEAN_ACTIVE_BASE_OPACITY,
          pulseOpacity: FLOW_LIGHT_PULSE_OPACITY,
          flowRate: FLOW_LIGHT_OCEAN_RATE,
        })
      : createStaticFlowLineMaterial({ baseOpacity: 0.98 });
    const lines = new THREE.LineSegments(geometry, material);
    lines.renderOrder = OCEAN_CURRENT_RENDER_ORDER;
    if (buildFlowLights) {
      addFlowLightParticleOverlay(lines, {
        sampleStride: FLOW_LIGHT_OCEAN_PARTICLE_SAMPLE_STRIDE,
        pointSize: FLOW_LIGHT_OCEAN_PARTICLE_SIZE,
        flowRate: FLOW_LIGHT_OCEAN_RATE,
      });
    }
    return lines;
  }

  const useLayerSplit = Boolean(workerResult.useLayerSplit);
  const flowlineCount = Number(workerResult.flowlineCount || oceanMeta?.streamline_count || oceanMeta?.flowline_count || 0);
  const segmentCount = Number(workerResult.segmentCount || 0);

  if (!useLayerSplit) {
    const layer = workerResult.layers?.all;
    const lines = createOceanCurrentLinesFromArrays(
      layer?.positions,
      layer?.colors,
      layer?.flowDistances,
      layer?.flowGlows
    );
    if (!lines) return null;
    lines.scale.y = Number(controlsUI.exaggeration.value);
    lines.visible = controlsUI.showOceanCurrents.checked;
    lines.userData.flowlineCount = flowlineCount;
    lines.userData.segmentCount = segmentCount;
    return lines;
  }

  const group = new THREE.Group();
  const layerMeshes = {};
  for (const layerName of OCEAN_CURRENT_LAYER_ORDER) {
    const layer = workerResult.layers?.[layerName];
    const lines = createOceanCurrentLinesFromArrays(
      layer?.positions,
      layer?.colors,
      layer?.flowDistances,
      layer?.flowGlows
    );
    if (!lines) continue;
    group.add(lines);
    layerMeshes[layerName] = lines;
  }
  if (!Object.keys(layerMeshes).length) return null;
  group.scale.y = Number(controlsUI.exaggeration.value);
  group.renderOrder = OCEAN_CURRENT_RENDER_ORDER;
  group.userData.layerMeshes = layerMeshes;
  group.userData.flowlineCount = flowlineCount;
  group.userData.segmentCount = segmentCount;
  updateOceanCurrentLayerVisibility(group, oceanMeta);
  return group;
}

function resizeRendererToViewer() {
  if (!renderer || !camera) return;
  const w = Math.max(1, viewerEl.clientWidth);
  const h = Math.max(1, viewerEl.clientHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

function syncRendererSizeToViewerIfNeeded() {
  if (!renderer || !camera) return;
  const targetWidth = Math.max(1, viewerEl.clientWidth);
  const targetHeight = Math.max(1, viewerEl.clientHeight);
  const targetAspect = targetWidth / targetHeight;
  if (Math.abs(camera.aspect - targetAspect) > 0.001) {
    resizeRendererToViewer();
  }
}

function requestViewerResize() {
  if (viewerResizeAnimationFrame) return;
  viewerResizeAnimationFrame = window.requestAnimationFrame(() => {
    viewerResizeAnimationFrame = 0;
    resizeRendererToViewer();
  });
}

function scheduleViewerResizeStabilization() {
  requestViewerResize();
  window.requestAnimationFrame(() => {
    requestViewerResize();
  });
  if (viewerResizeStabilizeTimer) {
    window.clearTimeout(viewerResizeStabilizeTimer);
  }
  viewerResizeStabilizeTimer = window.setTimeout(() => {
    viewerResizeStabilizeTimer = null;
    requestViewerResize();
  }, 140);
}

function bindViewerResizeHandling() {
  window.addEventListener("resize", requestViewerResize);
  if ("ResizeObserver" in window && !viewerResizeObserver) {
    viewerResizeObserver = new ResizeObserver(() => {
      requestViewerResize();
    });
    viewerResizeObserver.observe(viewerEl);
    viewerResizeObserver.observe(viewerShellEl);
  }

  scheduleViewerResizeStabilization();
  if (document.readyState !== "complete") {
    window.addEventListener("load", scheduleViewerResizeStabilization, { once: true });
  }
}

function cleanupViewerResizeHandling() {
  window.removeEventListener("resize", requestViewerResize);
  if (viewerResizeObserver) {
    viewerResizeObserver.disconnect();
    viewerResizeObserver = null;
  }
  if (viewerResizeAnimationFrame) {
    window.cancelAnimationFrame(viewerResizeAnimationFrame);
    viewerResizeAnimationFrame = 0;
  }
  if (viewerResizeStabilizeTimer) {
    window.clearTimeout(viewerResizeStabilizeTimer);
    viewerResizeStabilizeTimer = null;
  }
}

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a2b43);
  scene.fog = new THREE.Fog(0x0a2b43, 80, 240);

  const initialPose = getDefaultCameraPose();
  const initialViewerWidth = Math.max(1, viewerEl.clientWidth);
  const initialViewerHeight = Math.max(1, viewerEl.clientHeight);
  camera = new THREE.PerspectiveCamera(initialPose.fov, initialViewerWidth / initialViewerHeight, 0.1, 200);
  camera.position.set(...initialPose.position);

  const lowEndDevice = detectLowEndDevice();
  const coarsePointer = isCoarsePointerInput();
  const mobileHighQuality = detectMobileHighQualityDevice();
  let maxPixelRatio;
  if (coarsePointer) {
    if (mobileHighQuality) {
      maxPixelRatio = 2.05;
    } else {
      maxPixelRatio = lowEndDevice ? 1.4 : 1.75;
    }
    if (isShowcaseMode) {
      maxPixelRatio = Math.min(maxPixelRatio, 1.55);
    }
    if (isPreviewMode) {
      maxPixelRatio = Math.min(maxPixelRatio, 1.15);
    }
  } else {
    maxPixelRatio = lowEndDevice ? 1.5 : isShowcaseMode ? 1.7 : 2.0;
    if (isPreviewMode) {
      maxPixelRatio = Math.min(maxPixelRatio, 1.25);
    }
  }
  renderer = new THREE.WebGLRenderer({
    antialias: !lowEndDevice && !isPreviewMode,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewerEl.appendChild(renderer.domElement);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  resizeRendererToViewer();
  bindViewerResizeHandling();

  orbit = new OrbitControlsCtor(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  orbit.minDistance = isShowcaseMode ? 12 : 0.2;
  orbit.maxDistance = 360;
  orbit.target.set(...initialPose.target);
  orbit.enabled = true;
  renderer.domElement.style.touchAction = "none";

  const ambient = new THREE.AmbientLight(0xddefff, 0.75);
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(12, 18, 8);
  const fill = new THREE.DirectionalLight(0x95c9e7, 0.35);
  fill.position.set(-14, 6, -9);
  scene.add(ambient, key, fill);
}

function resetCameraToDefaultPose() {
  if (!camera || !orbit) return;
  stopShowcaseAutoOrbit({ syncBasePose: false });
  resetRecordingMotionState({ syncBasePose: false });
  const pose = getDefaultCameraPose();
  applyCameraPose(pose);
  captureRecordingBasePoseFromCurrentView();
  captureShowcaseBasePoseFromCurrentView();
  if ((isShowcaseMode && (!interactionGateEnabled || !interactionGateActive)) || isPreviewMode) {
    startShowcaseAutoOrbit();
  }
  updateRecordingControlsUi();
}

function waitForAnimationFrames(frameCount = 2) {
  const total = Math.max(1, Math.round(Number(frameCount) || 1));
  return new Promise((resolve) => {
    let remaining = total;
    const tick = () => {
      remaining -= 1;
      if (remaining <= 0) {
        resolve();
        return;
      }
      window.requestAnimationFrame(tick);
    };
    window.requestAnimationFrame(tick);
  });
}

function waitUntilCondition(predicate, timeoutMs = 30000, errorMessage = "Timed out waiting for condition.") {
  return new Promise((resolve, reject) => {
    const start = performance.now();
    const tick = () => {
      try {
        if (predicate()) {
          resolve();
          return;
        }
      } catch (error) {
        reject(error);
        return;
      }
      if (performance.now() - start >= timeoutMs) {
        reject(new Error(errorMessage));
        return;
      }
      window.setTimeout(tick, 50);
    };
    tick();
  });
}

function coerceVec3(value, fallback) {
  if (Array.isArray(value) && value.length >= 3) {
    const nums = value.slice(0, 3).map((entry) => Number(entry));
    if (nums.every((entry) => Number.isFinite(entry))) return nums;
  }
  if (value && typeof value === "object") {
    const x = Number(value.x);
    const y = Number(value.y);
    const z = Number(value.z);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      return [x, y, z];
    }
  }
  return Array.isArray(fallback) ? [...fallback] : [0, 0, 0];
}

function getCurrentCameraPose() {
  if (!camera || !orbit) return null;
  return {
    position: [Number(camera.position.x), Number(camera.position.y), Number(camera.position.z)],
    target: [Number(orbit.target.x), Number(orbit.target.y), Number(orbit.target.z)],
    fov: Number(camera.fov),
  };
}

function applyCameraPose(pose = null) {
  if (!camera || !orbit) return null;
  const fallbackPose = getCurrentCameraPose() || getDefaultCameraPose();
  const nextPose = pose || fallbackPose;
  const position = coerceVec3(nextPose.position, fallbackPose.position);
  const target = coerceVec3(nextPose.target, fallbackPose.target);
  const fovValue = Number(nextPose.fov);
  const fov = Number.isFinite(fovValue) ? fovValue : Number(fallbackPose.fov);
  const previousDamping = orbit.enableDamping;
  orbit.enableDamping = false;
  orbit.update();
  camera.position.set(...position);
  camera.fov = fov;
  camera.updateProjectionMatrix();
  orbit.target.set(...target);
  orbit.update();
  orbit.enableDamping = previousDamping;
  return getCurrentCameraPose();
}

function hasPendingLayerLoads() {
  return Boolean(
    riseLoadPromise ||
      velocityLoadPromise ||
      hydrologyLoadPromise ||
      oceanCurrentLoadPromise ||
      refinedBasinLoadPromise ||
      polarFeaturesController?.getState().loading
  );
}

function isLoadingOverlayVisible() {
  return Boolean(loadingOverlayEl && !loadingOverlayEl.classList.contains("hidden"));
}

function isExplorerReady() {
  return Boolean(camera && orbit && currentCoreContext) && !isLoadingOverlayVisible() && !hasPendingLayerLoads();
}

function hasFlowLightGeometry(object) {
  let found = false;
  object?.traverse((child) => {
    if (found || !child.geometry || !child.material?.uniforms?.uFlowTime) return;
    found = Boolean(child.geometry.getAttribute("flowDistance") && child.geometry.getAttribute("flowGlow"));
  });
  return found;
}

function hasFlowLightParticleOverlay(object) {
  let found = false;
  object?.traverse((child) => {
    found ||= Boolean(child.userData?.isFlowLightParticleOverlay);
  });
  return found;
}

function setFlowLightParticleOverlaysVisible(object, visible) {
  object?.traverse((child) => {
    if (child.userData?.isFlowLightParticleOverlay) child.visible = visible;
  });
}

async function rebuildStaticFlowLightLayers() {
  if (!isFlowLightAnimationEnabled()) return;

  if (flowlineMesh && velocityField && !hasFlowLightGeometry(flowlineMesh)) {
    selectedFlowlineHighlight = disposeObject3D(selectedFlowlineHighlight);
    selectedFlowlineState = null;
    const rebuiltFlowlines = buildFlowlineMesh(velocityField);
    flowlineMesh = disposeMesh(flowlineMesh);
    if (rebuiltFlowlines) {
      flowlineMesh = rebuiltFlowlines;
      scene.add(flowlineMesh);
    }
    updateFlowlineVisibility();
  }

  if (!oceanCurrentMesh || hasFlowLightGeometry(oceanCurrentMesh) || !currentCoreContext) return;
  const context = currentCoreContext;
  oceanCurrentMesh = disposeObject3D(oceanCurrentMesh);
  context.oceanCurrentLoaded = false;
  context.oceanCurrentUnavailable = false;
  currentOceanCurrentMeta = null;
  await ensureOceanCurrentsLoaded({ trigger: "toggle" });
}

function collectExplorerState() {
  const polarFeatureState = polarFeaturesController?.getState() || {
    featureLayers: {
      researchStations: { enabled: false, loaded: false, totalCount: 0, visibleCount: 0 },
      geographicNames: { enabled: false, loaded: false, totalCount: 0, visibleCount: 0 },
    },
    search: { query: "", resultCount: 0, activeIndex: -1 },
    selectedFeature: null,
  };
  const toggles = {
    showBed: Boolean(controlsUI.showBed?.checked),
    showIce: Boolean(controlsUI.showIce?.checked),
    showIceBottom: Boolean(controlsUI.showIceBottom?.checked),
    showVelocity: Boolean(controlsUI.showVelocity?.checked),
    showFlowline: Boolean(controlsUI.showFlowline?.checked),
    showBasalFriction: Boolean(controlsUI.showBasalFriction?.checked),
    showBasalMelt: Boolean(controlsUI.showBasalMelt?.checked),
    showThermalDriving: Boolean(controlsUI.showThermalDriving?.checked),
    showOceanCurrents: Boolean(controlsUI.showOceanCurrents?.checked),
    showOceanLayerSurface: Boolean(controlsUI.showOceanLayerSurface?.checked),
    showOceanLayerUpper: Boolean(controlsUI.showOceanLayerUpper?.checked),
    showOceanLayerMid: Boolean(controlsUI.showOceanLayerMid?.checked),
    showOceanLayerLower: Boolean(controlsUI.showOceanLayerLower?.checked),
    showRefinedBasins: Boolean(controlsUI.showRefinedBasins?.checked),
    showEffectivePressure: Boolean(controlsUI.showEffectivePressure?.checked),
    showSubglacialChannels: Boolean(controlsUI.showSubglacialChannels?.checked),
    showSea: Boolean(controlsUI.showSea?.checked),
    showIsostaticRebound: Boolean(controlsUI.showIsostaticRebound?.checked),
    highlightEmergentLand: Boolean(controlsUI.highlightEmergentLand?.checked),
    wireframe: Boolean(controlsUI.wireframe?.checked),
    showResearchStations: Boolean(controlsUI.showResearchStations?.checked),
    showGeographicNames: Boolean(controlsUI.showGeographicNames?.checked),
  };
  return {
    ready: isExplorerReady(),
    mode: isShowcaseMode ? "showcase" : isPreviewMode ? "preview" : "interactive",
    region: currentRegionKey,
    dataset: currentDatasetKey,
    status: statusEl?.textContent || "",
    recordingMode: {
      enabled: Boolean(recordingModeEnabled),
      panelVisible: Boolean(recordingModeEnabled && recordingMotionState.panelVisible),
    },
    flowAnimation: {
      enabled: isFlowLightAnimationEnabled(),
      reducedMotion: Boolean(flowMotionMediaQuery.matches),
      timeSeconds: Number(flowLightElapsedSeconds.toFixed(3)),
      iceReady: hasFlowLightGeometry(flowlineMesh),
      oceanReady: hasFlowLightGeometry(oceanCurrentMesh),
      iceParticles: hasFlowLightParticleOverlay(flowlineMesh),
      oceanParticles: hasFlowLightParticleOverlay(oceanCurrentMesh),
    },
    captureMotion: getRecordingStateSnapshot(),
    isostaticRebound: {
      enabled: Boolean(controlsUI.showIsostaticRebound?.checked),
      available: Boolean(controlsUI.showIsostaticRebound && !controlsUI.showIsostaticRebound.disabled),
      solved: Boolean(currentCoreContext?.reboundUplift),
      // `model` and `seaLevelMeters` report the requested settings, which change the
      // moment a control moves. The `solved*` pair reports what the cached uplift
      // field was actually computed with, so a caller can tell a pending re-solve
      // from a finished one instead of racing it.
      model: getReboundModelKey(),
      seaLevelMeters: getReboundSeaLevelMeters(),
      solvedModel: currentCoreContext?.reboundStats?.model || null,
      solvedSeaLevelMeters: Number.isFinite(currentCoreContext?.reboundStats?.seaLevelMeters)
        ? currentCoreContext.reboundStats.seaLevelMeters
        : null,
      pendingResolve: Boolean(
        currentCoreContext?.reboundUplift && currentCoreContext.reboundSolveKey !== getReboundSolveKey()
      ),
      progressPercent: Math.round(getReboundFraction() * 100),
      highlightEmergentLand: Boolean(controlsUI.highlightEmergentLand?.checked),
      maxUpliftMeters: Number.isFinite(currentCoreContext?.reboundStats?.maxUpliftMeters)
        ? Number(currentCoreContext.reboundStats.maxUpliftMeters.toFixed(1))
        : null,
      emergentAreaKm2: Number.isFinite(currentCoreContext?.reboundStats?.emergentAreaKm2)
        ? Math.round(currentCoreContext.reboundStats.emergentAreaKm2)
        : null,
      seaLevelEquivalentMeters: Number.isFinite(currentCoreContext?.reboundStats?.sleMeters)
        ? Number(currentCoreContext.reboundStats.sleMeters.toFixed(2))
        : null,
    },
    featureLayers: polarFeatureState.featureLayers,
    search: polarFeatureState.search,
    selectedFeature: polarFeatureState.selectedFeature,
    focusedExistingLabelId: polarFeatureState.focusedExistingLabelId,
    selectionOverlayCount: polarFeatureState.selectionOverlayCount,
    toggles,
    controls: { ...toggles },
    legends: {
      bed: Boolean(controlsUI.bedLegendSection && !controlsUI.bedLegendSection.hidden),
      velocity: Boolean(controlsUI.velocityLegendSection && !controlsUI.velocityLegendSection.hidden),
      basalFriction: Boolean(controlsUI.basalFrictionLegendSection && !controlsUI.basalFrictionLegendSection.hidden),
      basalMelt: Boolean(controlsUI.basalMeltLegendSection && !controlsUI.basalMeltLegendSection.hidden),
      thermalDriving: Boolean(controlsUI.thermalDrivingLegendSection && !controlsUI.thermalDrivingLegendSection.hidden),
      oceanCurrents: Boolean(controlsUI.oceanLegendSection && !controlsUI.oceanLegendSection.hidden),
      effectivePressure: Boolean(controlsUI.effectivePressureLegendSection && !controlsUI.effectivePressureLegendSection.hidden),
      subglacialChannels: Boolean(controlsUI.channelLegendSection && !controlsUI.channelLegendSection.hidden),
    },
    camera:
      camera && orbit
        ? {
            position: {
              x: Number(camera.position.x.toFixed(2)),
              y: Number(camera.position.y.toFixed(2)),
              z: Number(camera.position.z.toFixed(2)),
            },
            target: {
              x: Number(orbit.target.x.toFixed(2)),
              y: Number(orbit.target.y.toFixed(2)),
              z: Number(orbit.target.z.toFixed(2)),
            },
            fov: Number(camera.fov.toFixed(2)),
          }
        : null,
    grid: currentCoreContext
      ? {
          nx: currentCoreContext.nx,
          ny: currentCoreContext.ny,
          dx_m: currentCoreContext.meta.grid.dx_m,
          dy_m: currentCoreContext.meta.grid.dy_m,
          origin: "Scene origin is centered on the grid midpoint; +x follows increasing projected x and +z follows increasing grid row.",
        }
      : null,
    meshes: {
      bed: Boolean(bedMesh && bedMesh.visible),
      isostaticRebound: Boolean(bedMesh && bedMesh.visible && isReboundActive()),
      ice: Boolean(iceMesh && iceMesh.visible),
      iceBottom: Boolean(iceBottomMesh && iceBottomMesh.visible),
      velocity: Boolean(velocitySurfaceMesh && velocitySurfaceMesh.visible),
      flowline: Boolean(flowlineMesh && flowlineMesh.visible),
      basalFriction: Boolean(basalFrictionMesh && basalFrictionMesh.visible),
      basalMelt: Boolean(basalMeltMesh && basalMeltMesh.visible),
      thermalDriving: Boolean(thermalDrivingMesh && thermalDrivingMesh.visible),
      oceanCurrents: Boolean(oceanCurrentMesh && oceanCurrentMesh.visible),
      basins: Boolean(
        (refinedBasinBedLines && refinedBasinBedLines.visible) ||
          (refinedBasinSurfaceLines && refinedBasinSurfaceLines.visible) ||
          (refinedBasinBedLabels && refinedBasinBedLabels.visible) ||
          (refinedBasinSurfaceLabels && refinedBasinSurfaceLabels.visible)
      ),
      hydrology: Boolean(
        (effectivePressureMesh && effectivePressureMesh.visible) ||
          (subglacialChannelMesh && subglacialChannelMesh.visible)
      ),
    },
    selectedFlowline: selectedFlowlineState
      ? {
          index: Number(selectedFlowlineState.index),
          lengthKm: Number(selectedFlowlineState.lengthKm),
        }
      : null,
  };
}

function renderLoop(frameTimeMs = performance.now()) {
  const deltaMs = lastRenderFrameTimeMs ? frameTimeMs - lastRenderFrameTimeMs : 1000 / 60;
  lastRenderFrameTimeMs = frameTimeMs;
  syncRendererSizeToViewerIfNeeded();
  stepRuntime(deltaMs);
  renderer.render(scene, camera);
  animationHandle = requestAnimationFrame(renderLoop);
}

function disposeMesh(mesh) {
  return disposeObject3D(mesh);
}

function disposeTexture(texture) {
  if (!texture) return null;
  if (typeof texture.dispose === "function") {
    texture.dispose();
  }
  return null;
}

function disposeObject3D(object) {
  if (!object) return null;
  scene.remove(object);
  object.traverse((child) => {
    if (child.geometry && typeof child.geometry.dispose === "function") {
      child.geometry.dispose();
    }
    if (!child.material) return;
    if (Array.isArray(child.material)) {
      for (const material of child.material) {
        if (material && typeof material.dispose === "function") material.dispose();
      }
      return;
    }
    if (typeof child.material.dispose === "function") {
      child.material.dispose();
    }
  });
  return null;
}

function clearRefinedBasinOverlays() {
  refinedBasinBedLines = disposeObject3D(refinedBasinBedLines);
  refinedBasinSurfaceLines = disposeObject3D(refinedBasinSurfaceLines);
  refinedBasinBedLabels = disposeObject3D(refinedBasinBedLabels);
  refinedBasinSurfaceLabels = disposeObject3D(refinedBasinSurfaceLabels);
  for (const texture of refinedBasinLabelTextureCache.values()) texture.dispose();
  refinedBasinLabelTextureCache.clear();
  refinedBasinLabelTexturePixels = 0;
}

function clearModelMeshes() {
  polarFeaturesController?.clearScene();
  bedMesh = disposeMesh(bedMesh);
  iceMesh = disposeMesh(iceMesh);
  iceBottomMesh = disposeMesh(iceBottomMesh);
  iceSideMesh = disposeMesh(iceSideMesh);
  velocitySurfaceMesh = disposeMesh(velocitySurfaceMesh);
  velocityDataTexture = disposeTexture(velocityDataTexture);
  basalFrictionMesh = disposeMesh(basalFrictionMesh);
  basalMeltMesh = disposeMesh(basalMeltMesh);
  basalMeltDataTexture = disposeTexture(basalMeltDataTexture);
  thermalDrivingMesh = disposeMesh(thermalDrivingMesh);
  thermalDrivingDataTexture = disposeTexture(thermalDrivingDataTexture);
  effectivePressureMesh = disposeMesh(effectivePressureMesh);
  subglacialChannelMesh = disposeMesh(subglacialChannelMesh);
  selectedFlowlineHighlight = disposeObject3D(selectedFlowlineHighlight);
  selectedFlowlineState = null;
  flowlineMesh = disposeMesh(flowlineMesh);
  oceanCurrentMesh = disposeObject3D(oceanCurrentMesh);
  seaLevelMesh = disposeMesh(seaLevelMesh);
  clearRefinedBasinOverlays();
  velocityField = null;
  refreshFlowlineGuidanceUi();
}

function buildSelectedFlowlineHighlightObject(flowlineSummary) {
  if (!THREE || !flowlineSummary || !Array.isArray(flowlineSummary.highlightPositions) || flowlineSummary.highlightPositions.length < 6) {
    return null;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(flowlineSummary.highlightPositions), 3));
  const material = new THREE.LineBasicMaterial({
    color: 0xf7feff,
    transparent: true,
    opacity: 0.98,
    depthWrite: false,
    depthTest: true,
  });
  const highlight = new THREE.Line(geometry, material);
  highlight.renderOrder = 18;
  highlight.scale.y = Number(controlsUI.exaggeration.value);
  return highlight;
}

function updateSelectedFlowlineVisualState() {
  if (!selectedFlowlineHighlight) return;
  selectedFlowlineHighlight.visible = Boolean(selectedFlowlineState && flowlineMesh && flowlineMesh.visible);
  selectedFlowlineHighlight.scale.y = Number(controlsUI.exaggeration.value);
}

function clearSelectedFlowline({ updateMeta = true } = {}) {
  selectedFlowlineState = null;
  selectedFlowlineHighlight = disposeObject3D(selectedFlowlineHighlight);
  if (updateMeta) {
    updateMetaFromCurrentState();
  }
  refreshFlowlineGuidanceUi();
}

function setSelectedFlowline(flowlineSummary) {
  if (!flowlineSummary) {
    clearSelectedFlowline();
    return;
  }
  selectedFlowlineState = flowlineSummary;
  selectedFlowlineHighlight = disposeObject3D(selectedFlowlineHighlight);
  const highlight = buildSelectedFlowlineHighlightObject(flowlineSummary);
  if (highlight) {
    selectedFlowlineHighlight = highlight;
    scene.add(selectedFlowlineHighlight);
  }
  metaSectionOpenState.set("selected-flowline", true);
  updateSelectedFlowlineVisualState();
  updateMetaFromCurrentState();
  refreshFlowlineGuidanceUi();
}

function pickVisibleFlowlineAtClientPoint(clientX, clientY) {
  if (!(renderer && camera && flowlineMesh && flowlineMesh.visible && flowlineRaycaster && flowlinePointerNdc)) {
    return false;
  }

  const rect = renderer.domElement.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return false;
  if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) {
    return false;
  }

  flowlinePointerNdc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  flowlineRaycaster.setFromCamera(flowlinePointerNdc, camera);
  flowlineRaycaster.params.Line.threshold = clamp(
    camera.position.distanceTo(orbit?.target || new THREE.Vector3()) * 0.0045,
    0.2,
    1.25
  );

  const intersections = flowlineRaycaster.intersectObject(flowlineMesh, false);
  if (!intersections.length) return false;

  const hit = intersections[0];
  const segmentOrdinal = Math.max(0, Math.floor(Number(hit.index || 0) / 2));
  const flowlineIndex = Number(flowlineMesh.userData?.segmentFlowlineIndices?.[segmentOrdinal]);
  if (!(flowlineIndex >= 0)) return false;

  const flowlineSummary = flowlineMesh.userData?.flowlines?.[flowlineIndex];
  if (!flowlineSummary) return false;

  setSelectedFlowline(flowlineSummary);
  return true;
}

function projectPs71PointToGrid(context, xMeters, yMeters) {
  if (!context?.meta?.grid) return null;
  const grid = context.meta.grid;
  if (!Number.isFinite(xMeters) || !Number.isFinite(yMeters)) return null;
  if (!Number.isFinite(grid.x0_m) || !Number.isFinite(grid.y0_m)) return null;
  if (!Number.isFinite(grid.dx_m) || !Number.isFinite(grid.dy_m)) return null;
  if (Math.abs(grid.dx_m) < 1e-9 || Math.abs(grid.dy_m) < 1e-9) return null;

  const col = (xMeters - grid.x0_m) / grid.dx_m;
  const row = (yMeters - grid.y0_m) / grid.dy_m;
  if (!Number.isFinite(col) || !Number.isFinite(row)) return null;
  if (col < 0 || row < 0 || col > context.nx - 1 || row > context.ny - 1) return null;
  return { col, row };
}

function gridToSceneXZ(context, col, row) {
  const halfX = (context.nx - 1) / 2;
  const halfY = (context.ny - 1) / 2;
  const x = ((col - halfX) * context.meta.grid.dx_m) / context.baseConfig.horizontalMetersPerUnit;
  const z = ((row - halfY) * Math.abs(context.meta.grid.dy_m)) / context.baseConfig.horizontalMetersPerUnit;
  return { x, z };
}

function sampleGridHeightNearest(context, col, row, layerKey) {
  const c = Math.round(col);
  const r = Math.round(row);
  if (c < 0 || r < 0 || c >= context.nx || r >= context.ny) return Number.NaN;
  const idx = r * context.nx + c;

  if (layerKey === "bed") {
    if (!context.bedValid[idx]) return Number.NaN;
    const value = context.bedHeights[idx];
    return Number.isFinite(value) ? value : Number.NaN;
  }

  const value = context.surfaceHeights[idx];
  return Number.isFinite(value) ? value : Number.NaN;
}

function getPolarFeatureScenePoint(feature) {
  const context = currentCoreContext;
  if (!context || feature?.region !== currentRegionKey) return null;
  const gridPoint = projectPs71PointToGrid(context, Number(feature.x_m), Number(feature.y_m));
  if (!gridPoint) return null;
  const scenePoint = gridToSceneXZ(context, gridPoint.col, gridPoint.row);
  const seaLevelKinds = new Set(["ocean", "sea", "fjord", "strait"]);
  const isUnderseaBasin = feature.kind === "basin" && String(feature.id || "").includes("gebco");
  let heightMeters = seaLevelKinds.has(feature.kind) || isUnderseaBasin
    ? 0
    : sampleGridHeightNearest(context, gridPoint.col, gridPoint.row, "surface");
  if (!Number.isFinite(heightMeters)) {
    heightMeters = sampleGridHeightNearest(context, gridPoint.col, gridPoint.row, "bed");
  }
  if (!Number.isFinite(heightMeters)) heightMeters = 0;
  return {
    x: scenePoint.x,
    baseY: (heightMeters + 220) / context.baseConfig.verticalMetersPerUnit,
    z: scenePoint.z,
  };
}

function focusPolarFeature(feature, point) {
  if (!camera || !orbit || !point) return;
  const exaggeration = Number(controlsUI.exaggeration.value) || 1;
  const target = new THREE.Vector3(point.x, point.baseY * exaggeration, point.z);
  const direction = camera.position.clone().sub(orbit.target);
  if (direction.lengthSq() < 1e-8) direction.set(1, 0.72, 1);
  const areaKm2 = Number(feature?.area_km2);
  const basinDistance = feature?.layer === "refined_basins" && Number.isFinite(areaKm2)
    ? clamp(8 + Math.sqrt(Math.max(0, areaKm2)) / 70, 10, 24)
    : 0;
  const distance = basinDistance || clamp(direction.length() * 0.62, 7.5, 17);
  direction.normalize();
  const position = target.clone().addScaledVector(direction, distance);
  applyCameraPose({
    position: position.toArray(),
    target: target.toArray(),
    fov: Math.min(Number(camera.fov) || 42, 42),
  });
  viewerInteracted = true;
}

function loadRefinedBasinDataset(basinsUrl) {
  const cached = refinedBasinDataCache.get(basinsUrl);
  if (cached) return Promise.resolve(cached);
  const pending = refinedBasinDataLoadPromises.get(basinsUrl);
  if (pending) return pending;
  const loadPromise = fetchRefinedBasinJson(
    basinsUrl,
    errorLabel("explorer.errors.failedToLoadBasinBoundaries"),
  )
    .then((basinData) => {
      refinedBasinDataCache.set(basinsUrl, basinData);
      return basinData;
    })
    .finally(() => refinedBasinDataLoadPromises.delete(basinsUrl));
  refinedBasinDataLoadPromises.set(basinsUrl, loadPromise);
  return loadPromise;
}

async function activateRefinedBasinFeature(_feature, isCancelled = () => false) {
  const context = currentCoreContext;
  if (!context || !context.dataset.capabilities.refinedBasins || controlsUI.showRefinedBasins.disabled) {
    throw new Error("Refined basins are unavailable for the selected dataset.");
  }
  if (!controlsUI.showBed.checked && !controlsUI.showIce.checked) {
    controlsUI.showIce.checked = true;
    syncIceSurfaceVisibilityFromControls();
  }
  controlsUI.showRefinedBasins.checked = true;
  const loaded = await ensureRefinedBasinsLoaded({ trigger: "toggle" });
  if (isCancelled() || !loaded) return false;
  updateRefinedBasinVisibility();
  updateMetaFromCurrentState();
  return true;
}

function initializePolarFeatures() {
  polarFeaturesController = createPolarFeaturesController({
    THREE,
    scene,
    locale: pageLocale,
    elements: {
      searchInput: controlsUI.polarSearchInput,
      results: controlsUI.polarSearchResults,
      searchStatus: controlsUI.polarSearchStatus,
      details: controlsUI.polarFeatureDetails,
      stationToggle: controlsUI.showResearchStations,
      namesToggle: controlsUI.showGeographicNames,
      basinToggle: controlsUI.showRefinedBasins,
    },
    dataUrls: polarFeatureDataUrls,
    getRegion: () => currentRegionKey,
    changeRegion: async (nextRegionKey) => {
      if (nextRegionKey === currentRegionKey) return;
      if (!controlsUI.regionPreset || lockedRegionKey) {
        throw new Error(t("explorer.errors.regionSwitchUnavailable"));
      }
      controlsUI.regionPreset.value = nextRegionKey;
      controlsUI.regionPreset.dispatchEvent(new Event("change"));
    },
    waitForRegionReady: (regionKey, isCancelled = () => false) =>
      waitUntilCondition(
        () =>
          isCancelled() ||
          (currentRegionKey === regionKey &&
            currentCoreContext?.generation === loadGeneration &&
            !isLoadingOverlayVisible()),
        90000,
        t("explorer.errors.regionLoadTimedOut", { region: regionKey }),
      ),
    getScenePoint: getPolarFeatureScenePoint,
    getExaggeration: () => Number(controlsUI.exaggeration.value) || 1,
    focusScenePoint: focusPolarFeature,
    activateRefinedBasinFeature,
  });
  polarFeaturesController.initialize();
}

function sampleRefinedBasinPoint(context, xMeters, yMeters, layerKey, offsetMeters) {
  const gridPoint = projectPs71PointToGrid(context, xMeters, yMeters);
  if (!gridPoint) return null;
  const heightMeters = sampleGridHeightNearest(context, gridPoint.col, gridPoint.row, layerKey);
  if (!Number.isFinite(heightMeters)) return null;
  const scenePoint = gridToSceneXZ(context, gridPoint.col, gridPoint.row);
  return {
    x: scenePoint.x,
    y: (heightMeters + offsetMeters) / context.baseConfig.verticalMetersPerUnit,
    z: scenePoint.z,
  };
}

function getRefinedBasinLabelTexture(
  labelText,
  {
    keyPrefix = "refined-basin",
    fontSize = 44,
    textColor = "#f6f9ff",
    strokeColor = "rgba(7, 17, 26, 0.95)",
    strokeWidth = 7,
    fontFamily = "system-ui, -apple-system, Segoe UI, sans-serif",
  } = {}
) {
  const key = `${keyPrefix}|${labelText}`;
  const cached = refinedBasinLabelTextureCache.get(key);
  if (cached) return cached;

  const canvas = document.createElement("canvas");
  const probe = canvas.getContext("2d");
  if (!probe) return null;

  const padding = 14;
  probe.font = `600 ${fontSize}px ${fontFamily}`;
  const initialTextWidth = Math.max(1, Math.ceil(probe.measureText(labelText).width));
  const maximumTextWidth = MAX_REFINED_BASIN_LABEL_CANVAS_WIDTH - padding * 2 - strokeWidth * 2;
  const renderedFontSize = initialTextWidth > maximumTextWidth
    ? Math.max(20, Math.floor(fontSize * maximumTextWidth / initialTextWidth))
    : fontSize;
  probe.font = `600 ${renderedFontSize}px ${fontFamily}`;
  const textWidth = Math.max(1, Math.ceil(probe.measureText(labelText).width));
  const canvasWidth = Math.min(MAX_REFINED_BASIN_LABEL_CANVAS_WIDTH, textWidth + padding * 2 + strokeWidth * 2);
  const canvasHeight = Math.ceil(renderedFontSize * 1.45 + padding * 2);
  const texturePixels = canvasWidth * canvasHeight;
  if (refinedBasinLabelTexturePixels + texturePixels > MAX_REFINED_BASIN_LABEL_TEXTURE_PIXELS) return null;
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.font = `600 ${renderedFontSize}px ${fontFamily}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = strokeWidth;
  ctx.strokeStyle = strokeColor;
  ctx.fillStyle = textColor;

  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  ctx.strokeText(labelText, centerX, centerY);
  ctx.fillText(labelText, centerX, centerY);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  refinedBasinLabelTextureCache.set(key, texture);
  refinedBasinLabelTexturePixels += texturePixels;
  return texture;
}

function createRefinedBasinLabelSprite(labelText, labelStyle = {}) {
  const texture = getRefinedBasinLabelTexture(labelText, labelStyle);
  if (!texture) return null;

  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      depthTest: false,
    })
  );
  const worldScale = Number.isFinite(labelStyle.worldScale) ? labelStyle.worldScale : REFINED_BASIN_LABEL_WORLD_SCALE;
  const width = texture.image?.width || 256;
  const height = texture.image?.height || 64;
  sprite.scale.set(width * worldScale, height * worldScale, 1);
  return sprite;
}

function buildRefinedBasinLineGroup(context, basins, layerKey, options = {}) {
  const offsetMeters = Number(options.offsetMeters || 0);
  const widthMeters = Number.isFinite(options.widthMeters) ? Math.max(0, options.widthMeters) : 0;
  const halfWidthUnits = (widthMeters / context.baseConfig.horizontalMetersPerUnit) * 0.5;
  const depthTest = options.depthTest !== false;
  const polygonOffsetFactor = Number.isFinite(options.polygonOffsetFactor) ? options.polygonOffsetFactor : -2;
  const polygonOffsetUnits = Number.isFinite(options.polygonOffsetUnits) ? options.polygonOffsetUnits : -2;
  const positions = [];

  for (const basin of basins) {
    const segments = Array.isArray(basin?.segments_xy_m) ? basin.segments_xy_m : [];
    for (const segment of segments) {
      if (!Array.isArray(segment) || segment.length < 2) continue;
      let prev = null;
      for (const point of segment) {
        if (!Array.isArray(point) || point.length < 2) {
          prev = null;
          continue;
        }
        const sample = sampleRefinedBasinPoint(context, Number(point[0]), Number(point[1]), layerKey, offsetMeters);
        if (!sample) {
          prev = null;
          continue;
        }
        if (prev) {
          const dx = sample.x - prev.x;
          const dz = sample.z - prev.z;
          const length = Math.hypot(dx, dz);
          if (length > 1e-8 && halfWidthUnits > 0) {
            const nx = -dz / length;
            const nz = dx / length;
            const left0x = prev.x + nx * halfWidthUnits;
            const left0z = prev.z + nz * halfWidthUnits;
            const right0x = prev.x - nx * halfWidthUnits;
            const right0z = prev.z - nz * halfWidthUnits;
            const left1x = sample.x + nx * halfWidthUnits;
            const left1z = sample.z + nz * halfWidthUnits;
            const right1x = sample.x - nx * halfWidthUnits;
            const right1z = sample.z - nz * halfWidthUnits;

            positions.push(
              left0x,
              prev.y,
              left0z,
              left1x,
              sample.y,
              left1z,
              right0x,
              prev.y,
              right0z,
              right0x,
              prev.y,
              right0z,
              left1x,
              sample.y,
              left1z,
              right1x,
              sample.y,
              right1z
            );
          }
        }
        prev = sample;
      }
    }
  }

  if (!positions.length) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
  const material = new THREE.MeshBasicMaterial({
    color: options.color || 0xffffff,
    transparent: true,
    opacity: Number.isFinite(options.opacity) ? options.opacity : 0.95,
    depthWrite: false,
    depthTest,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor,
    polygonOffsetUnits,
  });
  const lines = new THREE.Mesh(geometry, material);
  lines.scale.y = Number(controlsUI.exaggeration.value);
  lines.renderOrder = Number.isFinite(options.renderOrder) ? options.renderOrder : 36;
  return lines;
}

function buildRefinedBasinLabelGroup(context, basins, layerKey, options = {}) {
  const group = new THREE.Group();
  const offsetMeters = Number(options.offsetMeters || 0);
  const style = options.labelStyle || {};

  for (const basin of basins.slice(0, MAX_REFINED_BASIN_LABELS)) {
    if (refinedBasinLabelTexturePixels >= MAX_REFINED_BASIN_LABEL_TEXTURE_PIXELS) break;
    const labelPoint = basin?.label_xy_m;
    if (!Array.isArray(labelPoint) || labelPoint.length < 2) continue;
    const name = String(basin?.name || "").trim();
    if (!name) continue;

    const sample = sampleRefinedBasinPoint(context, Number(labelPoint[0]), Number(labelPoint[1]), layerKey, offsetMeters);
    if (!sample) continue;

    const sprite = createRefinedBasinLabelSprite(name, style);
    if (!sprite) continue;
    sprite.userData.baseY = sample.y;
    sprite.position.set(sample.x, sample.y, sample.z);
    sprite.renderOrder = Number.isFinite(options.renderOrder) ? options.renderOrder : 44;
    group.add(sprite);
  }

  return group.children.length ? group : null;
}

function getRefinedBasinLabelStyle(context, layerKey) {
  const regionKey = context?.dataset?.regionKey || "antarctica";
  const isBedLayer = layerKey === "bed";
  const isGreenland = regionKey === "greenland";
  return {
    keyPrefix: `${regionKey}-refined-basin-${isBedLayer ? "bed" : "surface"}`,
    textColor: isBedLayer ? "#ffe8a3" : "#ffd9d5",
    strokeColor: isBedLayer ? "rgba(22, 34, 44, 0.96)" : "rgba(20, 32, 46, 0.96)",
    fontSize: isGreenland ? 56 : 44,
    strokeWidth: isGreenland ? 8 : 7,
    worldScale: isGreenland ? GREENLAND_REFINED_BASIN_LABEL_WORLD_SCALE : REFINED_BASIN_LABEL_WORLD_SCALE,
  };
}

function updateRefinedBasinLabelPositionsForExaggeration() {
  const exaggeration = Number(controlsUI.exaggeration.value);
  const apply = (group) => {
    if (!group) return;
    for (const sprite of group.children) {
      const baseY = Number(sprite.userData?.baseY);
      if (!Number.isFinite(baseY)) continue;
      sprite.position.y = baseY * exaggeration;
    }
  };
  apply(refinedBasinBedLabels);
  apply(refinedBasinSurfaceLabels);
}

function updateRefinedBasinVisibility() {
  const enabled = controlsUI.showRefinedBasins.checked;
  const showBed = enabled && controlsUI.showBed.checked;
  const showSurface = enabled && controlsUI.showIce.checked;
  const showBedLines = showBed && !showSurface;
  const showBedLabels = showBed && !showSurface;

  if (refinedBasinBedLines) refinedBasinBedLines.visible = showBedLines;
  if (refinedBasinSurfaceLines) refinedBasinSurfaceLines.visible = showSurface;
  if (refinedBasinBedLabels) refinedBasinBedLabels.visible = showBedLabels;
  if (refinedBasinSurfaceLabels) refinedBasinSurfaceLabels.visible = showSurface;
}

function buildRefinedBasinOverlays(context, dataset) {
  const basins = Array.isArray(dataset?.basins) ? dataset.basins : [];
  if (!basins.length) return false;

  clearRefinedBasinOverlays();

  refinedBasinBedLines = buildRefinedBasinLineGroup(context, basins, "bed", {
    offsetMeters: REFINED_BASIN_BED_OFFSET_M,
    widthMeters: REFINED_BASIN_BED_LINE_WIDTH_M,
    color: 0xf6d365,
    opacity: 0.9,
    polygonOffsetFactor: -12,
    polygonOffsetUnits: -12,
    renderOrder: 38,
  });
  refinedBasinSurfaceLines = buildRefinedBasinLineGroup(context, basins, "surface", {
    offsetMeters: REFINED_BASIN_SURFACE_OFFSET_M,
    widthMeters: REFINED_BASIN_SURFACE_LINE_WIDTH_M,
    color: 0xff5f56,
    opacity: 0.95,
    renderOrder: 39,
  });
  refinedBasinBedLabels = buildRefinedBasinLabelGroup(context, basins, "bed", {
    offsetMeters: REFINED_BASIN_BED_LABEL_OFFSET_M,
    renderOrder: 46,
    labelStyle: getRefinedBasinLabelStyle(context, "bed"),
  });
  refinedBasinSurfaceLabels = buildRefinedBasinLabelGroup(context, basins, "surface", {
    offsetMeters: REFINED_BASIN_SURFACE_LABEL_OFFSET_M,
    renderOrder: 47,
    labelStyle: getRefinedBasinLabelStyle(context, "surface"),
  });

  if (refinedBasinBedLines) scene.add(refinedBasinBedLines);
  if (refinedBasinSurfaceLines) scene.add(refinedBasinSurfaceLines);
  if (refinedBasinBedLabels) scene.add(refinedBasinBedLabels);
  if (refinedBasinSurfaceLabels) scene.add(refinedBasinSurfaceLabels);

  updateRefinedBasinLabelPositionsForExaggeration();
  updateRefinedBasinVisibility();

  return Boolean(refinedBasinBedLines || refinedBasinSurfaceLines || refinedBasinBedLabels || refinedBasinSurfaceLabels);
}

async function ensureRefinedBasinsLoaded({ trigger = "prefetch" } = {}) {
  const context = currentCoreContext;
  if (!context || context.generation !== loadGeneration) return false;
  if (!context.dataset.capabilities.refinedBasins) return false;
  const basinsUrl = context.dataset.refinedBasinsUrl;
  if (!basinsUrl) return false;
  if (refinedBasinUnavailableUrls.has(basinsUrl)) return false;

  const basinStatusLabel = String(context.dataset.basinStatusLabel || "refined basins");
  const basinStatusTitle = basinStatusLabel.charAt(0).toUpperCase() + basinStatusLabel.slice(1);

  const overlaysReady = refinedBasinBedLines || refinedBasinSurfaceLines || refinedBasinBedLabels || refinedBasinSurfaceLabels;
  if (overlaysReady) {
    currentRefinedBasinData = refinedBasinDataCache.get(basinsUrl) || currentRefinedBasinData;
    updateRefinedBasinLabelPositionsForExaggeration();
    updateRefinedBasinVisibility();
    updateMetaFromCurrentState();
    return true;
  }

  if (refinedBasinLoadPromise && refinedBasinLoadingUrl === basinsUrl) {
    return refinedBasinLoadPromise;
  }

  const pendingPromise = (async () => {
    const showStatus = trigger === "toggle";
    if (showStatus) {
      statusEl.textContent = t("explorer.status.loadingBasin", { label: basinStatusLabel });
    }

    const basinData = await loadRefinedBasinDataset(basinsUrl);
    currentRefinedBasinData = basinData;

    if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    const built = buildRefinedBasinOverlays(context, basinData);
    if (!built) {
      throw new Error("Basin dataset is empty.");
    }

    if (showStatus) {
      statusEl.textContent = getReadyStatusText(context);
    }
    updateMetaFromCurrentState();
    return true;
  })()
    .catch((error) => {
      console.error("Basin layer load failed:", error);
      if (context === currentCoreContext && context.generation === loadGeneration) {
        refinedBasinUnavailableUrls.add(basinsUrl);
        controlsUI.showRefinedBasins.checked = false;
        clearRefinedBasinOverlays();
        const readyText = getReadyStatusText(context);
        const transientText = t("explorer.status.refinedBasinsUnavailable", { label: basinStatusTitle });
        statusEl.textContent = transientText;
        updateRegionLayerAvailability(currentRegionKey);
        window.setTimeout(() => {
          if (statusEl.textContent === transientText) {
            statusEl.textContent = readyText;
          }
        }, 1800);
      }
      return false;
    })
    .finally(() => {
      if (refinedBasinLoadPromise === pendingPromise) {
        refinedBasinLoadPromise = null;
        refinedBasinLoadingUrl = null;
      }
    });

  refinedBasinLoadingUrl = basinsUrl;
  refinedBasinLoadPromise = pendingPromise;
  return refinedBasinLoadPromise;
}

function updateIceSideVisibility() {
  if (!iceSideMesh) return;
  iceSideMesh.visible = controlsUI.showIce.checked && controlsUI.showIceBottom.checked;
}

function syncIceMaterialMode(opacityValue = Number(controlsUI.iceOpacity?.value)) {
  const requestedOpacity = clamp01(Number(opacityValue));

  const applyModeToMaterial = (material, nextOpacity) => {
    if (!material) return;
    let needsUpdate = false;
    if (material.transparent !== true) {
      material.transparent = true;
      needsUpdate = true;
    }
    if (material.depthWrite !== true) {
      material.depthWrite = true;
      needsUpdate = true;
    }
    if (Math.abs(material.opacity - nextOpacity) > 1e-6) {
      material.opacity = nextOpacity;
    }
    if (needsUpdate) {
      material.needsUpdate = true;
    }
  };

  if (iceMesh) {
    applyModeToMaterial(iceMesh.material, requestedOpacity);
    iceMesh.renderOrder = ICE_SURFACE_RENDER_ORDER;
  }

  if (iceSideMesh) {
    applyModeToMaterial(iceSideMesh.material, requestedOpacity);
    iceSideMesh.renderOrder = ICE_SIDE_STRICT_OCCLUSION_RENDER_ORDER;
  }
}

function syncIceSurfaceVisibilityFromControls() {
  if (iceMesh) iceMesh.visible = controlsUI.showIce.checked;
  if (iceBottomMesh) iceBottomMesh.visible = controlsUI.showIceBottom.checked;
  updateIceSideVisibility();
  updateRefinedBasinVisibility();
}

function hideIceSurfaceForRiseOverlay() {
  if (!controlsUI.showIce.checked) return;
  controlsUI.showIce.checked = false;
  syncIceSurfaceVisibilityFromControls();
}

function hideBedStressOverlaysExcept(activeKey = "") {
  let changed = false;
  if (controlsUI.showIsostaticRebound?.checked) {
    controlsUI.showIsostaticRebound.checked = false;
    updateReboundControlsUi();
    applyReboundGeometry({ recomputeNormals: true });
    if (iceSideMesh) updateIceSideVisibility();
    changed = true;
  }
  if (activeKey !== "basalFriction" && controlsUI.showBasalFriction.checked) {
    controlsUI.showBasalFriction.checked = false;
    changed = true;
  }
  if (activeKey !== "effectivePressure" && controlsUI.showEffectivePressure.checked) {
    controlsUI.showEffectivePressure.checked = false;
    changed = true;
  }
  if (basalFrictionMesh) basalFrictionMesh.visible = controlsUI.showBasalFriction.checked;
  if (effectivePressureMesh) effectivePressureMesh.visible = controlsUI.showEffectivePressure.checked;
  return changed;
}

function updateFlowlineVisibility() {
  if (!controlsUI.showFlowline.checked) {
    if (flowlineMesh) flowlineMesh.visible = false;
    updateSelectedFlowlineVisualState();
    refreshFlowlineGuidanceUi();
    return;
  }

  if (flowlineMesh) {
    flowlineMesh.visible = true;
    updateSelectedFlowlineVisualState();
    refreshFlowlineGuidanceUi();
    return;
  }

  if (!velocityField) {
    refreshFlowlineGuidanceUi();
    return;
  }

  const pendingGeneration = loadGeneration;
  const previousStatus = statusEl.textContent;
  statusEl.textContent = t("explorer.status.computingFlowlines");

  window.setTimeout(() => {
    if (pendingGeneration !== loadGeneration || !velocityField || !controlsUI.showFlowline.checked) {
      if (statusEl.textContent === t("explorer.status.computingFlowlines")) {
        statusEl.textContent = previousStatus;
      }
      return;
    }

    const built = buildFlowlineMesh(velocityField);
    if (!built) {
      controlsUI.showFlowline.checked = false;
      if (statusEl.textContent === t("explorer.status.computingFlowlines")) {
        statusEl.textContent = previousStatus;
      }
      return;
    }

    flowlineMesh = built;
    scene.add(flowlineMesh);
    flowlineMesh.visible = true;
    updateSelectedFlowlineVisualState();
    refreshFlowlineGuidanceUi();
    if (statusEl.textContent === t("explorer.status.computingFlowlines")) {
      statusEl.textContent = previousStatus;
    }
  }, 0);
}

function reportLoadError(error) {
  console.error(error);
  setLoadingOverlayVisible(false);
  statusEl.textContent = t("explorer.status.loadFailed");
  if (controlsUI.flowlineProfileCardMount) {
    controlsUI.flowlineProfileCardMount.innerHTML = "";
    controlsUI.flowlineProfileCardMount.hidden = true;
  }
  metaListEl.innerHTML = `<li><strong>${t("explorer.meta.errorLabel")}:</strong> ${escapeHtml(
    localizeErrorMessage(error.message)
  )}</li>`;
}

function formatCoord(value, digits = 2) {
  const text = Number(value).toFixed(digits);
  if (digits <= 0) return text;
  return text.replace(/\.?0+$/, "");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderSourceValue(source) {
  if (!source) return "";
  if (typeof source === "string") return escapeHtml(source);
  const text = escapeHtml(source.text || "");
  const url = escapeHtml(source.url || "");
  if (!url) return text;
  return `<a class="meta-link" href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
}

function renderSourceLine(label, source, suffix = "") {
  return `<li class="meta-compact"><strong>${escapeHtml(label)}:</strong> ${renderSourceValue(source)}${suffix}</li>`;
}

function renderMetaSection(sectionId, title, items, options = {}) {
  if (!items.length) return "";
  const defaultOpen = Boolean(options.defaultOpen);
  const isOpen = metaSectionOpenState.has(sectionId) ? metaSectionOpenState.get(sectionId) : defaultOpen;
  return `
    <li class="meta-section">
      <details class="meta-section-details"${isOpen ? " open" : ""} data-section-id="${escapeHtml(sectionId)}">
        <summary class="meta-section-summary">
          <span>${escapeHtml(title)}</span>
          <span class="meta-section-icon" aria-hidden="true"></span>
        </summary>
        <ul class="meta-section-items">
          ${items.join("")}
        </ul>
      </details>
    </li>
  `;
}

function createFlowlineEmptyStateSvgMarkup(flowlinesVisible) {
  const lineStroke = flowlinesVisible ? "rgba(62, 142, 176, 0.96)" : "rgba(104, 126, 139, 0.58)";
  const lineGlow = flowlinesVisible ? "rgba(166, 226, 248, 0.38)" : "rgba(195, 208, 217, 0.22)";
  const previewOpacity = flowlinesVisible ? "1" : "0.6";
  return `
    <svg class="meta-flowline-empty-graphic" viewBox="0 0 220 92" aria-hidden="true">
      <rect x="0" y="0" width="220" height="92" rx="10" fill="rgba(223, 241, 249, 0.52)"></rect>
      <path d="M 64 55 C 86 37, 101 31, 124 34" fill="none" stroke="${lineGlow}" stroke-width="10" stroke-linecap="round"></path>
      <path d="M 64 55 C 86 37, 101 31, 124 34" fill="none" stroke="${lineStroke}" stroke-width="3.5" stroke-linecap="round"></path>
      <path d="M 38 24 L 54 38 L 46 40 L 52 56 L 44 59 L 38 43 L 31 49 Z" fill="rgba(245, 252, 255, 0.96)" stroke="rgba(50, 77, 95, 0.48)" stroke-width="1.4"></path>
      <path d="M 54 30 C 79 30, 97 27, 126 27" fill="none" stroke="rgba(72, 110, 132, 0.4)" stroke-width="1.9" stroke-linecap="round" stroke-dasharray="4.5 5"></path>
      <path d="M 123 23 L 132 27 L 123 31" fill="none" stroke="rgba(72, 110, 132, 0.4)" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"></path>
      <g opacity="${previewOpacity}">
        <rect x="144" y="17" width="54" height="54" rx="8" fill="rgba(255, 255, 255, 0.76)" stroke="rgba(133, 178, 201, 0.5)" stroke-width="1.5"></rect>
        <path d="M 150 59 L 190 59 L 190 63 L 150 63 Z" fill="rgba(143, 101, 58, 0.3)"></path>
        <path d="M 150 55 C 160 50, 172 52, 190 46 L 190 59 L 150 59 Z" fill="rgba(143, 101, 58, 0.34)"></path>
        <path d="M 150 55 C 160 50, 172 52, 190 46" fill="none" stroke="rgba(120, 78, 36, 0.94)" stroke-width="1.8" stroke-linecap="round"></path>
        <path d="M 150 43 C 161 34, 174 38, 190 29 L 190 46 C 172 52, 160 50, 150 55 Z" fill="rgba(133, 193, 221, 0.5)"></path>
        <path d="M 150 43 C 161 34, 174 38, 190 29" fill="none" stroke="rgba(246, 252, 255, 0.98)" stroke-width="2.2" stroke-linecap="round"></path>
      </g>
    </svg>
  `;
}

function renderFlowlineEmptyStateCard(flowlinesVisible) {
  const title = flowlinesVisible
    ? t("explorer.meta.selectedFlowlineEmptyTitle")
    : t("explorer.meta.selectedFlowlineDisabledTitle");
  const body = flowlinesVisible
    ? t("explorer.meta.selectedFlowlineEmptyBody")
    : t("explorer.meta.selectedFlowlineDisabledBody");
  return `
    <div class="meta-flowline-empty" role="note">
      <div class="meta-flowline-empty-title">${escapeHtml(title)}</div>
      <div class="meta-flowline-empty-body">${escapeHtml(body)}</div>
      ${createFlowlineEmptyStateSvgMarkup(flowlinesVisible)}
    </div>
  `;
}

function buildSvgPath(points) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${formatCoord(point.x, 2)} ${formatCoord(point.y, 2)}`)
    .join(" ");
}

function createFlowlineProfileSvgMarkup(flowlineSummary) {
  const samples = Array.isArray(flowlineSummary?.samples) ? flowlineSummary.samples : [];
  if (samples.length < 2) return "";

  const width = 336;
  const height = 188;
  const padLeft = 38;
  const padRight = 12;
  const padTop = 24;
  const padBottom = 32;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const elevations = [];

  for (const sample of samples) {
    const surface = Number(sample.surface);
    let bottom = Number(sample.bottom);
    let bed = Number(sample.bed);
    if (!Number.isFinite(surface)) continue;
    if (!Number.isFinite(bottom) || bottom > surface) bottom = surface;
    if (!Number.isFinite(bed) || bed > bottom) bed = bottom;
    elevations.push(surface, bottom, bed);
  }
  if (elevations.length < 4) return "";

  let minElevation = Math.min(...elevations);
  let maxElevation = Math.max(...elevations);
  const padElevation = Math.max(25, (maxElevation - minElevation) * 0.08);
  minElevation -= padElevation;
  maxElevation += padElevation;
  if (Math.abs(maxElevation - minElevation) < 1e-6) {
    maxElevation += 1;
    minElevation -= 1;
  }

  const totalDistanceKm = Math.max(1e-6, Number(flowlineSummary.lengthKm) || 0);
  const toX = (distanceKm, fallbackIndex) =>
    padLeft + plotWidth * (totalDistanceKm > 0 ? clamp01(distanceKm / totalDistanceKm) : fallbackIndex / Math.max(1, samples.length - 1));
  const toY = (elevation) => padTop + ((maxElevation - elevation) / (maxElevation - minElevation)) * plotHeight;

  const surfacePoints = [];
  const bottomPoints = [];
  const bedPoints = [];
  samples.forEach((sample, index) => {
    const surface = Number(sample.surface);
    let bottom = Number(sample.bottom);
    let bed = Number(sample.bed);
    if (!Number.isFinite(surface)) return;
    if (!Number.isFinite(bottom) || bottom > surface) bottom = surface;
    if (!Number.isFinite(bed) || bed > bottom) bed = bottom;
    const x = toX(Number(sample.distanceKm) || 0, index);
    surfacePoints.push({ x, y: toY(surface) });
    bottomPoints.push({ x, y: toY(bottom) });
    bedPoints.push({ x, y: toY(bed) });
  });
  if (surfacePoints.length < 2 || bottomPoints.length < 2 || bedPoints.length < 2) return "";

  const fillPath = `${buildSvgPath(surfacePoints)} ${buildSvgPath([...bottomPoints].reverse()).replace(/^M /, "L ")} Z`;
  const chartBottomY = padTop + plotHeight;
  const chartLeftX = padLeft;
  const chartRightX = padLeft + plotWidth;
  const yAxisLabelX = chartLeftX;
  const yAxisLabelY = 11;
  const yTickLabelX = chartLeftX - 6;
  const bedFillPath = `${buildSvgPath(bedPoints)} L ${formatCoord(
    bedPoints[bedPoints.length - 1].x,
    2
  )} ${formatCoord(chartBottomY, 2)} L ${formatCoord(bedPoints[0].x, 2)} ${formatCoord(chartBottomY, 2)} Z`;
  const surfacePath = buildSvgPath(surfacePoints);
  const bottomPath = buildSvgPath(bottomPoints);
  const bedPath = buildSvgPath(bedPoints);
  const gradientId = `flowline-silhouette-fill-${Number(flowlineSummary.index || 0)}`;
  const ariaLabel = escapeHtml(t("explorer.meta.selectedFlowlinePreview"));
  const axisColor = "rgba(85, 114, 132, 0.72)";
  const gridColor = "rgba(129, 159, 176, 0.22)";
  const tickColor = "rgba(88, 116, 131, 0.86)";
  const unitColor = "rgba(100, 128, 142, 0.78)";
  const fontFamily = "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace";
  const yTicks = [maxElevation, (maxElevation + minElevation) / 2, minElevation];
  const xTicks = [0, totalDistanceKm / 2, totalDistanceKm];
  const yTickMarkup = yTicks
    .map((value) => {
      const y = toY(value);
      return `
        <line x1="${formatCoord(chartLeftX, 2)}" y1="${formatCoord(y, 2)}" x2="${formatCoord(chartRightX, 2)}" y2="${formatCoord(y, 2)}" stroke="${gridColor}" stroke-width="1"></line>
        <line x1="${formatCoord(chartLeftX - 4, 2)}" y1="${formatCoord(y, 2)}" x2="${formatCoord(chartLeftX, 2)}" y2="${formatCoord(y, 2)}" stroke="${axisColor}" stroke-width="1.4"></line>
        <text x="${formatCoord(yTickLabelX, 2)}" y="${formatCoord(y + 3.4, 2)}" text-anchor="end" fill="${tickColor}" font-family="${fontFamily}" font-size="10.2">${escapeHtml(formatCoord(value, 0))}</text>
      `;
    })
    .join("");
  const xTickMarkup = xTicks
    .map((value, index) => {
      const x = toX(value, index);
      return `
        <line x1="${formatCoord(x, 2)}" y1="${formatCoord(chartBottomY, 2)}" x2="${formatCoord(x, 2)}" y2="${formatCoord(chartBottomY + 4, 2)}" stroke="${axisColor}" stroke-width="1.4"></line>
        <text x="${formatCoord(x, 2)}" y="${formatCoord(chartBottomY + 16, 2)}" text-anchor="middle" fill="${tickColor}" font-family="${fontFamily}" font-size="10.2">${escapeHtml(formatCoord(value, value >= 100 ? 0 : 1))}</text>
      `;
    })
    .join("");

  return `
    <svg class="meta-flowline-silhouette" viewBox="0 0 ${width} ${height}" role="img" aria-label="${ariaLabel}">
      <defs>
        <linearGradient id="${gradientId}" x1="0%" x2="0%" y1="0%" y2="100%">
          <stop offset="0%" stop-color="#e0f6ff" stop-opacity="0.95" />
          <stop offset="60%" stop-color="#aadbf2" stop-opacity="0.88" />
          <stop offset="100%" stop-color="#76abca" stop-opacity="0.92" />
        </linearGradient>
      </defs>
      ${yTickMarkup}
      <path d="${bedFillPath}" fill="rgba(143, 101, 58, 0.38)"></path>
      <path d="${fillPath}" fill="url(#${gradientId})"></path>
      <path d="${bedPath}" fill="none" stroke="rgba(120, 78, 36, 0.94)" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"></path>
      <path d="${bottomPath}" fill="none" stroke="rgba(44, 79, 107, 0.62)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"></path>
      <path d="${surfacePath}" fill="none" stroke="rgba(248, 253, 255, 0.98)" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"></path>
      <line x1="${formatCoord(chartLeftX, 2)}" y1="${formatCoord(chartBottomY, 2)}" x2="${formatCoord(chartRightX, 2)}" y2="${formatCoord(chartBottomY, 2)}" stroke="${axisColor}" stroke-width="1.5"></line>
      <line x1="${formatCoord(chartLeftX, 2)}" y1="${formatCoord(padTop, 2)}" x2="${formatCoord(chartLeftX, 2)}" y2="${formatCoord(chartBottomY, 2)}" stroke="${axisColor}" stroke-width="1.5"></line>
      ${xTickMarkup}
      <text x="${formatCoord(yAxisLabelX, 2)}" y="${formatCoord(yAxisLabelY, 2)}" text-anchor="middle" fill="${unitColor}" font-family="${fontFamily}" font-size="10.2">Y (m)</text>
      <text x="${formatCoord(chartRightX, 2)}" y="${formatCoord(chartBottomY + 29, 2)}" text-anchor="end" fill="${unitColor}" font-family="${fontFamily}" font-size="10.2">X (km)</text>
    </svg>
  `;
}

function renderSelectedFlowlineCard(selectedFlowline) {
  if (!selectedFlowline) {
    return renderFlowlineEmptyStateCard(Boolean(controlsUI.showFlowline?.checked));
  }
  const profileSvg = createFlowlineProfileSvgMarkup(selectedFlowline);
  if (!profileSvg) return "";
  return `<div class="flowline-inline-card">${profileSvg}</div>`;
}

function bindMetaSectionToggles() {
  metaListEl.querySelectorAll(".meta-section-details").forEach((detailsEl) => {
    detailsEl.addEventListener("toggle", () => {
      const sectionId = detailsEl.dataset.sectionId || "";
      if (!sectionId) return;
      metaSectionOpenState.set(sectionId, detailsEl.open);
    });
  });
}

async function exportCurrentViewPose() {
  if (!camera || !orbit) return;

  const poseText = [
    `position: new THREE.Vector3(${formatCoord(camera.position.x)}, ${formatCoord(camera.position.y)}, ${formatCoord(camera.position.z)}),`,
    `target: new THREE.Vector3(${formatCoord(orbit.target.x)}, ${formatCoord(orbit.target.y)}, ${formatCoord(orbit.target.z)}),`,
    `fov: ${formatCoord(camera.fov, 1)},`,
  ].join("\n");

  console.log("[View Pose]\n" + poseText);

  let copied = false;
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(poseText);
      copied = true;
    } catch (error) {
      console.warn("Clipboard write failed, pose still logged in console.", error);
    }
  }

  const previous = statusEl.textContent;
  const message = copied ? t("explorer.status.viewCopied") : t("explorer.status.viewLogged");
  statusEl.textContent = message;
  window.setTimeout(() => {
    if (statusEl.textContent === message) {
      statusEl.textContent = previous;
    }
  }, 1800);
}

function fullscreenApiAvailable() {
  if (isShowcaseMode) return false;
  const hasRequest =
    typeof viewerShellEl?.requestFullscreen === "function" ||
    typeof viewerShellEl?.webkitRequestFullscreen === "function";
  const explicitlyDisabled = document.fullscreenEnabled === false && document.webkitFullscreenEnabled === false;
  return hasRequest && !explicitlyDisabled;
}

function getFullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function isViewerFullscreen() {
  return getFullscreenElement() === viewerShellEl;
}

async function requestViewerFullscreen() {
  if (typeof viewerShellEl.requestFullscreen === "function") {
    return viewerShellEl.requestFullscreen();
  }
  if (typeof viewerShellEl.webkitRequestFullscreen === "function") {
    return viewerShellEl.webkitRequestFullscreen();
  }
  throw new Error(t("explorer.errors.fullscreenUnavailable"));
}

async function exitFullscreenMode() {
  if (typeof document.exitFullscreen === "function") {
    return document.exitFullscreen();
  }
  if (typeof document.webkitExitFullscreen === "function") {
    return document.webkitExitFullscreen();
  }
  return undefined;
}

function updateFullscreenControls() {
  const supported = fullscreenApiAvailable();
  const active = isViewerFullscreen();
  const label = active ? t("explorer.fullscreen.exit") : t("explorer.fullscreen.enter");
  const hiddenInShowcase = isShowcaseMode;
  const hideOnMobileUnsupported = !supported && isCoarsePointerInput() && compactViewportQuery.matches;

  const apply = (buttonEl) => {
    if (!buttonEl) return;
    buttonEl.hidden = hiddenInShowcase || hideOnMobileUnsupported;
    if (hiddenInShowcase || hideOnMobileUnsupported) {
      buttonEl.disabled = true;
      buttonEl.setAttribute("aria-hidden", "true");
      buttonEl.setAttribute("aria-pressed", "false");
      return;
    }
    buttonEl.disabled = !supported;
    buttonEl.textContent = supported ? label : t("explorer.fullscreen.unavailable");
    buttonEl.removeAttribute("aria-hidden");
    buttonEl.setAttribute("aria-pressed", active ? "true" : "false");
  };

  apply(controlsUI.fullscreenToggle);
  apply(controlsUI.viewerFullscreenToggle);
  viewerShellEl.classList.toggle("is-fullscreen", active);
}

async function toggleFullscreenMode() {
  if (!fullscreenApiAvailable()) return;
  try {
    if (isViewerFullscreen()) {
      await exitFullscreenMode();
    } else {
      await requestViewerFullscreen();
    }
  } catch (error) {
    console.warn("Fullscreen toggle failed:", error);
    const previousStatus = statusEl.textContent;
    const message = t("explorer.fullscreen.blocked");
    statusEl.textContent = message;
    window.setTimeout(() => {
      if (statusEl.textContent === message) {
        statusEl.textContent = previousStatus;
      }
    }, 1800);
  }
}

function handleFullscreenChange() {
  updateFullscreenControls();
  resizeRendererToViewer();
}

function setMobilePanelOffset(offsetPx) {
  if (!panelEl) return;
  const clampedOffset = clamp(offsetPx, 0, mobilePanelClosedOffsetPx);
  mobilePanelOffsetPx = clampedOffset;
  panelEl.style.setProperty("--mobile-panel-offset", `${clampedOffset}px`);
  const openThreshold = Math.max(8, mobilePanelClosedOffsetPx * 0.18);
  mobilePanelOpen = clampedOffset <= openThreshold;
  document.body.classList.toggle("mobile-panel-open", mobilePanelOpen);
}

function syncMobilePanelBounds({ preservePosition = true } = {}) {
  if (!mobileDrawerEnabled || !panelEl) return;
  const previousClosedOffset = mobilePanelClosedOffsetPx;
  const previousOffset = mobilePanelOffsetPx;
  const panelHeight = Math.ceil(panelEl.getBoundingClientRect().height);
  mobilePanelClosedOffsetPx = Math.max(0, panelHeight - MOBILE_PANEL_PEEK_PX);

  let nextOffset = mobilePanelOpen ? 0 : mobilePanelClosedOffsetPx;
  if (preservePosition && previousClosedOffset > 0) {
    const ratio = clamp(previousOffset / previousClosedOffset, 0, 1);
    nextOffset = ratio * mobilePanelClosedOffsetPx;
  }
  setMobilePanelOffset(nextOffset);
}

function setMobilePanelOpen(open) {
  if (!mobileDrawerEnabled || !panelEl) {
    mobilePanelOpen = false;
    document.body.classList.remove("mobile-panel-open");
    return;
  }
  syncMobilePanelBounds({ preservePosition: true });
  setMobilePanelOffset(open ? 0 : mobilePanelClosedOffsetPx);
}

function bindMobileDrawerDrag() {
  if (!panelHeaderEl || !panelEl) return;

  const start = (event) => {
    if (!mobileDrawerEnabled) return;
    if (event.pointerType === "mouse" && event.button !== 0) return;
    if (event.target instanceof Element && event.target.closest("button, input, select, textarea, a, label")) return;
    const now = performance.now();
    syncMobilePanelBounds({ preservePosition: true });
    mobilePanelDragState = {
      pointerId: event.pointerId,
      startY: event.clientY,
      currentY: event.clientY,
      lastTime: now,
      startOffset: mobilePanelOffsetPx,
      displayOffset: mobilePanelOffsetPx,
      velocityPxPerMs: 0,
      moved: false,
    };
    mobilePanelIgnoreTap = false;
    document.body.classList.add("mobile-panel-dragging");
    panelHeaderEl.setPointerCapture(event.pointerId);
    event.preventDefault();
  };

  const move = (event) => {
    if (!mobilePanelDragState) return;
    if (event.pointerId !== mobilePanelDragState.pointerId) return;
    const now = performance.now();
    const deltaY = event.clientY - mobilePanelDragState.startY;
    if (Math.abs(deltaY) > MOBILE_PANEL_DRAG_MOVE_THRESHOLD_PX) {
      mobilePanelDragState.moved = true;
    }

    const rawOffset = clamp(mobilePanelDragState.startOffset + deltaY, 0, mobilePanelClosedOffsetPx);
    const dt = Math.max(8, now - mobilePanelDragState.lastTime);
    const dy = event.clientY - mobilePanelDragState.currentY;
    const instantVelocity = dy / dt;
    mobilePanelDragState.velocityPxPerMs = mobilePanelDragState.velocityPxPerMs * 0.62 + instantVelocity * 0.38;

    const speedNorm = clamp(
      Math.abs(mobilePanelDragState.velocityPxPerMs) / MOBILE_PANEL_DRAG_FAST_SPEED_PX_PER_MS,
      0,
      1
    );
    const followGain =
      MOBILE_PANEL_DRAG_SLOW_GAIN + (MOBILE_PANEL_DRAG_FAST_GAIN - MOBILE_PANEL_DRAG_SLOW_GAIN) * speedNorm;

    const nextDisplayOffset =
      mobilePanelDragState.displayOffset + (rawOffset - mobilePanelDragState.displayOffset) * followGain;
    mobilePanelDragState.displayOffset = nextDisplayOffset;
    setMobilePanelOffset(nextDisplayOffset);
    mobilePanelDragState.currentY = event.clientY;
    mobilePanelDragState.lastTime = now;
    event.preventDefault();
  };

  const end = (event) => {
    if (!mobilePanelDragState) return;
    if (event.pointerId !== mobilePanelDragState.pointerId) return;
    const releaseRawOffset = clamp(
      mobilePanelDragState.startOffset + (event.clientY - mobilePanelDragState.startY),
      0,
      mobilePanelClosedOffsetPx
    );
    const releaseVelocity = mobilePanelDragState.velocityPxPerMs;
    const dragTravel = Math.abs(event.clientY - mobilePanelDragState.startY);
    mobilePanelIgnoreTap = mobilePanelDragState.moved;
    mobilePanelDragState = null;
    document.body.classList.remove("mobile-panel-dragging");
    setMobilePanelOffset(releaseRawOffset);

    const shouldFling =
      dragTravel >= MOBILE_PANEL_FLING_MIN_TRAVEL_PX &&
      Math.abs(releaseVelocity) >= MOBILE_PANEL_FLING_SPEED_PX_PER_MS;
    if (shouldFling) {
      setMobilePanelOpen(releaseVelocity < 0);
    } else {
      const snapRange = 12;
      if (mobilePanelOffsetPx <= snapRange) {
        setMobilePanelOpen(true);
      } else if (mobilePanelOffsetPx >= mobilePanelClosedOffsetPx - snapRange) {
        setMobilePanelOpen(false);
      }
    }

    try {
      panelHeaderEl.releasePointerCapture(event.pointerId);
    } catch (_err) {
      // No-op: pointer capture may already be released by browser.
    }
  };

  panelHeaderEl.addEventListener("pointerdown", start);
  panelHeaderEl.addEventListener("pointermove", move);
  panelHeaderEl.addEventListener("pointerup", end);
  panelHeaderEl.addEventListener("pointercancel", end);
  panelHeaderEl.addEventListener("click", (event) => {
    if (!mobileDrawerEnabled) return;
    if (event.target instanceof Element && event.target.closest("button, input, select, textarea, a, label")) return;
    if (mobilePanelIgnoreTap) {
      mobilePanelIgnoreTap = false;
      return;
    }
    setMobilePanelOpen(!mobilePanelOpen);
  });
}

function updateInteractionHint(active) {
  if (!interactionHintEl) return;
  if (isPreviewMode) {
    interactionHintEl.innerHTML = "";
    return;
  }
  if (isShowcaseMode) {
    interactionHintEl.innerHTML = active ? SHOWCASE_HINT_ACTIVE : SHOWCASE_HINT_IDLE;
    return;
  }
  if (!interactionGateEnabled) {
    interactionHintEl.innerHTML = DEFAULT_INTERACTION_HINT;
    return;
  }
  interactionHintEl.innerHTML = active ? TOUCH_HINT_ACTIVE : TOUCH_HINT_IDLE;
}

function syncInteractionToggleButton() {
  if (!controlsUI.interactionToggle) return;
  const showToggle = interactionGateEnabled && isCoarsePointerInput();
  controlsUI.interactionToggle.classList.toggle("is-hidden", !showToggle);
  controlsUI.interactionToggle.disabled = !showToggle;
  controlsUI.interactionToggle.textContent = interactionGateActive
    ? t("explorer.interaction.scrollPage")
    : t("explorer.interaction.enable3d");
  controlsUI.interactionToggle.setAttribute("aria-pressed", interactionGateActive ? "true" : "false");
}

function getIdleViewerTouchAction() {
  return interactionGateEnabled && isCoarsePointerInput() ? "pan-y" : "auto";
}

function refreshResponsiveLayout() {
  syncPointerModeClass();
  const previouslyEnabled = mobileDrawerEnabled;
  mobileDrawerEnabled = shouldUseMobileDrawer();
  document.body.classList.toggle("mobile-drawer", mobileDrawerEnabled);
  if (statusEl) {
    statusEl.hidden = mobileDrawerEnabled;
  }

  if (controlsUI.panelCloseButton) {
    controlsUI.panelCloseButton.hidden = !mobileDrawerEnabled;
    controlsUI.panelCloseButton.disabled = !mobileDrawerEnabled;
  }

  if (!mobileDrawerEnabled) {
    document.body.classList.remove("mobile-panel-dragging");
    if (panelEl) {
      panelEl.style.removeProperty("--mobile-panel-offset");
    }
    setMobilePanelOpen(false);
    mobilePanelClosedOffsetPx = 0;
    mobilePanelOffsetPx = 0;
    mobilePanelIgnoreTap = false;
  } else {
    syncMobilePanelBounds({ preservePosition: previouslyEnabled });
    if (!previouslyEnabled) {
      setMobilePanelOpen(false);
    }
  }
  updateFullscreenControls();
  scheduleTickLegendLayout(controlsUI?.velocityLegendLabels);
  scheduleTickLegendLayout(controlsUI?.basalFrictionLegendLabels, { minGapPx: 8 });
  scheduleTickLegendLayout(controlsUI?.basalMeltLegendLabels);
  if (shouldUseShowcaseMobileLinkout() && interactionGateActive) {
    setInteractionGateActive(false);
  }
  if (renderer && interactionGateEnabled && !interactionGateActive) {
    renderer.domElement.style.touchAction = getIdleViewerTouchAction();
  }
  updateInteractionHint(getShowcaseInteractionHintActiveState());
  syncInteractionToggleButton();
  syncDesktopInteractiveShowcasePrompt();
}

function setInteractionGateActive(active) {
  if (!orbit || !renderer || !viewerShellEl) return;
  clearShowcaseInteractionResumeTimer();
  if (!interactionGateEnabled) {
    interactionGateActive = true;
    interactionGateTapState = null;
    orbit.enabled = true;
    renderer.domElement.style.touchAction = "none";
    viewerShellEl.classList.remove("showcase-idle");
    updateInteractionHint(true);
    syncInteractionToggleButton();
    syncDesktopInteractiveShowcasePrompt();
    if (isShowcaseMode) {
      stopShowcaseAutoOrbit({ syncBasePose: false });
    }
    return;
  }

  interactionGateActive = Boolean(active);
  interactionGateTapState = null;
  orbit.enabled = interactionGateActive;
  renderer.domElement.style.touchAction = interactionGateActive ? "none" : getIdleViewerTouchAction();
  if (isShowcaseMode) {
    viewerShellEl.classList.toggle("showcase-idle", !interactionGateActive);
  } else {
    viewerShellEl.classList.remove("showcase-idle");
  }
  updateInteractionHint(interactionGateActive);
  syncInteractionToggleButton();
  syncDesktopInteractiveShowcasePrompt();
  if (isShowcaseMode) {
    if (interactionGateActive) {
      stopShowcaseAutoOrbit({ syncBasePose: true });
    } else {
      startShowcaseAutoOrbit();
    }
  }
}

function bindInteractionGate() {
  if (!renderer || !viewerShellEl || !orbit) return;

  const canvas = renderer.domElement;
  const pointerIsInsideViewer = (event) => {
    if (!event || !Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return false;
    const hit = document.elementFromPoint(event.clientX, event.clientY);
    return Boolean(hit && viewerShellEl.contains(hit));
  };

  const shouldUseTapToArm = (event) => {
    if (!interactionGateEnabled || !isCoarsePointerInput()) return false;
    return !event || event.pointerType !== "mouse";
  };

  const clearTapCandidate = (pointerId = null) => {
    if (pointerId !== null && interactionGateTapState && interactionGateTapState.pointerId !== pointerId) return;
    interactionGateTapState = null;
  };

  const activate = ({ closeMobilePanel = true } = {}) => {
    if (shouldUseShowcaseMobileLinkout()) return;
    if (!interactionGateEnabled || interactionGateActive) return;
    if (closeMobilePanel && mobileDrawerEnabled) {
      setMobilePanelOpen(false);
    }
    setInteractionGateActive(true);
  };

  const deactivate = () => {
    clearTapCandidate();
    if (!interactionGateEnabled || !interactionGateActive) return;
    setInteractionGateActive(false);
  };

  interactionGateEnabled = shouldUseInteractionGate();
  if (isDesktopInteractiveShowcase()) {
    setShowcaseUserInteracting(false);
  }

  // Coarse pointers must tap first so a page scroll swipe never doubles as an orbit gesture.
  canvas.addEventListener(
    "pointerdown",
    (event) => {
      if (!interactionGateEnabled || interactionGateActive) return;
      if (shouldUseTapToArm(event)) {
        if (event.isPrimary === false || (interactionGateTapState && interactionGateTapState.pointerId !== event.pointerId)) {
          clearTapCandidate();
          return;
        }
        interactionGateTapState = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          moved: false,
        };
        return;
      }
      activate();
    },
    { capture: true }
  );
  canvas.addEventListener(
    "pointermove",
    (event) => {
      if (!interactionGateTapState || event.pointerId !== interactionGateTapState.pointerId) return;
      if (
        Math.abs(event.clientX - interactionGateTapState.startX) > INTERACTION_TAP_MOVE_THRESHOLD_PX ||
        Math.abs(event.clientY - interactionGateTapState.startY) > INTERACTION_TAP_MOVE_THRESHOLD_PX
      ) {
        interactionGateTapState.moved = true;
      }
    },
    { capture: true }
  );
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      deactivate();
    }
  });
  window.addEventListener("blur", deactivate);

  orbit.addEventListener("start", () => {
    if (isDesktopInteractiveShowcase()) {
      noteDesktopInteractiveShowcaseActivity({ scheduleResume: false });
      return;
    }
    clearShowcaseInteractionResumeTimer();
  });
  orbit.addEventListener("change", () => {
    if (!isDesktopInteractiveShowcase() || !showcaseMotionState.userInteracting) return;
    showcaseLastActivityAtMs = performance.now();
  });
  orbit.addEventListener("end", () => {
    scheduleShowcaseInteractionResume();
  });

  if (isDesktopInteractiveShowcase()) {
    const scheduleDesktopInteractiveResumeIfNeeded = () => {
      if (!showcaseMotionState.userInteracting) return;
      scheduleShowcaseInteractionResume();
    };
    const bindDesktopInteractiveResumeEvents = (eventTarget) => {
      if (!eventTarget || typeof eventTarget.addEventListener !== "function") return;
      eventTarget.addEventListener("mouseup", scheduleDesktopInteractiveResumeIfNeeded, { passive: true });
      eventTarget.addEventListener("pointerup", scheduleDesktopInteractiveResumeIfNeeded, { passive: true });
      eventTarget.addEventListener("pointercancel", scheduleDesktopInteractiveResumeIfNeeded, { passive: true });
    };
    bindDesktopInteractiveResumeEvents(canvas);
    bindDesktopInteractiveResumeEvents(window);
    bindDesktopInteractiveResumeEvents(document);
    try {
      if (window.top && window.top !== window) {
        bindDesktopInteractiveResumeEvents(window.top);
        bindDesktopInteractiveResumeEvents(window.top.document);
      }
    } catch (_error) {
      // Cross-origin parents are ignored; local listeners still cover standalone mode.
    }
    canvas.addEventListener(
      "click",
      (event) => {
        if (showcaseMotionState.userInteracting) return;
        if (!pointerIsInsideViewer(event)) return;
        noteDesktopInteractiveShowcaseActivity();
      },
      { capture: true }
    );
    canvas.addEventListener(
      "wheel",
      () => {
        if (!showcaseMotionState.userInteracting) return;
        noteDesktopInteractiveShowcaseActivity();
      },
      { passive: true }
    );
  }

  if (isShowcaseMode) {
    viewerShellEl.addEventListener("mouseleave", deactivate);
    canvas.addEventListener(
      "pointerup",
      (event) => {
        if (interactionGateTapState && event.pointerId === interactionGateTapState.pointerId) {
          const tapCandidate = interactionGateTapState;
          clearTapCandidate(event.pointerId);
          if (!tapCandidate.moved && pointerIsInsideViewer(event)) {
            activate();
            return;
          }
        }
        if (interactionGateActive && !pointerIsInsideViewer(event)) {
          deactivate();
        }
      },
      { capture: true }
    );
    canvas.addEventListener(
      "pointercancel",
      (event) => {
        clearTapCandidate(event.pointerId);
        deactivate();
      },
      { capture: true }
    );
    canvas.addEventListener("lostpointercapture", () => {
      clearTapCandidate();
      window.requestAnimationFrame(() => {
        if (!viewerShellEl.matches(":hover")) {
          deactivate();
        }
      });
    });
  } else {
    canvas.addEventListener(
      "pointerup",
      (event) => {
        if (!interactionGateTapState || event.pointerId !== interactionGateTapState.pointerId) return;
        const tapCandidate = interactionGateTapState;
        clearTapCandidate(event.pointerId);
        if (!tapCandidate.moved && pointerIsInsideViewer(event)) {
          activate();
        }
      },
      { capture: true }
    );
    canvas.addEventListener(
      "pointercancel",
      (event) => {
        clearTapCandidate(event.pointerId);
      },
      { capture: true }
    );
  }

  window.addEventListener("keydown", (event) => {
    const activeTag = document.activeElement?.tagName || "";
    if (activeTag === "INPUT" || activeTag === "SELECT" || activeTag === "TEXTAREA") return;
    if (event.shiftKey && event.code === "KeyV") {
      event.preventDefault();
      exportCurrentViewPose();
    }
  });

  // In an iframe, wheel events do not reach the parent page by default.
  // Forward wheel scrolling to the parent page while 3D interaction is idle.
  if (isShowcaseMode) {
    canvas.addEventListener(
      "wheel",
      (event) => {
        if (!shouldForwardShowcaseWheelToPage()) return;
        if (event.ctrlKey || event.metaKey) return;

        event.preventDefault();
        event.stopImmediatePropagation();

        if (window.parent && window.parent !== window) {
          try {
            window.parent.postMessage(
              {
                type: SHOWCASE_SCROLL_MESSAGE_TYPE,
                deltaX: event.deltaX,
                deltaY: event.deltaY,
              },
              "*"
            );
            return;
          } catch (_err) {
            // Fall through to local scrolling when postMessage is unavailable.
          }
        }

        window.scrollBy({ left: event.deltaX, top: event.deltaY, behavior: "auto" });
      },
      { capture: true, passive: false }
    );
  }

  if (isDesktopInteractiveShowcase()) {
    setShowcaseUserInteracting(false);
  } else {
    setInteractionGateActive(interactionGateEnabled ? false : true);
  }
}

function bindFlowlinePicking() {
  if (!renderer || !THREE) return;
  if (!flowlineRaycaster) flowlineRaycaster = new THREE.Raycaster();
  if (!flowlinePointerNdc) flowlinePointerNdc = new THREE.Vector2();

  const canvas = renderer.domElement;

  const clearGesture = (pointerId = null) => {
    if (pointerId !== null && flowlinePickGesture && flowlinePickGesture.pointerId !== pointerId) return;
    flowlinePickGesture = null;
  };

  canvas.addEventListener(
    "pointerdown",
    (event) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      flowlinePickGesture = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        interactionReady: !interactionGateEnabled || interactionGateActive,
      };
    },
    { passive: true }
  );

  canvas.addEventListener(
    "pointerup",
    (event) => {
      if (!flowlinePickGesture || flowlinePickGesture.pointerId !== event.pointerId) return;
      const gesture = flowlinePickGesture;
      clearGesture(event.pointerId);

      const moved =
        Math.abs(event.clientX - gesture.startX) > FLOWLINE_PICK_MOVE_THRESHOLD_PX ||
        Math.abs(event.clientY - gesture.startY) > FLOWLINE_PICK_MOVE_THRESHOLD_PX;
      if (moved || !gesture.interactionReady) return;

      pickVisibleFlowlineAtClientPoint(event.clientX, event.clientY);
    },
    { passive: true }
  );

  canvas.addEventListener(
    "pointercancel",
    (event) => {
      clearGesture(event.pointerId);
    },
    { passive: true }
  );

  canvas.addEventListener("lostpointercapture", () => {
    clearGesture();
  });
}

function bindBackgroundWarmupTrigger() {
  if (isShowcaseMode || !renderer) return;
  const canvas = renderer.domElement;
  const markInteraction = () => {
    viewerInteracted = true;
    scheduleBackgroundWarmup();
  };
  canvas.addEventListener("pointerdown", markInteraction, { passive: true });
  canvas.addEventListener("wheel", markInteraction, { passive: true });
  canvas.addEventListener("touchstart", markInteraction, { passive: true });
}

function bindRecordingMode() {
  updateRecordingControlsUi();
  if (!recordingModeEnabled || !renderer) return;

  const canvas = renderer.domElement;
  const interruptAutoMotion = () => {
    if (!recordingMotionState.playing) return;
    stopRecordingPlayback({ syncBasePose: true });
  };
  canvas.addEventListener("pointerdown", interruptAutoMotion, { passive: true });
  canvas.addEventListener("wheel", interruptAutoMotion, { passive: true });
  canvas.addEventListener("touchstart", interruptAutoMotion, { passive: true });

  if (controlsUI.capturePlayButton) {
    controlsUI.capturePlayButton.addEventListener("click", () => {
      toggleRecordingPlayback();
    });
  }

  if (controlsUI.captureResetButton) {
    controlsUI.captureResetButton.addEventListener("click", () => {
      resetCameraToDefaultPose();
    });
  }

  if (controlsUI.captureHideButton) {
    controlsUI.captureHideButton.addEventListener("click", () => {
      setRecordingPanelVisible(false);
    });
  }

  if (controlsUI.captureOrbitToggle) {
    controlsUI.captureOrbitToggle.addEventListener("change", () => {
      recordingMotionState.orbitEnabled = Boolean(controlsUI.captureOrbitToggle.checked);
      if (!recordingMotionState.orbitEnabled && !recordingMotionState.zoomEnabled && recordingMotionState.playing) {
        stopRecordingPlayback({ syncBasePose: true });
      }
      updateRecordingControlsUi();
    });
  }

  if (controlsUI.captureZoomToggle) {
    controlsUI.captureZoomToggle.addEventListener("change", () => {
      recordingMotionState.zoomEnabled = Boolean(controlsUI.captureZoomToggle.checked);
      if (!recordingMotionState.orbitEnabled && !recordingMotionState.zoomEnabled && recordingMotionState.playing) {
        stopRecordingPlayback({ syncBasePose: true });
      }
      updateRecordingControlsUi();
    });
  }

  if (controlsUI.captureDirectionCw) {
    controlsUI.captureDirectionCw.addEventListener("click", () => {
      setRecordingDirection("cw");
    });
  }

  if (controlsUI.captureDirectionCcw) {
    controlsUI.captureDirectionCcw.addEventListener("click", () => {
      setRecordingDirection("ccw");
    });
  }

  if (controlsUI.captureSpeed) {
    controlsUI.captureSpeed.addEventListener("input", () => {
      recordingMotionState.speed = clamp(Number(controlsUI.captureSpeed.value), 0.25, 2);
      updateRecordingControlsUi();
    });
  }

  if (controlsUI.captureZoomAmount) {
    controlsUI.captureZoomAmount.addEventListener("input", () => {
      recordingMotionState.zoomAmount = clamp(Number(controlsUI.captureZoomAmount.value) / 100, 0, 0.3);
      updateRecordingControlsUi();
    });
  }

  for (const button of controlsUI.captureManualButtons) {
    const action = button.dataset.captureManual;
    if (!action) continue;
    let activePointerId = null;

    const release = (event = null) => {
      if (event && activePointerId !== null && event.pointerId !== activePointerId) return;
      setRecordingManualSource("manualButtonState", action, false);
      activePointerId = null;
    };

    button.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      activePointerId = event.pointerId;
      setRecordingManualSource("manualButtonState", action, true);
      button.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    button.addEventListener("pointerup", release);
    button.addEventListener("pointercancel", release);
    button.addEventListener("lostpointercapture", () => {
      release();
    });
  }

  window.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (isEditableElementActive()) return;

    if (event.code === "KeyP" && !event.repeat) {
      event.preventDefault();
      toggleRecordingPlayback();
      return;
    }

    if (event.code === "KeyH" && !event.repeat) {
      event.preventDefault();
      setRecordingPanelVisible(!recordingMotionState.panelVisible);
      return;
    }

    const keyMap = {
      KeyJ: "orbitLeft",
      KeyL: "orbitRight",
      KeyI: "tiltUp",
      KeyK: "tiltDown",
      KeyU: "zoomOut",
      KeyO: "zoomIn",
    };
    const action = keyMap[event.code];
    if (!action) return;
    event.preventDefault();
    setRecordingManualSource("manualKeyState", action, true);
  });

  window.addEventListener("keyup", (event) => {
    const keyMap = {
      KeyJ: "orbitLeft",
      KeyL: "orbitRight",
      KeyI: "tiltUp",
      KeyK: "tiltDown",
      KeyU: "zoomOut",
      KeyO: "zoomIn",
    };
    const action = keyMap[event.code];
    if (!action) return;
    setRecordingManualSource("manualKeyState", action, false);
  });

  window.addEventListener("blur", () => {
    clearRecordingManualState();
  });
}

function bindUI() {
  syncFlowLightAnimationPreference();
  const syncFlowLightForMotionPreference = () => syncFlowLightAnimationPreference();
  if (typeof flowMotionMediaQuery.addEventListener === "function") {
    flowMotionMediaQuery.addEventListener("change", syncFlowLightForMotionPreference);
  } else if (typeof flowMotionMediaQuery.addListener === "function") {
    flowMotionMediaQuery.addListener(syncFlowLightForMotionPreference);
  }

  bindMobileDrawerDrag();
  if (panelEl && !mobilePanelResizeObserver && "ResizeObserver" in window) {
    mobilePanelResizeObserver = new ResizeObserver(() => {
      if (!mobileDrawerEnabled || mobilePanelDragState) return;
      syncMobilePanelBounds({ preservePosition: true });
    });
    mobilePanelResizeObserver.observe(panelEl);
  }

  if (controlsUI.panelCloseButton) {
    controlsUI.panelCloseButton.addEventListener("click", () => {
      setMobilePanelOpen(false);
    });
  }

  if (controlsUI.interactionToggle) {
    controlsUI.interactionToggle.addEventListener("click", () => {
      if (!interactionGateEnabled) return;
      if (!interactionGateActive) {
        if (mobileDrawerEnabled) {
          setMobilePanelOpen(false);
        }
        setInteractionGateActive(true);
        return;
      }
      setInteractionGateActive(false);
    });
  }

  if (!lockedRegionKey && controlsUI.regionPreset) {
    controlsUI.regionPreset.addEventListener("change", () => {
      const nextRegionKey = getRegionConfig(controlsUI.regionPreset.value).key;
      const preferredDatasetKey = datasetSelectionByRegion[nextRegionKey] || getDefaultDatasetKey(nextRegionKey);
      currentRegionKey = nextRegionKey;
      if (controlsUI.showOceanCurrents.checked) {
        restoreOceanCurrentLayerSelection(currentRegionKey);
      }
      currentDatasetKey = refreshRegionUi(currentRegionKey, preferredDatasetKey);
      pendingViewResetOnLoad = true;
      loadAndBuildMeshes(currentDatasetKey).catch(reportLoadError);
    });
  }

  if (!lockedDatasetKey) {
    controlsUI.resolutionPreset.addEventListener("change", () => {
      currentDatasetKey = controlsUI.resolutionPreset.value;
      datasetSelectionByRegion[currentRegionKey] = currentDatasetKey;
      loadAndBuildMeshes(currentDatasetKey).catch(reportLoadError);
    });
  }

  controlsUI.exaggeration.addEventListener("input", () => {
    const value = Number(controlsUI.exaggeration.value);
    controlsUI.exaggerationValue.textContent = `${value.toFixed(1)}x`;
  if (bedMesh) bedMesh.scale.y = value;
  if (iceMesh) iceMesh.scale.y = value;
  if (iceBottomMesh) iceBottomMesh.scale.y = value;
  if (iceSideMesh) iceSideMesh.scale.y = value;
  if (velocitySurfaceMesh) velocitySurfaceMesh.scale.y = value;
  if (basalFrictionMesh) basalFrictionMesh.scale.y = value;
  if (basalMeltMesh) basalMeltMesh.scale.y = value;
    if (thermalDrivingMesh) thermalDrivingMesh.scale.y = value;
    if (effectivePressureMesh) effectivePressureMesh.scale.y = value;
    if (subglacialChannelMesh) subglacialChannelMesh.scale.y = value;
    if (flowlineMesh) flowlineMesh.scale.y = value;
    if (selectedFlowlineHighlight) selectedFlowlineHighlight.scale.y = value;
    if (oceanCurrentMesh) oceanCurrentMesh.scale.y = value;
    if (refinedBasinBedLines) refinedBasinBedLines.scale.y = value;
    if (refinedBasinSurfaceLines) refinedBasinSurfaceLines.scale.y = value;
    updateRefinedBasinLabelPositionsForExaggeration();
    updateReboundSeaPlane();
    polarFeaturesController?.updateExaggeration();
  });

  controlsUI.iceOpacity.addEventListener("input", () => {
    const value = Number(controlsUI.iceOpacity.value);
    controlsUI.iceOpacityValue.textContent = value.toFixed(2);
    syncIceMaterialMode(value);
    if (iceBottomMesh) iceBottomMesh.material.opacity = Math.max(0.12, value * 0.82);
  });

  controlsUI.showBed.addEventListener("change", () => {
    if (bedMesh) bedMesh.visible = controlsUI.showBed.checked;
    updateRefinedBasinVisibility();
    updateLegendVisibility();
  });
  controlsUI.showIce.addEventListener("change", () => {
    syncIceSurfaceVisibilityFromControls();
    updateMetaFromCurrentState();
  });
  controlsUI.showIceBottom.addEventListener("change", () => {
    syncIceSurfaceVisibilityFromControls();
    updateMetaFromCurrentState();
  });
  controlsUI.showVelocity.addEventListener("change", () => {
    updateLegendVisibility();
    if (controlsUI.showVelocity.checked && !velocitySurfaceMesh) {
      ensureVelocityLoaded({ trigger: "toggle" });
      return;
    }
    if (velocitySurfaceMesh) velocitySurfaceMesh.visible = controlsUI.showVelocity.checked;
    updateMetaFromCurrentState();
  });
  controlsUI.showFlowline.addEventListener("change", () => {
    updateLegendVisibility();
    if (controlsUI.showFlowline.checked && !velocityField) {
      refreshFlowlineGuidanceUi();
      updateMetaFromCurrentState();
      ensureVelocityLoaded({ trigger: "toggle" });
      return;
    }
    updateFlowlineVisibility();
    updateMetaFromCurrentState();
  });
  controlsUI.animateFlow.addEventListener("change", () => {
    syncFlowLightAnimationPreference();
    revealFlowlineForAnimation();
    rebuildStaticFlowLightLayers().catch((error) => {
      console.error("Failed to rebuild animated flow layers:", error);
    });
  });
  controlsUI.showBasalFriction.addEventListener("change", () => {
    if (controlsUI.showBasalFriction.checked) {
      hideBedStressOverlaysExcept("basalFriction");
    }
    updateLegendVisibility();
    if (controlsUI.showBasalFriction.checked && !basalFrictionMesh) {
      ensureBasalFrictionLoaded({ trigger: "toggle" });
      updateMetaFromCurrentState();
      return;
    }
    if (basalFrictionMesh) basalFrictionMesh.visible = controlsUI.showBasalFriction.checked;
    updateMetaFromCurrentState();
  });
  controlsUI.showBasalMelt.addEventListener("change", () => {
    if (controlsUI.showBasalMelt.checked && controlsUI.showThermalDriving.checked) {
      controlsUI.showThermalDriving.checked = false;
    }
    if (controlsUI.showBasalMelt.checked) {
      hideIceSurfaceForRiseOverlay();
    }
    updateLegendVisibility();
    if (controlsUI.showBasalMelt.checked && !basalMeltMesh) {
      ensureRiseLoaded({ trigger: "toggle" });
      updateMetaFromCurrentState();
      return;
    }
    updateRiseOverlayVisibility();
    updateMetaFromCurrentState();
  });
  controlsUI.showThermalDriving.addEventListener("change", () => {
    if (controlsUI.showThermalDriving.checked && controlsUI.showBasalMelt.checked) {
      controlsUI.showBasalMelt.checked = false;
    }
    if (controlsUI.showThermalDriving.checked) {
      hideIceSurfaceForRiseOverlay();
    }
    updateLegendVisibility();
    if (controlsUI.showThermalDriving.checked && !thermalDrivingMesh) {
      ensureRiseLoaded({ trigger: "toggle" });
      updateMetaFromCurrentState();
      return;
    }
    updateRiseOverlayVisibility();
    updateMetaFromCurrentState();
  });
  controlsUI.showOceanCurrents.addEventListener("change", async () => {
    if (controlsUI.showOceanCurrents.checked) {
      restoreOceanCurrentLayerSelection();
    } else {
      rememberOceanCurrentLayerSelection();
      for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
        const control = getOceanCurrentLayerControl(layer);
        if (control) control.checked = false;
      }
    }
    updateOceanCurrentLayerControls();
    updateLegendVisibility();
    if (controlsUI.showOceanCurrents.checked && !oceanCurrentMesh) {
      await ensureOceanCurrentsLoaded({ trigger: "toggle" });
      updateMetaFromCurrentState();
      return;
    }
    updateOceanCurrentLayerVisibility();
    updateMetaFromCurrentState();
  });
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    const control = getOceanCurrentLayerControl(layer);
    if (!control) continue;
    control.addEventListener("change", () => {
      rememberOceanCurrentLayerSelection();
      updateOceanCurrentLayerVisibility();
      updateMetaFromCurrentState();
    });
  }
  controlsUI.showRefinedBasins.addEventListener("change", () => {
    if (!controlsUI.showRefinedBasins.checked) {
      updateRefinedBasinVisibility();
      return;
    }
    const overlaysReady = refinedBasinBedLines || refinedBasinSurfaceLines || refinedBasinBedLabels || refinedBasinSurfaceLabels;
    if (!overlaysReady) {
      ensureRefinedBasinsLoaded({ trigger: "toggle" });
      return;
    }
    updateRefinedBasinVisibility();
  });
  controlsUI.showEffectivePressure.addEventListener("change", () => {
    if (controlsUI.showEffectivePressure.checked) {
      hideBedStressOverlaysExcept("effectivePressure");
    }
    updateLegendVisibility();
    if (controlsUI.showEffectivePressure.checked && !effectivePressureMesh) {
      ensureHydrologyLoaded({ trigger: "toggle" });
      return;
    }
    if (effectivePressureMesh) effectivePressureMesh.visible = controlsUI.showEffectivePressure.checked;
    updateMetaFromCurrentState();
  });
  controlsUI.showSubglacialChannels.addEventListener("change", () => {
    updateLegendVisibility();
    if (controlsUI.showSubglacialChannels.checked && !subglacialChannelMesh) {
      ensureHydrologyLoaded({ trigger: "toggle" });
      return;
    }
    if (subglacialChannelMesh) subglacialChannelMesh.visible = controlsUI.showSubglacialChannels.checked;
    updateMetaFromCurrentState();
  });
  controlsUI.showSea.addEventListener("change", () => {
    if (seaLevelMesh) seaLevelMesh.visible = controlsUI.showSea.checked;
  });

  controlsUI.showIsostaticRebound?.addEventListener("change", () => {
    const enabled = controlsUI.showIsostaticRebound.checked;
    if (enabled) {
      hideOverlaysForRebound();
      // Without a waterline the uplift is invisible, so surface the sea plane.
      if (controlsUI.showSea && !controlsUI.showSea.checked) {
        controlsUI.showSea.checked = true;
        if (seaLevelMesh) seaLevelMesh.visible = true;
      }
    } else if (iceSideMesh) {
      updateIceSideVisibility();
    }
    updateReboundControlsUi();
    updateLegendVisibility();
    if (enabled && currentCoreContext?.reboundSolveKey !== getReboundSolveKey()) {
      ensureIsostaticReboundLoaded({ trigger: "toggle" });
      return;
    }
    applyReboundGeometry({ recomputeNormals: true });
    updateMetaFromCurrentState();
  });

  controlsUI.reboundProgress?.addEventListener("input", () => {
    updateReboundControlsUi();
    if (isReboundActive()) scheduleReboundGeometryUpdate({ recomputeNormals: false });
  });
  controlsUI.reboundProgress?.addEventListener("change", () => {
    updateReboundControlsUi();
    if (!isReboundActive()) return;
    scheduleReboundGeometryUpdate({ recomputeNormals: true });
    updateMetaFromCurrentState();
  });

  controlsUI.reboundModel?.addEventListener("change", () => {
    updateReboundControlsUi();
    if (!controlsUI.showIsostaticRebound?.checked) return;
    ensureIsostaticReboundLoaded({ trigger: "resolve" });
  });

  controlsUI.reboundSeaLevel?.addEventListener("input", () => {
    updateReboundControlsUi();
    // The datum shifts the waterline immediately; the deflection it induces is a
    // second-order correction (~26 m of ocean-floor subsidence for 58 m of rise), so
    // it is re-solved once the slider settles rather than on every frame.
    if (isReboundActive()) scheduleReboundGeometryUpdate({ recomputeNormals: false });
  });
  controlsUI.reboundSeaLevel?.addEventListener("change", () => {
    updateReboundControlsUi();
    if (!controlsUI.showIsostaticRebound?.checked) return;
    if (reboundSeaLevelDebounce !== null) window.clearTimeout(reboundSeaLevelDebounce);
    reboundSeaLevelDebounce = window.setTimeout(() => {
      reboundSeaLevelDebounce = null;
      ensureIsostaticReboundLoaded({ trigger: "resolve" });
    }, 180);
  });

  controlsUI.highlightEmergentLand?.addEventListener("change", () => {
    if (isReboundActive()) applyReboundGeometry({ recomputeNormals: false });
  });

  controlsUI.wireframe.addEventListener("change", () => {
    const enable = controlsUI.wireframe.checked;
    if (bedMesh) bedMesh.material.wireframe = enable;
    if (iceMesh) iceMesh.material.wireframe = enable;
    if (iceBottomMesh) iceBottomMesh.material.wireframe = enable;
    if (iceSideMesh) iceSideMesh.material.wireframe = enable;
    if (velocitySurfaceMesh) velocitySurfaceMesh.material.wireframe = enable;
    if (basalFrictionMesh) basalFrictionMesh.material.wireframe = enable;
    if (basalMeltMesh) basalMeltMesh.material.wireframe = enable;
    if (thermalDrivingMesh) thermalDrivingMesh.material.wireframe = enable;
    if (effectivePressureMesh) effectivePressureMesh.material.wireframe = enable;
  });

  controlsUI.resetView.addEventListener("click", () => {
    resetCameraToDefaultPose();
  });

  if (controlsUI.fullscreenToggle) {
    controlsUI.fullscreenToggle.addEventListener("click", () => {
      toggleFullscreenMode();
    });
  }
  if (controlsUI.viewerFullscreenToggle) {
    controlsUI.viewerFullscreenToggle.addEventListener("click", () => {
      toggleFullscreenMode();
    });
  }

  window.addEventListener("keydown", (event) => {
    if (event.code !== "KeyF") return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const activeTag = document.activeElement?.tagName || "";
    if (activeTag === "INPUT" || activeTag === "SELECT" || activeTag === "TEXTAREA") return;
    event.preventDefault();
    toggleFullscreenMode();
  });

  refreshResponsiveLayout();
  window.addEventListener("resize", refreshResponsiveLayout);
  updateFullscreenControls();
}

function updateMeta(meta, dataset, velocityMeta, basalFrictionMeta, riseMeta, hydrologyMeta, oceanCurrentMeta, velocityMedianSpeed) {
  const region = getRegionConfig(dataset.regionKey);
  const capabilities = dataset.capabilities || region.capabilities;
  const grid = meta.grid;
  const bedStats = meta.fields.find((f) => f.name === "bed").stats_m;
  const thickStats = meta.fields.find((f) => f.name === "thickness").stats_m;
  const speedField = velocityMeta?.fields?.find((f) => f.name === "speed");
  const speedStats = speedField?.stats_m_per_year;
  const speedQuantiles = speedField?.quantiles_m_per_year;
  const basalFrictionField = basalFrictionMeta?.fields?.find((f) => f.name === "basal_friction");
  const basalFrictionStats = basalFrictionField?.stats_mpa;
  const basalFrictionQuantiles = basalFrictionField?.quantiles_mpa;
  const riseBasalField = riseMeta?.fields?.find((f) => f.name === "ismr");
  const riseBasalStats = riseBasalField?.stats_m_per_year;
  const riseBasalQuantiles = riseBasalField?.quantiles_m_per_year;
  const riseThermalField = riseMeta?.fields?.find((f) => f.name === "tstar_zice");
  const riseThermalStats = riseThermalField?.stats_c;
  const riseThermalQuantiles = riseThermalField?.quantiles_c;
  const riseDraftField = riseMeta?.fields?.find((f) => f.name === "zice");
  const riseDraftStats = riseDraftField?.stats_m;
  const riseDraftQuantiles = riseDraftField?.quantiles_m;
  const riseCoverage = riseMeta?.coverage;
  const pressureField = hydrologyMeta?.fields?.find((f) => f.name === "effective_pressure");
  const pressureStats = pressureField?.stats_pa;
  const pressureQuantiles = pressureField?.quantiles_pa;
  const channelField = hydrologyMeta?.fields?.find((f) => f.name === "channel_discharge");
  const channelStats = channelField?.stats_m3_per_s;
  const channelQuantiles = channelField?.quantiles_m3_per_s;
  const channelCoverage = hydrologyMeta?.coverage;
  const oceanStreamlineCount = Number(oceanCurrentMeta?.streamline_count || oceanCurrentMeta?.flowline_count || 0);
  const oceanCurrentSegmentCount = Number(oceanCurrentMeta?.segment_count || 0);
  const oceanCurrentSpeedStats = oceanCurrentMeta?.fields?.find((f) => f.name === "speed_mps")?.stats;
  const oceanCurrentThetaStats = oceanCurrentMeta?.fields?.find((f) => f.name === "theta_c_summary")?.stats;
  const oceanCurrentSalinityStats = oceanCurrentMeta?.fields?.find((f) => f.name === "sal_psu_summary")?.stats;
  const oceanCurrentCoverage = oceanCurrentMeta?.coverage;
  const oceanCurrentSeedStrategy = oceanCurrentMeta?.sampling?.seed_strategy || "";
  const oceanCurrentSeedBuckets =
    oceanCurrentCoverage?.streamlines_by_seed_bucket || oceanCurrentCoverage?.streamlines_by_seed_depth || {};
  const oceanCurrentBucketEntries = Object.entries(oceanCurrentSeedBuckets);
  const oceanCurrentLayerCount = new Set(
    oceanCurrentBucketEntries.map(([key]) => getOceanCurrentLayerFromBucketKey(key)).filter((value) => value.length > 0)
  ).size;
  const cavityMargin80kmStreamlineCount = oceanCurrentBucketEntries.reduce(
    (sum, [key, value]) => (key.startsWith("cavity_margin_80km_") ? sum + Number(value || 0) : sum),
    0
  );
  const remoteOpenOceanStreamlineCount = oceanCurrentBucketEntries.reduce(
    (sum, [key, value]) => (key.startsWith("remote_") ? sum + Number(value || 0) : sum),
    0
  );
  const oceanCurrentDepthMin = Number(oceanCurrentCoverage?.depth_min_m);
  const oceanCurrentDepthMax = Number(oceanCurrentCoverage?.depth_max_m);
  const oceanCurrentReference = oceanCurrentMeta?.source_time_label
    ? String(oceanCurrentMeta.source_time_label)
    : oceanCurrentMeta?.source_time_utc
    ? new Date(oceanCurrentMeta.source_time_utc).toLocaleDateString(numberLocale, {
        month: "long",
        year: "numeric",
        timeZone: "UTC",
      })
    : "";
  const refinedBasinCount = Number(currentRefinedBasinData?.basin_count);
  const stepKm = Math.abs(grid.dx_m) / 1000;
  const fmtInt = (value) => Math.round(Number(value)).toLocaleString(numberLocale);
  const fmtMpa = (value) =>
    (Number(value) / 1e6).toLocaleString(numberLocale, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtQ = (value) =>
    Number(value).toLocaleString(numberLocale, { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  const fmtShort = (value, digits = 2) =>
    Number(value).toLocaleString(numberLocale, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const quantileEntry = (label, value, formatter, suffix = "") =>
    Number.isFinite(Number(value)) ? `${label} ${formatter(Number(value))}${suffix}` : "";
  const quantileSummary = (entries) => entries.filter(Boolean).join("; ");
  const availableOnDemandText = t("explorer.meta.availableOnDemand");
  const notApplicableText = "n/a";
  const projectionLabel =
    dataset.regionKey === "greenland"
      ? "EPSG:3413 polar stereographic"
      : dataset.regionKey === "antarctica"
      ? "EPSG:3031 polar stereographic"
      : "Polar stereographic";
  const oceanCurrentReferenceText = oceanCurrentReference
    ? isChineseLocale
      ? `${oceanCurrentReference} 模式场`
      : `${oceanCurrentReference} model field`
    : isChineseLocale
    ? "模式场"
    : "model field";
  const qrfCoverageFraction = Number(meta.qrf_coverage?.applied_grounded_ice_fraction);
  const qrfHybridSummary = meta.hybridization
    ? isChineseLocale
      ? `QRF 床面用于 ${
          Number.isFinite(qrfCoverageFraction) ? fmtShort(qrfCoverageFraction * 100, 1) : "n/a"
        }% 的接地冰；冰面和掩膜采用 BedMachine Greenland v6，浮冰、非冰区和 QRF 空缺区保留其床面和厚度。`
      : `QRF bed for ${
          Number.isFinite(qrfCoverageFraction) ? fmtShort(qrfCoverageFraction * 100, 1) : "n/a"
        }% of grounded ice; BedMachine Greenland v6 supplies surface and mask, with its bed and thickness retained over floating ice, non-ice, and QRF gaps.`
    : "";
  const oceanCurrentSummary = !oceanCurrentMeta
    ? availableOnDemandText
    : oceanCurrentSeedStrategy === "merged_streamline_sets" &&
      cavityMargin80kmStreamlineCount > 0 &&
      remoteOpenOceanStreamlineCount > 0
    ? isChineseLocale
      ? `基于年平均模式场预计算 ${fmtInt(
          oceanStreamlineCount
        )} 条海洋流线，覆盖四个垂向层（表层、上层、中层、下层），其中 ${fmtInt(
          cavityMargin80kmStreamlineCount
        )} 条位于冰架腔体及 80 km 沿岸带，${fmtInt(remoteOpenOceanStreamlineCount)} 条位于外海。`
      : `Precomputed ${fmtInt(
          oceanStreamlineCount
        )} ocean streamlines from annual mean model fields across four vertical layers (surface, upper, mid, lower), including ${fmtInt(
          cavityMargin80kmStreamlineCount
        )} in the ice-shelf cavity and 80-km coastal zone, and ${fmtInt(
          remoteOpenOceanStreamlineCount
        )} in the open ocean.`
    : dataset.regionKey === "greenland" && oceanCurrentLayerCount >= 4
    ? isChineseLocale
      ? `基于 2025 年 8 月模式场预计算 ${fmtInt(
          oceanStreamlineCount
        )} 条海洋流线，覆盖四个垂向层（表层、上层、中层、下层）。`
      : `Precomputed ${fmtInt(
          oceanStreamlineCount
        )} ocean streamlines from August 2025 model fields across four vertical layers (surface, upper, mid, lower).`
    : isChineseLocale
    ? `${fmtInt(oceanStreamlineCount)} 条流线${
        oceanCurrentLayerCount > 1 ? `，覆盖 ${fmtInt(oceanCurrentLayerCount)} 个深度层` : ""
      }（${fmtInt(oceanCurrentSegmentCount)} 个线段），绘制自 ${oceanCurrentReferenceText}。`
    : `${fmtInt(oceanStreamlineCount)} streamlines${
        oceanCurrentLayerCount > 1 ? ` across ${fmtInt(oceanCurrentLayerCount)} depth layers` : ""
      } (${fmtInt(oceanCurrentSegmentCount)} segments), plotted from the ${oceanCurrentReferenceText}.`;

  const geometryItems = [
    `<li><strong>${t("explorer.meta.preset")}:</strong> ${region.label} ${dataset.label} (${stepKm.toFixed(
      stepKm >= 10 ? 0 : 1
    )} km grid)</li>`,
    `<li><strong>${t("explorer.meta.grid")}:</strong> ${fmtInt(grid.nx)} x ${fmtInt(grid.ny)}</li>`,
    `<li><strong>${t("explorer.meta.projection")}:</strong> ${projectionLabel}</li>`,
    `<li><strong>${t("explorer.meta.bedElevation")}:</strong> ${fmtInt(bedStats.min)} to ${fmtInt(bedStats.max)} m</li>`,
    `<li><strong>${t("explorer.meta.maxIceThickness")}:</strong> ${fmtInt(thickStats.max)} m</li>`,
    `<li><strong>${t("explorer.meta.meanIceThickness")}:</strong> ${
      Number.isFinite(thickStats.mean) ? `${fmtInt(thickStats.mean)} m` : notApplicableText
    }</li>`,
    ...(qrfHybridSummary
      ? [
          `<li><strong>${isChineseLocale ? "地形组成" : "Terrain composition"}:</strong> ${qrfHybridSummary}</li>`,
        ]
      : []),
  ];

  const velocityItems = capabilities.velocity
    ? [
        `<li><strong>${t("explorer.meta.surfaceSpeedRange")}:</strong> ${
          speedStats ? `${fmtInt(speedStats.min)} to ${fmtInt(speedStats.max)} m/yr` : notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.speedQuantiles")}:</strong> ${
          quantileSummary([
            quantileEntry(
              "median",
              Number.isFinite(Number(speedQuantiles?.median)) ? speedQuantiles.median : velocityMedianSpeed,
              fmtInt,
              " m/yr"
            ),
            quantileEntry("P90", speedQuantiles?.q90, fmtInt, " m/yr"),
            quantileEntry("P95", speedQuantiles?.q95, fmtInt, " m/yr"),
            quantileEntry("P99", speedQuantiles?.q99, fmtInt, " m/yr"),
          ]) || availableOnDemandText
        }</li>`,
      ]
    : [`<li><strong>${t("explorer.meta.status")}:</strong> ${t("explorer.meta.notYetAdded", { region: region.label })}</li>`];

  const basalFrictionItems = capabilities.basalFriction
    ? [
        `<li><strong>${t("explorer.meta.invertedFriction")}:</strong> ${t("explorer.meta.invertedFrictionSummary")}</li>`,
        `<li><strong>${t("explorer.meta.frictionRange")}:</strong> ${
          basalFrictionStats
            ? `${fmtShort(basalFrictionStats.min, 3)} to ${fmtShort(basalFrictionStats.max, 2)} MPa`
            : notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.frictionQuantiles")}:</strong> ${
          quantileSummary([
            quantileEntry("median", basalFrictionQuantiles?.median, (value) => fmtShort(value, 3), " MPa"),
            quantileEntry("P90", basalFrictionQuantiles?.q90, (value) => fmtShort(value, 3), " MPa"),
            quantileEntry("P95", basalFrictionQuantiles?.q95, (value) => fmtShort(value, 3), " MPa"),
            quantileEntry("P99", basalFrictionQuantiles?.q99, (value) => fmtShort(value, 3), " MPa"),
          ]) || notApplicableText
        }</li>`,
      ]
    : [`<li><strong>${t("explorer.meta.status")}:</strong> ${t("explorer.meta.notYetAdded", { region: region.label })}</li>`];

  const riseItems = capabilities.rise
    ? [
        `<li><strong>${t("explorer.meta.basalMeltRate")}:</strong> ${
          riseBasalStats ? `${fmtShort(riseBasalStats.min, 3)} to ${fmtShort(riseBasalStats.max, 3)} m/yr` : availableOnDemandText
        }</li>`,
        `<li><strong>${t("explorer.meta.meltQuantiles")}:</strong> ${
          quantileSummary([
            quantileEntry("median", riseBasalQuantiles?.median, (value) => fmtShort(value, 3), " m/yr"),
            quantileEntry("P90", riseBasalQuantiles?.q90, (value) => fmtShort(value, 3), " m/yr"),
            quantileEntry("P95", riseBasalQuantiles?.q95, (value) => fmtShort(value, 3), " m/yr"),
            quantileEntry("P99", riseBasalQuantiles?.q99, (value) => fmtShort(value, 3), " m/yr"),
          ]) || availableOnDemandText
        }</li>`,
        `<li><strong>${t("explorer.meta.thermalDriving")}:</strong> ${
          riseThermalStats ? `${fmtShort(riseThermalStats.min, 3)} to ${fmtShort(riseThermalStats.max, 3)} °C` : availableOnDemandText
        }</li>`,
        `<li><strong>${t("explorer.meta.thermalQuantiles")}:</strong> ${
          quantileSummary([
            quantileEntry("median", riseThermalQuantiles?.median, (value) => fmtShort(value, 3), " °C"),
            quantileEntry("P90", riseThermalQuantiles?.q90, (value) => fmtShort(value, 3), " °C"),
            quantileEntry("P95", riseThermalQuantiles?.q95, (value) => fmtShort(value, 3), " °C"),
            quantileEntry("P99", riseThermalQuantiles?.q99, (value) => fmtShort(value, 3), " °C"),
          ]) || availableOnDemandText
        }</li>`,
        `<li><strong>${t("explorer.meta.iceDraft")}:</strong> ${
          riseDraftStats ? `${fmtShort(riseDraftStats.min, 0)} to ${fmtShort(riseDraftStats.max, 0)} m` : availableOnDemandText
        }</li>`,
        `<li><strong>${t("explorer.meta.draftQuantiles")}:</strong> ${
          quantileSummary([
            quantileEntry("median", riseDraftQuantiles?.median, (value) => fmtShort(value, 0), " m"),
            quantileEntry("P90", riseDraftQuantiles?.q90, (value) => fmtShort(value, 0), " m"),
            quantileEntry("P95", riseDraftQuantiles?.q95, (value) => fmtShort(value, 0), " m"),
            quantileEntry("P99", riseDraftQuantiles?.q99, (value) => fmtShort(value, 0), " m"),
          ]) || availableOnDemandText
        }</li>`,
      ]
    : [];

  const oceanCurrentItems = capabilities.oceanCurrents
    ? [
        `<li><strong>${t("explorer.meta.oceanStreamlines")}:</strong> ${oceanCurrentSummary}</li>`,
        `<li><strong>${t("explorer.meta.depthSpan")}:</strong> ${
          Number.isFinite(oceanCurrentDepthMin) && Number.isFinite(oceanCurrentDepthMax)
            ? `${fmtShort(oceanCurrentDepthMin, 0)} to ${fmtShort(oceanCurrentDepthMax, 0)} m`
            : notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.horizontalSpeed")}:</strong> ${
          oceanCurrentSpeedStats ? `${fmtShort(oceanCurrentSpeedStats.min)} to ${fmtShort(oceanCurrentSpeedStats.max)} m/s` : notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.waterMassColor")}:</strong> ${
          oceanCurrentThetaStats && oceanCurrentSalinityStats
            ? t("explorer.meta.oceanWaterMassSummary", {
                thetaMin: fmtShort(oceanCurrentThetaStats.min, 1),
                thetaMax: fmtShort(oceanCurrentThetaStats.max, 1),
                salinityMin: fmtShort(oceanCurrentSalinityStats.min, 1),
                salinityMax: fmtShort(oceanCurrentSalinityStats.max, 1),
              })
            : t("explorer.meta.fourCornerPalette")
        }</li>`,
      ]
    : [];

  const hydrologyItems = capabilities.hydrology
    ? [
        `<li><strong>${t("explorer.meta.effectivePressureRange")}:</strong> ${
          pressureStats ? `${fmtMpa(pressureStats.min)} to ${fmtMpa(pressureStats.max)} MPa` : notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.pressureQuantiles")}:</strong> ${
          quantileSummary([
            quantileEntry("median", pressureQuantiles?.median, fmtMpa, " MPa"),
            quantileEntry("P90", pressureQuantiles?.q90, fmtMpa, " MPa"),
            quantileEntry("P95", pressureQuantiles?.q95, fmtMpa, " MPa"),
            quantileEntry("P99", pressureQuantiles?.q99, fmtMpa, " MPa"),
          ]) || notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.channelDischargeRange")}:</strong> ${
          channelStats ? `${fmtQ(channelStats.min)} to ${fmtQ(channelStats.max)} m3/s` : notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.dischargeQuantiles")}:</strong> ${
          quantileSummary([
            quantileEntry("median", channelQuantiles?.median, fmtQ, " m3/s"),
            quantileEntry("P90", channelQuantiles?.q90, fmtQ, " m3/s"),
            quantileEntry("P95", channelQuantiles?.q95, fmtQ, " m3/s"),
            quantileEntry("P99", channelQuantiles?.q99, fmtQ, " m3/s"),
          ]) || notApplicableText
        }</li>`,
        `<li><strong>${t("explorer.meta.renderedChannels")}:</strong> ${
          channelCoverage ? fmtInt(channelCoverage.channel_segment_count_unique) : notApplicableText
        }</li>`,
      ]
    : [`<li><strong>${t("explorer.meta.status")}:</strong> ${t("explorer.meta.notYetAdded", { region: region.label })}</li>`];

  const sourceLines = [renderSourceLine(t("explorer.meta.sourceBedIceGeometry"), dataset.sources.geometry)];
  if (dataset.regionKey === "antarctica" && dataset.sources.basins) {
    sourceLines.push(renderSourceLine(t("explorer.meta.sourceBasinBoundaries"), dataset.sources.basins));
  }
  if (capabilities.velocity && dataset.sources.velocity) {
    sourceLines.push(renderSourceLine(t("explorer.meta.sourceSurfaceVelocity"), dataset.sources.velocity));
  }
  if (capabilities.basalFriction && dataset.sources.basalFriction) {
    sourceLines.push(renderSourceLine(t("explorer.meta.sourceBasalFriction"), dataset.sources.basalFriction));
  }
  if (capabilities.hydrology && dataset.sources.hydrology) {
    sourceLines.push(renderSourceLine(t("explorer.meta.sourceSubglacialHydrology"), dataset.sources.hydrology));
  }
  if (capabilities.oceanCurrents && dataset.sources.oceanCurrents) {
    const oceanSuffix =
      oceanCurrentCoverage?.streamlines_by_seed_bucket ||
      oceanCurrentCoverage?.segments_by_seed_bucket ||
      oceanCurrentCoverage?.streamlines_by_seed_depth ||
      oceanCurrentCoverage?.segments_by_seed_depth ||
      oceanCurrentCoverage?.counts_by_depth
        ? ` (${fmtInt(oceanStreamlineCount)} 3D streamlines)`
        : "";
    sourceLines.push(renderSourceLine(t("explorer.meta.sourceOceanCirculation"), dataset.sources.oceanCurrents, oceanSuffix));
  }
  if (capabilities.rise && dataset.sources.rise) {
    sourceLines.push(renderSourceLine(t("explorer.meta.sourceMeltDrivers"), dataset.sources.rise, " (15,269 ice-shelf cells)"));
  }
  if (!capabilities.velocity && !capabilities.basalFriction && !capabilities.hydrology && !capabilities.refinedBasins) {
    sourceLines.push(
      `<li class="meta-compact"><strong>${t("explorer.meta.currentScope", {
        region: region.label,
      })}:</strong> ${t("explorer.meta.currentScopeSummary")}</li>`
    );
  }


  const reboundStats = currentCoreContext?.reboundStats;
  const reboundActive = Boolean(controlsUI.showIsostaticRebound?.checked);
  const fmtMillions = (value, digits = 3) =>
    Number(value).toLocaleString(numberLocale, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  const fmtArea = (value) =>
    Number.isFinite(value)
      ? t("explorer.meta.reboundMillionKm2", { value: fmtMillions(value / 1e6) })
      : notApplicableText;
  const reboundItems =
    reboundActive && reboundStats
      ? [
          `<li><strong>${t("explorer.meta.reboundModelLabel")}:</strong> ${
            reboundStats.model === "local"
              ? t("explorer.meta.reboundModelLocal")
              : t("explorer.meta.reboundModelFlexural")
          }</li>`,
          ...(reboundStats.model === "local"
            ? []
            : [
                `<li class="meta-compact"><strong>${t("explorer.meta.reboundRigidity")}:</strong> ${
                  reboundStats.flexuralRigidityNm.toExponential(0)
                } N m; <strong>${t("explorer.meta.reboundLengthScale")}:</strong> ${fmtInt(
                  reboundStats.flexuralLengthScaleKm
                )} km</li>`,
              ]),
          `<li class="meta-compact"><strong>${t("explorer.meta.reboundProgressLabel")}:</strong> ${Math.round(
            getReboundFraction() * 100
          )}%; <strong>${t("explorer.meta.reboundRelaxation")}:</strong> ${fmtInt(
            reboundStats.relaxationTimeYears
          )} yr; <strong>${t("explorer.meta.reboundSeaLevelLabel")}:</strong> +${fmtInt(
            reboundStats.seaLevelMeters
          )} m</li>`,
          `<li><strong>${t("explorer.meta.reboundMaxUplift")}:</strong> ${fmtInt(
            reboundStats.maxUpliftMeters
          )} m at ${formatCoord(reboundStats.maxUpliftXMeters / 1000, 0)}, ${formatCoord(
            reboundStats.maxUpliftYMeters / 1000,
            0
          )} km</li>`,
          `<li><strong>${t("explorer.meta.reboundMeanGroundedUplift")}:</strong> ${fmtInt(
            reboundStats.meanGroundedUpliftMeters
          )} m</li>`,
          `<li><strong>${t("explorer.meta.reboundEmergent")}:</strong> ${fmtArea(
            reboundStats.emergentAreaKm2
          )}</li>`,
          `<li class="meta-compact"><strong>${t("explorer.meta.reboundLandBefore")}:</strong> ${fmtArea(
            reboundStats.landAreaNowKm2
          )}; <strong>${t("explorer.meta.reboundLandAfter")}:</strong> ${fmtArea(
            reboundStats.landAreaAfterKm2
          )}</li>`,
          `<li><strong>${t("explorer.meta.reboundMarineUnderIce")}:</strong> ${fmtArea(
            reboundStats.marineUnderIceAfterAreaKm2
          )}</li>`,
          ...(reboundStats.closedBasinAreaKm2 > 0
            ? [
                `<li><strong>${t("explorer.meta.reboundClosedBasins")}:</strong> ${fmtArea(
                  reboundStats.closedBasinAreaKm2
                )}</li>`,
              ]
            : []),
          ...(reboundStats.submergedAreaKm2 > 0
            ? [
                `<li><strong>${t("explorer.meta.reboundDrowned")}:</strong> ${fmtInt(
                  reboundStats.submergedAreaKm2
                )} km²</li>`,
              ]
            : []),
          `<li><strong>${t("explorer.meta.reboundDeepest")}:</strong> ${fmtInt(
            reboundStats.deepestGroundedBedAfterMeters
          )} m</li>`,
          `<li class="meta-compact"><strong>${t(
            "explorer.meta.reboundVolumeAboveFlotation"
          )}:</strong> ${t("explorer.meta.reboundVolumeOf", {
            above: fmtMillions(reboundStats.volumeAboveFlotationKm3 / 1e6, 2),
            total: fmtMillions(reboundStats.iceVolumeKm3 / 1e6, 2),
          })}</li>`,
          `<li><strong>${t("explorer.meta.reboundSle")}:</strong> ${fmtShort(
            reboundStats.sleMeters,
            1
          )} m (${
            dataset.regionKey === "greenland"
              ? t("explorer.meta.reboundSlePublishedGreenland")
              : t("explorer.meta.reboundSlePublishedAntarctica")
          })</li>`,
          `<li class="meta-compact"><strong>${t("explorer.meta.reboundSolveGrid")}:</strong> ${
            reboundStats.solveCellKm > 0
              ? t("explorer.meta.reboundSolveGridValue", {
                  cell: fmtShort(reboundStats.solveCellKm, 0),
                  sizeX: reboundStats.fftSizeX,
                  sizeY: reboundStats.fftSizeY,
                })
              : t("explorer.meta.reboundSolveGridPointwise", { cell: fmtShort(stepKm, 0) })
          }; <strong>${t("explorer.meta.reboundConvergence")}:</strong> ${t(
            "explorer.meta.reboundConvergenceValue",
            {
              iterations: reboundStats.iterations,
              residual: fmtShort(reboundStats.residualMeters, 3),
            }
          )}</li>`,
          `<li><strong>${t("explorer.meta.reboundMethod")}:</strong> ${t("explorer.meta.reboundMethodText")}</li>`,
          `<li><strong>${t("explorer.meta.reboundAssumptions")}:</strong> ${t(
            "explorer.meta.reboundAssumptionsText"
          )}</li>`,
        ]
      : [];

  const selectedFlowlineCard =
    capabilities.velocity && capabilities.flowline && controlsUI.showFlowline.checked
      ? renderSelectedFlowlineCard(selectedFlowlineState)
      : "";

  if (controlsUI.flowlineProfileCardMount) {
    controlsUI.flowlineProfileCardMount.innerHTML = selectedFlowlineCard;
    controlsUI.flowlineProfileCardMount.hidden = !selectedFlowlineCard;
  }

  const sections = [
    renderMetaSection("geometry-grid", t("explorer.meta.geometrySection"), geometryItems),
    renderMetaSection("surface-velocity", t("explorer.meta.velocitySection"), velocityItems),
    renderMetaSection("basal-friction", t("explorer.meta.basalFrictionSection"), basalFrictionItems),
    renderMetaSection("subglacial-hydrology", t("explorer.meta.hydrologySection"), hydrologyItems),
    renderMetaSection("ocean-circulation", t("explorer.meta.oceanSection"), oceanCurrentItems),
    renderMetaSection("basal-melt", t("explorer.meta.riseSection"), riseItems),
    renderMetaSection("isostatic-rebound", t("explorer.meta.isostaticReboundSection"), reboundItems, {
      defaultOpen: true,
    }),
    renderMetaSection("sources", t("explorer.meta.sourcesSection"), sourceLines),
  ].filter(Boolean);

  metaListEl.innerHTML = sections.join("");
  bindMetaSectionToggles();
  if (fieldStatsEl) {
    const fieldParts = [
      `${region.label}`,
      `${dataset.label}`,
      `${t("explorer.meta.fieldStatsBed")}: ${Math.round(bedStats.min)} m to ${Math.round(bedStats.max)} m`,
      `${t("explorer.meta.fieldStatsIceMax")}: ${Math.round(thickStats.max)} m`,
      `${t("explorer.meta.fieldStatsVelocityMax")}: ${
        capabilities.velocity && speedStats ? `${Math.round(speedStats.max)} m/yr` : notApplicableText
      }`,
    ];
    if (capabilities.oceanCurrents && oceanStreamlineCount > 0) {
      fieldParts.push(`${t("explorer.meta.fieldStatsOceanStreamlines")}: ${fmtInt(oceanStreamlineCount)}`);
    }
    if (capabilities.basalFriction && controlsUI.showBasalFriction.checked && basalFrictionQuantiles) {
      fieldParts.push(
        `${t("explorer.meta.fieldStatsTauHighEnd")}: ${fmtShort(basalFrictionQuantiles.q995, 2)} MPa ${t(
          "explorer.meta.fieldStatsTauHighEndSuffix"
        )}`
      );
    }
    if (capabilities.rise && controlsUI.showBasalMelt.checked && riseBasalStats) {
      fieldParts.push(
        `${t("explorer.meta.fieldStatsBasalMeltMax")}: ${fmtShort(riseBasalStats.max, 1)} m/yr ${t(
          "explorer.meta.fieldStatsBasalMeltMaxSuffix"
        )}`
      );
    }
    if (capabilities.rise && controlsUI.showThermalDriving.checked && riseThermalStats) {
      fieldParts.push(
        `${t("explorer.meta.fieldStatsThermalDrivingMax")}: ${fmtShort(riseThermalStats.max, 2)} °C ${t(
          "explorer.meta.fieldStatsThermalDrivingMaxSuffix"
        )}`
      );
    }
    if (capabilities.refinedBasins && Number.isFinite(refinedBasinCount) && refinedBasinCount > 0) {
      fieldParts.push(`${t("explorer.meta.fieldStatsBasins")}: ${fmtInt(refinedBasinCount)}`);
    }
    fieldStatsEl.textContent = fieldParts.join(" | ");
  }
}

function updateMetaFromCurrentState() {
  updateOceanCurrentLayerControls();
  updateLegendVisibility();
  updateBasalFrictionLegend();
  updateBasalMeltLegend();
  updateThermalDrivingLegend();
  if (!currentCoreContext) {
    if (controlsUI.flowlineProfileCardMount) {
      controlsUI.flowlineProfileCardMount.innerHTML = "";
      controlsUI.flowlineProfileCardMount.hidden = true;
    }
    updateOceanCurrentLegend();
    return;
  }
  updateMeta(
    currentCoreContext.meta,
    currentCoreContext.dataset,
    currentVelocityMeta,
    currentBasalFrictionMeta,
    currentRiseMeta,
    currentHydrologyMeta,
    currentOceanCurrentMeta,
    currentVelocityMedianSpeed
  );
  updateOceanCurrentLegend();
}


// ------------------------------------------------------------------ isostatic rebound
//
// Recomputes the bed (and the residual ice riding on it) from a cached equilibrium
// uplift field. The solve itself lives in js/gia-rebound.js and runs in a dedicated
// module worker; everything below is presentation.

const REBOUND_WORKER_PATH = "gia-rebound-worker.js";
const EMERGENT_HIGHLIGHT_RGB = [0.99, 0.6, 0.21];
const EMERGENT_HIGHLIGHT_BLEND = 0.55;

function loadReboundModule() {
  if (reboundModule) return Promise.resolve(reboundModule);
  if (!reboundModulePromise) {
    reboundModulePromise = import(assetUrl("js/gia-rebound.js")).then((module) => {
      reboundModule = module;
      return module;
    });
  }
  return reboundModulePromise;
}

function ensureReboundWorker() {
  if (reboundWorker || reboundWorkerUnavailable) return reboundWorker;
  try {
    // A module worker keeps the solver a plain importable ES module, which is what
    // makes it unit-testable outside the browser.
    reboundWorker = new Worker(assetUrl(REBOUND_WORKER_PATH), { type: "module" });
  } catch (error) {
    console.warn("Isostatic-rebound worker unavailable, solving on the main thread.", error);
    reboundWorkerUnavailable = true;
    return null;
  }
  reboundWorker.addEventListener("message", (event) => {
    const data = event.data || {};
    const pending = reboundWorkerPending.get(data.id);
    if (!pending) return;
    if (data.kind === "progress") {
      if (typeof pending.onProgress === "function") {
        pending.onProgress(data.progress, localizeWorkerStage(data.stageKey || "", data.stage || ""));
      }
      return;
    }
    reboundWorkerPending.delete(data.id);
    if (data.kind === "result" && data.ok) {
      pending.resolve(data.result);
      return;
    }
    pending.reject(new Error(localizeErrorMessage(data?.error?.message || t("explorer.errors.workerTaskFailed"))));
  });
  reboundWorker.addEventListener("error", (event) => {
    const message = localizeErrorMessage(event?.message || t("explorer.errors.workerCrashed"));
    reboundWorkerPending.forEach((pending) => pending.reject(new Error(message)));
    reboundWorkerPending.clear();
    reboundWorkerUnavailable = true;
    reboundWorker = null;
  });
  return reboundWorker;
}

function terminateReboundWorker() {
  if (!reboundWorker) return;
  reboundWorker.terminate();
  reboundWorker = null;
  reboundWorkerPending.forEach((pending) => pending.reject(new Error(t("explorer.errors.workerTerminated"))));
  reboundWorkerPending.clear();
}

function runReboundWorkerTask(payload, onProgress = null) {
  const worker = ensureReboundWorker();
  if (!worker) return null;
  const id = ++reboundWorkerRequestSeq;
  return new Promise((resolve, reject) => {
    reboundWorkerPending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, task: "solveIsostaticRebound", payload });
  });
}

function isReboundCapable(regionKey = currentRegionKey, datasetKey = currentDatasetKey) {
  const region = getRegionConfig(regionKey);
  const dataset = getDatasetConfig(region.key, datasetKey);
  const capabilities = dataset?.capabilities || region.capabilities;
  return Boolean(!isShowcaseMode && capabilities.isostaticRebound);
}

function isReboundActive() {
  return Boolean(controlsUI.showIsostaticRebound?.checked && currentCoreContext?.reboundUplift);
}

function getReboundStandardParallel(regionKey = currentRegionKey) {
  return regionKey === "greenland" ? 70 : -71;
}

function getReboundFraction() {
  const value = Number(controlsUI.reboundProgress?.value);
  return Number.isFinite(value) ? clamp01(value / 100) : 1;
}

function getReboundSeaLevelMeters() {
  const value = Number(controlsUI.reboundSeaLevel?.value);
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

function getReboundModelKey() {
  return controlsUI.reboundModel?.value === "local" ? "local" : "flexural";
}

/** Identifies a cached solve; the uplift field depends on the model and the datum. */
function getReboundSolveKey() {
  return `${getReboundModelKey()}|${getReboundSeaLevelMeters()}`;
}

function blendEmergentHighlight(rgb, blend) {
  return [
    lerp(rgb[0], EMERGENT_HIGHLIGHT_RGB[0], blend),
    lerp(rgb[1], EMERGENT_HIGHLIGHT_RGB[1], blend),
    lerp(rgb[2], EMERGENT_HIGHLIGHT_RGB[2], blend),
  ];
}

/**
 * Rewrite the bed mesh's vertex heights and colours for the current scenario, plus
 * the residual ice surfaces. Vertex normals are the expensive part, so they are only
 * recomputed when the interaction settles rather than on every drag frame.
 */
function applyReboundGeometry({ recomputeNormals = true } = {}) {
  const context = currentCoreContext;
  if (!context || !bedMesh) return;

  const active = isReboundActive();
  const fraction = active ? getReboundFraction() : 0;
  const seaLevelMeters = active ? getReboundSeaLevelMeters() : 0;
  const highlight = Boolean(active && controlsUI.highlightEmergentLand?.checked);
  const {
    cellCount,
    bedHeights,
    bedValid,
    surfaceHeights,
    iceBottomHeights,
    iceBottomValid,
    thickness,
    mask,
    iceValid,
  } = context;
  const uplift = active ? context.reboundUplift : null;
  const emergent = active ? context.reboundEmergent : null;
  const verticalMetersPerUnit = context.baseConfig.verticalMetersPerUnit;

  const positions = bedMesh.geometry.getAttribute("position");
  const colors = bedMesh.geometry.getAttribute("color");
  for (let index = 0; index < cellCount; index += 1) {
    const bed = bedHeights[index];
    const height = uplift && Number.isFinite(bed) ? bed + fraction * uplift[index] : bed;
    positions.array[3 * index + 1] = bedValid[index] ? height / verticalMetersPerUnit : 0;

    // Re-datum the colour ramp so the GMT_relief land/ocean break lands on the
    // waterline rather than on the present-day zero.
    let rgb = bedColor(height - seaLevelMeters);
    if (highlight && emergent && emergent[index]) {
      rgb = blendEmergentHighlight(rgb, EMERGENT_HIGHLIGHT_BLEND * fraction);
    }
    colors.array[3 * index] = Math.round(clamp01(rgb[0]) * 255);
    colors.array[3 * index + 1] = Math.round(clamp01(rgb[1]) * 255);
    colors.array[3 * index + 2] = Math.round(clamp01(rgb[2]) * 255);
  }
  positions.needsUpdate = true;
  colors.needsUpdate = true;
  if (recomputeNormals) bedMesh.geometry.computeVertexNormals();

  if (!reboundModule || !iceMesh || !iceBottomMesh) {
    updateReboundSeaPlane();
    return;
  }

  const icePositions = iceMesh.geometry.getAttribute("position");
  const iceBottomPositions = iceBottomMesh.geometry.getAttribute("position");

  if (!active) {
    // Restore the dataset's own ice geometry rather than re-deriving it at fraction
    // zero: BedMachine's shelf draft is not exactly the hydrostatic value, so
    // re-deriving would leave the ice a few metres off its pristine position.
    for (let index = 0; index < cellCount; index += 1) {
      icePositions.array[3 * index + 1] = iceValid[index]
        ? surfaceHeights[index] / verticalMetersPerUnit
        : 0;
      iceBottomPositions.array[3 * index + 1] = iceBottomValid[index]
        ? iceBottomHeights[index] / verticalMetersPerUnit
        : 0;
    }
  } else {
    const reboundedBed = context.reboundScratchBed || new Float32Array(cellCount);
    context.reboundScratchBed = reboundedBed;
    reboundModule.applyReboundFraction({ cellCount, bedHeights, uplift, fraction, out: reboundedBed });
    const surfaces = reboundModule.deriveReboundedIceSurfaces({
      cellCount,
      reboundedBedHeights: reboundedBed,
      thickness,
      mask,
      iceValid,
      fraction,
      seaLevelMeters,
      surfaceOut: context.reboundScratchSurface || new Float32Array(cellCount),
      bottomOut: context.reboundScratchBottom || new Float32Array(cellCount),
    });
    context.reboundScratchSurface = surfaces.surface;
    context.reboundScratchBottom = surfaces.bottom;

    for (let index = 0; index < cellCount; index += 1) {
      const top = surfaces.surface[index];
      const bottom = surfaces.bottom[index];
      icePositions.array[3 * index + 1] = Number.isFinite(top) ? top / verticalMetersPerUnit : 0;
      iceBottomPositions.array[3 * index + 1] = Number.isFinite(bottom)
        ? bottom / verticalMetersPerUnit
        : 0;
    }
  }
  icePositions.needsUpdate = true;
  iceBottomPositions.needsUpdate = true;
  if (recomputeNormals) {
    iceMesh.geometry.computeVertexNormals();
    iceBottomMesh.geometry.computeVertexNormals();
  }

  // The ice side skirt is built from boundary edges rather than one vertex per cell,
  // so it cannot be rewritten in place; hide it while the scenario is running.
  if (iceSideMesh) iceSideMesh.visible = active ? false : iceSideMesh.visible;

  // Fade the ice out as it thins away. At full deglaciation the ice meshes collapse
  // onto the bed with zero thickness, and leaving them at the user's opacity would
  // veil the rebounded bed with a uniform tint.
  const iceOpacity = Number(controlsUI.iceOpacity.value);
  const remainingIce = active ? 1 - fraction : 1;
  syncIceMaterialMode(iceOpacity * remainingIce);
  if (iceBottomMesh) {
    iceBottomMesh.material.opacity = active
      ? iceOpacity * 0.82 * remainingIce
      : Math.max(0.12, iceOpacity * 0.82);
  }
  updateReboundSeaPlane();
}

function scheduleReboundGeometryUpdate({ recomputeNormals = false } = {}) {
  reboundGeometryWantsNormals = reboundGeometryWantsNormals || recomputeNormals;
  if (reboundGeometryFrame !== null) return;
  reboundGeometryFrame = window.requestAnimationFrame(() => {
    reboundGeometryFrame = null;
    const wantsNormals = reboundGeometryWantsNormals;
    reboundGeometryWantsNormals = false;
    applyReboundGeometry({ recomputeNormals: wantsNormals });
  });
}

function updateReboundSeaPlane() {
  if (!seaLevelMesh || !currentCoreContext) return;
  const seaLevelMeters = isReboundActive() ? getReboundSeaLevelMeters() : 0;
  const exaggeration = Number(controlsUI.exaggeration.value);
  // seaLevelMesh is a flat plane with no scale.y of its own, so the exaggeration the
  // terrain meshes get from scale.y has to be folded into its position instead.
  seaLevelMesh.position.y =
    (seaLevelMeters / currentCoreContext.baseConfig.verticalMetersPerUnit) * exaggeration;
}

function hideOverlaysForRebound() {
  // Every one of these overlays is baked onto the present-day bed or ice surface, so
  // it would float or sink once the bed moves. Ocean streamlines are placed against
  // the present bathymetry for the same reason.
  let changed = false;
  for (const key of [
    "showVelocity",
    "showBasalFriction",
    "showEffectivePressure",
    "showSubglacialChannels",
    "showBasalMelt",
    "showThermalDriving",
    "showOceanCurrents",
    "showFlowline",
    "showRefinedBasins",
  ]) {
    const control = controlsUI[key];
    if (control?.checked) {
      control.checked = false;
      changed = true;
    }
  }
  if (!changed) return false;
  if (velocitySurfaceMesh) velocitySurfaceMesh.visible = false;
  if (basalFrictionMesh) basalFrictionMesh.visible = false;
  if (effectivePressureMesh) effectivePressureMesh.visible = false;
  if (subglacialChannelMesh) subglacialChannelMesh.visible = false;
  updateRiseOverlayVisibility();
  updateOceanCurrentLayerControls();
  updateOceanCurrentLayerVisibility();
  updateRefinedBasinVisibility();
  updateFlowlineVisibility();
  return true;
}

function formatReboundYears(years) {
  if (!Number.isFinite(years)) return "";
  if (years >= 1000) {
    return t("explorer.rebound.kiloyears", {
      value: (years / 1000).toLocaleString(numberLocale, {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }),
    });
  }
  return t("explorer.rebound.years", { value: Math.round(years / 10) * 10 });
}

function updateReboundControlsUi() {
  const enabled = Boolean(controlsUI.showIsostaticRebound?.checked);
  if (controlsUI.isostaticReboundControls) controlsUI.isostaticReboundControls.hidden = !enabled;

  const percent = Math.round(getReboundFraction() * 100);
  if (controlsUI.reboundProgressValue) controlsUI.reboundProgressValue.textContent = `${percent}%`;
  const seaLevelMeters = getReboundSeaLevelMeters();
  if (controlsUI.reboundSeaLevelValue) {
    controlsUI.reboundSeaLevelValue.textContent = `+${seaLevelMeters} m`;
  }

  if (controlsUI.reboundProgressNote) {
    const relaxation = reboundModule?.REBOUND_RELAXATION_TIME_YEARS || 3000;
    const years = reboundModule ? reboundModule.reboundElapsedYears(getReboundFraction()) : Number.NaN;
    controlsUI.reboundProgressNote.textContent =
      percent >= 100
        ? t("explorer.rebound.progressNoteComplete", { tau: relaxation })
        : t("explorer.rebound.progressNote", {
            percent,
            elapsed: formatReboundYears(years),
            tau: relaxation,
          });
  }

  if (controlsUI.reboundModelNote) {
    const stats = currentCoreContext?.reboundStats;
    controlsUI.reboundModelNote.textContent =
      getReboundModelKey() === "local"
        ? t("explorer.rebound.modelNoteLocal")
        : t("explorer.rebound.modelNoteFlexural", {
            lengthScale: Math.round(stats?.flexuralLengthScaleKm || 133),
          });
  }

  if (controlsUI.reboundSeaLevelNote) {
    const stats = currentCoreContext?.reboundStats;
    controlsUI.reboundSeaLevelNote.textContent =
      seaLevelMeters > 0
        ? t("explorer.rebound.seaLevelNoteRaised", { datum: seaLevelMeters })
        : t("explorer.rebound.seaLevelNoteZero", {
            sle: (stats?.sleMeters || 0).toLocaleString(numberLocale, {
              minimumFractionDigits: 1,
              maximumFractionDigits: 1,
            }),
          });
  }
}

async function ensureIsostaticReboundLoaded({ trigger = "toggle" } = {}) {
  const context = currentCoreContext;
  if (!context || context.generation !== loadGeneration) return false;
  if (!isReboundCapable()) {
    context.reboundUnavailable = true;
    return false;
  }
  const solveKey = getReboundSolveKey();
  if (context.reboundUplift && context.reboundSolveKey === solveKey) return true;
  if (reboundLoadPromise) {
    // A solve is already running for whatever the controls said when it started.
    // Wait for it, then re-check: if the user has since changed the model or the
    // datum, that in-flight result is stale and a fresh solve is needed. Returning
    // the running promise directly would leave the display pinned to the old
    // settings with no way to recover.
    const settled = await reboundLoadPromise;
    if (
      context === currentCoreContext &&
      context.generation === loadGeneration &&
      !context.reboundUnavailable &&
      context.reboundSolveKey !== getReboundSolveKey()
    ) {
      return ensureIsostaticReboundLoaded({ trigger });
    }
    return settled;
  }

  const showOverlay = trigger === "toggle" || trigger === "resolve";
  reboundLoadPromise = (async () => {
    if (showOverlay) {
      setLoadingOverlayVisible(true);
      updateLoadingProgress(0.1, t("explorer.loading.solvingIsostaticRebound"));
      statusEl.textContent = t("explorer.status.loadingIsostaticRebound");
    }

    const module = await loadReboundModule();
    if (context !== currentCoreContext || context.generation !== loadGeneration) return false;

    const payload = {
      nx: context.nx,
      ny: context.ny,
      cellCount: context.cellCount,
      grid: context.meta.grid,
      bedHeights: context.bedHeights,
      surfaceHeights: context.surfaceHeights,
      thickness: context.thickness,
      mask: context.mask,
      model: getReboundModelKey(),
      seaLevelMeters: getReboundSeaLevelMeters(),
      standardParallelDegrees: getReboundStandardParallel(context.dataset.regionKey),
      reportProgress: showOverlay,
    };
    const onProgress = showOverlay
      ? (progress, stage) => {
          updateLoadingProgress(
            0.12 + clamp01(progress) * 0.82,
            stage || t("explorer.loading.solvingIsostaticRebound")
          );
        }
      : null;

    let result = null;
    const workerTask = runReboundWorkerTask(payload, onProgress);
    if (workerTask) {
      try {
        result = await workerTask;
      } catch (workerError) {
        console.warn("Isostatic-rebound worker failed, retrying on the main thread.", workerError);
        reboundWorkerUnavailable = true;
      }
    }
    if (!result) {
      // Main-thread fallback: a sub-second blocking solve behind the loading overlay
      // is preferable to losing the layer on browsers without module workers.
      await nextAnimationFrame();
      result = module.solveIsostaticRebound({ ...payload, onProgress: null });
    }
    if (context !== currentCoreContext || context.generation !== loadGeneration) return false;

    context.reboundUplift = result.uplift;
    context.reboundEmergent = result.emergent;
    context.reboundStats = result.stats;
    context.reboundSolveKey = solveKey;
    context.reboundUnavailable = false;

    applyReboundGeometry({ recomputeNormals: true });
    updateReboundControlsUi();
    updateMetaFromCurrentState();
    if (showOverlay) {
      statusEl.textContent = getReadyStatusText(context);
      updateLoadingProgress(1, t("explorer.loading.isostaticReboundReady"));
      setLoadingOverlayVisible(false);
    }
    return true;
  })()
    .catch((error) => {
      console.error("Isostatic-rebound solve failed:", error);
      if (context === currentCoreContext && context.generation === loadGeneration) {
        context.reboundUnavailable = true;
        if (controlsUI.showIsostaticRebound) {
          controlsUI.showIsostaticRebound.checked = false;
          controlsUI.showIsostaticRebound.disabled = true;
        }
        updateReboundControlsUi();
        applyReboundGeometry({ recomputeNormals: true });
        setTransientStatus(t("explorer.status.isostaticReboundUnavailable"));
        setLoadingOverlayVisible(false);
      }
      return false;
    })
    .finally(() => {
      if (context === currentCoreContext) reboundLoadPromise = null;
    });

  return reboundLoadPromise;
}

function buildCoreSceneFromContext(context) {
  const { baseConfig, bedHeights, bedValid, surfaceHeights, iceValid, thickness, mask, iceBottomHeights, iceBottomValid } =
    context;
  const exaggeration = Number(controlsUI.exaggeration.value);
  const wireframe = controlsUI.wireframe.checked;

  clearModelMeshes();

  const bedGeometry = buildSurfaceGeometry({
    ...baseConfig,
    heights: bedHeights,
    valid: bedValid,
    colorFn: (height) => bedColor(height),
  });
  const bedMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.92,
    metalness: 0.02,
    flatShading: false,
  });
  bedMesh = new THREE.Mesh(bedGeometry, bedMaterial);
  bedMesh.scale.y = exaggeration;
  bedMesh.visible = controlsUI.showBed.checked;
  bedMesh.material.wireframe = wireframe;
  bedMesh.renderOrder = 1;
  scene.add(bedMesh);

  const iceGeometry = buildSurfaceGeometry({
    ...baseConfig,
    heights: surfaceHeights,
    valid: iceValid,
    extraField: thickness,
    mask,
    colorFn: (_height, thick, maskValue) => iceColor(thick, maskValue),
  });
  const iceMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    transparent: true,
    opacity: Number(controlsUI.iceOpacity.value),
    depthWrite: false,
    roughness: 0.38,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  iceMesh = new THREE.Mesh(iceGeometry, iceMaterial);
  iceMesh.scale.y = exaggeration;
  iceMesh.visible = controlsUI.showIce.checked;
  iceMesh.material.wireframe = wireframe;
  iceMesh.renderOrder = ICE_SURFACE_RENDER_ORDER;
  scene.add(iceMesh);

  const iceBottomGeometry = buildSurfaceGeometry({
    ...baseConfig,
    heights: iceBottomHeights,
    valid: iceBottomValid,
    extraField: thickness,
    mask,
    colorFn: (_height, thick, maskValue) => iceBottomColor(thick, maskValue),
  });
  const iceBottomMaterial = new THREE.MeshStandardMaterial({
    vertexColors: true,
    transparent: true,
    opacity: Math.max(0.12, Number(controlsUI.iceOpacity.value) * 0.82),
    depthWrite: false,
    roughness: 0.44,
    metalness: 0.02,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  iceBottomMesh = new THREE.Mesh(iceBottomGeometry, iceBottomMaterial);
  iceBottomMesh.scale.y = exaggeration;
  iceBottomMesh.visible = controlsUI.showIceBottom.checked;
  iceBottomMesh.material.wireframe = wireframe;
  iceBottomMesh.renderOrder = ICE_BOTTOM_RENDER_ORDER;
  scene.add(iceBottomMesh);

  const iceSideGeometry = buildIceSideGeometry({
    ...baseConfig,
    topHeights: surfaceHeights,
    bottomHeights: iceBottomHeights,
    valid: iceValid,
    extraField: thickness,
    mask,
    colorFn: (_height, thick, maskValue) => iceColor(thick, maskValue),
  });
  const iceSideMaterial = iceMaterial.clone();
  iceSideMaterial.opacity = Number(controlsUI.iceOpacity.value);
  iceSideMaterial.depthWrite = false;
  iceSideMesh = new THREE.Mesh(iceSideGeometry, iceSideMaterial);
  iceSideMesh.scale.y = exaggeration;
  iceSideMesh.material.wireframe = wireframe;
  iceSideMesh.renderOrder = ICE_SIDE_STRICT_OCCLUSION_RENDER_ORDER;
  updateIceSideVisibility();
  syncIceMaterialMode(Number(controlsUI.iceOpacity.value));
  scene.add(iceSideMesh);

  const widthUnits = ((context.nx - 1) * Math.abs(context.meta.grid.dx_m)) / baseConfig.horizontalMetersPerUnit;
  const depthUnits = ((context.ny - 1) * Math.abs(context.meta.grid.dy_m)) / baseConfig.horizontalMetersPerUnit;
  const seaGeometry = new THREE.PlaneGeometry(widthUnits, depthUnits, 1, 1);
  seaGeometry.rotateX(-Math.PI / 2);
  const seaMaterial = new THREE.MeshBasicMaterial({
    color: 0x4ca8d8,
    transparent: true,
    opacity: 0.13,
    side: THREE.DoubleSide,
    depthWrite: false,
    depthTest: true,
  });
  seaLevelMesh = new THREE.Mesh(seaGeometry, seaMaterial);
  seaLevelMesh.position.y = 0;
  seaLevelMesh.renderOrder = 24;
  seaLevelMesh.visible = controlsUI.showSea.checked;
  scene.add(seaLevelMesh);
}

function buildRiseOverlayGeometry(context, heights, validMask) {
  const vertexCount = context.cellCount;
  const positions = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const indices = [];
  const halfX = (context.nx - 1) / 2;
  const halfY = (context.ny - 1) / 2;
  const absDy = Math.abs(context.meta.grid.dy_m);

  for (let row = 0; row < context.ny; row += 1) {
    for (let col = 0; col < context.nx; col += 1) {
      const index = row * context.nx + col;
      positions[3 * index] = ((col - halfX) * context.meta.grid.dx_m) / context.baseConfig.horizontalMetersPerUnit;
      positions[3 * index + 1] = validMask[index] ? heights[index] / context.baseConfig.verticalMetersPerUnit : 0;
      positions[3 * index + 2] = ((row - halfY) * absDy) / context.baseConfig.horizontalMetersPerUnit;
      uvs[2 * index] = col / Math.max(1, context.nx - 1);
      uvs[2 * index + 1] = row / Math.max(1, context.ny - 1);
    }
  }

  for (let row = 0; row < context.ny - 1; row += 1) {
    for (let col = 0; col < context.nx - 1; col += 1) {
      const i0 = row * context.nx + col;
      const i1 = i0 + 1;
      const i2 = i0 + context.nx;
      const i3 = i2 + 1;
      if (validMask[i0] && validMask[i2] && validMask[i1]) {
        indices.push(i0, i2, i1);
      }
      if (validMask[i1] && validMask[i2] && validMask[i3]) {
        indices.push(i1, i2, i3);
      }
    }
  }

  if (!indices.length) {
    return null;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  return geometry;
}

function buildRiseOverlayColorAttribute(values, validMask, colorFn) {
  const colors = new Uint8Array(validMask.length * 3);
  for (let i = 0; i < validMask.length; i += 1) {
    if (!validMask[i]) continue;
    const rgb = colorFn(values[i]);
    colors[3 * i] = Math.round(clamp01(rgb[0]) * 255);
    colors[3 * i + 1] = Math.round(clamp01(rgb[1]) * 255);
    colors[3 * i + 2] = Math.round(clamp01(rgb[2]) * 255);
  }
  return new THREE.BufferAttribute(colors, 3, true);
}

function buildRiseOverlayMesh(context, values, { kind, riseMeta, fallbackColorFn }) {
  if (!context || !values) {
    return { mesh: null, dataTexture: null };
  }

  const overlayHeights = new Float32Array(context.cellCount);
  const overlayValid = new Uint8Array(context.cellCount);
  for (let i = 0; i < context.cellCount; i += 1) {
    const isShelf = context.iceBottomValid[i] && context.mask[i] === 3 && context.riseMask?.[i] === 2;
    const value = values[i];
    if (isShelf && Number.isFinite(value)) {
      overlayHeights[i] = context.iceBottomHeights[i] + RISE_SURFACE_OFFSET_M;
      overlayValid[i] = 1;
    }
  }

  const geometry = buildRiseOverlayGeometry(context, overlayHeights, overlayValid);
  if (!geometry) {
    return { mesh: null, dataTexture: null };
  }

  let dataTexture = null;
  let material = null;

  try {
    dataTexture = createScalarFieldDataTexture(values, overlayValid, context.nx, context.ny, renderer);
    if (dataTexture) {
      material = createRiseFieldShaderMaterial(dataTexture, context.nx, context.ny, { kind, riseMeta });
    }
  } catch (error) {
    console.warn(`Continuous ${kind} sampling failed; using vertex-colored fallback.`, error);
  }

  if (!material) {
    dataTexture = disposeTexture(dataTexture);
    if (!riseContinuousSamplingWarningIssued) {
      console.warn("Continuous RISE sampling unavailable; using vertex-colored overlay fallback.");
      riseContinuousSamplingWarningIssued = true;
    }
    geometry.setAttribute("color", buildRiseOverlayColorAttribute(values, overlayValid, fallbackColorFn));
    material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.98,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
      depthWrite: false,
    });
  }

  const mesh = new THREE.Mesh(geometry, material);
  mesh.scale.y = Number(controlsUI.exaggeration.value);
  mesh.material.wireframe = controlsUI.wireframe.checked;
  mesh.renderOrder = 10;
  return { mesh, dataTexture };
}

function updateRiseOverlayVisibility() {
  if (basalMeltMesh) basalMeltMesh.visible = controlsUI.showBasalMelt.checked;
  if (thermalDrivingMesh) thermalDrivingMesh.visible = controlsUI.showThermalDriving.checked;
}

async function ensureHydrologyColorTables() {
  if (effectivePressureLut.length && channelDischargeLut.length) return;
  try {
    if (!effectivePressureLut.length) {
      await loadEffectivePressureColorTable();
    }
  } catch (colorTableError) {
    console.warn("Failed to load cmocean_dense.rgb, using fallback effective-pressure colors.", colorTableError);
    effectivePressureLut = [];
  }
  try {
    if (!channelDischargeLut.length) {
      await loadChannelDischargeColorTable();
    }
  } catch (colorTableError) {
    console.warn("Failed to load cmocean_matter.rgb, using fallback channel colors.", colorTableError);
    channelDischargeLut = [];
  }
  updateEffectivePressureLegend();
  updateChannelLegend();
}

function primeLayerMetadata(context) {
  if (!context) return;

  const velocityMetaPromise = !context.dataset.velocityMetaUrl
    ? Promise.resolve(null)
    : context.velocityPrefetchMeta
    ? Promise.resolve(context.velocityPrefetchMeta)
    : fetchJsonStrict(
        context.dataset.velocityMetaUrl,
        errorLabel("explorer.errors.failedToLoadVelocityMetadata")
      ).catch((error) => {
        console.warn("Velocity metadata prefetch failed:", error);
        return null;
      });

  const basalFrictionMetaPromise = !context.dataset.basalFrictionMetaUrl
    ? Promise.resolve(null)
    : context.basalFrictionPrefetchMeta
    ? Promise.resolve(context.basalFrictionPrefetchMeta)
    : fetchJsonStrict(
        context.dataset.basalFrictionMetaUrl,
        errorLabel("explorer.errors.failedToLoadBasalFrictionMetadata")
      ).catch((error) => {
        console.warn("Basal-friction metadata prefetch failed:", error);
        return null;
      });

  const hydrologyMetaPromise = !context.dataset.hydrologyMetaUrl
    ? Promise.resolve(null)
    : context.hydrologyPrefetchMeta
    ? Promise.resolve(context.hydrologyPrefetchMeta)
    : fetchJsonStrict(
        context.dataset.hydrologyMetaUrl,
        errorLabel("explorer.errors.failedToLoadHydrologyMetadata")
      ).catch((error) => {
        console.warn("Hydrology metadata prefetch failed:", error);
        return null;
      });

  const riseMetaPromise = !context.dataset.riseMetaUrl
    ? Promise.resolve(null)
    : context.risePrefetchMeta
    ? Promise.resolve(context.risePrefetchMeta)
    : fetchJsonStrict(
        context.dataset.riseMetaUrl,
        errorLabel("explorer.errors.failedToLoadRiseMetadata")
      ).catch((error) => {
        console.warn("RISE metadata prefetch failed:", error);
        return null;
      });

  const oceanCurrentMetaPromise = !context.dataset.oceanCurrentsMetaUrl
    ? Promise.resolve(null)
    : context.oceanCurrentPrefetchMeta
    ? Promise.resolve(context.oceanCurrentPrefetchMeta)
    : fetchJsonStrict(
        context.dataset.oceanCurrentsMetaUrl,
        errorLabel("explorer.errors.failedToLoadOceanCurrentMetadata")
      ).catch((error) => {
        console.warn("Ocean-current metadata prefetch failed:", error);
        return null;
      });

  Promise.all([
    velocityMetaPromise,
    basalFrictionMetaPromise,
    riseMetaPromise,
    hydrologyMetaPromise,
    oceanCurrentMetaPromise,
  ]).then(
    ([velocityMeta, basalFrictionMeta, riseMeta, hydrologyMeta, oceanCurrentMeta]) => {
    if (context !== currentCoreContext || context.generation !== loadGeneration) return;

    let hasUpdate = false;
    if (velocityMeta) {
      context.velocityPrefetchMeta = velocityMeta;
      currentVelocityMeta = velocityMeta;
      hasUpdate = true;
    }
    if (basalFrictionMeta) {
      context.basalFrictionPrefetchMeta = basalFrictionMeta;
      currentBasalFrictionMeta = basalFrictionMeta;
      hasUpdate = true;
    }
    if (riseMeta) {
      context.risePrefetchMeta = riseMeta;
      currentRiseMeta = riseMeta;
      hasUpdate = true;
    }
    if (hydrologyMeta) {
      context.hydrologyPrefetchMeta = hydrologyMeta;
      currentHydrologyMeta = hydrologyMeta;
      hasUpdate = true;
    }
    if (oceanCurrentMeta) {
      context.oceanCurrentPrefetchMeta = oceanCurrentMeta;
      currentOceanCurrentMeta = oceanCurrentMeta;
      hasUpdate = true;
    }
    if (hasUpdate) {
      updateMetaFromCurrentState();
    }
  }
  );
}

function getVelocityLayerMeshStride(context) {
  if (!context?.dataset) return 1;
  const datasetStride = Number(context.dataset.velocityMeshStride);
  if (Number.isFinite(datasetStride) && datasetStride > 0) {
    return Math.max(1, Math.round(datasetStride));
  }
  return context.dataset.id === "hd" || context.dataset.id === "bedmap3-hd" ? HD_VELOCITY_MESH_STRIDE : 1;
}

function getHydrologyLayerMeshStride(context) {
  if (!context?.dataset) return 1;
  return context.dataset.id === "hd" || context.dataset.id === "bedmap3-hd" ? HD_HYDROLOGY_MESH_STRIDE : 1;
}

function gridsMatch(left, right) {
  return ["nx", "ny", "x0_m", "y0_m", "dx_m", "dy_m"].every(
    (key) => Number(left?.[key]) === Number(right?.[key])
  );
}

function getBasalFrictionLayerMeshStride(context) {
  return getHydrologyLayerMeshStride(context);
}

async function buildVelocityLayerFromPayload(
  context,
  velocityMeta,
  velocityBuffer,
  { showOverlay = false, updateStatus = true } = {}
) {
  if (!gridsMatch(velocityMeta.grid, context.meta.grid)) {
    throw new Error("Velocity grid is not aligned to the active terrain grid.");
  }
  if (!(velocityBuffer instanceof ArrayBuffer)) {
    throw new Error("Velocity payload is invalid.");
  }

  if (showOverlay) {
    updateLoadingProgress(0.58, t("explorer.loading.decodingVelocityField"));
  }

  const workerResult = await runGeometryWorkerTask(
    "buildVelocity",
    {
      velocityMeta,
      velocityBuffer,
      surfaceHeights: context.surfaceHeights,
      iceValid: context.iceValid,
      nx: context.nx,
      ny: context.ny,
      grid: context.meta.grid,
      cellCount: context.cellCount,
      baseConfig: context.baseConfig,
      meshStride: getVelocityLayerMeshStride(context),
      reportProgress: showOverlay,
    },
    {
      transfer: [velocityBuffer],
      onProgress: showOverlay
        ? (progress, stage) => {
            const p = 0.58 + clamp01(progress) * 0.38;
            updateLoadingProgress(p, stage || t("explorer.loading.buildingVelocityMesh"));
          }
        : null,
    }
  );
  if (context !== currentCoreContext || context.generation !== loadGeneration) return false;

  velocitySurfaceMesh = disposeMesh(velocitySurfaceMesh);
  flowlineMesh = disposeMesh(flowlineMesh);
  selectedFlowlineHighlight = disposeObject3D(selectedFlowlineHighlight);
  selectedFlowlineState = null;
  velocityDataTexture = disposeTexture(velocityDataTexture);
  const velocitySurfaceGeometry = new THREE.BufferGeometry();
  velocitySurfaceGeometry.setAttribute("position", new THREE.BufferAttribute(workerResult.positions, 3));
  velocitySurfaceGeometry.setAttribute("uv", new THREE.BufferAttribute(workerResult.uvs, 2));
  velocitySurfaceGeometry.setIndex(new THREE.BufferAttribute(workerResult.indices, 1));
  let velocitySurfaceMaterial = null;
  if (supportsContinuousVelocitySampling(renderer, context.nx, context.ny)) {
    velocityDataTexture = createVelocityDataTexture(workerResult, context.nx, context.ny, renderer);
    if (velocityDataTexture) {
      velocitySurfaceMaterial = createVelocitySurfaceShaderMaterial(velocityDataTexture, context.nx, context.ny);
    }
  }
  if (!velocitySurfaceMaterial) {
    if (!velocityContinuousSamplingWarningIssued) {
      console.warn("Continuous velocity sampling unavailable; using vertex-colored velocity surface fallback.");
      velocityContinuousSamplingWarningIssued = true;
    }
    velocitySurfaceGeometry.setAttribute("color", new THREE.BufferAttribute(workerResult.colors, 3, true));
    velocitySurfaceMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: VELOCITY_SURFACE_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
  }
  velocitySurfaceMesh = new THREE.Mesh(velocitySurfaceGeometry, velocitySurfaceMaterial);
  velocitySurfaceMesh.scale.y = Number(controlsUI.exaggeration.value);
  velocitySurfaceMesh.visible = controlsUI.showVelocity.checked;
  velocitySurfaceMesh.material.wireframe = controlsUI.wireframe.checked;
  velocitySurfaceMesh.renderOrder = 12;
  scene.add(velocitySurfaceMesh);

  velocityField = {
    nx: context.nx,
    ny: context.ny,
    dxMeters: context.meta.grid.dx_m,
    dyMeters: context.meta.grid.dy_m,
    horizontalMetersPerUnit: context.baseConfig.horizontalMetersPerUnit,
    verticalMetersPerUnit: context.baseConfig.verticalMetersPerUnit,
    surfaceHeights: context.surfaceHeights,
    iceValid: context.iceValid,
    iceBottomHeights: context.iceBottomHeights,
    iceBottomValid: context.iceBottomValid,
    bedHeights: context.bedHeights,
    bedValid: context.bedValid,
    thickness: context.thickness,
    velocityX: workerResult.velocityX,
    velocityY: workerResult.velocityY,
    velocitySpeed: workerResult.velocitySpeed,
    velocityValid: workerResult.velocityValid,
  };

  context.velocityLoaded = true;
  context.velocityPrefetchMeta = null;
  context.velocityPrefetchBuffer = null;
  currentVelocityMeta = velocityMeta;
  currentVelocityMedianSpeed = Number(workerResult.velocityMedianSpeed);
  updateMetaFromCurrentState();
  updateFlowlineVisibility();
  if (updateStatus) {
    statusEl.textContent = getReadyStatusText(context);
  }
  if (showOverlay) {
    updateLoadingProgress(1, t("explorer.loading.velocityLayerReady"));
    setLoadingOverlayVisible(false);
  }
  return true;
}

async function ensureBasalFrictionLoaded({ trigger = "prefetch" } = {}) {
  const context = currentCoreContext;
  if (!context || context.generation !== loadGeneration) return false;
  if (!context.dataset.capabilities.basalFriction || !context.dataset.basalFrictionMetaUrl || !context.dataset.basalFrictionBinUrl) {
    context.basalFrictionUnavailable = true;
    return false;
  }
  if (context.basalFrictionLoaded) {
    if (basalFrictionMesh) basalFrictionMesh.visible = controlsUI.showBasalFriction.checked;
    return true;
  }
  if (context.basalFrictionUnavailable) return false;

  const showOverlay = trigger === "toggle";
  const buildLayer = trigger === "toggle" || trigger === "warmup";

  if (basalFrictionLoadPromise) {
    const settled = await basalFrictionLoadPromise;
    if (
      trigger === "toggle" &&
      context === currentCoreContext &&
      context.generation === loadGeneration &&
      !context.basalFrictionLoaded &&
      !context.basalFrictionUnavailable
    ) {
      return ensureBasalFrictionLoaded({ trigger: "toggle" });
    }
    return settled;
  }

  basalFrictionLoadPromise = (async () => {
    if (showOverlay) {
      setLoadingOverlayVisible(true);
      updateLoadingProgress(0.08, t("explorer.loading.loadingBasalFrictionMetadata"));
      statusEl.textContent = t("explorer.status.loadingBasalFriction");
    }

    let basalFrictionMeta = context.basalFrictionPrefetchMeta;
    let basalFrictionBuffer = context.basalFrictionPrefetchBuffer;
    if (!basalFrictionMeta) {
      basalFrictionMeta = await fetchJsonStrict(
        context.dataset.basalFrictionMetaUrl,
        errorLabel("explorer.errors.failedToLoadBasalFrictionMetadata")
      );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }
    if (context === currentCoreContext && context.generation === loadGeneration && basalFrictionMeta) {
      currentBasalFrictionMeta = basalFrictionMeta;
      updateBasalFrictionLegend();
      updateMetaFromCurrentState();
    }

    if (!basalFrictionBuffer) {
      basalFrictionBuffer = showOverlay
        ? await fetchArrayBufferWithProgress(
            context.dataset.basalFrictionBinUrl,
            0.18,
            0.74,
            t("explorer.loading.downloadingBasalFrictionField"),
            errorLabel("explorer.errors.failedToLoadBasalFrictionField")
          )
        : await fetchArrayBufferStrict(
            context.dataset.basalFrictionBinUrl,
            errorLabel("explorer.errors.failedToLoadBasalFrictionField")
          );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }

    if (!buildLayer) {
      context.basalFrictionPrefetchMeta = basalFrictionMeta;
      context.basalFrictionPrefetchBuffer = basalFrictionBuffer;
      return true;
    }

    context.basalFrictionPrefetchMeta = null;
    context.basalFrictionPrefetchBuffer = null;
    if (!gridsMatch(basalFrictionMeta.grid, context.meta.grid)) {
      throw new Error("Basal-friction grid is not aligned to the active terrain grid.");
    }

    if (showOverlay) {
      updateLoadingProgress(0.76, t("explorer.loading.processingBasalFrictionField"));
    }

    const workerResult = await runGeometryWorkerTask(
      "buildBasalFriction",
      {
        basalFrictionMeta,
        basalFrictionBuffer,
        nx: context.nx,
        ny: context.ny,
        grid: context.meta.grid,
        cellCount: context.cellCount,
        baseConfig: context.baseConfig,
        bedHeights: context.bedHeights,
        bedValid: context.bedValid,
        mask: context.mask,
        meshStride: getBasalFrictionLayerMeshStride(context),
        reportProgress: showOverlay,
      },
      {
        transfer: [basalFrictionBuffer],
        onProgress: showOverlay
          ? (progress, stage) => {
              const p = 0.76 + clamp01(progress) * 0.22;
              updateLoadingProgress(p, stage || t("explorer.loading.processingBasalFrictionField"));
            }
          : null,
      }
    );
    if (context !== currentCoreContext || context.generation !== loadGeneration) return false;

    basalFrictionMesh = disposeMesh(basalFrictionMesh);
    const meshStride = getBasalFrictionLayerMeshStride(context);
    const polygonOffset = meshStride > 1 ? -4 : -2;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(workerResult.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(workerResult.colors, 3, true));
    geometry.setIndex(new THREE.BufferAttribute(workerResult.indices, 1));
    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: polygonOffset,
      polygonOffsetUnits: polygonOffset,
    });
    basalFrictionMesh = new THREE.Mesh(geometry, material);
    basalFrictionMesh.scale.y = Number(controlsUI.exaggeration.value);
    basalFrictionMesh.visible = controlsUI.showBasalFriction.checked;
    basalFrictionMesh.material.wireframe = controlsUI.wireframe.checked;
    basalFrictionMesh.renderOrder = 6;
    scene.add(basalFrictionMesh);

    context.basalFrictionLoaded = true;
    context.basalFrictionPrefetchMeta = null;
    context.basalFrictionPrefetchBuffer = null;
    currentBasalFrictionMeta = basalFrictionMeta;
    updateBasalFrictionLegend();
    updateMetaFromCurrentState();
    if (showOverlay) {
      statusEl.textContent = getReadyStatusText(context);
      updateLoadingProgress(1, t("explorer.loading.basalFrictionLayerReady"));
      setLoadingOverlayVisible(false);
    }
    return true;
  })()
    .catch((error) => {
      console.error("Basal-friction layer load failed:", error);
      if (context === currentCoreContext && context.generation === loadGeneration && trigger === "toggle") {
        context.basalFrictionUnavailable = true;
        controlsUI.showBasalFriction.checked = false;
        controlsUI.showBasalFriction.disabled = true;
        basalFrictionMesh = disposeMesh(basalFrictionMesh);
        const readyText = getReadyStatusText(context);
        const transientText = t("explorer.status.basalFrictionUnavailable");
        statusEl.textContent = transientText;
        window.setTimeout(() => {
          if (statusEl.textContent === transientText) {
            statusEl.textContent = readyText;
          }
        }, 1800);
        setLoadingOverlayVisible(false);
      }
      return false;
    })
    .finally(() => {
      if (context === currentCoreContext) {
        basalFrictionLoadPromise = null;
      }
    });

  return basalFrictionLoadPromise;
}

async function ensureRiseLoaded({ trigger = "prefetch" } = {}) {
  const context = currentCoreContext;
  if (!context || context.generation !== loadGeneration) return false;
  if (!context.dataset.capabilities.rise || !context.dataset.riseMetaUrl || !context.dataset.riseBinUrl) {
    context.riseUnavailable = true;
    return false;
  }
  if (context.riseLoaded) {
    updateRiseOverlayVisibility();
    return true;
  }
  if (context.riseUnavailable) return false;

  const showOverlay = trigger === "toggle";

  if (riseLoadPromise) {
    const settled = await riseLoadPromise;
    if (trigger === "toggle" && context === currentCoreContext && context.generation === loadGeneration) {
      updateRiseOverlayVisibility();
    }
    return settled;
  }

  riseLoadPromise = (async () => {
    try {
      if (showOverlay) {
        updateLoadingProgress(0.56, t("explorer.loading.loadingRiseMetadata"));
        setLoadingOverlayVisible(true);
      }

      const riseMeta =
        context.risePrefetchMeta ||
        (await fetchJsonStrict(
          context.dataset.riseMetaUrl,
          errorLabel("explorer.errors.failedToLoadRiseMetadata")
        ));
      if (showOverlay) {
        updateLoadingProgress(0.68, t("explorer.loading.downloadingRiseOverlayPackage"));
      }
      const riseBuffer =
        context.risePrefetchBuffer ||
        (showOverlay
          ? await fetchArrayBufferWithProgress(
              context.dataset.riseBinUrl,
              0.68,
              0.9,
              t("explorer.loading.downloadingRiseOverlayPackage"),
              errorLabel("explorer.errors.failedToLoadRiseOverlayPackage")
            )
          : await fetchArrayBufferStrict(
              context.dataset.riseBinUrl,
              errorLabel("explorer.errors.failedToLoadRiseOverlayPackage")
            ));

      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
      if (riseMeta.grid.nx !== context.nx || riseMeta.grid.ny !== context.ny) {
        throw new Error("RISE grid is not aligned to the active BedMachine grid.");
      }

      const riseMask = parseField(riseMeta, riseBuffer, "mask");
      const riseIceshelfId = parseField(riseMeta, riseBuffer, "iceshelf_id");
      const riseIceDraft = decodeFieldToFloat32(riseMeta, riseBuffer, "zice");
      const riseBasalMelt = decodeFieldToFloat32(riseMeta, riseBuffer, "ismr");
      const riseThermalDriving = decodeFieldToFloat32(riseMeta, riseBuffer, "tstar_zice");

      if (
        riseMask.length !== context.cellCount ||
        riseIceshelfId.length !== context.cellCount ||
        riseIceDraft.length !== context.cellCount ||
        riseBasalMelt.length !== context.cellCount ||
        riseThermalDriving.length !== context.cellCount
      ) {
        throw new Error("Unexpected field length in the RISE package.");
      }

      context.riseMask = riseMask;
      context.riseIceshelfId = riseIceshelfId;
      context.riseIceDraft = riseIceDraft;
      context.riseBasalMelt = riseBasalMelt;
      context.riseThermalDriving = riseThermalDriving;

      basalMeltMesh = disposeMesh(basalMeltMesh);
      basalMeltDataTexture = disposeTexture(basalMeltDataTexture);
      thermalDrivingMesh = disposeMesh(thermalDrivingMesh);
      thermalDrivingDataTexture = disposeTexture(thermalDrivingDataTexture);

      if (showOverlay) {
        updateLoadingProgress(0.93, t("explorer.loading.buildingRiseOverlayMeshes"));
      }

      const basalMeltLayer = buildRiseOverlayMesh(context, riseBasalMelt, {
        kind: "basal_melt",
        riseMeta,
        fallbackColorFn: basalMeltColor,
      });
      const thermalDrivingLayer = buildRiseOverlayMesh(context, riseThermalDriving, {
        kind: "thermal_driving",
        riseMeta,
        fallbackColorFn: thermalDrivingColor,
      });
      basalMeltMesh = basalMeltLayer.mesh;
      basalMeltDataTexture = basalMeltLayer.dataTexture;
      thermalDrivingMesh = thermalDrivingLayer.mesh;
      thermalDrivingDataTexture = thermalDrivingLayer.dataTexture;

      if (basalMeltMesh) scene.add(basalMeltMesh);
      if (thermalDrivingMesh) scene.add(thermalDrivingMesh);

      context.riseLoaded = true;
      context.risePrefetchMeta = null;
      context.risePrefetchBuffer = null;
      currentRiseMeta = riseMeta;
      updateRiseOverlayVisibility();
      updateMetaFromCurrentState();
      statusEl.textContent = getReadyStatusText(context);
      if (showOverlay) {
        updateLoadingProgress(1, t("explorer.loading.riseOverlaysReady"));
        setLoadingOverlayVisible(false);
      }
      return true;
    } catch (error) {
      if (context === currentCoreContext && context.generation === loadGeneration) {
        context.riseUnavailable = true;
        controlsUI.showBasalMelt.checked = false;
        controlsUI.showThermalDriving.checked = false;
        basalMeltMesh = disposeMesh(basalMeltMesh);
        basalMeltDataTexture = disposeTexture(basalMeltDataTexture);
        thermalDrivingMesh = disposeMesh(thermalDrivingMesh);
        thermalDrivingDataTexture = disposeTexture(thermalDrivingDataTexture);
        updateMetaFromCurrentState();
      }
      if (showOverlay) {
        reportLoadError(error);
      } else {
        console.warn("RISE overlay load failed:", error);
      }
      return false;
    } finally {
      if (context === currentCoreContext && context.generation === loadGeneration) {
        riseLoadPromise = null;
      }
    }
  })();
  return riseLoadPromise;
}

async function ensureOceanCurrentsLoaded({ trigger = "prefetch" } = {}) {
  const context = currentCoreContext;
  if (!context || context.generation !== loadGeneration) return false;
  if (
    !context.dataset.capabilities.oceanCurrents ||
    !context.dataset.oceanCurrentsMetaUrl ||
    !context.dataset.oceanCurrentsBinUrl
  ) {
    context.oceanCurrentUnavailable = true;
    return false;
  }
  if (context.oceanCurrentLoaded) {
    updateOceanCurrentLayerVisibility();
    return true;
  }
  if (context.oceanCurrentUnavailable) return false;

  const updateStatus = trigger === "toggle";
  const showOverlay = updateStatus && OCEAN_CURRENT_BLOCKING_OVERLAY_ENABLED;
  const buildLayer = trigger === "toggle" || trigger === "warmup";

  if (oceanCurrentLoadPromise) {
    const settled = await oceanCurrentLoadPromise;
    if (
      trigger === "toggle" &&
      context === currentCoreContext &&
      context.generation === loadGeneration &&
      !context.oceanCurrentLoaded &&
      !context.oceanCurrentUnavailable
    ) {
      return ensureOceanCurrentsLoaded({ trigger: "toggle" });
    }
    return settled;
  }

  oceanCurrentLoadPromise = (async () => {
    const hasPrefetchedPayload = Boolean(context.oceanCurrentPrefetchMeta && context.oceanCurrentPrefetchBuffer);

    if (showOverlay) {
      setLoadingOverlayVisible(true);
      updateLoadingProgress(
        hasPrefetchedPayload ? 0.34 : 0.12,
        hasPrefetchedPayload
          ? t("explorer.loading.applyingPrefetchedOceanData")
          : t("explorer.loading.loadingOceanCurrentMetadata")
      );
      statusEl.textContent = t("explorer.status.loadingOceanStreamlines");
    }

    let oceanMeta = context.oceanCurrentPrefetchMeta;
    let oceanBuffer = context.oceanCurrentPrefetchBuffer;

    if (!oceanMeta) {
      oceanMeta = await fetchJsonStrict(
        context.dataset.oceanCurrentsMetaUrl,
        errorLabel("explorer.errors.failedToLoadOceanCurrentMetadata")
      );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }
    if (context === currentCoreContext && context.generation === loadGeneration && oceanMeta) {
      currentOceanCurrentMeta = oceanMeta;
      updateMetaFromCurrentState();
    }

    if (!oceanBuffer) {
      oceanBuffer = showOverlay
        ? await fetchArrayBufferWithProgress(
            context.dataset.oceanCurrentsBinUrl,
            0.2,
            0.78,
            t("explorer.loading.downloadingOceanCurrentVectors"),
            errorLabel("explorer.errors.failedToLoadOceanCurrentVectors")
          )
        : await fetchArrayBufferStrict(
            context.dataset.oceanCurrentsBinUrl,
            errorLabel("explorer.errors.failedToLoadOceanCurrentVectors")
          );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }

    if (!buildLayer) {
      context.oceanCurrentPrefetchMeta = oceanMeta;
      context.oceanCurrentPrefetchBuffer = oceanBuffer;
      return true;
    }

    context.oceanCurrentPrefetchMeta = null;
    context.oceanCurrentPrefetchBuffer = null;

    if (showOverlay) {
      updateLoadingProgress(0.86, t("explorer.loading.buildingOceanStreamlines"));
    }

    let builtMesh = null;
    const allowMainThreadFallback = canUseMainThreadOceanCurrentFallback(oceanMeta, oceanBuffer);
    try {
      const workerResult = await runGeometryWorkerTask(
        "buildOceanCurrents",
        {
          oceanMeta,
          oceanBuffer,
          mask: context.mask,
          bedHeights: context.bedHeights,
          bedValid: context.bedValid,
          nx: context.nx,
          ny: context.ny,
          grid: context.meta.grid,
          baseConfig: context.baseConfig,
          buildFlowLights: isFlowLightAnimationEnabled(),
          reportProgress: showOverlay,
        },
        {
          transfer: allowMainThreadFallback ? [] : [oceanBuffer],
          onProgress: showOverlay
            ? (progress, stage) => {
                const p = 0.86 + clamp01(progress) * 0.12;
                updateLoadingProgress(p, stage || t("explorer.loading.buildingOceanStreamlines"));
              }
            : null,
        }
      );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
      builtMesh = buildOceanCurrentMeshFromWorkerResult(oceanMeta, workerResult);
    } catch (workerError) {
      if (!allowMainThreadFallback) {
        console.error("Ocean-current worker build failed for a large dataset; skipping main-thread fallback.", workerError);
        throw workerError;
      }
      console.warn("Ocean-current worker build failed, falling back to main-thread build:", workerError);
      builtMesh = buildOceanCurrentMesh(context, oceanMeta, oceanBuffer);
    }
    if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    if (!builtMesh) {
      throw new Error("Ocean-current dataset is empty after regional clipping.");
    }

    oceanCurrentMesh = disposeObject3D(oceanCurrentMesh);
    oceanCurrentMesh = builtMesh;
    currentOceanCurrentMeta = oceanMeta;
    context.oceanCurrentLoaded = true;
    scene.add(oceanCurrentMesh);
    updateOceanCurrentLayerVisibility();
    updateMetaFromCurrentState();
    if (isFlowLightAnimationEnabled() && !hasFlowLightGeometry(oceanCurrentMesh)) {
      window.setTimeout(() => {
        if (context !== currentCoreContext || context.generation !== loadGeneration) return;
        rebuildStaticFlowLightLayers().catch((error) => {
          console.error("Failed to upgrade ocean flow lights after loading:", error);
        });
      }, 0);
    }
    if (showOverlay) {
      statusEl.textContent = getReadyStatusText(context);
      updateLoadingProgress(1, t("explorer.loading.oceanStreamlinesReady"));
      setLoadingOverlayVisible(false);
    }
    return true;
  })()
    .catch((error) => {
      console.error("Ocean-current layer load failed:", error);
      if (context === currentCoreContext && context.generation === loadGeneration && trigger === "toggle") {
        context.oceanCurrentUnavailable = true;
        controlsUI.showOceanCurrents.checked = false;
        oceanCurrentMesh = disposeObject3D(oceanCurrentMesh);
        const readyText = getReadyStatusText(context);
        const transientText = t("explorer.status.oceanUnavailable");
        statusEl.textContent = transientText;
        window.setTimeout(() => {
          if (statusEl.textContent === transientText) {
            statusEl.textContent = readyText;
          }
        }, 1800);
        setLoadingOverlayVisible(false);
      }
      return false;
    })
    .finally(() => {
      if (context === currentCoreContext) {
        oceanCurrentLoadPromise = null;
      }
    });

  return oceanCurrentLoadPromise;
}

async function ensureVelocityLoaded({ trigger = "prefetch" } = {}) {
  const context = currentCoreContext;
  if (!context || context.generation !== loadGeneration) return false;
  if (!context.dataset.capabilities.velocity || !context.dataset.velocityMetaUrl || !context.dataset.velocityBinUrl) {
    context.velocityUnavailable = true;
    return false;
  }
  if (context.velocityLoaded) {
    if (velocitySurfaceMesh) velocitySurfaceMesh.visible = controlsUI.showVelocity.checked;
    if (controlsUI.showFlowline.checked) updateFlowlineVisibility();
    return true;
  }
  if (context.velocityUnavailable) return false;

  const updateStatus = trigger === "toggle";
  const showOverlay = updateStatus && VELOCITY_BLOCKING_OVERLAY_ENABLED;
  const buildLayer = trigger === "toggle" || trigger === "warmup";

  if (velocityLoadPromise) {
    const settled = await velocityLoadPromise;
    if (
      trigger === "toggle" &&
      context === currentCoreContext &&
      context.generation === loadGeneration &&
      !context.velocityLoaded &&
      !context.velocityUnavailable
    ) {
      return ensureVelocityLoaded({ trigger: "toggle" });
    }
    return settled;
  }

  velocityLoadPromise = (async () => {
    const hasPrefetchedPayload = Boolean(context.velocityPrefetchMeta && context.velocityPrefetchBuffer);

    if (updateStatus) {
      if (showOverlay) {
        setLoadingOverlayVisible(true);
        updateLoadingProgress(
          hasPrefetchedPayload ? 0.34 : 0.1,
          hasPrefetchedPayload
            ? t("explorer.loading.applyingPrefetchedVelocityData")
            : t("explorer.loading.loadingVelocityMetadata")
        );
      }
      statusEl.textContent = t("explorer.status.loadingVelocityLayer");
    }

    let velocityMeta = context.velocityPrefetchMeta;
    let velocityBuffer = context.velocityPrefetchBuffer;

    if (!velocityMeta) {
      velocityMeta = await fetchJsonStrict(
        context.dataset.velocityMetaUrl,
        errorLabel("explorer.errors.failedToLoadVelocityMetadata")
      );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }
    if (context === currentCoreContext && context.generation === loadGeneration && velocityMeta) {
      currentVelocityMeta = velocityMeta;
      updateMetaFromCurrentState();
    }

    if (!velocityBuffer) {
      velocityBuffer = showOverlay
        ? await fetchArrayBufferWithProgress(
            context.dataset.velocityBinUrl,
            0.18,
            0.52,
            t("explorer.loading.downloadingVelocityField"),
            errorLabel("explorer.errors.failedToLoadVelocityField")
          )
        : await fetchArrayBufferStrict(
            context.dataset.velocityBinUrl,
            errorLabel("explorer.errors.failedToLoadVelocityField")
          );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }

    if (!buildLayer) {
      context.velocityPrefetchMeta = velocityMeta;
      context.velocityPrefetchBuffer = velocityBuffer;
      return true;
    }

    context.velocityPrefetchMeta = null;
    context.velocityPrefetchBuffer = null;
    return buildVelocityLayerFromPayload(context, velocityMeta, velocityBuffer, {
      showOverlay,
      updateStatus,
    });
  })()
    .catch((error) => {
      console.error("Velocity layer load failed:", error);
      if (context === currentCoreContext && context.generation === loadGeneration && trigger === "toggle") {
        context.velocityUnavailable = true;
        controlsUI.showVelocity.checked = false;
        controlsUI.showFlowline.checked = false;
        velocitySurfaceMesh = disposeMesh(velocitySurfaceMesh);
        flowlineMesh = disposeMesh(flowlineMesh);
        selectedFlowlineHighlight = disposeObject3D(selectedFlowlineHighlight);
        selectedFlowlineState = null;
        velocityDataTexture = disposeTexture(velocityDataTexture);
        velocityField = null;
        updateMetaFromCurrentState();
        const readyText = getReadyStatusText(context);
        const transientText = t("explorer.status.velocityUnavailable");
        statusEl.textContent = transientText;
        window.setTimeout(() => {
          if (statusEl.textContent === transientText) {
            statusEl.textContent = readyText;
          }
        }, 1800);
        setLoadingOverlayVisible(false);
      }
      return false;
    })
    .finally(() => {
      if (context === currentCoreContext) {
        velocityLoadPromise = null;
      }
    });

  return velocityLoadPromise;
}

async function ensureHydrologyLoaded({ trigger = "prefetch" } = {}) {
  const context = currentCoreContext;
  if (!context || context.generation !== loadGeneration) return false;
  if (!context.dataset.capabilities.hydrology || !context.dataset.hydrologyMetaUrl || !context.dataset.hydrologyBinUrl) {
    context.hydrologyUnavailable = true;
    return false;
  }
  if (context.hydrologyLoaded) {
    if (effectivePressureMesh) effectivePressureMesh.visible = controlsUI.showEffectivePressure.checked;
    if (subglacialChannelMesh) subglacialChannelMesh.visible = controlsUI.showSubglacialChannels.checked;
    return true;
  }
  if (context.hydrologyUnavailable) return false;

  const showOverlay = trigger === "toggle";
  const buildLayer = trigger === "toggle" || trigger === "warmup";

  if (hydrologyLoadPromise) {
    const settled = await hydrologyLoadPromise;
    if (
      trigger === "toggle" &&
      context === currentCoreContext &&
      context.generation === loadGeneration &&
      !context.hydrologyLoaded &&
      !context.hydrologyUnavailable
    ) {
      return ensureHydrologyLoaded({ trigger: "toggle" });
    }
    return settled;
  }

  hydrologyLoadPromise = (async () => {
    if (showOverlay) {
      setLoadingOverlayVisible(true);
      updateLoadingProgress(0.08, t("explorer.loading.loadingHydrologyMetadata"));
      statusEl.textContent = t("explorer.status.loadingHydrology");
    }

    let hydrologyMeta = context.hydrologyPrefetchMeta;
    let hydrologyBuffer = context.hydrologyPrefetchBuffer;
    if (!hydrologyMeta) {
      hydrologyMeta = await fetchJsonStrict(
        context.dataset.hydrologyMetaUrl,
        errorLabel("explorer.errors.failedToLoadHydrologyMetadata")
      );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }
    if (context === currentCoreContext && context.generation === loadGeneration && hydrologyMeta) {
      currentHydrologyMeta = hydrologyMeta;
      updateMetaFromCurrentState();
    }

    if (!hydrologyBuffer) {
      hydrologyBuffer = showOverlay
        ? await fetchArrayBufferWithProgress(
            context.dataset.hydrologyBinUrl,
            0.18,
            0.75,
            t("explorer.loading.downloadingHydrologyField"),
            errorLabel("explorer.errors.failedToLoadHydrologyField")
          )
        : await fetchArrayBufferStrict(
            context.dataset.hydrologyBinUrl,
            errorLabel("explorer.errors.failedToLoadHydrologyField")
          );
      if (context !== currentCoreContext || context.generation !== loadGeneration) return false;
    }

    if (!buildLayer) {
      context.hydrologyPrefetchMeta = hydrologyMeta;
      context.hydrologyPrefetchBuffer = hydrologyBuffer;
      return true;
    }

    context.hydrologyPrefetchMeta = null;
    context.hydrologyPrefetchBuffer = null;
    if (!gridsMatch(hydrologyMeta.grid, context.meta.grid)) {
      throw new Error("Hydrology grid is not aligned to the active terrain grid.");
    }

    await ensureHydrologyColorTables();
    if (context !== currentCoreContext || context.generation !== loadGeneration) return false;

    if (showOverlay) {
      updateLoadingProgress(0.76, t("explorer.loading.processingHydrologyField"));
    }

    const workerResult = await runGeometryWorkerTask(
      "buildHydrology",
      {
        hydrologyMeta,
        hydrologyBuffer,
        nx: context.nx,
        ny: context.ny,
        grid: context.meta.grid,
        cellCount: context.cellCount,
        baseConfig: context.baseConfig,
        bedHeights: context.bedHeights,
        bedValid: context.bedValid,
        effectivePressureLut,
        channelDischargeLut,
        meshStride: getHydrologyLayerMeshStride(context),
        reportProgress: showOverlay,
      },
      {
        transfer: [hydrologyBuffer],
        onProgress: showOverlay
          ? (progress, stage) => {
              const p = 0.76 + clamp01(progress) * 0.22;
              updateLoadingProgress(p, stage || t("explorer.loading.processingHydrologyField"));
            }
          : null,
      }
    );
    if (context !== currentCoreContext || context.generation !== loadGeneration) return false;

    effectivePressureMesh = disposeMesh(effectivePressureMesh);
    subglacialChannelMesh = disposeMesh(subglacialChannelMesh);
    const meshStride = getHydrologyLayerMeshStride(context);
    const pressurePolygonOffset = meshStride > 1 ? -3 : -1;
    const channelPolygonOffset = meshStride > 1 ? -5 : -2;

    const effectivePressureGeometry = new THREE.BufferGeometry();
    effectivePressureGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(workerResult.effectivePressurePositions, 3)
    );
    effectivePressureGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(workerResult.effectivePressureColors, 3, true)
    );
    effectivePressureGeometry.setIndex(new THREE.BufferAttribute(workerResult.effectivePressureIndices, 1));
    const effectivePressureMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      side: THREE.DoubleSide,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: pressurePolygonOffset,
      polygonOffsetUnits: pressurePolygonOffset,
    });
    effectivePressureMesh = new THREE.Mesh(effectivePressureGeometry, effectivePressureMaterial);
    effectivePressureMesh.scale.y = Number(controlsUI.exaggeration.value);
    effectivePressureMesh.visible = controlsUI.showEffectivePressure.checked;
    effectivePressureMesh.material.wireframe = controlsUI.wireframe.checked;
    effectivePressureMesh.renderOrder = 7;
    scene.add(effectivePressureMesh);

    if (workerResult.channelPositions && workerResult.channelColors && workerResult.channelPositions.length) {
      const channelGeometry = new THREE.BufferGeometry();
      channelGeometry.setAttribute("position", new THREE.BufferAttribute(workerResult.channelPositions, 3));
      channelGeometry.setAttribute("color", new THREE.BufferAttribute(workerResult.channelColors, 3));
      const channelMaterial = new THREE.MeshBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.98,
        side: THREE.DoubleSide,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: channelPolygonOffset,
        polygonOffsetUnits: channelPolygonOffset,
      });
      subglacialChannelMesh = new THREE.Mesh(channelGeometry, channelMaterial);
      subglacialChannelMesh.visible = controlsUI.showSubglacialChannels.checked;
      subglacialChannelMesh.scale.y = Number(controlsUI.exaggeration.value);
      subglacialChannelMesh.userData.channelCount = workerResult.channelCount || 0;
      subglacialChannelMesh.renderOrder = 8;
      scene.add(subglacialChannelMesh);
    } else {
      controlsUI.showSubglacialChannels.checked = false;
      controlsUI.showSubglacialChannels.disabled = true;
    }

    context.hydrologyLoaded = true;
    context.hydrologyPrefetchMeta = null;
    context.hydrologyPrefetchBuffer = null;
    currentHydrologyMeta = hydrologyMeta;
    updateMetaFromCurrentState();
    if (showOverlay) {
      statusEl.textContent = getReadyStatusText(context);
      updateLoadingProgress(1, t("explorer.loading.hydrologyLayerReady"));
      setLoadingOverlayVisible(false);
    }
    return true;
  })()
    .catch((error) => {
      console.error("Hydrology layer load failed:", error);
      if (context === currentCoreContext && context.generation === loadGeneration && trigger === "toggle") {
        context.hydrologyUnavailable = true;
        controlsUI.showEffectivePressure.checked = false;
        controlsUI.showSubglacialChannels.checked = false;
        controlsUI.showEffectivePressure.disabled = true;
        controlsUI.showSubglacialChannels.disabled = true;
        effectivePressureMesh = disposeMesh(effectivePressureMesh);
        subglacialChannelMesh = disposeMesh(subglacialChannelMesh);
        const readyText = getReadyStatusText(context);
        const transientText = t("explorer.status.hydrologyUnavailable");
        statusEl.textContent = transientText;
        window.setTimeout(() => {
          if (statusEl.textContent === transientText) {
            statusEl.textContent = readyText;
          }
        }, 1800);
        setLoadingOverlayVisible(false);
      }
      return false;
    })
    .finally(() => {
      if (context === currentCoreContext) {
        hydrologyLoadPromise = null;
      }
    });

  return hydrologyLoadPromise;
}

function canWarmupInBackground() {
  if (isShowcaseMode) return false;
  if (!currentCoreContext) return false;
  if (backgroundWarmupStarted) return false;
  if (currentCoreContext.dataset?.disableBackgroundWarmup) return false;
  // HD warmup can starve on-demand layer toggles because it competes for the same worker.
  if (currentCoreContext.dataset?.id === "hd") return false;
  const connection = navigator.connection;
  if (connection?.saveData) return false;
  const networkType = String(connection?.effectiveType || "");
  if (networkType.includes("2g")) return false;
  return true;
}

function startBackgroundWarmup() {
  if (!canWarmupInBackground()) return;
  const context = currentCoreContext;
  backgroundWarmupStarted = true;
  backgroundWarmupScheduled = false;
  if (backgroundWarmupTimer !== null) {
    window.clearTimeout(backgroundWarmupTimer);
    backgroundWarmupTimer = null;
  }

  (async () => {
    if (!context || context !== currentCoreContext || context.generation !== loadGeneration) return;
    await ensureVelocityLoaded({ trigger: "warmup" });
    if (!context || context !== currentCoreContext || context.generation !== loadGeneration) return;
    await nextAnimationFrame();
    await ensureBasalFrictionLoaded({ trigger: "warmup" });
    if (!context || context !== currentCoreContext || context.generation !== loadGeneration) return;
    await nextAnimationFrame();
    await ensureHydrologyLoaded({ trigger: "warmup" });
    if (!context || context !== currentCoreContext || context.generation !== loadGeneration) return;
    await nextAnimationFrame();
    await ensureOceanCurrentsLoaded({ trigger: "warmup" });
  })().catch((error) => {
    console.warn("Background warmup failed:", error);
  });
}

function scheduleBackgroundWarmup() {
  if (!canWarmupInBackground()) return;
  if (backgroundWarmupScheduled) return;
  backgroundWarmupScheduled = true;

  const run = () => {
    backgroundWarmupScheduled = false;
    startBackgroundWarmup();
  };

  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(run, { timeout: 2600 });
  } else {
    window.setTimeout(run, 1200);
  }
}

async function loadAndBuildMeshes(datasetKey = currentDatasetKey) {
  const effectiveRegionKey = lockedRegionKey || currentRegionKey;
  const region = getRegionConfig(effectiveRegionKey);
  const effectiveDatasetKey =
    lockedDatasetKey || datasetKey || datasetSelectionByRegion[region.key] || getDefaultDatasetKey(region.key);
  const dataset = getDatasetConfig(region.key, effectiveDatasetKey);
  if (!dataset) {
    throw new Error(`Unknown terrain preset: ${String(effectiveDatasetKey)}`);
  }

  const generation = ++loadGeneration;
  currentRegionKey = region.key;
  currentDatasetKey = dataset.id;
  datasetSelectionByRegion[currentRegionKey] = currentDatasetKey;
  currentCoreContext = null;
  currentVelocityMeta = null;
  currentBasalFrictionMeta = null;
  currentRiseMeta = null;
  currentHydrologyMeta = null;
  currentOceanCurrentMeta = null;
  currentVelocityMedianSpeed = Number.NaN;
  currentRefinedBasinData = null;
  velocityLoadPromise = null;
  basalFrictionLoadPromise = null;
  riseLoadPromise = null;
  hydrologyLoadPromise = null;
  oceanCurrentLoadPromise = null;
  refinedBasinLoadPromise = null;
  refinedBasinLoadingUrl = null;
  backgroundWarmupStarted = false;
  backgroundWarmupScheduled = false;
  viewerInteracted = false;
  resetRecordingMotionState({ syncBasePose: false });
  if (backgroundWarmupTimer !== null) {
    window.clearTimeout(backgroundWarmupTimer);
    backgroundWarmupTimer = null;
  }
  terminateGeometryWorker();
  terminateReboundWorker();
  reboundLoadPromise = null;
  if (reboundGeometryFrame !== null) {
    window.cancelAnimationFrame(reboundGeometryFrame);
    reboundGeometryFrame = null;
  }
  if (reboundSeaLevelDebounce !== null) {
    window.clearTimeout(reboundSeaLevelDebounce);
    reboundSeaLevelDebounce = null;
  }
  currentDatasetKey = refreshRegionUi(currentRegionKey, currentDatasetKey);
  updateRegionControlAvailability({ isLoading: true });
  updateResolutionControlAvailability({ isLoading: true });
  statusEl.textContent = getLoadingStatusText(region, dataset);
  setLoadingOverlayVisible(true);
  updateLoadingProgress(0.02, t("explorer.loading.preparingDataStreams"));

  try {
    const metaPromise = fetchJsonStrict(
      dataset.metaUrl,
      errorLabel("explorer.errors.failedToLoadMetadata")
    );
    const bufferPromise = fetchArrayBufferWithProgress(
      dataset.binUrl,
      0.08,
      0.66,
      t("explorer.loading.downloadingTerrainPackage"),
      errorLabel("explorer.errors.failedToLoadTerrainPackage")
    );
    const [meta, buffer] = await Promise.all([metaPromise, bufferPromise]);
    if (generation !== loadGeneration) return;

    updateLoadingProgress(0.74, t("explorer.loading.decodingTerrainFields"));
    const nx = meta.grid.nx;
    const ny = meta.grid.ny;
    const cellCount = nx * ny;

    const bedInt = parseField(meta, buffer, "bed");
    const surfaceInt = parseField(meta, buffer, "surface");
    const thicknessInt = parseField(meta, buffer, "thickness");
    const mask = parseField(meta, buffer, "mask");
    if (
      bedInt.length !== cellCount ||
      surfaceInt.length !== cellCount ||
      thicknessInt.length !== cellCount ||
      mask.length !== cellCount
    ) {
      throw new Error("Unexpected field length in data package.");
    }

    const bedHeights = decodeFieldToFloat32(meta, buffer, "bed");
    const surfaceHeights = decodeFieldToFloat32(meta, buffer, "surface");
    const thickness = decodeFieldToFloat32(meta, buffer, "thickness");
    const iceBottomHeights = new Float32Array(cellCount);
    const bedValid = new Uint8Array(cellCount);
    const iceValid = new Uint8Array(cellCount);
    const iceBottomValid = new Uint8Array(cellCount);
    for (let i = 0; i < cellCount; i += 1) {
      bedValid[i] = Number(Number.isFinite(bedHeights[i]));
      iceValid[i] = Number(isIceCoveredMask(mask[i]) && Number.isFinite(surfaceHeights[i]) && thickness[i] > 0);
      if (iceValid[i]) {
        const baseHeight = surfaceHeights[i] - thickness[i];
        if (Number.isFinite(baseHeight)) {
          iceBottomHeights[i] = baseHeight;
          iceBottomValid[i] = 1;
        } else {
          iceBottomHeights[i] = Number.NaN;
          iceBottomValid[i] = 0;
        }
      } else {
        iceBottomHeights[i] = Number.NaN;
        iceBottomValid[i] = 0;
      }
    }

    const widthMeters = Math.abs(meta.grid.dx_m) * Math.max(1, nx - 1);
    const depthMeters = Math.abs(meta.grid.dy_m) * Math.max(1, ny - 1);
    const horizontalMetersPerUnit = Math.max(1, Math.max(widthMeters, depthMeters) / TARGET_WORLD_EXTENT_UNITS);
    const verticalMetersPerUnit = Math.max(1, horizontalMetersPerUnit / BASE_HORIZONTAL_VERTICAL_SCALE_RATIO);
    const baseConfig = {
      nx,
      ny,
      dxMeters: meta.grid.dx_m,
      dyMeters: meta.grid.dy_m,
      horizontalMetersPerUnit,
      verticalMetersPerUnit,
    };

    if (generation !== loadGeneration) return;
    currentCoreContext = {
      generation,
      dataset,
      meta,
      nx,
      ny,
      cellCount,
      baseConfig,
      bedHeights,
      surfaceHeights,
      thickness,
      mask,
      bedValid,
      iceValid,
      iceBottomHeights,
      iceBottomValid,
      velocityLoaded: false,
      velocityUnavailable: false,
      velocityPrefetchMeta: null,
      velocityPrefetchBuffer: null,
      basalFrictionLoaded: false,
      basalFrictionUnavailable: false,
      basalFrictionPrefetchMeta: null,
      basalFrictionPrefetchBuffer: null,
      riseLoaded: false,
      riseUnavailable: false,
      risePrefetchMeta: null,
      risePrefetchBuffer: null,
      riseMask: null,
      riseIceshelfId: null,
      riseIceDraft: null,
      riseBasalMelt: null,
      riseThermalDriving: null,
      hydrologyLoaded: false,
      hydrologyUnavailable: false,
      hydrologyPrefetchMeta: null,
      hydrologyPrefetchBuffer: null,
      oceanCurrentLoaded: false,
      oceanCurrentUnavailable: false,
      oceanCurrentPrefetchMeta: null,
      oceanCurrentPrefetchBuffer: null,
      reboundUplift: null,
      reboundEmergent: null,
      reboundStats: null,
      reboundSolveKey: "",
      reboundUnavailable: false,
      reboundScratchBed: null,
      reboundScratchSurface: null,
      reboundScratchBottom: null,
    };
    primeLayerMetadata(currentCoreContext);

    updateLoadingProgress(0.88, t("explorer.loading.buildingBaseMeshes"));
    buildCoreSceneFromContext(currentCoreContext);
    await polarFeaturesController?.onTerrainReady();
    if (generation !== loadGeneration || currentCoreContext?.generation !== generation) return;
    if (pendingViewResetOnLoad) {
      resetCameraToDefaultPose();
      pendingViewResetOnLoad = false;
    }
    captureRecordingBasePoseFromCurrentView();
    captureShowcaseBasePoseFromCurrentView();
    if ((isShowcaseMode && (!interactionGateEnabled || !interactionGateActive)) || isPreviewMode) {
      startShowcaseAutoOrbit();
    }
    updateRecordingControlsUi();
    updateMetaFromCurrentState();
    updateFlowlineVisibility();
    statusEl.textContent = getReadyStatusText(currentCoreContext);
    updateLoadingProgress(1, t("explorer.loading.coreTerrainReady"));

    window.setTimeout(() => {
      if (generation === loadGeneration) {
        setLoadingOverlayVisible(false);
      }
    }, 120);

    if (controlsUI.showVelocity.checked || controlsUI.showFlowline.checked) {
      ensureVelocityLoaded({ trigger: "toggle" });
    }
    if (controlsUI.showBasalFriction.checked && !controlsUI.showBasalFriction.disabled) {
      ensureBasalFrictionLoaded({ trigger: "toggle" });
    }
    if (
      (controlsUI.showBasalMelt.checked || controlsUI.showThermalDriving.checked) &&
      !controlsUI.showBasalMelt.disabled &&
      !controlsUI.showThermalDriving.disabled
    ) {
      ensureRiseLoaded({ trigger: "toggle" });
    }
    if (controlsUI.showEffectivePressure.checked || controlsUI.showSubglacialChannels.checked) {
      ensureHydrologyLoaded({ trigger: "toggle" });
    }
    if (controlsUI.showOceanCurrents.checked && !controlsUI.showOceanCurrents.disabled) {
      ensureOceanCurrentsLoaded({ trigger: "toggle" });
    }
    if (controlsUI.showRefinedBasins.checked && !controlsUI.showRefinedBasins.disabled) {
      ensureRefinedBasinsLoaded({ trigger: "toggle" });
    }
    if (controlsUI.showIsostaticRebound?.checked && !controlsUI.showIsostaticRebound.disabled) {
      ensureIsostaticReboundLoaded({ trigger: "toggle" });
    } else {
      updateReboundControlsUi();
    }
    if (!isShowcaseMode && !isPreviewMode) {
      const warmupDelay = viewerInteracted ? 180 : 900;
      backgroundWarmupTimer = window.setTimeout(() => {
        backgroundWarmupTimer = null;
        scheduleBackgroundWarmup();
      }, warmupDelay);
    }
  } finally {
    updateRegionLayerAvailability(currentRegionKey);
    updateRegionControlAvailability();
    updateResolutionControlAvailability();
  }
}

async function main() {
  try {
    updateLoadingProgress(0.01, t("explorer.loading.initializingRuntime"));
    bindRuntimeTheme();
    await loadThreeRuntime();
    updateLoadingProgress(0.04, t("explorer.loading.initializingRenderer"));

    initScene();
    initializePolarFeatures();
    bindUI();
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    bindInteractionGate();
    bindFlowlinePicking();
    bindBackgroundWarmupTrigger();
    bindRecordingMode();
    renderLoop();

    await Promise.all([loadBedColorTable(), ensureHydrologyColorTables()]);
    updateBedLegend();
    updateVelocityLegend();
    updateBasalFrictionLegend();
    updateBasalMeltLegend();
    updateThermalDrivingLegend();
    updateOceanCurrentLegend();
    updateEffectivePressureLegend();
    updateChannelLegend();

    await loadAndBuildMeshes(lockedDatasetKey || currentDatasetKey);
  } catch (error) {
    reportLoadError(error);
  }
}

window.render_game_to_text = () => JSON.stringify(collectExplorerState());

window.advanceTime = (ms = 16) => {
  const totalMs = Math.max(0, Number(ms) || 0);
  const steps = Math.max(1, Math.round(totalMs / (1000 / 60)));
  const stepMs = totalMs > 0 ? totalMs / steps : 1000 / 60;
  for (let i = 0; i < steps; i += 1) {
    stepRuntime(stepMs);
  }
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
};

window.addEventListener("beforeunload", () => {
  if (runtimeThemeObserver) {
    runtimeThemeObserver.disconnect();
  }
  if (runtimeThemeStorageHandler) {
    window.removeEventListener("storage", runtimeThemeStorageHandler);
  }
  if (runtimeThemeMediaHandler) {
    runtimeThemeMediaQuery.removeEventListener("change", runtimeThemeMediaHandler);
  }
  if (animationHandle) cancelAnimationFrame(animationHandle);
  cleanupViewerResizeHandling();
  polarFeaturesController?.destroy();
  polarFeaturesController = null;
  clearModelMeshes();
  terminateGeometryWorker();
  if (renderer) renderer.dispose();
});

main();
