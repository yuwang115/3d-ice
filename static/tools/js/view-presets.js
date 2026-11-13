/**
 * Named views of the explorer: the home page's showcase heroes (`mode=showcase`) and the
 * research page's `?preset=` views. Each names a region and dataset, the layer toggles to
 * set, the vertical exaggeration and the ice opacity, and for a showcase the camera. A view
 * may also name the basin layer's set (`basinSet`); one that does not shows the default.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner, where
 * a test holds every view to the panel's one-of choices.
 */

function deepFreeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

export const SHOWCASE_PRESETS = deepFreeze({
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
});

export function getShowcasePreset(presetKey) {
  return SHOWCASE_PRESETS[presetKey] || SHOWCASE_PRESETS["home-hero"];
}

const EXPLORER_PRESETS_DRAFT = {
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
  // Mankoff et al. (2020)'s outlet catchments, the Greenland basin layer's second set.
  "greenland-outlet-catchments": {
    regionKey: "greenland",
    datasetKey: "3km",
    exaggeration: 4.2,
    iceOpacity: 1,
    toggles: {
      showBed: true,
      showIce: true,
      showIceBottom: true,
      showRefinedBasins: true,
    },
    basinSet: "mankoff2020",
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

/**
 * The views the home page's data tower opens, one per layer of the stack. Each turns on its
 * layer as the panel would, with what switching it on there turns on too: the sea plane for
 * the sea ice, the rebound and the projection, and the flowlines for the projection. A layer
 * drawn on the bed hides the ice above it, as the ice-shelf base's colourings do themselves.
 */
const TOWER_VIEWS = {
  "ice-surface": {},
  "surface-change": { showSurfaceChange: true },
  "sea-ice": { showSeaIce: true, showSea: true },
  projection: { showIceProjection: true, showFlowline: true, showSea: true },
  "basal-melt": { showIce: false, showIceBottom: false, showBasalMelt: true },
  "basal-friction": { showIce: false, showIceBottom: false, showBasalFriction: true },
  "bed-topography": { showIce: false, showIceBottom: false },
  "isostatic-rebound": { showIsostaticRebound: true, showSea: true },
  "grounding-line": { showGroundingLine: true },
  "basin-boundary": { showRefinedBasins: true },
};

const TOWER_REGIONS = {
  antarctica: {
    datasetKey: "balanced",
    exaggeration: 4.8,
    views: ["ice-surface", "surface-change", "sea-ice", "projection", "basal-melt", "basal-friction", "bed-topography", "isostatic-rebound", "grounding-line"],
  },
  greenland: {
    datasetKey: "3km",
    exaggeration: 4.2,
    views: ["ice-surface", "sea-ice", "basal-friction", "bed-topography", "isostatic-rebound", "grounding-line", "basin-boundary"],
  },
};

function towerPresets() {
  const presets = {};
  for (const [regionKey, region] of Object.entries(TOWER_REGIONS)) {
    for (const view of region.views) {
      presets[`${regionKey}-${view}`] = {
        regionKey,
        datasetKey: region.datasetKey,
        exaggeration: region.exaggeration,
        iceOpacity: 1,
        toggles: { showBed: true, showIce: true, showIceBottom: true, ...TOWER_VIEWS[view] },
      };
    }
  }
  return presets;
}

// The data tower's views, after the named ones above (which stay as they were).
export const EXPLORER_PRESETS = deepFreeze({ ...EXPLORER_PRESETS_DRAFT, ...towerPresets() });

export function getExplorerPreset(presetKey) {
  return EXPLORER_PRESETS[presetKey] || null;
}
