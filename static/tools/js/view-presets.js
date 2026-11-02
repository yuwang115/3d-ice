/**
 * Named views of the explorer: the home page's showcase heroes (`mode=showcase`) and the
 * research page's `?preset=` views. Each names a region and dataset, the layer toggles to
 * set, the vertical exaggeration and the ice opacity, and for a showcase the camera.
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

export const EXPLORER_PRESETS = deepFreeze({
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
});

export function getExplorerPreset(presetKey) {
  return EXPLORER_PRESETS[presetKey] || null;
}
