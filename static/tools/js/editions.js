/**
 * Edition profiles: which datasets, layers and runtime behaviours a 3D ICE page offers.
 *
 * The research edition is the full explorer. The public edition is the same runtime cut
 * down to a core set of layers, with the guided tour and the layer explainers on top. A
 * page names its edition with <html data-edition="...">, and everything else follows from
 * its profile here, so the two editions share one runtime instead of forking it.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

export const EDITION_KEYS = Object.freeze({
  research: "research",
  public: "public",
});

/**
 * The package URLs each layer capability owns, at region or dataset level. Switching a
 * capability off drops its URLs as well, so the runtime cannot fetch, prefetch or prime
 * the metadata of a package that the edition does not show.
 */
const CAPABILITY_URL_KEYS = Object.freeze({
  velocity: Object.freeze(["velocityMetaUrl", "velocityBinUrl"]),
  basalFriction: Object.freeze(["basalFrictionMetaUrl", "basalFrictionBinUrl"]),
  rise: Object.freeze(["riseMetaUrl", "riseBinUrl"]),
  oceanCurrents: Object.freeze(["oceanCurrentsMetaUrl", "oceanCurrentsBinUrl"]),
  hydrology: Object.freeze(["hydrologyMetaUrl", "hydrologyBinUrl"]),
  refinedBasins: Object.freeze(["refinedBasinsUrl"]),
  isostaticRebound: Object.freeze(["reboundMetaUrl", "reboundBinUrl"]),
});

function deepFreeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

const PROFILES = deepFreeze({
  [EDITION_KEYS.research]: {
    key: EDITION_KEYS.research,
    // null offers every dataset in the registry.
    datasets: null,
    disabledCapabilities: [],
    // Every control is in the research page's markup, so a missing one is a bug to
    // surface rather than paper over.
    standInControls: false,
    fixedControls: {},
    // Layers fetched in the background after the first interaction, in this order.
    backgroundWarmup: ["velocity", "basalFriction", "hydrology", "oceanCurrents"],
    flowlinePicking: true,
    recording: true,
    detailedStatus: true,
    guide: false,
  },
  [EDITION_KEYS.public]: {
    key: EDITION_KEYS.public,
    // The default terrain of each region: BedMachine Antarctica v4 on the 10 km grid and
    // BedMachine Greenland v6 on the 3 km grid, both light enough for phones.
    datasets: {
      antarctica: ["balanced"],
      greenland: ["3km"],
    },
    disabledCapabilities: ["basalFriction", "rise", "hydrology", "refinedBasins"],
    standInControls: true,
    // State of the controls this edition's page leaves out. Anything not listed stands in
    // as an unchecked checkbox, so the layer it drives stays off.
    fixedControls: {
      // The runtime still fills the dataset picker, with the one dataset on offer.
      resolutionPreset: { tag: "select" },
      // The ocean legend is drawn with a 2D context, which only a canvas has.
      oceanLegendCanvas: { tag: "canvas" },
      showIceBottom: { type: "checkbox", checked: true },
      animateFlow: { type: "checkbox", checked: true },
      highlightEmergentLand: { type: "checkbox", checked: true },
      // The four ocean depth bands ride on the one ocean-current toggle.
      showOceanLayerSurface: { type: "checkbox", checked: true },
      showOceanLayerUpper: { type: "checkbox", checked: true },
      showOceanLayerMid: { type: "checkbox", checked: true },
      showOceanLayerLower: { type: "checkbox", checked: true },
      // Only the published Earth response of Paxman et al. (2022) is offered; it sets its
      // own ice-free sea surface, so the datum stays at zero.
      reboundModel: { type: "text", value: "paxman2022" },
      reboundSeaLevel: { type: "range", min: 0, max: 70, value: 0 },
    },
    // Just what the flowlines need. The Antarctic ocean package is tens of megabytes, too
    // much to fetch unasked on a phone, so it loads when its layer is switched on.
    backgroundWarmup: ["velocity"],
    flowlinePicking: false,
    recording: false,
    detailedStatus: false,
    guide: true,
  },
});

const EMPTY_STAND_IN = Object.freeze({});

/** @param {unknown} value @returns {"research" | "public"} */
export function resolveEditionKey(value) {
  return value === EDITION_KEYS.public ? EDITION_KEYS.public : EDITION_KEYS.research;
}

/** @param {unknown} value */
export function getEditionProfile(value) {
  return PROFILES[resolveEditionKey(value)];
}

/**
 * The state a missing control should stand in with, or null when the edition expects the
 * page to supply every control.
 */
export function getStandInSpec(profile, controlId) {
  if (!profile?.standInControls) return null;
  return profile.fixedControls[controlId] || EMPTY_STAND_IN;
}

function withoutCapabilities(config, disabledCapabilities) {
  if (!disabledCapabilities.length) return { ...config };
  const next = { ...config };
  if (config.capabilities) {
    next.capabilities = { ...config.capabilities };
    for (const capability of disabledCapabilities) next.capabilities[capability] = false;
  }
  for (const capability of disabledCapabilities) {
    for (const urlKey of CAPABILITY_URL_KEYS[capability] || []) delete next[urlKey];
  }
  return next;
}

/**
 * Restrict a region registry (the runtime's REGIONS) to what an edition offers. Returns
 * new region, dataset and capability objects and leaves the input untouched.
 */
export function applyEditionToRegions(regions, profile) {
  const disabledCapabilities = [...(profile?.disabledCapabilities || [])];
  return Object.fromEntries(
    Object.entries(regions).map(([regionKey, region]) => {
      const offered = profile?.datasets?.[regionKey] || null;
      const datasets = Object.fromEntries(
        Object.entries(region.datasets || {})
          .filter(([datasetKey]) => !offered || offered.includes(datasetKey))
          .map(([datasetKey, dataset]) => [datasetKey, withoutCapabilities(dataset, disabledCapabilities)])
      );
      const defaultDatasetKey = datasets[region.defaultDatasetKey]
        ? region.defaultDatasetKey
        : Object.keys(datasets)[0];
      return [
        regionKey,
        { ...withoutCapabilities(region, disabledCapabilities), datasets, defaultDatasetKey },
      ];
    })
  );
}
