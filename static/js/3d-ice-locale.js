(function () {
  "use strict";

  const STORAGE_KEY = "3d-ice:locale";
  const DEFAULT_LOCALE = "en-US";
  const SUPPORTED_LOCALES = ["en-US", "zh-CN"];
  const OG_LOCALE_BY_LOCALE = {
    "en-US": "en_US",
    "zh-CN": "zh_CN",
  };
  const ROUTES = {
    home: {
      "en-US": "/",
      "zh-CN": "/zh/",
    },
    // yuwang.blog builds its copy of the home page from this repository and serves it here.
    landing: {
      "en-US": "/tools/3d-ice/",
      "zh-CN": "/zh/tools/3d-ice/",
    },
    explorer: {
      "en-US": "/tools/3D-interactive-cryosphere-explorer.html",
      "zh-CN": "/zh/tools/3D-interactive-cryosphere-explorer.html",
    },
    explore: {
      "en-US": "/explore/",
      "zh-CN": "/zh/explore/",
    },
    legacyRedirect: {
      "en-US": "/tools/3d-antarctica/",
      "zh-CN": "/zh/tools/3d-antarctica/",
    },
  };

  const MESSAGES = {
    "en-US": {
      shared: {
        switcherLabel: "Language",
        localeEnglish: "EN",
        localeChinese: "中文",
        localeEnglishLong: "English",
        localeChineseLong: "Simplified Chinese",
      },
      home: {
        switcherTitle: "Language",
      },
      explorer: {
        regions: {
          antarctica: {
            label: "Antarctica",
            intro:
              "Explore Antarctica from every angle. Rotate, zoom, and peel back the ice to uncover a hidden world in interactive 3D.",
            basinToggleLabel: "Show refined basins",
            basinStatusLabel: "refined basins",
          },
          greenland: {
            label: "Greenland",
            intro:
              "Explore Greenland from every angle. Rotate, zoom, and peel back the ice to uncover a hidden world in interactive 3D.",
            basinToggleLabel: "Show basins",
            basinStatusLabel: "basins",
          },
        },
        datasets: {
          antarctica: {
            balanced: {
              label: "BedMachine v4 — Balanced",
              summary: "10 km grid; ~3.0 MB",
            },
            hd: {
              label: "BedMachine v4 — HD",
              summary: "4 km grid; ~18.6 MB",
            },
            bedmap3: {
              label: "Bedmap3 — Balanced",
              summary: "10 km grid; ~3.0 MB",
            },
            "bedmap3-hd": {
              label: "Bedmap3 — HD",
              summary: "4 km grid; ~18.6 MB, desktop recommended",
            },
          },
          greenland: {
            "3km": {
              label: "Balanced",
              summary: "3 km grid; ~3.1 MB",
            },
            "1km": {
              label: "HD",
              summary: "1 km grid; ~28.2 MB",
            },
            qrf: {
              label: "QRF 2025 — Balanced",
              summary: "3 km grid; ~3.1 MB",
            },
            "qrf-hd": {
              label: "QRF 2025 — HD",
              summary: "1 km grid; ~28.2 MB, desktop recommended",
            },
          },
        },
        interaction: {
          showcaseIdleHtml:
            "<strong>Scroll page:</strong> swipe naturally.<br /><strong>Tap scene</strong> to enter 3D.",
          showcaseActiveHtml:
            "<strong>3D active:</strong> drag to orbit, pinch to zoom.<br /><strong>Tap Scroll</strong> to return to scrolling.",
          touchIdleHtml:
            "<strong>Scroll page:</strong> swipe naturally.<br /><strong>Tap scene</strong> to enable 3D gestures.",
          touchActiveHtml:
            "<strong>3D active:</strong> drag to orbit, pinch to zoom.<br /><strong>Tap Scroll</strong> to return to page scrolling.",
          enable3d: "Enable 3D",
          scrollPage: "Scroll Page",
        },
        drawer: {
          open: "Layers and places",
          close: "Done",
        },
        capture: {
          play: "Play",
          stop: "Stop",
          shown: "Capture controls shown.",
          hiddenHint: "Capture controls hidden. Press H to show.",
          enableOrbitOrZoomFirst: "Enable Orbit or Zoom Pulse first.",
        },
        fullscreen: {
          enter: "Enter Fullscreen",
          exit: "Exit Fullscreen",
          unavailable: "Fullscreen Unavailable",
          blocked: "Fullscreen request was blocked",
        },
        status: {
          loadingIsostaticRebound: "Solving isostatic rebound...",
          loadingIsostaticResponse: "Loading the published isostatic response...",
          loadingIceProjection: "Loading ice-sheet projection...",
          iceProjectionUnavailable: "Ice-sheet projection unavailable",
          isostaticReboundUnavailable: "Isostatic rebound unavailable",
          ready: "Ready",
          readyWithContext: "Ready ({region} {dataset})",
          previewReady: "Preview ready",
          loadingPreview: "Loading {region} preview...",
          loadingCoreTerrain: "Loading {region} {dataset} core terrain...",
          loadingRegion: "Loading {region}...",
          loadingBasin: "Loading {label}...",
          loadingBasalFriction: "Loading basal friction...",
          loadingSurfaceChange: "Loading observed elevation change...",
          surfaceChangeUnavailable: "Observed elevation change unavailable",
          loadingSeaIce: "Loading observed sea ice...",
          seaIceUnavailable: "Observed sea ice unavailable",
          loadingVelocityLayer: "Loading velocity layer...",
          loadingOceanStreamlines: "Loading ocean streamlines...",
          loadingHydrology: "Loading subglacial hydrology...",
          computingFlowlines: "Computing flowlines...",
          loadFailed: "Load failed",
          viewCopied: "View copied to clipboard",
          viewLogged: "View logged to console",
          refinedBasinsUnavailable: "{label} unavailable",
          velocityUnavailable: "Velocity layer unavailable",
          basalFrictionUnavailable: "Basal friction layer unavailable",
          oceanUnavailable: "Ocean streamlines unavailable",
          hydrologyUnavailable: "Hydrology layer unavailable",
          flowlineProfileHint: "Click any flowline to inspect the ice-and-bedrock profile",
        },
        loading: {
          solvingIsostaticRebound: "Solving isostatic rebound...",
          loadingIsostaticResponse: "Loading the published isostatic response...",
          downloadingIsostaticResponse: "Downloading isostatic response",
          downloadingIceProjection: "Downloading ice-sheet projection",
          buildingIceProjection: "Building ice-sheet projection...",
          iceProjectionReady: "Ice-sheet projection ready",
          isostaticReboundReady: "Isostatic rebound ready",
          initializingRuntime: "Initializing 3D runtime...",
          initializingRenderer: "Initializing renderer...",
          progressiveLoadingEnabled: "Progressive loading enabled",
          loadingThreeRuntime: "Loading Three.js runtime...",
          preparingDataStreams: "Preparing data streams...",
          downloadingTerrainPackage: "Downloading terrain package",
          decodingTerrainFields: "Decoding terrain fields...",
          buildingBaseMeshes: "Building base meshes...",
          coreTerrainReady: "Core terrain ready",
          loadingVelocityMetadata: "Loading velocity metadata...",
          applyingPrefetchedVelocityData: "Applying prefetched velocity data...",
          downloadingVelocityField: "Downloading velocity field",
          decodingVelocityField: "Decoding velocity field...",
          buildingVelocityMesh: "Building velocity mesh...",
          triangulatingVelocityMesh: "Triangulating velocity mesh...",
          finalizingVelocityLayer: "Finalizing velocity layer...",
          velocityLayerReady: "Velocity layer ready",
          loadingBasalFrictionMetadata: "Loading basal friction metadata...",
          loadingSurfaceChange: "Loading observed elevation change...",
          downloadingSurfaceChange: "Downloading observed elevation change",
          surfaceChangeReady: "Observed elevation change ready",
          loadingSeaIce: "Loading observed sea ice...",
          downloadingSeaIce: "Downloading observed sea ice",
          seaIceReady: "Observed sea ice ready",
          downloadingBasalFrictionField: "Downloading basal friction field",
          processingBasalFrictionField: "Processing basal friction field...",
          buildingBasalFrictionMesh: "Building basal friction mesh...",
          triangulatingBasalFrictionMesh: "Triangulating basal friction mesh...",
          finalizingBasalFrictionLayer: "Finalizing basal friction layer...",
          basalFrictionLayerReady: "Basal friction layer ready",
          loadingRiseMetadata: "Loading RISE metadata...",
          downloadingRiseOverlayPackage: "Downloading RISE overlay package...",
          buildingRiseOverlayMeshes: "Building RISE overlay meshes...",
          riseOverlaysReady: "RISE overlays ready",
          loadingOceanCurrentMetadata: "Loading ocean-current metadata...",
          applyingPrefetchedOceanData: "Applying prefetched ocean-current data...",
          downloadingOceanCurrentVectors: "Downloading ocean-current vectors",
          buildingOceanStreamlines: "Building ocean streamlines...",
          oceanStreamlinesReady: "Ocean streamlines ready",
          loadingHydrologyMetadata: "Loading hydrology metadata...",
          downloadingHydrologyField: "Downloading hydrology field",
          processingHydrologyField: "Processing hydrology field...",
          buildingHydrologyMesh: "Building hydrology mesh...",
          triangulatingHydrologyMesh: "Triangulating hydrology mesh...",
          buildingChannelRibbons: "Building channel ribbons...",
          finalizingHydrologyLayer: "Finalizing hydrology layer...",
          hydrologyLayerReady: "Hydrology layer ready",
        },
        rebound: {
          years: "about {value} yr",
          kiloyears: "about {value} kyr",
          progressNote:
            "{percent}% of the equilibrium rebound, reached {elapsed} after an instantaneous deglaciation (relaxation time {tau} yr). Ice thickness and bed relaxation advance together here; that coupling is illustrative, not a transient simulation.",
          progressNoteComplete:
            "Full equilibrium rebound. With this model's single relaxation time ({tau} yr) that is effectively reached 15-20 kyr after an instantaneous deglaciation; the real mantle relaxes over several timescales, and full re-equilibration takes of order 100 kyr. Beneath the Amundsen Sea Embayment the mantle is far weaker and responds within decades to centuries.",
          progressNotePublished:
            "{percent}% of the published equilibrium response. The bed and the ice advance together as a straight interpolation between today and full re-equilibration; the response has no single timescale, so no elapsed time is implied.",
          progressNotePublishedComplete:
            "Full re-equilibration, of order 100 kyr after the ice is gone (Paxman et al., 2022). Beneath the Amundsen Sea Embayment the mantle is far weaker and responds within decades to centuries.",
          modelNotePublished:
            "Published response of Paxman, Austermann & Hollyday (2022), grids v3: elastic-plate flexure with laterally variable elastic thickness ({teModel}), plus the post-LGM rebound still to come and the load of the seawater that floods the rebounded bed.",
          modelNotePublishedQrf:
            "The published grid was computed for the BedMachine Greenland v6 ice load; the QRF bed changes the flexural response by about 10 m RMS, well inside the model spread.",
          modelNoteFlexural:
            "Idealised: a thin elastic plate on a fluid asthenosphere, D = 1e25 N m, flexural length scale {lengthScale} km. Lithospheric strength spreads the response, lowering peak uplift; beyond the former margin, the forebulge that the ice had raised subsides slightly.",
          modelNoteLocal:
            "Idealised: Airy isostasy applied column by column, with no lithospheric strength. This is the upper bound on peak uplift and shows more short-wavelength detail than the solid Earth can actually support.",
          seaLevelNoteZero:
            "Ocean held at today's datum. Melting this ice would itself raise global mean sea level by about {sle} m, which this setting deliberately leaves out so the bed motion can be read on its own.",
          seaLevelNoteRaised:
            "Ocean raised by {datum} m, applied as a uniform global datum. Near the former ice sheet the sea surface also sits lower because the ice's gravitational pull is gone; once the mantle has fully relaxed the removed ice mass is largely compensated by inflow beneath, so that geoid drop is of order tens of metres rather than hundreds. The much larger near-field sea-level fall usually quoted is mostly the bedrock uplift, which this view already shows.",
          seaLevelNotePublished:
            "The sea surface is part of the published model: meltwater from both ice sheets (+{eustatic} m) plus the residual post-LGM geoid change, {gmin}-{gmax} m above today's over this ice sheet. Heights and emergence are shown relative to that ice-free sea surface; the datum slider applies to the idealised responses only.",
          seaLevelNotePublishedGeneric:
            "The sea surface is part of the published model: meltwater from both ice sheets plus the residual post-LGM geoid change. Heights and emergence are shown relative to that ice-free sea surface; the datum slider applies to the idealised responses only.",
          seaLevelValuePublished: "set by the model",
          legendNote:
            "Colours follow the active sea-level datum, so the land/ocean break tracks the waterline. Newly emergent land is tinted orange. Ice shelves contribute no uplift: floating ice already displaces its own weight of seawater.",
        },
        surfaceChange: {
          regionAis: "Antarctic Ice Sheet since 1995",
          regionWais: "West Antarctica",
          regionEais: "East Antarctica",
          regionApis: "Antarctic Peninsula",
          massValue: "{value}\u00a0\u00b1\u00a0{sigma}\u00a0Gt",
          readoutSeaLevel: "Sea level from Antarctica",
          seaLevelValue: "{value} mm",
          readoutPeriod: "Period",
          periodValue: "{first}–{last}, linear trend",
          readoutSignificant: "Cells beyond 2σ",
          noteCumulative: "Satellite radar altimetry (ERS-2, Envisat, CryoSat-2) processed by CPOM: how far the surface of the grounded ice has risen (blue) or fallen (red) since 1995, every 140 days. The totals are CPOM's own mass changes; 362.5 Gt of ice is 1 mm of global sea level. Ice shelves, islands and ice rises are not covered.",
          noteRate: "The linear rate of surface elevation change from 1995 to 2026, from CPOM's multi-mission radar altimetry. Ice shelves, islands and ice rises are not covered.",
          legendTitleCumulative: "Surface Elevation Change Since 1995",
          legendTitleRate: "Surface Elevation Change Rate",
          legendNoteCumulative: "Change in surface elevation since 1995, on a symmetric log scale that is linear within ±{softness} m. Faded: within 2σ of no change (σ at the end of the record); south of 88°S, which is always extrapolated; and south of 81.5°S until CryoSat-2 data arrive in late 2010, after which the change there still includes the extrapolated years. Ice without colour has no data.",
          legendNoteRate: "Rate of surface elevation change, 1995–2026, on a symmetric log scale that is linear within ±{softness} m/yr. Faded: within 2σ of no change, or extrapolated south of 88°S; south of 81.5°S the years before CryoSat-2 are extrapolated too. Ice without colour has no data.",
          sheetName: "Antarctic Ice Sheet",
          basinWhole: "Whole ice sheet",
          basinAllOf: "All of {region}",
          basinOption: "Basin {code}",
          basinName: "Basin {code}, {region}",
          readoutTotal: "{name}, {first}–{last}",
          chartLabel: "Cumulative mass change of {name} from {first} to {last}, with its 1σ band",
          chartNote: "CPOM's cumulative mass change, with a ±1σ band. Click the ice to pick one of its 18 IMBIE drainage basins; click it again for the whole ice sheet.",
        },
        seaIce: {
          arctic: "Arctic",
          antarctic: "Antarctic",
          extentLabel: "{region} extent",
          millionKm2: "{value} million km\u00b2",
          anomalyLabel: "vs {first}–{last}",
          anomalyValue: "{value} million km\u00b2 ({percent}%)",
          rankLabel: "Rank",
          rankLowest: "Lowest {month} of {count}",
          rankLow: "{ordinal} lowest {month} of {count}",
          rankHighest: "Highest {month} of {count}",
          rankHigh: "{ordinal} highest {month} of {count}",
          noDataGap: "No data: a gap in the satellite record",
          noDataOutside: "No data: the record runs from {first} to {last}",
          chartUnit: "M km\u00b2",
          chartTrend: "{value}% per decade",
          chartLabel: "{region} sea-ice extent in {month} from {first} to {last}, with its least-squares trend",
          chartNoteNorth:
            "Extent of the whole Arctic in this month of every year, from the same record: the ocean with at least 15% ice on at least half of the month's days. The Greenland view holds only part of it. Dashed: the least-squares trend since 1979, as a share of the 1981–2010 average.",
          chartNoteSouth:
            "Extent of the whole Southern Ocean in this month of every year, from the same record: the ocean with at least 15% ice on at least half of the month's days. Dashed: the least-squares trend since 1979, as a share of the 1981–2010 average.",
          note: "Monthly mean sea-ice concentration measured from space by passive microwave radiometers (SMMR, SSM/I, SSMIS; AMSR2 from 2025): the NOAA/NSIDC climate data record on a 25 km grid. Water with less than 15% ice is left clear. Near the coast, where land blurs the signal, the ice is carried in from the nearest observed cells.",
          legendNote:
            "Share of the sea surface covered by ice in the month; below 15% the water is left clear. Orange line: the month's median ice edge in 1981–2010, where ice was present in half of those years.",
        },
        projection: {
          play: "Play",
          pause: "Pause",
          scenarioAe10: "Low emissions (SSP1-2.6)",
          scenarioAe05: "High emissions (SSP5-8.5)",
          scenarioAe14: "High emissions with ice-shelf collapse",
          shortAe10: "SSP1-2.6",
          shortAe05: "SSP5-8.5",
          shortAe14: "+ shelf collapse",
          readoutSeaLevel: "Sea-level change from Antarctica",
          readoutRange: "Range of the models",
          readoutGroundedMap: "Grounded ice on the map",
          readoutGrounded: "Grounded ice",
          readoutFloating: "Floating ice",
          seaLevelValue: "{value} m",
          rangeValue: "{low} to {high} m",
          areaValue: "{value} million km\u00b2 ({change})",
          chartLabel:
            "Sea-level change from Antarctica from {first} to {last} for each scenario; the band is the range of the models",
          noteChange:
            "Red ice is thinner than in {first}, blue ice thicker. Grounded ice rests on the bed and floating ice sits at flotation. The flowlines follow the projected ice and turn brighter and faster where it speeds up.",
          noteType:
            "White is grounded ice, blue is floating ice, both from flotation on the bed. The flowlines follow the projected ice.",
          legendNote:
            "Change in ice thickness since {first}, on a symmetric log scale that is linear within \u00b110 m. Where the ice has gone, the bare bed or the open sea shows through.",
          ensembleNote:
            "Average change of {models} ISMIP6 ice sheet models (Seroussi et al. 2024) under UKESM1-0-LL climate forcing, applied to today's ice (BedMachine v4). The sea-level number is the average of the models' own results; because flotation is not linear, the averaged ice on the map implies somewhat more.",
        },
        legends: {
          warm: "Warm",
          cold: "Cold",
          fresh: "Fresh",
          salty: "Salty",
          effectivePressureNote:
            "Over-pressurised bed, where water pressure exceeds the ice overburden (N < 0), is not drawn.",
        },
        meta: {
          reboundMillionKm2: "{value} million km\u00b2",
          reboundVolumeOf: "{above} of {total} million km\u00b3",
          reboundSolveGridValue: "{cell} km grid, {sizeX} x {sizeY} transform",
          reboundSolveGridPointwise: "{cell} km grid, solved pointwise",
          reboundConvergenceValue: "{iterations} iterations, {residual} m residual",
          isostaticReboundSection: "Isostatic Rebound (modelled)",
          reboundModelLabel: "Earth response",
          reboundModelPublished: "Published: Paxman et al. (2022), grids v3",
          reboundModelFlexural: "Idealised regional flexure (elastic plate on a fluid asthenosphere)",
          reboundModelLocal: "Idealised local (Airy) isostasy",
          reboundElasticThickness: "Elastic thickness",
          reboundPublishedEarthModelValue: "{teModel}, laterally variable; densities ice 917, seawater 1028, mantle 3330 kg/m3",
          reboundSeaSurface: "Ice-free sea surface",
          reboundSeaSurfaceValue: "+{eustatic} m eustatic (both ice sheets) plus the residual post-LGM geoid: {gmin}-{gmax} m above today's under the grounded ice",
          reboundMaxTopographyChange: "Largest rise above the ice-free sea surface",
          reboundMeanTopographyChange: "Mean rise above the ice-free sea surface under grounded ice",
          reboundComponents: "Components under grounded ice",
          reboundComponentsValue: "ice unloading {iu} m (max {iuMax} m); post-LGM {lgm} m (max {lgmMax} m); water loading {wl} m (min {wlMin} m)",
          reboundSigma: "Earth-model spread (1 sigma) under grounded ice",
          reboundSigmaValue: "{mean} m mean, {max} m max",
          reboundLandAfterPublished: "Bed above the ice-free sea surface after rebound",
          reboundMarineUnderIcePublished: "Still below the ice-free sea surface under the present ice",
          reboundQrfLoadNote: "Load",
          reboundQrfLoadNoteValue: "computed for the BedMachine Greenland v6 ice load; the QRF bed changes the flexural response by about 10 m RMS",
          reboundRigidity: "Flexural rigidity",
          reboundLengthScale: "Flexural length scale",
          reboundRelaxation: "Relaxation time",
          reboundProgressLabel: "Scenario progress",
          reboundSeaLevelLabel: "Sea-level datum",
          reboundMaxUplift: "Maximum equilibrium uplift",
          reboundMeanGroundedUplift: "Mean uplift under grounded ice",
          reboundLandBefore: "Bed above sea level today",
          reboundLandAfter: "Bed above the datum after rebound",
          reboundEmergent: "Newly emergent land",
          reboundDrowned: "Present bedrock drowned by the datum",
          reboundMarineUnderIce: "Still below the datum under the present ice",
          reboundClosedBasins: "Closed basins below the datum",
          reboundDeepest: "Deepest grounded bed after rebound",
          reboundIceVolume: "Ice volume on the active grid",
          reboundVolumeAboveFlotation: "Volume above flotation",
          reboundSle: "Sea-level equivalent",
          reboundSlePublishedAntarctica:
            "above flotation, converted at freshwater density as Gregory et al. (2019) define sea-level equivalent; cf. 57.9 ± 0.9 m published for BedMachine Antarctica v1 (Morlighem et al., 2020). The 10 km and 4 km packages agree to 0.01%, and Bedmap3 gives 58.6 m on the same grids.",
          reboundSlePublishedGreenland:
            "above flotation, converted at freshwater density as Gregory et al. (2019) define sea-level equivalent; cf. 7.42 ± 0.05 m published for BedMachine Greenland v3 (Morlighem et al., 2017).",
          reboundSolveGrid: "Flexure solve grid",
          reboundConvergence: "Picard convergence",
          reboundAssumptions: "Assumptions",
          reboundAssumptionsText:
            "Idealised equilibrium response to removing the present ice load, not a transient simulation. The present bed is taken as being in balance with the present load, which it is not: part of the rebound from the Last Glacial Maximum is still to come (up to +68 m of bed elevation under the Ross and Weddell embayments, while the collapsing Laurentide forebulge lowers Greenland by up to 25 m; Paxman et al., 2022), and the Amundsen Sea Embayment is rising at up to 41 mm/yr in response to recent ice loss over a weak mantle. Sea-level fingerprinting, geoid change and rotational feedback are omitted, and a single global rigidity and mantle density replace real lateral structure.",
          reboundAssumptionsTextPublished:
            "Fully re-equilibrated response to removing both ice sheets. The post-LGM correction uses a one-dimensional mantle viscosity profile, which is least reliable where the upper mantle is weak, as beneath West Antarctica and East Greenland. The sea surface rises by the eustatic amount plus the residual post-LGM geoid change; thermosteric and dynamic sea level, erosion and sedimentation are not included, and every point below the ice-free sea surface is loaded with seawater.",
          reboundMethod: "Method",
          reboundMethodText:
            "Thin-plate flexure D grad^4 u + rho_m g u = sigma_now - sigma_after, solved spectrally, with the post-deglaciation water load resolved by Picard iteration over the ocean-connected footprint. Densities: ice 917, seawater 1027, mantle 3300 kg/m3.",
          reboundMethodTextPublished:
            "Published total isostatic response T = R - G, point-sampled at this grid's nodes, where R is the solid-surface displacement and G the sea-surface change. The bed is drawn at bed + T, so heights read directly against the ice-free sea surface. Grid files: NSF Arctic Data Center, doi:10.18739/A22Z12R8C (CC BY 4.0).",
          errorLabel: "Error",
          geometrySection: "Geometry & Grid",
          velocitySection: "Ice Surface Velocity (observed)",
          basalFrictionSection: "Basal Friction (inverted)",
          hydrologySection: "Subglacial Hydrology (simulated)",
          oceanSection: "Ocean Circulation (simulated)",
          riseSection: "Ice-Shelf Basal Melt (simulated)",
          sourcesSection: "Data Sources",
          availableOnDemand: "Available on demand",
          notYetAdded: "Not yet added for {region}",
          preset: "Preset",
          grid: "Grid",
          projection: "Projection",
          bedElevation: "Bed Elevation",
          maxIceThickness: "Max Ice Thickness",
          meanIceThickness: "Mean Ice Thickness",
          surfaceSpeedRange: "Surface Speed Range",
          speedQuantiles: "Speed Quantiles",
          selectedFlowlineSection: "Ice Flowline Profile",
          selectedFlowlineHint: "Click a visible ice flowline to preview its along-flow silhouette here.",
          selectedFlowlineEmptyTitle: "Pick a flowline",
          selectedFlowlineEmptyBody:
            "Click any visible ice flowline to reveal its along-flow silhouette, including surface, ice base, and bedrock.",
          selectedFlowlineDisabledTitle: "Flowlines are hidden",
          selectedFlowlineDisabledBody: "Turn on Show Flowlines, then click a line to preview its profile silhouette.",
          selectedFlowlineLabel: "Flowline {index}",
          selectedFlowlineLength: "Along-flow Length",
          selectedFlowlineSpeedRange: "Flow Speed",
          selectedFlowlineSurfaceSpan: "Surface Elevation",
          selectedFlowlineThickness: "Ice Thickness",
          selectedFlowlinePreview: "Profile Silhouette",
          status: "Status",
          invertedFriction:
            "Inverted Friction",
          invertedFrictionSummary:
            "Median of ensemble inversions using Elmer/Ice, based on the Shallow Shelf Approximation",
          frictionRange: "Friction Range",
          frictionQuantiles: "Friction Quantiles",
          basalMeltRate: "Basal Melt Rate",
          meltQuantiles: "Melt Quantiles",
          thermalDriving: "Thermal Driving",
          thermalQuantiles: "Thermal Quantiles",
          iceDraft: "Ice Draft",
          draftQuantiles: "Draft Quantiles",
          oceanStreamlines: "Ocean streamlines",
          depthSpan: "3D Depth Span",
          horizontalSpeed: "Horizontal Speed",
          waterMassColor: "Water-Mass Color",
          fourCornerPalette: "Four-corner temperature-salinity palette",
          effectivePressureRange: "Effective Pressure Range",
          pressureQuantiles: "Pressure Quantiles",
          channelDischargeRange: "Channel Discharge Range",
          dischargeQuantiles: "Discharge Quantiles",
          renderedChannels: "Rendered Channels",
          sourceBedIceGeometry: "Bed/Ice Geometry",
          sourceBasinBoundaries: "Basin Boundaries",
          sourceSurfaceVelocity: "Surface Velocity",
          sourceBasalFriction: "Basal Friction",
          sourceSubglacialHydrology: "Subglacial Hydrology",
          sourceOceanCirculation: "Ocean Circulation",
          sourceMeltDrivers: "Melt Drivers",
          // Suffixes bring their own spacing, which differs between the locales.
          iceShelfCellCount: " ({count} ice-shelf cells)",
          streamlineCountSuffix: " ({count} 3D streamlines)",
          gridSpacingSuffix: " ({spacing} km grid)",
          valueRange: "{min} to {max}",
          quantileMedian: "median",
          notApplicable: "n/a",
          polarStereographic: "{epsg} polar stereographic",
          polarStereographicGeneric: "Polar stereographic",
          atGridPosition: "{value} m at x = {x} km, y = {y} km ({epsg})",
          yearsValue: "{value} yr",
          sourceIsostaticResponse: "Isostatic Response",
          sourceIceProjection: "Ice-sheet Projection",
          iceProjectionSection: "Ice-sheet projection",
          sourceSurfaceChange: "Surface Elevation Change",
          surfaceChangeSection: "Observed elevation change (CPOM)",
          sourceSeaIce: "Sea Ice",
          seaIceSection: "Observed sea ice (NSIDC)",
          seaIceMonth: "Month shown",
          seaIceMonthValue: "{date} (one of {count} months, {first} to {last})",
          seaIceExtent: "Extent and area",
          seaIceExtentValueNorth: "{extent} and {area} million km\u00b2, whole Arctic",
          seaIceExtentValueSouth: "{extent} and {area} million km\u00b2, whole Southern Ocean",
          seaIceTrend: "{month}: average and trend",
          seaIceTrendValue: "{mean} million km\u00b2 in 1981–2010; {slope} million km\u00b2 ({percent}%) per decade, {first}–{last}",
          seaIceSensor: "Sensor",
          seaIceCoast: "Near the coast",
          seaIceCoastValue: "{share}% of the 25 km cells take their concentration from their neighbours, where NSIDC's land mask leaves them no ocean cell",
          seaIceIndex: "Against the Sea Ice Index",
          seaIceIndexValue: "on average {percent}% above that of NSIDC's Sea Ice Index, which averages daily extents from the NASA Team algorithm; the ice the map draws, every cell whose monthly mean reaches 15%, is {meanField}% above it",
          seaIceMethod: "Method",
          seaIceMethodValue: "the map draws the monthly mean CDR concentration (the higher of the NASA Team and Bootstrap estimates in each cell), resampled bilinearly from NSIDC's polar stereographic grid (Hughes 1980 ellipsoid) onto the terrain's ocean; the extent counts the hemisphere's ocean cells with at least 15% ice on at least half of the month's days (the files' QA flag), on NSIDC's grid",
          surfaceChangeEpoch: "Epoch shown",
          surfaceChangeEpochValue: "{date} (one of {count} 140-day epochs, {first} to {last})",
          surfaceChangeRange: "Change on the map",
          surfaceChangeRangeValue: "{min} to {max} {unit}",
          surfaceChangeSignificant: "Beyond 2σ",
          surfaceChangeSignificantValue: "{share}% of the cells drawn",
          surfaceChangeExtrapolated: "Extrapolated near the pole",
          surfaceChangeExtrapolatedValue: "south of 81.5°S before {date}, south of 88°S throughout",
          surfaceChangePeriod: "Period",
          surfaceChangeCells: "Coverage",
          surfaceChangeCellsValue: "grounded ice in CPOM's mask (BedMachine v2), drawn on every explorer ice cell it covers by at least half, floating cells next to the grounding line included; no ice shelves, islands or ice rises",
          surfaceChangeMethod: "Method",
          surfaceChangeMethodValue: "plane fit in each 5 km cell, 140-day epochs, IJ05 R2 glacial isostatic adjustment removed, missions cross-calibrated (Shepherd et al. 2019); the change is relative to CPOM's 1995 reference level, so the first epoch is not zero; area mean onto the explorer's grid",
          surfaceChangeMassSince: "Mass change since 1995 (CPOM)",
          surfaceChangeMassPeriod: "Mass change {first}–{last} (CPOM)",
          surfaceChangeMassPart: "{name} {value}",
          surfaceChangeMassSeparator: "; ",
          surfaceChangeSelected: "Selected",
          surfaceChangeSelectedValue: "{name}: {value}",
          iceProjectionExperiment: "Experiment",
          iceProjectionExperimentValue: "ISMIP6 {id}: {climate} {scenario}",
          iceProjectionModels: "Models averaged",
          iceProjectionModelsValue: "{count}, the main submission of each group that ran all three scenarios: {names}",
          iceProjectionYear: "Year shown",
          iceProjectionYearMeanValue: "{year}: sea-level rise {sle} m (models {low} to {high} m)",
          iceProjectionSeaLevel: "Sea-level contribution",
          iceProjectionSeaLevelMeanValue:
            "Mean of the models' published values (Seroussi & Pelle 2024) relative to {first}, control run not subtracted: {end} m by {last}. The averaged ice on the map implies {packed} m.",
          iceProjectionMethod: "Method",
          iceProjectionMethodValue:
            "Each of the {count} models resampled conservatively to 10 km; their mean change since 2015 in thickness and in depth-averaged speed is applied to BedMachine v4 ice. Thickening is added; thinning is scaled by BedMachine over the models' own 2015 thickness, so each cell keeps the share of its ice that the models keep, and where the models average under 10 m the ice is gone. The map's ice-volume change is {captured}% of the models' mean.",
          iceProjectionKeyframes: "Keyframes",
          iceProjectionKeyframesMeanValue:
            "{count} keyframes every {interval} years, {first}-{last}, blended linearly; {cells} cells of present-day ice",
          currentScope: "Current {region} scope",
          currentScopeSummary: "Bed topography, ice surface, and ice base are enabled in this version.",
          fieldStatsBed: "Bed",
          fieldStatsIceMax: "Ice max",
          fieldStatsVelocityMax: "Velocity max",
          fieldStatsOceanStreamlines: "Ocean streamlines",
          fieldStatsTauHighEnd: "Tau_b",
          fieldStatsTauHighEndSuffix: "high-end",
          fieldStatsBasalMeltMax: "Basal melt",
          fieldStatsBasalMeltMaxSuffix: "max",
          fieldStatsThermalDrivingMax: "Thermal driving",
          fieldStatsThermalDrivingMaxSuffix: "max",
          fieldStatsBasins: "Basins",
          oceanWaterMassSummary:
            "Cold-fresh cyan, cold-salty indigo, warm-fresh green, warm-salty orange-red ({thetaMin} to {thetaMax} °C; {salinityMin} to {salinityMax} PSU)",
        },
        errors: {
          reboundGridTooSmall: "Isostatic-rebound solver needs a grid of at least 2x2 cells.",
          failedToLoadIceProjection: "Failed to load ice-sheet projection ({status})",
          iceProjectionGridMismatch: "The ice-sheet projection is on a different grid from the terrain.",
          reboundTransformTooLarge: "Isostatic-rebound solver grid exceeds the supported transform size.",
          workerTaskFailed: "Worker task failed",
          workerCrashed: "Geometry worker crashed",
          workerTerminated: "Geometry worker terminated",
          fullscreenUnavailable: "Fullscreen API unavailable.",
          regionSwitchUnavailable: "Region switching is unavailable in this explorer mode.",
          regionLoadTimedOut: "Timed out loading {region}.",
          layersLoadTimedOut: "Timed out waiting for the layers to load.",
          failedToLoadMetadata: "Failed to load metadata ({status})",
          failedToLoadTerrainPackage: "Failed to load terrain package ({status})",
          failedToLoadVelocityMetadata: "Failed to load velocity metadata ({status})",
          failedToLoadBasalFrictionMetadata: "Failed to load basal-friction metadata ({status})",
          failedToLoadHydrologyMetadata: "Failed to load hydrology metadata ({status})",
          failedToLoadRiseMetadata: "Failed to load RISE metadata ({status})",
          failedToLoadOceanCurrentMetadata: "Failed to load ocean-current metadata ({status})",
          failedToLoadVelocityField: "Failed to load velocity field ({status})",
          failedToLoadBasalFrictionField: "Failed to load basal-friction field ({status})",
          failedToLoadSurfaceChangeMetadata: "Failed to load surface-change metadata ({status})",
          failedToLoadSurfaceChangeField: "Failed to load surface-change field ({status})",
          failedToLoadSeaIceMetadata: "Failed to load sea-ice metadata ({status})",
          failedToLoadSeaIceField: "Failed to load sea-ice field ({status})",
          failedToLoadHydrologyField: "Failed to load hydrology field ({status})",
          failedToLoadReboundMetadata: "Failed to load isostatic-response metadata ({status})",
          failedToLoadReboundField: "Failed to load isostatic-response field ({status})",
          failedToLoadRiseOverlayPackage: "Failed to load RISE overlay package ({status})",
          failedToLoadOceanCurrentVectors: "Failed to load ocean-current vectors ({status})",
          failedToLoadBasinBoundaries: "Failed to load basin boundaries ({status})",
          failedToLoadBedColorTable: "Failed to load GMT_relief color table ({status})",
          failedToLoadEffectivePressureColorTable: "Failed to load cmocean_dense color table ({status})",
          failedToLoadChannelColorTable: "Failed to load cmocean_matter color table ({status})",
          unexpectedFieldLength: "Unexpected field length in data package.",
          unexpectedRiseFieldLength: "Unexpected field length in the RISE package.",
          riseGridMisaligned: "RISE grid is not aligned to the active BedMachine grid.",
          velocityGridMisaligned: "Velocity grid is not aligned to BedMachine grid.",
          velocityPayloadInvalid: "Velocity payload is invalid.",
          velocityTextureFieldLengthMismatch: "Velocity texture field length mismatch.",
          scalarFieldTextureLengthMismatch: "Scalar field texture length mismatch.",
          basalFrictionGridMisaligned: "Basal-friction grid is not aligned to BedMachine grid.",
          hydrologyGridMisaligned: "Hydrology grid is not aligned to BedMachine grid.",
          oceanPackageMisaligned: "Ocean-current package fields are misaligned.",
          oceanDatasetEmpty: "Ocean-current dataset is empty after regional clipping.",
          basinDatasetEmpty: "Basin dataset is empty.",
        },
      },
      worker: {
        progress: {
          reboundSolvingFlexure: "Solving isostatic rebound...",
          oceanDecodingPackage: "Decoding ocean-current package...",
          oceanScanningSegments: "Scanning ocean-current segments...",
          oceanBuildingGeometry: "Building ocean streamline geometry...",
          oceanFinalizing: "Finalizing ocean streamlines...",
          velocityDecodingField: "Decoding velocity field...",
          velocityBuildingMesh: "Building velocity mesh...",
          velocityTriangulatingMesh: "Triangulating velocity mesh...",
          velocityFinalizing: "Finalizing velocity layer...",
          hydrologyProcessingField: "Processing hydrology field...",
          hydrologyBuildingMesh: "Building hydrology mesh...",
          hydrologyTriangulatingMesh: "Triangulating hydrology mesh...",
          hydrologyBuildingChannels: "Building channel ribbons...",
          hydrologyFinalizing: "Finalizing hydrology layer...",
          basalFrictionProcessingField: "Processing basal friction field...",
          basalFrictionBuildingMesh: "Building basal friction mesh...",
          basalFrictionTriangulatingMesh: "Triangulating basal friction mesh...",
          basalFrictionFinalizing: "Finalizing basal friction layer...",
        },
      },
    },
    "zh-CN": {
      shared: {
        switcherLabel: "语言",
        localeEnglish: "EN",
        localeChinese: "中文",
        localeEnglishLong: "英文",
        localeChineseLong: "简体中文",
      },
      home: {
        switcherTitle: "语言",
      },
      explorer: {
        regions: {
          antarctica: {
            label: "南极洲",
            intro:
              "全方位，探秘南极。旋转、缩放、穿透重重冰盖，在 3D 交互中，唤醒沉睡的冰下世界。",
            basinToggleLabel: "显示细化流域",
            basinStatusLabel: "细化流域",
          },
          greenland: {
            label: "格陵兰",
            intro:
              "全方位，探秘格陵兰。旋转、缩放、穿透重重冰盖，在 3D 交互中，唤醒沉睡的冰下世界。",
            basinToggleLabel: "显示流域",
            basinStatusLabel: "流域",
          },
        },
        datasets: {
          antarctica: {
            balanced: {
              label: "BedMachine v4 — 标准",
              summary: "10 km 网格；约 3.0 MB",
            },
            hd: {
              label: "BedMachine v4 — 高清",
              summary: "4 km 网格；约 18.6 MB，推荐桌面端",
            },
            bedmap3: {
              label: "Bedmap3 — 标准",
              summary: "10 km 网格；约 3.0 MB",
            },
            "bedmap3-hd": {
              label: "Bedmap3 — 高清",
              summary: "4 km 网格；约 18.6 MB，推荐桌面端",
            },
          },
          greenland: {
            "3km": {
              label: "标准",
              summary: "3 km 网格；约 3.1 MB",
            },
            "1km": {
              label: "高清",
              summary: "1 km 网格；约 28.2 MB",
            },
            qrf: {
              label: "QRF 2025 — 标准",
              summary: "3 km 网格；约 3.1 MB",
            },
            "qrf-hd": {
              label: "QRF 2025 — 高清",
              summary: "1 km 网格；约 28.2 MB，推荐桌面端",
            },
          },
        },
        interaction: {
          showcaseIdleHtml:
            "<strong>页面滚动：</strong>自然滑动即可。<br /><strong>轻触场景</strong>进入 3D。",
          showcaseActiveHtml:
            "<strong>3D 已启用：</strong>拖拽旋转，双指缩放。<br /><strong>轻触滚动</strong>返回页面滚动。",
          touchIdleHtml:
            "<strong>页面滚动：</strong>自然滑动即可。<br /><strong>轻触场景</strong>启用 3D 手势。",
          touchActiveHtml:
            "<strong>3D 已启用：</strong>拖拽旋转，双指缩放。<br /><strong>轻触滚动</strong>返回页面滚动。",
          enable3d: "启用 3D",
          scrollPage: "滚动页面",
        },
        drawer: {
          open: "图层与地点",
          close: "完成",
        },
        capture: {
          play: "播放",
          stop: "停止",
          shown: "已显示录制控制面板。",
          hiddenHint: "已隐藏录制控制面板。按 H 可再次显示。",
          enableOrbitOrZoomFirst: "请先启用轨道旋转或缩放脉冲。",
        },
        fullscreen: {
          enter: "进入全屏",
          exit: "退出全屏",
          unavailable: "当前无法全屏",
          blocked: "全屏请求被浏览器拦截",
        },
        status: {
          loadingIsostaticRebound: "正在求解地壳回弹...",
          loadingIsostaticResponse: "正在加载已发表的地壳均衡响应...",
          loadingIceProjection: "正在加载冰盖未来预估...",
          iceProjectionUnavailable: "冰盖未来预估不可用",
          isostaticReboundUnavailable: "地壳回弹图层不可用",
          ready: "就绪",
          readyWithContext: "就绪（{region} {dataset}）",
          previewReady: "预览已就绪",
          loadingPreview: "正在加载 {region} 预览...",
          loadingCoreTerrain: "正在加载 {region} {dataset} 核心地形...",
          loadingRegion: "正在加载 {region}...",
          loadingBasin: "正在加载 {label}...",
          loadingBasalFriction: "正在加载基底摩擦...",
          loadingSurfaceChange: "正在加载观测到的高程变化...",
          surfaceChangeUnavailable: "观测到的高程变化图层当前不可用",
          loadingSeaIce: "正在加载观测到的海冰...",
          seaIceUnavailable: "观测到的海冰图层当前不可用",
          loadingVelocityLayer: "正在加载流速图层...",
          loadingOceanStreamlines: "正在加载海洋流线...",
          loadingHydrology: "正在加载冰下水文...",
          computingFlowlines: "正在计算流线...",
          loadFailed: "加载失败",
          viewCopied: "视角参数已复制到剪贴板",
          viewLogged: "视角参数已输出到控制台",
          refinedBasinsUnavailable: "{label}当前不可用",
          velocityUnavailable: "流速图层当前不可用",
          basalFrictionUnavailable: "基底摩擦图层当前不可用",
          oceanUnavailable: "海洋流线当前不可用",
          hydrologyUnavailable: "水文图层当前不可用",
          flowlineProfileHint: "点击任意流线查看冰体与基岩剖面",
        },
        loading: {
          solvingIsostaticRebound: "正在求解地壳回弹...",
          loadingIsostaticResponse: "正在加载已发表的地壳均衡响应...",
          downloadingIsostaticResponse: "正在下载地壳均衡响应",
          downloadingIceProjection: "正在下载冰盖未来预估",
          buildingIceProjection: "正在构建冰盖未来预估...",
          iceProjectionReady: "冰盖未来预估已就绪",
          isostaticReboundReady: "地壳回弹已就绪",
          initializingRuntime: "正在初始化 3D 运行时...",
          initializingRenderer: "正在初始化渲染器...",
          progressiveLoadingEnabled: "已启用渐进式加载",
          loadingThreeRuntime: "正在加载 Three.js 运行时...",
          preparingDataStreams: "正在准备数据流...",
          downloadingTerrainPackage: "正在下载地形数据包",
          decodingTerrainFields: "正在解码地形字段...",
          buildingBaseMeshes: "正在构建基础网格...",
          coreTerrainReady: "核心地形已就绪",
          loadingVelocityMetadata: "正在加载流速元数据...",
          applyingPrefetchedVelocityData: "正在应用预取的流速数据...",
          downloadingVelocityField: "正在下载流速场",
          decodingVelocityField: "正在解码流速场...",
          buildingVelocityMesh: "正在构建流速网格...",
          triangulatingVelocityMesh: "正在三角化流速网格...",
          finalizingVelocityLayer: "正在完成流速图层...",
          velocityLayerReady: "流速图层已就绪",
          loadingBasalFrictionMetadata: "正在加载基底摩擦元数据...",
          loadingSurfaceChange: "正在加载观测到的高程变化...",
          downloadingSurfaceChange: "正在下载观测到的高程变化",
          surfaceChangeReady: "观测到的高程变化图层已就绪",
          loadingSeaIce: "正在加载观测到的海冰...",
          downloadingSeaIce: "正在下载观测到的海冰",
          seaIceReady: "观测到的海冰图层已就绪",
          downloadingBasalFrictionField: "正在下载基底摩擦场",
          processingBasalFrictionField: "正在处理基底摩擦场...",
          buildingBasalFrictionMesh: "正在构建基底摩擦网格...",
          triangulatingBasalFrictionMesh: "正在三角化基底摩擦网格...",
          finalizingBasalFrictionLayer: "正在完成基底摩擦图层...",
          basalFrictionLayerReady: "基底摩擦图层已就绪",
          loadingRiseMetadata: "正在加载 RISE 元数据...",
          downloadingRiseOverlayPackage: "正在下载 RISE 叠加层数据包...",
          buildingRiseOverlayMeshes: "正在构建 RISE 叠加层网格...",
          riseOverlaysReady: "RISE 叠加层已就绪",
          loadingOceanCurrentMetadata: "正在加载海洋流场元数据...",
          applyingPrefetchedOceanData: "正在应用预取的海洋流场数据...",
          downloadingOceanCurrentVectors: "正在下载海洋流场矢量",
          buildingOceanStreamlines: "正在构建海洋流线...",
          oceanStreamlinesReady: "海洋流线已就绪",
          loadingHydrologyMetadata: "正在加载水文元数据...",
          downloadingHydrologyField: "正在下载水文字段",
          processingHydrologyField: "正在处理水文字段...",
          buildingHydrologyMesh: "正在构建水文网格...",
          triangulatingHydrologyMesh: "正在三角化水文网格...",
          buildingChannelRibbons: "正在构建通道带状网格...",
          finalizingHydrologyLayer: "正在完成水文图层...",
          hydrologyLayerReady: "水文图层已就绪",
        },
        rebound: {
          years: "约 {value} 年",
          kiloyears: "约 {value} 千年",
          progressNote:
            "达到平衡态回弹的 {percent}%，相当于瞬时冰消后 {elapsed}（松弛时间 {tau} 年）。此处冰厚减薄与基岩回弹同步推进；这一耦合仅为示意，并非瞬态模拟。",
          progressNoteComplete:
            "完全平衡态回弹。按本模型单一的松弛时间（{tau} 年），瞬时冰消后约 15-20 千年即可达到；但真实地幔有多个松弛时间尺度，完全重新平衡需要约 10 万年量级。阿蒙森海湾之下的地幔要软弱得多，其响应时间仅为数十年至数百年。",
          progressNotePublished:
            "已发表平衡态响应的 {percent}%。基岩与冰体在当前状态与完全重新平衡之间按线性插值同步推进；该响应没有单一的时间尺度，因此不对应具体的经过时间。",
          progressNotePublishedComplete:
            "完全重新平衡，约在冰体消失后 10 万年量级达到（Paxman 等，2022）。阿蒙森海湾之下的地幔要软弱得多，其响应时间仅为数十年至数百年。",
          modelNotePublished:
            "Paxman、Austermann 与 Hollyday（2022）发表的响应，网格第 3 版：弹性厚度横向变化（{teModel}）的弹性板挠曲，加上尚未完成的末次盛冰期后回弹，以及淹没回弹后基岩的海水载荷。",
          modelNotePublishedQrf:
            "已发表网格按 BedMachine Greenland v6 的冰载荷计算；QRF 基岩使挠曲响应改变约 10 m（均方根），远小于模型离散度。",
          modelNoteFlexural:
            "理想化：流体软流圈之上的薄弹性板，D = 1e25 N m，挠曲特征长度 {lengthScale} km。岩石圈强度使响应向外扩展，因而峰值抬升更低；原冰缘之外，冰载荷曾抬起的前缘隆起会轻微沉降。",
          modelNoteLocal:
            "理想化：逐列应用 Airy 地壳均衡，不考虑岩石圈强度。这给出峰值抬升的上界，且其短波细节超出固体地球实际能够支撑的程度。",
          seaLevelNoteZero:
            "海面保持在当前基准。融化这些冰本身会使全球平均海平面上升约 {sle} m，此设置有意将其排除，以便单独解读基岩的运动。",
          seaLevelNoteRaised:
            "海面抬升 {datum} m，按均匀的全球基准施加。在原冰盖附近，由于冰体的引力吸引消失，海面也会相对偏低；但在地幔完全松弛之后，被移除的冰质量已在其下方由物质流入大致补偿，因此该大地水准面降幅仅为数十米量级，而非数百米。通常引用的近场海平面大幅下降，主要来自基岩抬升——而这一部分本视图已经呈现。",
          seaLevelNotePublished:
            "海面是已发表模型的一部分：两个冰盖的融水（+{eustatic} m）加上末次盛冰期后残余的大地水准面变化，在该冰盖上方比现今高 {gmin}-{gmax} m。高程与出露均相对这一无冰海面显示；海平面基准滑块仅适用于理想化响应。",
          seaLevelNotePublishedGeneric:
            "海面是已发表模型的一部分：两个冰盖的融水加上末次盛冰期后残余的大地水准面变化。高程与出露均相对这一无冰海面显示；海平面基准滑块仅适用于理想化响应。",
          seaLevelValuePublished: "由模型给定",
          legendNote:
            "配色跟随当前的海平面基准，因此陆海分界线始终与水线一致。新出露的陆地以橙色标示。冰架不产生抬升：漂浮的冰早已排开与自身等重的海水。",
        },
        surfaceChange: {
          regionAis: "1995 年以来南极冰盖",
          regionWais: "西南极",
          regionEais: "东南极",
          regionApis: "南极半岛",
          massValue: "{value}\u00a0\u00b1\u00a0{sigma}\u00a0Gt",
          readoutSeaLevel: "南极对海平面的贡献",
          seaLevelValue: "{value} mm",
          readoutPeriod: "时段",
          periodValue: "{first}–{last} 年，线性趋势",
          readoutSignificant: "超过 2σ 的网格",
          noteCumulative: "由 CPOM 处理的卫星雷达测高数据（ERS-2、Envisat、CryoSat-2）：自 1995 年以来，每 140 天接地冰表面升高（蓝）或降低（红）了多少。合计值是 CPOM 自己的质量变化；362.5 Gt 冰相当于全球海平面 1 mm。不含冰架、岛屿和冰隆。",
          noteRate: "CPOM 多卫星雷达测高给出的 1995–2026 年冰面高程线性变化速率。不含冰架、岛屿和冰隆。",
          legendTitleCumulative: "相对 1995 年的冰面高程变化",
          legendTitleRate: "冰面高程变化速率",
          legendNoteCumulative: "相对 1995 年的冰面高程变化，采用对称对数色标，±{softness} m 以内为线性。淡化表示：变化在 2σ 以内（σ 取记录末期的值）；88°S 以南，那里始终是外推值；以及 2010 年末 CryoSat-2 数据到来之前的 81.5°S 以南，此后那里的变化仍包含此前外推的年份。未着色的冰面没有数据。",
          legendNoteRate: "1995–2026 年冰面高程变化速率，采用对称对数色标，±{softness} m/yr 以内为线性。淡化表示变化在 2σ 以内，或是 88°S 以南的外推值；81.5°S 以南 CryoSat-2 之前的年份同样是外推。未着色的冰面没有数据。",
          sheetName: "南极冰盖",
          basinWhole: "整个冰盖",
          basinAllOf: "{region}全部",
          basinOption: "{code} 流域",
          basinName: "{region} {code} 流域",
          readoutTotal: "{name}，{first}–{last} 年",
          chartLabel: "{name} {first}–{last} 年的累积质量变化及其 1σ 范围",
          chartNote: "CPOM 的累积质量变化，阴影为 ±1σ。点击冰面可选择 18 个 IMBIE 流域之一，再次点击则回到整个冰盖。",
        },
        seaIce: {
          arctic: "北极",
          antarctic: "南极",
          extentLabel: "{region}海冰范围",
          millionKm2: "{value} 百万 km\u00b2",
          anomalyLabel: "相对 {first}–{last} 年",
          anomalyValue: "{value} 百万 km\u00b2（{percent}%）",
          rankLabel: "排名",
          rankLowest: "{count} 个{month}中最低",
          rankLow: "{count} 个{month}中第 {place} 低",
          rankHighest: "{count} 个{month}中最高",
          rankHigh: "{count} 个{month}中第 {place} 高",
          noDataGap: "无数据：卫星记录在此中断",
          noDataOutside: "无数据：记录始于 {first}，止于 {last}",
          chartUnit: "百万 km\u00b2",
          chartTrend: "每十年 {value}%",
          chartLabel: "{first} 至 {last} 年历年{month}的{region}海冰范围及其最小二乘趋势",
          chartNoteNorth:
            "整个北极历年该月份的海冰范围，来自同一数据记录：当月至少一半天数海冰覆盖率达到 15% 的海域。格陵兰视图只包含其中一部分。虚线：1979 年以来的最小二乘趋势，以 1981–2010 年平均值的百分比表示。",
          chartNoteSouth:
            "整个南大洋历年该月份的海冰范围，来自同一数据记录：当月至少一半天数海冰覆盖率达到 15% 的海域。虚线：1979 年以来的最小二乘趋势，以 1981–2010 年平均值的百分比表示。",
          note: "卫星被动微波辐射计（SMMR、SSM/I、SSMIS；2025 年起为 AMSR2）测得的月平均海冰密集度，来自 NOAA/NSIDC 气候数据记录，25 km 网格。海冰覆盖不足 15% 的海面不着色。近岸处陆地会干扰信号，那里的海冰取自最近的观测网格。",
          legendNote:
            "该月海面被海冰覆盖的比例；低于 15% 的海面不着色。橙线：1981–2010 年该月的中位冰缘线，即这些年份中有一半出现海冰的边界。",
        },
        projection: {
          play: "播放",
          pause: "暂停",
          scenarioAe10: "低排放（SSP1-2.6）",
          scenarioAe05: "高排放（SSP5-8.5）",
          scenarioAe14: "高排放并伴随冰架崩解",
          shortAe10: "SSP1-2.6",
          shortAe05: "SSP5-8.5",
          shortAe14: "+冰架崩解",
          readoutSeaLevel: "南极造成的海平面变化",
          readoutRange: "各模型的范围",
          readoutGroundedMap: "地图上的接地冰",
          readoutGrounded: "接地冰",
          readoutFloating: "漂浮冰",
          seaLevelValue: "{value} m",
          rangeValue: "{low} 至 {high} m",
          areaValue: "{value} 百万 km\u00b2（{change}）",
          chartLabel: "{first}-{last} 年各情景下南极造成的海平面变化；阴影为各模型的范围",
          noteChange:
            "红色表示冰比 {first} 年更薄，蓝色表示更厚。接地冰坐落在基岩上，漂浮冰处于静力漂浮状态。流线随预估的冰面移动，在冰加速的地方更亮、流得更快。",
          noteType: "白色为接地冰，蓝色为漂浮冰，二者都由基岩上的漂浮条件推得。流线随预估的冰面移动。",
          legendNote:
            "相对 {first} 年的冰厚变化，采用对称对数刻度，在 \u00b110 m 以内为线性。冰已消失之处显露出裸露的基岩或开阔的海面。",
          ensembleNote:
            "{models} 个 ISMIP6 冰盖模型（Seroussi 等，2024）在 UKESM1-0-LL 气候强迫下的平均变化，套用到今天的冰盖（BedMachine v4）上。海平面数字是各模型自身结果的平均；由于漂浮条件是非线性的，地图上平均后的冰盖对应的海平面上升略高于这个数字。",
        },
        legends: {
          warm: "暖",
          cold: "冷",
          fresh: "淡",
          salty: "咸",
          effectivePressureNote: "水压超过冰体上覆压力的过压区域（N < 0）不予显示。",
        },
        meta: {
          reboundMillionKm2: "{value} \u767e\u4e07 km\u00b2",
          reboundVolumeOf: "{above} / {total} \u767e\u4e07 km\u00b3",
          reboundSolveGridValue: "{cell} km \u7f51\u683c\uff0c{sizeX} x {sizeY} \u53d8\u6362",
          reboundSolveGridPointwise: "{cell} km \u7f51\u683c\uff0c\u9010\u70b9\u6c42\u89e3",
          reboundConvergenceValue: "{iterations} \u6b21\u8fed\u4ee3\uff0c\u6b8b\u5dee {residual} m",
          isostaticReboundSection: "地壳回弹（模拟）",
          reboundModelLabel: "固体地球响应",
          reboundModelPublished: "已发表：Paxman 等（2022），网格第 3 版",
          reboundModelFlexural: "理想化区域挠曲（弹性板 + 流体软流圈）",
          reboundModelLocal: "理想化局地（Airy）地壳均衡",
          reboundElasticThickness: "弹性厚度",
          reboundPublishedEarthModelValue: "{teModel}，横向变化；密度取值：冰 917、海水 1028、地幔 3330 kg/m3",
          reboundSeaSurface: "无冰海面",
          reboundSeaSurfaceValue: "+{eustatic} m 全球平均上升（两个冰盖）加上末次盛冰期后残余的大地水准面变化：在接地冰上方比现今高 {gmin}-{gmax} m",
          reboundMaxTopographyChange: "相对无冰海面的最大抬升",
          reboundMeanTopographyChange: "接地冰之下相对无冰海面的平均抬升",
          reboundComponents: "接地冰之下的各分量",
          reboundComponentsValue: "冰卸载 {iu} m（最大 {iuMax} m）；末次盛冰期后残余 {lgm} m（最大 {lgmMax} m）；水载荷 {wl} m（最小 {wlMin} m）",
          reboundSigma: "接地冰之下的地球模型离散度（1 sigma）",
          reboundSigmaValue: "平均 {mean} m，最大 {max} m",
          reboundLandAfterPublished: "回弹后高于无冰海面的基岩面积",
          reboundMarineUnderIcePublished: "现有冰盖之下仍低于无冰海面的区域",
          reboundQrfLoadNote: "载荷",
          reboundQrfLoadNoteValue: "按 BedMachine Greenland v6 的冰载荷计算；QRF 基岩使挠曲响应改变约 10 m（均方根）",
          reboundRigidity: "挠曲刚度",
          reboundLengthScale: "挠曲特征长度",
          reboundRelaxation: "松弛时间",
          reboundProgressLabel: "情景进度",
          reboundSeaLevelLabel: "海平面基准",
          reboundMaxUplift: "平衡态最大抬升",
          reboundMeanGroundedUplift: "接地冰之下的平均抬升",
          reboundLandBefore: "当前高于海平面的基岩面积",
          reboundLandAfter: "回弹后高于基准面的基岩面积",
          reboundEmergent: "新出露陆地",
          reboundDrowned: "被抬升基准面淹没的现有基岩",
          reboundMarineUnderIce: "现有冰盖之下仍低于基准面的区域",
          reboundClosedBasins: "低于基准面的封闭盆地",
          reboundDeepest: "回弹后最深的接地基岩",
          reboundIceVolume: "当前网格上的冰体积",
          reboundVolumeAboveFlotation: "漂浮阈值以上体积",
          reboundSle: "等效海平面",
          reboundSlePublishedAntarctica:
            "为漂浮阈值以上的部分，按淡水密度换算（即 Gregory 等 2019 对海平面当量的定义）；可对比 Morlighem 等（2020）对 BedMachine Antarctica v1 发表的 57.9 ± 0.9 m。10 km 与 4 km 数据包的结果相差不到 0.01%，Bedmap3 在相同网格上给出 58.6 m。",
          reboundSlePublishedGreenland:
            "为漂浮阈值以上的部分，按淡水密度换算（即 Gregory 等 2019 对海平面当量的定义）；可对比 Morlighem 等（2017）对 BedMachine Greenland v3 发表的 7.42 ± 0.05 m。",
          reboundSolveGrid: "挠曲求解网格",
          reboundConvergence: "Picard 收敛情况",
          reboundAssumptions: "假设",
          reboundAssumptionsText:
            "这是移除现有冰载荷后的理想化平衡态响应，而非瞬态模拟。计算假设当前基岩与当前载荷处于平衡，但实际并非如此：末次盛冰期以来的回弹尚未完成（罗斯海与威德尔海湾之下基岩高程仍有最多 +68 m 的变化，而劳伦泰德冰盖前缘隆起的塌陷使格陵兰最多降低 25 m；Paxman 等，2022），阿蒙森海湾则因近期冰量损失叠加软弱地幔，正以最高约 41 mm/yr 的速率抬升。计算未包含海平面指纹、大地水准面变化与自转反馈，并以单一的全球刚度和地幔密度替代真实的横向结构。",
          reboundAssumptionsTextPublished:
            "移除两个冰盖后完全重新平衡的响应。末次盛冰期后残余项采用一维地幔黏度剖面，在上地幔软弱的地区（如西南极和东格陵兰之下）最不可靠。海面按全球平均上升量加末次盛冰期后残余大地水准面变化抬升；未包含热比容与动力海面、侵蚀与沉积，且无冰海面以下的所有位置都加载了海水。",
          reboundMethod: "方法",
          reboundMethodText:
            "薄板挠曲方程 D grad^4 u + rho_m g u = sigma_now - sigma_after，在谱域求解；冰消后的水载荷通过 Picard 迭代在与海洋连通的区域上求解。密度取值：冰 917、海水 1027、地幔 3300 kg/m3。",
          reboundMethodTextPublished:
            "已发表的总地壳均衡响应 T = R - G，在本网格节点上点采样，其中 R 为固体地表位移，G 为海面变化。基岩按 bed + T 绘制，因此高程可直接与无冰海面对照。网格文件：NSF Arctic Data Center，doi:10.18739/A22Z12R8C（CC BY 4.0）。",
          errorLabel: "错误",
          geometrySection: "几何与网格",
          velocitySection: "冰表流速（观测）",
          basalFrictionSection: "基底摩擦（反演）",
          hydrologySection: "冰下水文（模拟）",
          oceanSection: "海洋环流（模拟）",
          riseSection: "冰架底部融化（模拟）",
          sourcesSection: "数据来源",
          availableOnDemand: "按需加载",
          notYetAdded: "{region} 暂未加入",
          preset: "预设",
          grid: "网格",
          projection: "投影",
          bedElevation: "基岩高程",
          maxIceThickness: "最大冰厚",
          meanIceThickness: "平均冰厚",
          surfaceSpeedRange: "表面流速范围",
          speedQuantiles: "流速分位数",
          selectedFlowlineSection: "冰流线侧剖面",
          selectedFlowlineHint: "点击任意可见冰流线，即可在此预览对应的沿流向侧剖面剪影。",
          selectedFlowlineEmptyTitle: "选择一条流线",
          selectedFlowlineEmptyBody: "点击任意可见冰流线，即可查看包含冰表、冰底与基岩的沿流向侧剖面。",
          selectedFlowlineDisabledTitle: "流线图层已隐藏",
          selectedFlowlineDisabledBody: "请先开启“显示冰流线”，再点击流线查看对应剖面剪影。",
          selectedFlowlineLabel: "第 {index} 条流线",
          selectedFlowlineLength: "沿流向长度",
          selectedFlowlineSpeedRange: "流速范围",
          selectedFlowlineSurfaceSpan: "表面高程范围",
          selectedFlowlineThickness: "冰厚范围",
          selectedFlowlinePreview: "剖面剪影",
          status: "状态",
          invertedFriction: "反演摩擦",
          invertedFrictionSummary: "基于 Elmer/Ice 集合反演的中位值，采用浅冰架近似（SSA）",
          frictionRange: "摩擦范围",
          frictionQuantiles: "摩擦分位数",
          basalMeltRate: "底部融化速率",
          meltQuantiles: "融化分位数",
          thermalDriving: "热驱动",
          thermalQuantiles: "热驱动分位数",
          iceDraft: "冰吃水深度",
          draftQuantiles: "吃水分位数",
          oceanStreamlines: "海洋流线",
          depthSpan: "三维深度范围",
          horizontalSpeed: "水平流速",
          waterMassColor: "水团颜色",
          fourCornerPalette: "四角温盐配色",
          effectivePressureRange: "有效压力范围",
          pressureQuantiles: "压力分位数",
          channelDischargeRange: "通道流量范围",
          dischargeQuantiles: "流量分位数",
          renderedChannels: "渲染通道数",
          sourceBedIceGeometry: "基岩/冰体几何",
          sourceBasinBoundaries: "流域边界",
          sourceSurfaceVelocity: "表面流速",
          sourceBasalFriction: "基底摩擦",
          sourceSubglacialHydrology: "冰下水文",
          sourceOceanCirculation: "海洋环流",
          sourceMeltDrivers: "融化驱动",
          iceShelfCellCount: "（{count} 个冰架格点）",
          streamlineCountSuffix: "（{count} 条三维流线）",
          gridSpacingSuffix: "（{spacing} km 网格）",
          valueRange: "{min} 至 {max}",
          quantileMedian: "中位数",
          notApplicable: "无",
          polarStereographic: "{epsg} 极地立体投影",
          polarStereographicGeneric: "极地立体投影",
          atGridPosition: "{value} m，位于 x = {x} km、y = {y} km（{epsg}）",
          yearsValue: "{value} 年",
          sourceIsostaticResponse: "地壳均衡响应",
          sourceIceProjection: "冰盖未来预估",
          iceProjectionSection: "冰盖未来预估",
          sourceSurfaceChange: "冰面高程变化",
          surfaceChangeSection: "观测到的高程变化（CPOM）",
          sourceSeaIce: "海冰",
          seaIceSection: "观测到的海冰（NSIDC）",
          seaIceMonth: "当前月份",
          seaIceMonthValue: "{date}（共 {count} 个月，{first} 至 {last}）",
          seaIceExtent: "范围与面积",
          seaIceExtentValueNorth: "{extent} 与 {area} 百万 km\u00b2，整个北极",
          seaIceExtentValueSouth: "{extent} 与 {area} 百万 km\u00b2，整个南大洋",
          seaIceTrend: "{month}：平均与趋势",
          seaIceTrendValue: "1981–2010 年平均 {mean} 百万 km\u00b2；每十年 {slope} 百万 km\u00b2（{percent}%），{first}–{last} 年",
          seaIceSensor: "传感器",
          seaIceCoast: "近岸",
          seaIceCoastValue: "{share}% 的 25 km 网格在 NSIDC 陆地掩膜下没有相邻的海洋网格，其密集度取自相邻网格",
          seaIceIndex: "与 Sea Ice Index 比较",
          seaIceIndexValue: "平均比 NSIDC Sea Ice Index 高 {percent}%（后者为 NASA Team 算法的每日范围的月平均）；地图所画的海冰（月平均密集度达到 15% 的网格）比它高 {meanField}%",
          seaIceMethod: "方法",
          seaIceMethodValue: "地图绘制 CDR 月平均密集度（每个网格取 NASA Team 与 Bootstrap 估计中的较大值），从 NSIDC 极地立体网格（Hughes 1980 椭球）双线性重采样到地形的海洋上；海冰范围为整个半球上当月至少一半天数海冰覆盖率达到 15% 的海洋网格面积（取自文件的质量标志），在 NSIDC 网格上统计",
          surfaceChangeEpoch: "当前时期",
          surfaceChangeEpochValue: "{date}（共 {count} 个 140 天时期，{first} 至 {last}）",
          surfaceChangeRange: "图上的变化",
          surfaceChangeRangeValue: "{min} 至 {max} {unit}",
          surfaceChangeSignificant: "超过 2σ",
          surfaceChangeSignificantValue: "占已绘制网格的 {share}%",
          surfaceChangeExtrapolated: "极点附近的外推",
          surfaceChangeExtrapolatedValue: "{date}之前为 81.5°S 以南，全程为 88°S 以南",
          surfaceChangePeriod: "时段",
          surfaceChangeCells: "覆盖范围",
          surfaceChangeCellsValue: "CPOM 掩膜（BedMachine v2）中的接地冰，绘制在其覆盖至少一半的每个浏览器冰网格上，包括接地线旁的漂浮网格；不含冰架、岛屿和冰隆",
          surfaceChangeMethod: "方法",
          surfaceChangeMethodValue: "每个 5 km 网格做平面拟合，140 天一期，扣除 IJ05 R2 冰川均衡调整，多卫星交叉定标（Shepherd et al. 2019）；变化相对于 CPOM 的 1995 年参考面，因此第一期并不为零；按面积平均到浏览器网格",
          surfaceChangeMassSince: "1995 年以来的质量变化（CPOM）",
          surfaceChangeMassPeriod: "{first}–{last} 年质量变化（CPOM）",
          surfaceChangeMassPart: "{name} {value}",
          surfaceChangeMassSeparator: "；",
          surfaceChangeSelected: "所选范围",
          surfaceChangeSelectedValue: "{name}：{value}",
          iceProjectionExperiment: "试验",
          iceProjectionExperimentValue: "ISMIP6 {id}：{climate} {scenario}",
          iceProjectionModels: "参与平均的模型",
          iceProjectionModelsValue: "{count} 个，即三个情景都完成的各组主提交：{names}",
          iceProjectionYear: "当前年份",
          iceProjectionYearMeanValue: "{year} 年：海平面上升 {sle} m（各模型 {low} 至 {high} m）",
          iceProjectionSeaLevel: "海平面贡献",
          iceProjectionSeaLevelMeanValue:
            "各模型已发表数值（Seroussi & Pelle 2024）的平均，相对 {first} 年，未扣除控制试验：到 {last} 年为 {end} m。地图上平均后的冰盖对应 {packed} m。",
          iceProjectionMethod: "方法",
          iceProjectionMethodValue:
            "{count} 个模型各自保守地重采样到 10 km；它们自 2015 年以来冰厚与垂向平均流速的平均变化套用到 BedMachine v4 的冰盖上。增厚直接相加；减薄按 BedMachine 冰厚与模型自身 2015 年冰厚之比缩放，使每个格点保留的冰量比例与模型一致；模型平均冰厚不足 10 m 处，冰视为已消失。地图上的冰体积变化为模型平均值的 {captured}%。",
          iceProjectionKeyframes: "关键帧",
          iceProjectionKeyframesMeanValue: "{first}-{last} 年每 {interval} 年一帧，共 {count} 帧，帧间线性插值；今天的冰盖共 {cells} 个单元",
          currentScope: "当前 {region} 范围",
          currentScopeSummary: "本版本已启用基岩地形、冰表面和冰底界面。",
          fieldStatsBed: "基岩",
          fieldStatsIceMax: "最大冰厚",
          fieldStatsVelocityMax: "最大流速",
          fieldStatsOceanStreamlines: "海洋流线",
          fieldStatsTauHighEnd: "Tau_b",
          fieldStatsTauHighEndSuffix: "高端值",
          fieldStatsBasalMeltMax: "最大底融",
          fieldStatsBasalMeltMaxSuffix: "最大值",
          fieldStatsThermalDrivingMax: "最大热驱动",
          fieldStatsThermalDrivingMaxSuffix: "最大值",
          fieldStatsBasins: "流域数",
          oceanWaterMassSummary:
            "冷淡青色、冷咸靛色、暖淡绿色、暖咸橙红色（{thetaMin} 到 {thetaMax} °C；{salinityMin} 到 {salinityMax} PSU）",
        },
        errors: {
          reboundGridTooSmall: "地壳回弹求解器至少需要 2x2 的网格。",
          failedToLoadIceProjection: "加载冰盖未来预估失败（{status}）",
          iceProjectionGridMismatch: "冰盖未来预估与地形不在同一网格上。",
          reboundTransformTooLarge: "地壳回弹求解网格超出了支持的变换尺寸。",
          workerTaskFailed: "后台任务失败",
          workerCrashed: "几何 worker 已崩溃",
          workerTerminated: "几何 worker 已终止",
          fullscreenUnavailable: "当前浏览器不支持全屏 API。",
          regionSwitchUnavailable: "当前探索模式不支持切换区域。",
          regionLoadTimedOut: "加载 {region} 超时。",
          layersLoadTimedOut: "等待图层加载超时。",
          failedToLoadMetadata: "加载元数据失败（{status}）",
          failedToLoadTerrainPackage: "加载地形数据包失败（{status}）",
          failedToLoadVelocityMetadata: "加载流速元数据失败（{status}）",
          failedToLoadBasalFrictionMetadata: "加载基底摩擦元数据失败（{status}）",
          failedToLoadHydrologyMetadata: "加载水文元数据失败（{status}）",
          failedToLoadRiseMetadata: "加载 RISE 元数据失败（{status}）",
          failedToLoadOceanCurrentMetadata: "加载海洋流场元数据失败（{status}）",
          failedToLoadVelocityField: "加载流速场失败（{status}）",
          failedToLoadBasalFrictionField: "加载基底摩擦场失败（{status}）",
          failedToLoadSurfaceChangeMetadata: "加载高程变化元数据失败（{status}）",
          failedToLoadSurfaceChangeField: "加载高程变化数据失败（{status}）",
          failedToLoadSeaIceMetadata: "加载海冰元数据失败（{status}）",
          failedToLoadSeaIceField: "加载海冰数据失败（{status}）",
          failedToLoadHydrologyField: "加载水文字段失败（{status}）",
          failedToLoadReboundMetadata: "加载地壳均衡响应元数据失败（{status}）",
          failedToLoadReboundField: "加载地壳均衡响应字段失败（{status}）",
          failedToLoadRiseOverlayPackage: "加载 RISE 叠加层数据包失败（{status}）",
          failedToLoadOceanCurrentVectors: "加载海洋流场矢量失败（{status}）",
          failedToLoadBasinBoundaries: "加载流域边界失败（{status}）",
          failedToLoadBedColorTable: "加载 GMT_relief 配色表失败（{status}）",
          failedToLoadEffectivePressureColorTable: "加载 cmocean_dense 配色表失败（{status}）",
          failedToLoadChannelColorTable: "加载 cmocean_matter 配色表失败（{status}）",
          unexpectedFieldLength: "数据包中的字段长度异常。",
          unexpectedRiseFieldLength: "RISE 数据包中的字段长度异常。",
          riseGridMisaligned: "RISE 网格与当前 BedMachine 网格不对齐。",
          velocityGridMisaligned: "流速网格与当前 BedMachine 网格不对齐。",
          velocityPayloadInvalid: "流速数据包无效。",
          velocityTextureFieldLengthMismatch: "流速纹理字段长度不匹配。",
          scalarFieldTextureLengthMismatch: "标量场纹理长度不匹配。",
          basalFrictionGridMisaligned: "基底摩擦网格与当前 BedMachine 网格不对齐。",
          hydrologyGridMisaligned: "水文网格与当前 BedMachine 网格不对齐。",
          oceanPackageMisaligned: "海洋流场数据包字段未对齐。",
          oceanDatasetEmpty: "区域裁剪后海洋流场数据为空。",
          basinDatasetEmpty: "流域数据集为空。",
        },
      },
      worker: {
        progress: {
          reboundSolvingFlexure: "正在求解地壳回弹...",
          oceanDecodingPackage: "正在解码海洋流场数据包...",
          oceanScanningSegments: "正在扫描海洋流场线段...",
          oceanBuildingGeometry: "正在构建海洋流线几何...",
          oceanFinalizing: "正在完成海洋流线...",
          velocityDecodingField: "正在解码流速场...",
          velocityBuildingMesh: "正在构建流速网格...",
          velocityTriangulatingMesh: "正在三角化流速网格...",
          velocityFinalizing: "正在完成流速图层...",
          hydrologyProcessingField: "正在处理水文字段...",
          hydrologyBuildingMesh: "正在构建水文网格...",
          hydrologyTriangulatingMesh: "正在三角化水文网格...",
          hydrologyBuildingChannels: "正在构建通道带状网格...",
          hydrologyFinalizing: "正在完成水文图层...",
          basalFrictionProcessingField: "正在处理基底摩擦场...",
          basalFrictionBuildingMesh: "正在构建基底摩擦网格...",
          basalFrictionTriangulatingMesh: "正在三角化基底摩擦网格...",
          basalFrictionFinalizing: "正在完成基底摩擦图层...",
        },
      },
    },
  };

  const ERROR_PATTERN_BUILDERS = [
    { prefix: "Failed to load metadata", key: "explorer.errors.failedToLoadMetadata" },
    { prefix: "Failed to load velocity metadata", key: "explorer.errors.failedToLoadVelocityMetadata" },
    { prefix: "Failed to load basal-friction metadata", key: "explorer.errors.failedToLoadBasalFrictionMetadata" },
    { prefix: "Failed to load hydrology metadata", key: "explorer.errors.failedToLoadHydrologyMetadata" },
    { prefix: "Failed to load RISE metadata", key: "explorer.errors.failedToLoadRiseMetadata" },
    { prefix: "Failed to load ocean-current metadata", key: "explorer.errors.failedToLoadOceanCurrentMetadata" },
    { prefix: "Failed to load velocity field", key: "explorer.errors.failedToLoadVelocityField" },
    { prefix: "Failed to load basal-friction field", key: "explorer.errors.failedToLoadBasalFrictionField" },
    { prefix: "Failed to load surface-change metadata", key: "explorer.errors.failedToLoadSurfaceChangeMetadata" },
    { prefix: "Failed to load surface-change field", key: "explorer.errors.failedToLoadSurfaceChangeField" },
    { prefix: "Failed to load sea-ice metadata", key: "explorer.errors.failedToLoadSeaIceMetadata" },
    { prefix: "Failed to load sea-ice field", key: "explorer.errors.failedToLoadSeaIceField" },
    { prefix: "Failed to load hydrology field", key: "explorer.errors.failedToLoadHydrologyField" },
    { prefix: "Failed to load isostatic-response metadata", key: "explorer.errors.failedToLoadReboundMetadata" },
    { prefix: "Failed to load isostatic-response field", key: "explorer.errors.failedToLoadReboundField" },
    { prefix: "Failed to load RISE overlay package", key: "explorer.errors.failedToLoadRiseOverlayPackage" },
    { prefix: "Failed to load ocean-current vectors", key: "explorer.errors.failedToLoadOceanCurrentVectors" },
    { prefix: "Failed to load basin boundaries", key: "explorer.errors.failedToLoadBasinBoundaries" },
    { prefix: "Failed to load GMT_relief color table", key: "explorer.errors.failedToLoadBedColorTable" },
    { prefix: "Failed to load cmocean_dense color table", key: "explorer.errors.failedToLoadEffectivePressureColorTable" },
    { prefix: "Failed to load cmocean_matter color table", key: "explorer.errors.failedToLoadChannelColorTable" },
  ];
  const EXACT_ERROR_KEYS = new Map([
    ["Worker task failed", "explorer.errors.workerTaskFailed"],
    ["Geometry worker crashed", "explorer.errors.workerCrashed"],
    ["Geometry worker terminated", "explorer.errors.workerTerminated"],
    ["Unexpected field length in data package.", "explorer.errors.unexpectedFieldLength"],
    ["Unexpected field length in the RISE package.", "explorer.errors.unexpectedRiseFieldLength"],
    ["RISE grid is not aligned to the active BedMachine grid.", "explorer.errors.riseGridMisaligned"],
    ["Velocity grid is not aligned to BedMachine grid.", "explorer.errors.velocityGridMisaligned"],
    ["Velocity payload is invalid.", "explorer.errors.velocityPayloadInvalid"],
    ["Velocity texture field length mismatch.", "explorer.errors.velocityTextureFieldLengthMismatch"],
    ["Scalar field texture length mismatch.", "explorer.errors.scalarFieldTextureLengthMismatch"],
    ["Basal-friction grid is not aligned to BedMachine grid.", "explorer.errors.basalFrictionGridMisaligned"],
    ["Hydrology grid is not aligned to BedMachine grid.", "explorer.errors.hydrologyGridMisaligned"],
    ["Ocean-current package fields are misaligned.", "explorer.errors.oceanPackageMisaligned"],
    ["Ocean-current dataset is empty after regional clipping.", "explorer.errors.oceanDatasetEmpty"],
    ["Basin dataset is empty.", "explorer.errors.basinDatasetEmpty"],
  ]);

  function normalizeLocale(value) {
    const text = String(value || "").trim();
    if (!text) return DEFAULT_LOCALE;
    const lowered = text.toLowerCase();
    if (lowered === "zh" || lowered === "zh-cn" || lowered === "zh_cn") return "zh-CN";
    if (lowered === "en" || lowered === "en-us" || lowered === "en_us") return "en-US";
    return SUPPORTED_LOCALES.includes(text) ? text : DEFAULT_LOCALE;
  }

  function getMessages(locale) {
    return MESSAGES[normalizeLocale(locale)] || MESSAGES[DEFAULT_LOCALE];
  }

  function lookupMessage(locale, key) {
    return key.split(".").reduce((acc, segment) => (acc && Object.prototype.hasOwnProperty.call(acc, segment) ? acc[segment] : undefined), getMessages(locale));
  }

  function interpolate(template, vars) {
    return String(template).replace(/\{(\w+)\}/g, (_match, token) =>
      Object.prototype.hasOwnProperty.call(vars || {}, token) ? String(vars[token]) : ""
    );
  }

  function t(locale, key, vars) {
    const template = lookupMessage(locale, key);
    if (typeof template !== "string") return key;
    return interpolate(template, vars || {});
  }

  function getIntlLocale(locale) {
    return normalizeLocale(locale) === "zh-CN" ? "zh-CN" : "en-US";
  }

  function getStoredLocale() {
    try {
      return normalizeLocale(window.localStorage.getItem(STORAGE_KEY));
    } catch (_error) {
      return DEFAULT_LOCALE;
    }
  }

  function setStoredLocale(locale) {
    try {
      window.localStorage.setItem(STORAGE_KEY, normalizeLocale(locale));
    } catch (_error) {
      // Ignore storage failures.
    }
  }

  function normalizePathname(pathname) {
    const raw = String(pathname || "/");
    if (raw === "/") return "/";
    return raw.endsWith("/") ? raw.slice(0, -1) : raw;
  }

  function resolveRouteKey(pathname) {
    const normalized = normalizePathname(pathname);
    if (normalized === "" || normalized === "/" || normalized === "/index.html" || normalized === "/zh" || normalized === "/zh/index.html") {
      return "home";
    }
    if (
      normalized === "/tools/3d-ice" ||
      normalized === "/tools/3d-ice/index.html" ||
      normalized === "/zh/tools/3d-ice" ||
      normalized === "/zh/tools/3d-ice/index.html"
    ) {
      return "landing";
    }
    if (
      normalized === "/tools/3D-interactive-cryosphere-explorer.html" ||
      normalized === "/zh/tools/3D-interactive-cryosphere-explorer.html"
    ) {
      return "explorer";
    }
    if (
      normalized === "/explore" ||
      normalized === "/explore/index.html" ||
      normalized === "/zh/explore" ||
      normalized === "/zh/explore/index.html"
    ) {
      return "explore";
    }
    if (
      normalized === "/tools/3d-antarctica" ||
      normalized === "/tools/3d-antarctica/index.html" ||
      normalized === "/zh/tools/3d-antarctica" ||
      normalized === "/zh/tools/3d-antarctica/index.html"
    ) {
      return "legacyRedirect";
    }
    return null;
  }

  function buildLocaleUrl(targetLocale, inputUrl) {
    const locale = normalizeLocale(targetLocale);
    const url = new URL(inputUrl || window.location.href, window.location.href);
    const routeKey = resolveRouteKey(url.pathname);
    if (!routeKey) return url.toString();
    url.pathname = ROUTES[routeKey][locale];
    return url.toString();
  }

  function renderLocaleSwitcher(locale, inputUrl) {
    const currentLocale = normalizeLocale(locale);
    const links = SUPPORTED_LOCALES.map((nextLocale) => {
      const href = buildLocaleUrl(nextLocale, inputUrl);
      const shortLabel =
        nextLocale === "zh-CN" ? t(currentLocale, "shared.localeChinese") : t(currentLocale, "shared.localeEnglish");
      const longLabel =
        nextLocale === "zh-CN"
          ? t(currentLocale, "shared.localeChineseLong")
          : t(currentLocale, "shared.localeEnglishLong");
      return `
        <a
          class="explorer-locale-switcher__link${nextLocale === currentLocale ? " is-active" : ""}"
          href="${href}"
          hreflang="${nextLocale === "zh-CN" ? "zh-CN" : "en-US"}"
          lang="${nextLocale}"
          data-3d-ice-locale="${nextLocale}"
          aria-current="${nextLocale === currentLocale ? "true" : "false"}"
          title="${longLabel}"
        >${shortLabel}</a>
      `;
    }).join("");
    return `
      <div class="explorer-locale-switcher__label">${t(currentLocale, "shared.switcherLabel")}</div>
      <div class="explorer-locale-switcher__group" role="group" aria-label="${t(currentLocale, "shared.switcherLabel")}">
        ${links}
      </div>
    `;
  }

  function bindLocaleLinks(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-3d-ice-locale]").forEach((linkEl) => {
      linkEl.addEventListener("click", () => {
        const nextLocale = linkEl.getAttribute("data-3d-ice-locale");
        if (nextLocale) {
          setStoredLocale(nextLocale);
        }
      });
    });
  }

  function mountLocaleSwitcher(selectorOrElement, locale, inputUrl) {
    const element =
      typeof selectorOrElement === "string" ? document.querySelector(selectorOrElement) : selectorOrElement;
    if (!element) return;
    element.innerHTML = renderLocaleSwitcher(locale, inputUrl);
    bindLocaleLinks(element);
  }

  function relaxChineseHomeTitleWidth(locale) {
    if (normalizeLocale(locale) !== "zh-CN") return;
    // The home page, here or in yuwang.blog's copy at /zh/tools/3d-ice/.
    const route = resolveRouteKey(window.location.pathname || "/");
    if (route !== "home" && route !== "landing") return;
    const title = document.querySelector(".explorer-page-shell--ice .explorer-page-title");
    if (!title) return;
    title.style.maxWidth = "none";
  }

  function initPage(options) {
    const locale = normalizeLocale(options && options.locale);
    setStoredLocale(locale);
    const switchers = Array.isArray(options?.switchers) ? options.switchers : [];
    switchers.forEach((target) => mountLocaleSwitcher(target, locale, options?.url || window.location.href));
    relaxChineseHomeTitleWidth(locale);
  }

  function getLocalizedRegionInfo(locale, regionKey) {
    const region = lookupMessage(locale, `explorer.regions.${regionKey}`);
    if (!region || typeof region !== "object") return {};
    return region;
  }

  function getLocalizedDatasetInfo(locale, regionKey, datasetKey) {
    const dataset = lookupMessage(locale, `explorer.datasets.${regionKey}.${datasetKey}`);
    if (!dataset || typeof dataset !== "object") return {};
    return dataset;
  }

  function formatNumber(locale, value, options) {
    return Number(value).toLocaleString(getIntlLocale(locale), options || undefined);
  }

  function localizeWorkerStage(locale, stageKey, fallbackStage) {
    if (stageKey) {
      const key = `worker.progress.${stageKey}`;
      const value = lookupMessage(locale, key);
      if (typeof value === "string") return value;
    }
    return fallbackStage || "";
  }

  function localizeErrorMessage(locale, message) {
    if (!message) return "";
    const exactKey = EXACT_ERROR_KEYS.get(message);
    if (exactKey) return t(locale, exactKey);
    for (const entry of ERROR_PATTERN_BUILDERS) {
      if (message.startsWith(entry.prefix)) {
        const match = message.match(/\(([^)]+)\)\s*$/);
        return t(locale, entry.key, { status: match ? match[1] : "?" });
      }
    }
    return message;
  }

  window.__3dIceLocale = {
    STORAGE_KEY,
    DEFAULT_LOCALE,
    ROUTES,
    MESSAGES,
    normalizeLocale,
    getIntlLocale,
    getStoredLocale,
    setStoredLocale,
    buildLocaleUrl,
    renderLocaleSwitcher,
    mountLocaleSwitcher,
    bindLocaleLinks,
    initPage,
    getLocalizedRegionInfo,
    getLocalizedDatasetInfo,
    formatNumber,
    localizeWorkerStage,
    localizeErrorMessage,
    ogLocaleFor(locale) {
      return OG_LOCALE_BY_LOCALE[normalizeLocale(locale)] || OG_LOCALE_BY_LOCALE[DEFAULT_LOCALE];
    },
    t(locale, key, vars) {
      return t(locale, key, vars);
    },
  };
})();
