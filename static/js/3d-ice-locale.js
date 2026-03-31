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
    explorer: {
      "en-US": "/tools/3D-interactive-cryosphere-explorer.html",
      "zh-CN": "/zh/tools/3D-interactive-cryosphere-explorer.html",
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
              label: "Balanced",
              summary: "10 km grid; ~3.0 MB",
            },
            hd: {
              label: "HD",
              summary: "4 km grid; ~18.6 MB",
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
          ready: "Ready",
          readyWithContext: "Ready ({region} {dataset})",
          previewReady: "Preview ready",
          loadingPreview: "Loading {region} preview...",
          loadingCoreTerrain: "Loading {region} {dataset} core terrain...",
          loadingBasin: "Loading {label}...",
          loadingBasalFriction: "Loading basal friction...",
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
        legends: {
          warm: "Warm",
          cold: "Cold",
          fresh: "Fresh",
          salty: "Salty",
        },
        meta: {
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
          workerTaskFailed: "Worker task failed",
          workerCrashed: "Geometry worker crashed",
          workerTerminated: "Geometry worker terminated",
          fullscreenUnavailable: "Fullscreen API unavailable.",
          failedToLoadMetadata: "Failed to load metadata ({status})",
          failedToLoadTerrainPackage: "Failed to load terrain package ({status})",
          failedToLoadVelocityMetadata: "Failed to load velocity metadata ({status})",
          failedToLoadBasalFrictionMetadata: "Failed to load basal-friction metadata ({status})",
          failedToLoadHydrologyMetadata: "Failed to load hydrology metadata ({status})",
          failedToLoadRiseMetadata: "Failed to load RISE metadata ({status})",
          failedToLoadOceanCurrentMetadata: "Failed to load ocean-current metadata ({status})",
          failedToLoadVelocityField: "Failed to load velocity field ({status})",
          failedToLoadBasalFrictionField: "Failed to load basal-friction field ({status})",
          failedToLoadHydrologyField: "Failed to load hydrology field ({status})",
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
              label: "标准",
              summary: "10 km 网格；约 3.0 MB",
            },
            hd: {
              label: "高清",
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
          ready: "就绪",
          readyWithContext: "就绪（{region} {dataset}）",
          previewReady: "预览已就绪",
          loadingPreview: "正在加载 {region} 预览...",
          loadingCoreTerrain: "正在加载 {region} {dataset} 核心地形...",
          loadingBasin: "正在加载 {label}...",
          loadingBasalFriction: "正在加载基底摩擦...",
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
        legends: {
          warm: "暖",
          cold: "冷",
          fresh: "淡",
          salty: "咸",
        },
        meta: {
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
          workerTaskFailed: "后台任务失败",
          workerCrashed: "几何 worker 已崩溃",
          workerTerminated: "几何 worker 已终止",
          fullscreenUnavailable: "当前浏览器不支持全屏 API。",
          failedToLoadMetadata: "加载元数据失败（{status}）",
          failedToLoadTerrainPackage: "加载地形数据包失败（{status}）",
          failedToLoadVelocityMetadata: "加载流速元数据失败（{status}）",
          failedToLoadBasalFrictionMetadata: "加载基底摩擦元数据失败（{status}）",
          failedToLoadHydrologyMetadata: "加载水文元数据失败（{status}）",
          failedToLoadRiseMetadata: "加载 RISE 元数据失败（{status}）",
          failedToLoadOceanCurrentMetadata: "加载海洋流场元数据失败（{status}）",
          failedToLoadVelocityField: "加载流速场失败（{status}）",
          failedToLoadBasalFrictionField: "加载基底摩擦场失败（{status}）",
          failedToLoadHydrologyField: "加载水文字段失败（{status}）",
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
    { prefix: "Failed to load hydrology field", key: "explorer.errors.failedToLoadHydrologyField" },
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
      normalized === "/tools/3D-interactive-cryosphere-explorer.html" ||
      normalized === "/zh/tools/3D-interactive-cryosphere-explorer.html"
    ) {
      return "explorer";
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
    const pathname = window.location.pathname || "/";
    if (pathname !== "/zh/" && pathname !== "/zh/index.html") return;
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
