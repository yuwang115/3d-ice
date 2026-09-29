/**
 * Copy and stop definitions for the public edition's guided tour and layer explainers.
 *
 * TOUR_CHAPTERS says what each stop shows: the region, the layer toggles (see
 * TOUR_CONTROL_DEFAULTS in ./explore-tour.js), the sliders, where the camera goes and an
 * optional slider animation, played once the camera has arrived. A camera is either "default", the region's opening view, or
 * { lat, lon, fitKm, azimuthDeg, elevationDeg }: it frames a disc fitKm across round that
 * point, whatever the size of the screen, looking from azimuth 0 (the lower edge of the
 * map, as the default view does) or 90 (its right-hand edge). A stop without a region
 * stays in the current one.
 *
 * EXPLORE_CONTENT holds the words, one entry per locale; tests/js/explore-content.test.mjs
 * keeps the two locales in step. Numbers quoted without a citation were measured on the
 * packages this edition loads (10 km BedMachine Antarctica v4, 3 km BedMachine Greenland v6
 * and the velocity grids resampled onto them).
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

const SOURCES = Object.freeze({
  bedMachineAntarctica: "https://doi.org/10.1038/s41561-019-0510-8",
  bedMachineGreenland: "https://doi.org/10.1002/2017GL074954",
  antarcticVelocity: "https://doi.org/10.1029/2019GL083826",
  greenlandVelocity: "https://nsidc.org/data/NSIDC-0776/versions/2",
  waom: "https://doi.org/10.3389/fmars.2023.1027704",
  iceShelfThinning: "https://doi.org/10.1126/science.aaa0940",
  arcticOcean: "https://data.marine.copernicus.eu/product/ARCTIC_ANALYSISFORECAST_PHY_002_001/description",
  isostaticResponse: "https://doi.org/10.1038/s41598-022-15440-y",
  isostaticGrids: "https://doi.org/10.18739/A22Z12R8C",
  massBalance: "https://doi.org/10.5194/essd-15-1597-2023",
  comnap: "https://www.comnap.aq/antarctic-facilities-information",
  scarGazetteer: "https://data.aad.gov.au/aadc/gaz/scar/",
});

export const TOUR_CHAPTERS = deepFreeze([
  {
    id: "ice-continent",
    view: {
      region: "antarctica",
      controls: {},
      sliders: { exaggeration: 4.8, iceOpacity: 1 },
      camera: { lat: -90, lon: 0, fitKm: 5600, azimuthDeg: 0, elevationDeg: 55 },
    },
  },
  {
    id: "land-beneath",
    view: {
      region: "antarctica",
      controls: {},
      sliders: { exaggeration: 4.8, iceOpacity: 0.06 },
      // The same view as the first stop, so the ice visibly melts away from it.
      camera: { lat: -90, lon: 0, fitKm: 5600, azimuthDeg: 0, elevationDeg: 55 },
      animate: { control: "iceOpacity", from: 1, to: 0.06, durationMs: 3000 },
    },
  },
  {
    id: "rivers-of-ice",
    view: {
      region: "antarctica",
      controls: { showFlowline: true },
      sliders: { exaggeration: 4.8, iceOpacity: 1 },
      camera: { lat: -78, lon: -105, fitKm: 1800, azimuthDeg: -75, elevationDeg: 42 },
    },
  },
  {
    id: "floating-ice",
    view: {
      region: "antarctica",
      controls: { showSea: true },
      sliders: { exaggeration: 4.8, iceOpacity: 1 },
      camera: { lat: -80.5, lon: -175, fitKm: 2400, azimuthDeg: 0, elevationDeg: 50 },
    },
  },
  {
    id: "southern-ocean",
    view: {
      region: "antarctica",
      // Half-transparent ice and a lower relief let the currents under the shelves show.
      controls: { showOceanCurrents: true },
      sliders: { exaggeration: 3, iceOpacity: 0.6 },
      camera: { lat: -74, lon: -108, fitKm: 1900, azimuthDeg: -70, elevationDeg: 52 },
    },
  },
  {
    id: "without-ice",
    view: {
      region: "antarctica",
      controls: { showIsostaticRebound: true, showSea: true },
      sliders: { exaggeration: 4.8, iceOpacity: 1 },
      camera: { lat: -90, lon: 0, fitKm: 5600, azimuthDeg: 0, elevationDeg: 50 },
      animate: { control: "reboundProgress", from: 0, to: 100, durationMs: 9000 },
    },
  },
  {
    id: "greenland",
    view: {
      region: "greenland",
      controls: { showFlowline: true },
      sliders: { exaggeration: 4.2, iceOpacity: 1 },
      camera: { lat: 71, lon: -41, fitKm: 2900, azimuthDeg: 0, elevationDeg: 55 },
    },
  },
  {
    id: "your-turn",
    view: {
      controls: {},
      sliders: { exaggeration: 4.8, iceOpacity: 1 },
      camera: "default",
    },
  },
]);

const EN = {
  ui: {
    tourLabel: "Guided tour",
    start: "Start the tour",
    resume: "Resume the tour",
    toolbarButton: "Guided tour",
    counter: "Stop {current} of {total}",
    back: "Back",
    next: "Next",
    finish: "Finish",
    close: "Close the tour",
    collapse: "Hide the text",
    expand: "Show the text",
    replay: "Play again",
    loading: "Loading this view…",
    loadFailed: "This stop could not be shown. Try it again, or carry on exploring.",
    stopsLabel: "Tour stops",
    goToStop: "Stop {number}: {title}",
    sources: "Sources",
    infoButton: "About {name}",
    infoClose: "Close",
    reboundLegend: "Orange: land that would rise out of the sea.",
  },
  chapters: {
    "ice-continent": {
      title: "A continent buried in ice",
      body: [
        "Antarctica is covered by an ice sheet of about 13.5 million km², larger than the United States and Mexico put together. On average the ice is more than 2 km thick.",
        "It holds enough water to raise sea level around the world by about 58 metres, if all of it melted.",
        "Heights here are stretched about five times so that you can see the shape of the ice. At true scale the ice sheet would be as thin, compared with its width, as a sheet of paper on a dinner plate.",
      ],
      sources: [
        { text: "Ice and bedrock: BedMachine Antarctica v4 (Morlighem et al., 2020)", url: SOURCES.bedMachineAntarctica },
      ],
    },
    "land-beneath": {
      title: "The land beneath the ice",
      body: [
        "With the ice made almost see-through, the land beneath it appears: mountain ranges, valleys and deep basins that no one has ever seen. They were mapped by radar that sees through the ice, flown over it for decades.",
        "Shades of blue show bedrock below today's sea level. Almost half of the ice sheet rests on such a bed, and West Antarctica almost entirely. Beneath Denman Glacier in East Antarctica the bed drops more than 3.5 km below sea level, the deepest canyon on land.",
        "Under the highest part of the ice, Dome A, lie the Gamburtsev Mountains, a range about the size of the Alps buried completely by the ice.",
      ],
      sources: [
        { text: "Bedrock: BedMachine Antarctica v4 (Morlighem et al., 2020)", url: SOURCES.bedMachineAntarctica },
      ],
    },
    "rivers-of-ice": {
      title: "Rivers of ice",
      body: [
        "Ice flows. Pulled by its own weight, it spreads from the high interior towards the coast. Each line traces the path the ice takes, and its colour shows how fast it moves.",
        "Two-thirds of the ice creeps along at less than 10 metres a year. Near the coast the flow gathers into ice streams and glaciers that move hundreds of metres to kilometres a year. Pine Island Glacier, in West Antarctica, reaches about 4 km a year.",
      ],
      sources: [
        {
          text: "Ice velocity: MEaSUREs phase-based Antarctic ice velocity map (Mouginot et al., 2019)",
          url: SOURCES.antarcticVelocity,
        },
      ],
    },
    "floating-ice": {
      title: "Where the ice meets the sea",
      body: [
        "At the coast the ice lifts off its bed and floats on the sea as ice shelves. The line where it starts to float is called the grounding line.",
        "The Ross Ice Shelf, in front of you, is about the size of France and several hundred metres thick.",
        "Ice shelves hold back the ice flowing in behind them, like a cork in a bottle. Where they thin or break up, the glaciers behind them speed up and deliver more ice to the ocean.",
      ],
      sources: [
        { text: "Grounded and floating ice: BedMachine Antarctica v4 (Morlighem et al., 2020)", url: SOURCES.bedMachineAntarctica },
      ],
    },
    "southern-ocean": {
      title: "The ocean around Antarctica",
      body: [
        "Currents circle the continent and reach into the cavities beneath the ice shelves. Each line follows the moving water, and its colour shows how warm and how salty that water is.",
        "Where water from the deep ocean, a few degrees above freezing, reaches the ice shelves, as it does here in the Amundsen Sea, it melts them from below. The ice shelves of this coast are thinning faster than any others in Antarctica.",
      ],
      sources: [
        { text: "Ocean currents: WAOM2 ocean model, annual mean (Dias et al., 2023)", url: SOURCES.waom },
        { text: "Ice-shelf thinning: Paolo, Fricker & Padman (2015)", url: SOURCES.iceShelfThinning },
      ],
    },
    "without-ice": {
      title: "If the ice were gone",
      body: [
        "The ice is so heavy that it has pressed the Earth's crust down into the mantle beneath. Watch what happens as the ice melts away.",
        "Freed of its load, the land slowly rises again, by up to about a kilometre in places, over many thousands of years. Orange marks land that would rise out of the sea.",
        "Even then, much of West Antarctica would stay under water, leaving a scatter of islands. The sea here is drawn at its level in that ice-free world, raised by the meltwater of both ice sheets.",
      ],
      sources: [
        {
          text: "Isostatic response: Paxman, Austermann & Hollyday (2022), grid files v3 (CC BY 4.0)",
          url: SOURCES.isostaticResponse,
        },
      ],
    },
    greenland: {
      title: "Greenland",
      body: [
        "Greenland's ice sheet is about a seventh the size of Antarctica's, and on average about 1.6 km thick. It holds enough water to raise sea level by about 7.4 metres.",
        "Since the early 1990s it has lost almost twice as much ice as Antarctica, as warmer summers melt its surface and its glaciers speed up.",
        "Sermeq Kujalleq (Jakobshavn Isbræ), on the west coast, is one of the fastest glaciers on Earth: in this map it flows at up to about 14 km a year.",
      ],
      sources: [
        { text: "Ice and bedrock: BedMachine Greenland v6 (Morlighem et al., 2017)", url: SOURCES.bedMachineGreenland },
        { text: "Ice velocity: ITS_LIVE velocity mosaic, version 2", url: SOURCES.greenlandVelocity },
        { text: "Ice loss since 1992: IMBIE (Otosaka et al., 2023)", url: SOURCES.massBalance },
      ],
    },
    "your-turn": {
      title: "Your turn",
      body: [
        "That is the end of the tour. Now explore on your own: switch layers on and off, search for a research station or a mountain range, and drag to look around.",
        "Tap the i beside a layer to find out what it shows. The research edition has every dataset in 3D ICE, including the friction under the ice, its meltwater and more detailed maps.",
      ],
      sources: [],
    },
  },
  info: {
    seeThrough: {
      title: "See through the ice",
      body: [
        "Fades the ice so that you can see the bedrock underneath. At 0 the ice is invisible; at 1 it is solid.",
      ],
      sources: [
        { text: "BedMachine Antarctica v4 (Morlighem et al., 2020)", url: SOURCES.bedMachineAntarctica },
        { text: "BedMachine Greenland v6 (Morlighem et al., 2017)", url: SOURCES.bedMachineGreenland },
      ],
    },
    exaggeration: {
      title: "Stretch the heights",
      body: [
        "Ice sheets are thousands of kilometres wide but only a few kilometres thick. Stretching the heights makes their shape visible. At 1× you see the true proportions, which look almost flat.",
      ],
      sources: [],
    },
    iceSheet: {
      title: "The ice sheet",
      body: [
        "The top of the ice, shaded whiter where it is thicker. Ice resting on land is called grounded. Around the coast, ice floating on the sea forms ice shelves, drawn slightly bluer.",
      ],
      sources: [
        { text: "BedMachine Antarctica v4 (Morlighem et al., 2020)", url: SOURCES.bedMachineAntarctica },
        { text: "BedMachine Greenland v6 (Morlighem et al., 2017)", url: SOURCES.bedMachineGreenland },
      ],
    },
    bedrock: {
      title: "Bedrock",
      body: [
        "The height of the land under the ice and of the sea floor around it, in metres above or below today's sea level. Greens and browns lie above sea level, blues below it.",
      ],
      sources: [
        { text: "BedMachine Antarctica v4 (Morlighem et al., 2020)", url: SOURCES.bedMachineAntarctica },
        { text: "BedMachine Greenland v6 (Morlighem et al., 2017)", url: SOURCES.bedMachineGreenland },
      ],
    },
    iceFlow: {
      title: "Ice flow",
      body: [
        "Each line follows the direction the ice surface moves, as measured from satellites. Its colour shows the speed, from a few metres a year in the interior to kilometres a year in the fastest glaciers. The moving light shows which way the ice is going.",
      ],
      sources: [
        { text: "Antarctica: MEaSUREs phase-based ice velocity (Mouginot et al., 2019)", url: SOURCES.antarcticVelocity },
        { text: "Greenland: ITS_LIVE velocity mosaic, version 2", url: SOURCES.greenlandVelocity },
      ],
    },
    oceanCurrents: {
      title: "Ocean currents",
      body: [
        "Lines follow the currents, from near the surface down to deep water, as computed by an ocean model. Colour combines the water's temperature and saltiness; the legend below shows the scale. Around Antarctica the currents reach into the cavities under the ice shelves.",
      ],
      sources: [
        { text: "Antarctica: WAOM2 ocean model, annual mean (Dias et al., 2023)", url: SOURCES.waom },
        { text: "Greenland: Copernicus Marine Arctic Ocean analysis, monthly mean", url: SOURCES.arcticOcean },
      ],
    },
    seaLevel: {
      title: "Sea level",
      body: [
        "A flat, see-through surface at today's sea level. Everything beneath it is under water: the sea floor, and the bedrock under much of the ice.",
      ],
      sources: [],
    },
    rebound: {
      title: "Remove the ice",
      body: [
        "Shows the land as it would be long after all the ice had melted. Without the weight of the ice the crust slowly springs back, rising by up to about a kilometre. Orange marks land that would rise out of the sea, and the sea is drawn at its level in that ice-free world.",
        "The slider blends between today (0%) and that final state (100%). It is not a timeline: the real rebound takes many thousands of years.",
      ],
      sources: [
        {
          text: "Paxman, Austermann & Hollyday (2022), grid files v3 (CC BY 4.0)",
          url: SOURCES.isostaticResponse,
        },
      ],
    },
    places: {
      title: "Places",
      body: [
        "Research stations are the year-round and summer bases where scientists live and work. Place names cover mountains, glaciers, ice shelves and seas. Search for any of them to fly there.",
      ],
      sources: [
        { text: "Antarctic stations: COMNAP Antarctic Facilities List", url: SOURCES.comnap },
        { text: "Antarctic names: SCAR Composite Gazetteer of Antarctica", url: SOURCES.scarGazetteer },
      ],
    },
  },
};

const ZH = {
  ui: {
    tourLabel: "导览",
    start: "开始导览",
    resume: "继续导览",
    toolbarButton: "导览",
    counter: "第 {current} 站，共 {total} 站",
    back: "上一站",
    next: "下一站",
    finish: "完成",
    close: "关闭导览",
    collapse: "收起文字",
    expand: "展开文字",
    replay: "再看一遍",
    loading: "正在加载这一视图……",
    loadFailed: "这一站没能显示出来。可以再试一次，或者继续自由探索。",
    stopsLabel: "导览站点",
    goToStop: "第 {number} 站：{title}",
    sources: "资料来源",
    infoButton: "关于{name}",
    infoClose: "关闭",
    reboundLegend: "橙色：将会升出海面的陆地。",
  },
  chapters: {
    "ice-continent": {
      title: "被冰封的大陆",
      body: [
        "南极洲覆盖着一个面积约 1350 万平方公里的冰盖，比美国和墨西哥加起来还大。冰层平均厚度超过 2 公里。",
        "如果这些冰全部融化，全球海平面将上升约 58 米。",
        "为了看清冰盖的形状，这里把高度拉伸了约五倍。按真实比例，冰盖的厚度与宽度之比，就像餐盘上的一张纸那么薄。",
      ],
      sources: [
        { text: "冰层与基岩：BedMachine Antarctica v4（Morlighem 等，2020）", url: SOURCES.bedMachineAntarctica },
      ],
    },
    "land-beneath": {
      title: "冰下的大地",
      body: [
        "把冰层调得几乎透明，冰下的大地就显现出来：山脉、河谷和深邃的盆地，从来没有人亲眼见过。它们是几十年来用飞机搭载、能穿透冰层的雷达测绘出来的。",
        "蓝色表示低于今天海平面的基岩。将近一半的冰盖就压在这样的基岩上，西南极几乎全部如此。在东南极的登曼冰川下方，基岩深达海平面以下 3.5 公里以上，是陆地上最深的峡谷。",
        "在冰盖的最高处冰穹 A 之下，埋藏着甘布尔采夫山脉。这条山脉的规模与阿尔卑斯山相当，却完全被冰覆盖。",
      ],
      sources: [
        { text: "基岩：BedMachine Antarctica v4（Morlighem 等，2020）", url: SOURCES.bedMachineAntarctica },
      ],
    },
    "rivers-of-ice": {
      title: "冰的河流",
      body: [
        "冰是会流动的。在自身重量的作用下，冰从高耸的内陆向海岸扩展。每条线描绘出冰流动的路径，颜色表示流动的快慢。",
        "三分之二的冰每年移动不到 10 米。靠近海岸时，冰汇聚成冰流和冰川，每年移动数百米乃至数公里。西南极的松岛冰川每年流动约 4 公里。",
      ],
      sources: [
        {
          text: "冰流速度：MEaSUREs 基于相位的南极冰流速度图（Mouginot 等，2019）",
          url: SOURCES.antarcticVelocity,
        },
      ],
    },
    "floating-ice": {
      title: "冰与海相遇的地方",
      body: [
        "到了海岸，冰会离开基岩，漂浮在海面上，形成冰架。冰开始漂浮的那条线叫做接地线。",
        "眼前的罗斯冰架面积与法国相当，厚达数百米。",
        "冰架像瓶口的软木塞一样，挡住了后方流来的冰。冰架一旦变薄或崩解，后面的冰川就会加速，把更多的冰送进海洋。",
      ],
      sources: [
        { text: "接地冰与漂浮冰：BedMachine Antarctica v4（Morlighem 等，2020）", url: SOURCES.bedMachineAntarctica },
      ],
    },
    "southern-ocean": {
      title: "环绕南极的海洋",
      body: [
        "洋流环绕着南极大陆，还会深入冰架下方的空腔。每条线追踪流动的海水，颜色表示海水有多温暖、有多咸。",
        "来自深海、比冰点高出几度的海水一旦抵达冰架，就会从下方融化冰架，这里的阿蒙森海正是如此。这一带海岸的冰架，是整个南极洲变薄最快的。",
      ],
      sources: [
        { text: "洋流：WAOM2 海洋模式，年平均（Dias 等，2023）", url: SOURCES.waom },
        { text: "冰架变薄：Paolo、Fricker 与 Padman（2015）", url: SOURCES.iceShelfThinning },
      ],
    },
    "without-ice": {
      title: "如果冰消失了",
      body: [
        "冰实在太重，把地壳压进了下方的地幔。来看看冰融化时会发生什么。",
        "卸下重负之后，陆地会慢慢回升，有些地方最多可升高约 1 公里，这个过程要持续数千年乃至更久。橙色表示将会升出海面的陆地。",
        "即便如此，西南极的大部分仍会留在水下，只剩下星罗棋布的岛屿。这里的海面画在那个无冰世界的海平面上，其中包含了两大冰盖的融水。",
      ],
      sources: [
        {
          text: "地壳均衡响应：Paxman、Austermann 与 Hollyday（2022），网格文件 v3（CC BY 4.0）",
          url: SOURCES.isostaticResponse,
        },
      ],
    },
    greenland: {
      title: "格陵兰",
      body: [
        "格陵兰冰盖的面积约为南极冰盖的七分之一，平均厚度约 1.6 公里。如果全部融化，海平面将上升约 7.4 米。",
        "自 1990 年代初以来，格陵兰损失的冰几乎是南极的两倍：夏季变暖融化了冰面，冰川也在加速流动。",
        "西海岸的瑟梅克库亚莱克冰川（雅各布港冰川）是地球上流动最快的冰川之一，在这张图中最快每年流动约 14 公里。",
      ],
      sources: [
        { text: "冰层与基岩：BedMachine Greenland v6（Morlighem 等，2017）", url: SOURCES.bedMachineGreenland },
        { text: "冰流速度：ITS_LIVE 速度镶嵌图，第 2 版", url: SOURCES.greenlandVelocity },
        { text: "1992 年以来的冰量损失：IMBIE（Otosaka 等，2023）", url: SOURCES.massBalance },
      ],
    },
    "your-turn": {
      title: "轮到你了",
      body: [
        "导览到此结束。现在请自由探索：打开或关闭各个图层，搜索一个科考站或一座山脉，拖动画面四处看看。",
        "点击图层旁的 i，可以了解它显示的内容。专业版包含 3D ICE 的全部数据集，包括冰下的摩擦、冰下融水和更精细的地图。",
      ],
      sources: [],
    },
  },
  info: {
    seeThrough: {
      title: "透视冰层",
      body: ["让冰层逐渐变透明，就能看到下方的基岩。设为 0 时冰完全透明，设为 1 时冰不透明。"],
      sources: [
        { text: "BedMachine Antarctica v4（Morlighem 等，2020）", url: SOURCES.bedMachineAntarctica },
        { text: "BedMachine Greenland v6（Morlighem 等，2017）", url: SOURCES.bedMachineGreenland },
      ],
    },
    exaggeration: {
      title: "拉伸高度",
      body: [
        "冰盖宽达数千公里，厚度却只有几公里。把高度拉伸之后，才能看清它们的形状。设为 1× 时是真实比例，看起来几乎是平的。",
      ],
      sources: [],
    },
    iceSheet: {
      title: "冰盖",
      body: [
        "冰的上表面，冰越厚颜色越白。压在陆地上的冰称为接地冰；在海岸周围漂浮在海面上的冰形成冰架，颜色略偏蓝。",
      ],
      sources: [
        { text: "BedMachine Antarctica v4（Morlighem 等，2020）", url: SOURCES.bedMachineAntarctica },
        { text: "BedMachine Greenland v6（Morlighem 等，2017）", url: SOURCES.bedMachineGreenland },
      ],
    },
    bedrock: {
      title: "基岩",
      body: ["冰下陆地和周围海底的高度，以高于或低于今天海平面的米数表示。绿色和棕色高于海平面，蓝色低于海平面。"],
      sources: [
        { text: "BedMachine Antarctica v4（Morlighem 等，2020）", url: SOURCES.bedMachineAntarctica },
        { text: "BedMachine Greenland v6（Morlighem 等，2017）", url: SOURCES.bedMachineGreenland },
      ],
    },
    iceFlow: {
      title: "冰流",
      body: [
        "每条线沿着卫星测得的冰面运动方向延伸。颜色表示速度：内陆每年只有几米，最快的冰川每年可达数公里。流动的光点显示冰前进的方向。",
      ],
      sources: [
        { text: "南极：MEaSUREs 基于相位的冰流速度（Mouginot 等，2019）", url: SOURCES.antarcticVelocity },
        { text: "格陵兰：ITS_LIVE 速度镶嵌图，第 2 版", url: SOURCES.greenlandVelocity },
      ],
    },
    oceanCurrents: {
      title: "洋流",
      body: [
        "这些线追踪海洋模式计算出的洋流，从近海面一直到深层海水。颜色综合了海水的温度和咸度，刻度见下方图例。在南极周围，洋流会深入冰架下方的空腔。",
      ],
      sources: [
        { text: "南极：WAOM2 海洋模式，年平均（Dias 等，2023）", url: SOURCES.waom },
        { text: "格陵兰：哥白尼海洋服务北冰洋分析，月平均", url: SOURCES.arcticOcean },
      ],
    },
    seaLevel: {
      title: "海平面",
      body: ["一个位于今天海平面高度的半透明平面。它下方的一切都在水下：海底，以及许多冰层下方的基岩。"],
      sources: [],
    },
    rebound: {
      title: "移除冰层",
      body: [
        "显示冰全部融化很久以后陆地的样子。没有了冰的重量，地壳会慢慢回弹，最多升高约 1 公里。橙色表示将会升出海面的陆地，海面则画在那个无冰世界的海平面上。",
        "滑块在今天（0%）和最终状态（100%）之间过渡，并不代表时间进程：真实的回弹需要数千年乃至更久。",
      ],
      sources: [
        {
          text: "Paxman、Austermann 与 Hollyday（2022），网格文件 v3（CC BY 4.0）",
          url: SOURCES.isostaticResponse,
        },
      ],
    },
    places: {
      title: "地点",
      body: [
        "科考站是科学家常年或夏季生活、工作的基地。地名涵盖山脉、冰川、冰架和海域。搜索其中任何一个，就能飞过去看看。",
      ],
      sources: [
        { text: "南极科考站：COMNAP 南极设施名录", url: SOURCES.comnap },
        { text: "南极地名：SCAR 南极综合地名录", url: SOURCES.scarGazetteer },
      ],
    },
  },
};

function deepFreeze(value) {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}


export const EXPLORE_CONTENT = deepFreeze({
  "en-US": EN,
  "zh-CN": ZH,
});

/** The copy for a locale, falling back to English. */
export function getExploreContent(locale) {
  return EXPLORE_CONTENT[locale] || EXPLORE_CONTENT["en-US"];
}
