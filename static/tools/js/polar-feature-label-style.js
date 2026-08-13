const REGION_LABEL_METRICS = Object.freeze({
  antarctica: Object.freeze({ fontSize: 44, strokeWidth: 7, worldScale: 0.0076 }),
  greenland: Object.freeze({ fontSize: 56, strokeWidth: 8, worldScale: 0.0104 }),
});

export const POLAR_LABEL_STYLES = Object.freeze({
  research_stations: Object.freeze({ textColor: "#d8cfad" }),
  geographic_names: Object.freeze({ textColor: "#aec6d1" }),
  refined_basins: Object.freeze({ textColor: "#ffe8a3" }),
});

const POLAR_MARKER_SIZES = Object.freeze({
  research_stations: 3.75,
  geographic_names: 2.75,
});

export function getPolarFeatureMarkerStyle(layer) {
  const normalizedLayer = Object.hasOwn(POLAR_MARKER_SIZES, layer) ? layer : "geographic_names";
  return {
    color: POLAR_LABEL_STYLES[normalizedLayer].textColor,
    size: POLAR_MARKER_SIZES[normalizedLayer],
  };
}

export function getPolarFeatureMarkerMaterialOptions(layer, map = null) {
  const style = getPolarFeatureMarkerStyle(layer);
  return {
    color: style.color,
    size: style.size,
    map,
    alphaTest: map ? 0.05 : 0,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.96,
    depthTest: false,
    depthWrite: false,
  };
}

export function createPolarMarkerCanvas(documentRef) {
  const canvas = documentRef.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#ffffff";
  context.beginPath();
  context.arc(32, 32, 30, 0, Math.PI * 2);
  context.fill();
  return canvas;
}

export function getPolarFeatureLabelStyle(layer, region = "antarctica", { selected = false } = {}) {
  const color = POLAR_LABEL_STYLES[layer]?.textColor || POLAR_LABEL_STYLES.geographic_names.textColor;
  const metrics = REGION_LABEL_METRICS[region] || REGION_LABEL_METRICS.antarctica;
  return {
    textColor: color,
    strokeColor: selected ? "rgba(3, 14, 24, 0.99)" : "rgba(7, 17, 26, 0.95)",
    fontSize: metrics.fontSize + (selected ? 6 : 0),
    strokeWidth: metrics.strokeWidth + (selected ? 2 : 0),
    worldScale: metrics.worldScale,
    fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
  };
}

export function createPolarLabelCanvas(
  documentRef,
  labelText,
  { layer = "geographic_names", region = "antarctica", selected = false } = {},
) {
  const style = getPolarFeatureLabelStyle(layer, region, { selected });
  const text = String(labelText || "").trim().slice(0, 180);
  const canvas = documentRef.createElement("canvas");
  const probe = canvas.getContext("2d");
  if (!probe) return null;

  const padding = 14;
  probe.font = `600 ${style.fontSize}px ${style.fontFamily}`;
  const textWidth = Math.max(1, Math.ceil(probe.measureText(text).width));
  canvas.width = Math.min(1600, textWidth + padding * 2 + style.strokeWidth * 2);
  canvas.height = Math.ceil(style.fontSize * 1.45 + padding * 2);

  const context = canvas.getContext("2d");
  if (!context) return null;
  context.font = `600 ${style.fontSize}px ${style.fontFamily}`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.lineJoin = "round";
  context.lineWidth = style.strokeWidth;
  context.strokeStyle = style.strokeColor;
  context.fillStyle = style.textColor;
  context.strokeText(text, canvas.width / 2, canvas.height / 2);
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  return canvas;
}
