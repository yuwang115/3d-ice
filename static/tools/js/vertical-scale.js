/**
 * The scene's vertical scale.
 *
 * Terrain is built with one scene unit standing for 52 km horizontally but 3.8 km
 * vertically, so heights are already drawn BASE_HORIZONTAL_VERTICAL_SCALE_RATIO (about
 * 13.7) times taller than horizontal distances before the exaggeration slider multiplies
 * them. The true vertical exaggeration is the slider value times that ratio, and the
 * slider's label shows it.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

export const BASE_HORIZONTAL_VERTICAL_SCALE_RATIO = 52000 / 3800;

/** The true vertical exaggeration for a slider value. */
export function trueVerticalExaggeration(sliderValue) {
  return Number(sliderValue) * BASE_HORIZONTAL_VERTICAL_SCALE_RATIO;
}

/** The slider label: the true factor as a whole number, for example "66×". */
export function formatVerticalExaggeration(sliderValue) {
  const factor = trueVerticalExaggeration(sliderValue);
  return Number.isFinite(factor) ? `${Math.round(factor)}×` : "";
}
