/**
 * Write the explorer's thickness-change colour scale, which the colour bar of Figure 4 uses:
 * 256 RGB samples of the ramp and the positions of its ticks, from
 * static/tools/js/ice-projection.js.
 *
 *   node paper/ess/figures/sample_ramp.mjs [output json]   # default: paper/ess/data/thickness_change_ramp.json
 */
import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { thicknessChangeRampColor, thicknessChangeScaleT } from "../../../static/tools/js/ice-projection.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = process.argv[2] ? resolve(process.argv[2]) : resolve(HERE, "../data/thickness_change_ramp.json");
const SAMPLES = 256;
const TICKS_METERS = [-1000, -100, -10, 0, 10, 100, 1000];

const ramp = Array.from({ length: SAMPLES }, (_, i) => thicknessChangeRampColor(i / (SAMPLES - 1)));
const ticks = TICKS_METERS.map((meters) => ({ meters, t: thicknessChangeScaleT(meters) }));
writeFileSync(OUT, `${JSON.stringify({ ramp, ticks })}\n`);
console.log(`wrote ${OUT}`);
