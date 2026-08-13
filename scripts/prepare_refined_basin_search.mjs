import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createRefinedBasinSearchFeatures } from "../static/tools/js/polar-feature-search.js";
import { validateRefinedBasinDataset } from "../static/tools/js/polar-refined-basins.js";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.resolve(scriptDir, "../static/tools/data");
const configurations = [
  {
    region: "antarctica",
    source: "imbie_refined_basins_v2.json",
    output: "antarctica_refined_basins_search.json",
  },
  {
    region: "greenland",
    source: "greenland_basins_ps_v1_4_2.json",
    output: "greenland_refined_basins_search.json",
  },
];

for (const configuration of configurations) {
  const sourcePath = path.join(dataDir, configuration.source);
  const outputPath = path.join(dataDir, configuration.output);
  const source = validateRefinedBasinDataset(JSON.parse(await readFile(sourcePath, "utf8")));
  const items = createRefinedBasinSearchFeatures(source, configuration.region);
  const catalogue = {
    schema_version: 1,
    region: configuration.region,
    layer: "refined_basins",
    feature_count: items.length,
    source_feature_count: source.basin_count,
    source_filename: configuration.source,
    items,
  };
  await writeFile(outputPath, `${JSON.stringify(catalogue)}\n`, "utf8");
  process.stdout.write(`[3d-ice] wrote ${outputPath} (${items.length} refined basins)\n`);
}
