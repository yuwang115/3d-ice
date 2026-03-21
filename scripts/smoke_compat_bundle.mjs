import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { access, readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const distRoot = path.join(repoRoot, "dist");
const bundlePath = path.join(distRoot, "3d-ice-compat.tar.gz");
const checksumPath = `${bundlePath}.sha256`;

const requiredEntries = [
  "tools/3D-interactive-cryosphere-explorer.html",
  "tools/3d-antarctica/index.html",
  "tools/3d-ice-logo-light.jpg",
  "tools/3d-ice-logo.jpg",
  "tools/antarctica-geometry-worker.js",
  "tools/data/bedmachine_antarctica_v4_480.meta.json",
  "tools/data/greenland_ocean_currents_cmems_202508.meta.json",
  "tools/media/3d-ice/antarctica-velocity-preview.mp4",
  "tools/vendor/three/three.module.min.js",
];

async function ensureFile(filePath) {
  await access(filePath);
}

async function sha256(filePath) {
  return await new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(filePath);
    stream.on("error", reject);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

function listArchiveEntries(filePath) {
  const result = spawnSync("tar", ["-tzf", filePath], {
    cwd: repoRoot,
    stdio: "pipe",
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(result.stderr || `tar -tzf ${filePath} failed`);
  }
  return new Set(
    result.stdout
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
  );
}

async function main() {
  await ensureFile(bundlePath);
  await ensureFile(checksumPath);

  const entries = listArchiveEntries(bundlePath);
  for (const requiredEntry of requiredEntries) {
    if (!entries.has(requiredEntry)) {
      throw new Error(`Bundle is missing required entry: ${requiredEntry}`);
    }
  }

  const checksumContents = await readFile(checksumPath, "utf8");
  const expectedSha = checksumContents.trim().split(/\s+/)[0]?.toLowerCase();
  const actualSha = await sha256(bundlePath);
  if (!expectedSha || expectedSha !== actualSha) {
    throw new Error(`Checksum mismatch for ${bundlePath}`);
  }

  process.stdout.write("[3d-ice] compatibility bundle smoke test passed\n");
}

await main();
