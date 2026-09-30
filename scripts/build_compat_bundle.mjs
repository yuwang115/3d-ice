import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { cp, mkdir, readdir, rm, stat, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const staticRoot = path.join(repoRoot, "static");
const distRoot = path.join(repoRoot, "dist");
const stageRoot = path.join(distRoot, "stage");
const bundleName = "3d-ice-compat.tar.gz";
const checksumName = `${bundleName}.sha256`;
const manifestName = "3d-ice-compat-manifest.json";

/**
 * Files a host serves at the same paths as 3d-ice.com: both explorer editions in both
 * locales, and what the home page loads, typefaces included, so a copy of the page draws
 * the same weights. yuwang.blog mounts the bundle at its site root.
 */
const SERVED_PATHS = Object.freeze([
  "tools",
  "explore/index.html",
  "zh/explore/index.html",
  "zh/tools/3D-interactive-cryosphere-explorer.html",
  "js/3d-ice-locale.js",
  "js/3d-ice-home.js",
  "css/3d-ice-home.css",
  "css/3d-ice-type.css",
  "fonts/playfair-display-latin.woff2",
  "fonts/playfair-display-latin-ext.woff2",
  "fonts/space-grotesk-latin.woff2",
  "fonts/space-grotesk-latin-ext.woff2",
]);

/**
 * The home pages, for a host that builds its own copy of them (yuwang.blog does, at
 * /tools/3d-ice/). They are kept under home/ so that mounting the bundle at a site root
 * cannot replace that site's own home page.
 */
const EMBED_SOURCES = Object.freeze({
  "home/en-US.html": "index.html",
  "home/zh-CN.html": "zh/index.html",
});

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    stdio: "inherit",
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with status ${result.status}`);
  }
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

async function collectFiles(rootDir, currentDir = rootDir) {
  const entries = await readdir(currentDir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const absolutePath = path.join(currentDir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectFiles(rootDir, absolutePath)));
      continue;
    }
    const details = await stat(absolutePath);
    files.push({
      path: path.relative(rootDir, absolutePath).split(path.sep).join("/"),
      bytes: details.size,
    });
  }
  return files.sort((left, right) => left.path.localeCompare(right.path));
}

async function main() {
  const bundlePath = path.join(distRoot, bundleName);
  const checksumPath = path.join(distRoot, checksumName);
  const manifestPath = path.join(distRoot, manifestName);

  await rm(distRoot, { recursive: true, force: true });
  await mkdir(distRoot, { recursive: true });

  for (const servedPath of SERVED_PATHS) {
    await cp(path.join(staticRoot, servedPath), path.join(stageRoot, servedPath), { recursive: true });
  }
  for (const [bundleEntry, staticPath] of Object.entries(EMBED_SOURCES)) {
    await cp(path.join(staticRoot, staticPath), path.join(stageRoot, bundleEntry));
  }
  const topLevel = (await readdir(stageRoot)).sort();
  run("tar", ["-czf", bundlePath, "-C", stageRoot, ...topLevel]);

  const bundleSha = await sha256(bundlePath);
  const manifest = {
    bundle: bundleName,
    createdAt: new Date().toISOString(),
    sha256: bundleSha,
    // Paths relative to the bundle root.
    files: await collectFiles(stageRoot),
  };
  await rm(stageRoot, { recursive: true, force: true });

  await writeFile(checksumPath, `${bundleSha}  ${bundleName}\n`);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  process.stdout.write(`[3d-ice] built ${bundlePath}\n`);
}

await main();
