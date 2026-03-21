import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readdir, rm, stat, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const staticRoot = path.join(repoRoot, "static");
const toolsRoot = path.join(staticRoot, "tools");
const distRoot = path.join(repoRoot, "dist");
const bundleName = "3d-ice-compat.tar.gz";
const checksumName = `${bundleName}.sha256`;
const manifestName = "3d-ice-compat-manifest.json";

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

  run("tar", ["-czf", bundlePath, "-C", staticRoot, "tools"]);

  const bundleSha = await sha256(bundlePath);
  const manifest = {
    bundle: bundleName,
    createdAt: new Date().toISOString(),
    sha256: bundleSha,
    files: await collectFiles(toolsRoot),
  };

  await writeFile(checksumPath, `${bundleSha}  ${bundleName}\n`);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  process.stdout.write(`[3d-ice] built ${bundlePath}\n`);
}

await main();
