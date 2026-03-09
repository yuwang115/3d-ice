import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";

function parseArgs(argv) {
  const defaults = {
    baseUrl: "http://127.0.0.1:1313",
    sceneFile: new URL("./trailer/scenes.json", import.meta.url),
    outputDir: "output/trailer",
    fps: null,
    width: null,
    height: null,
    headless: true,
    sceneId: null,
    skipFfmpeg: false,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === "--base-url" && next) {
      defaults.baseUrl = next;
      i += 1;
    } else if (arg === "--scene-file" && next) {
      defaults.sceneFile = path.resolve(next);
      i += 1;
    } else if (arg === "--output-dir" && next) {
      defaults.outputDir = path.resolve(next);
      i += 1;
    } else if (arg === "--fps" && next) {
      defaults.fps = Number(next);
      i += 1;
    } else if (arg === "--width" && next) {
      defaults.width = Number(next);
      i += 1;
    } else if (arg === "--height" && next) {
      defaults.height = Number(next);
      i += 1;
    } else if (arg === "--headless" && next) {
      defaults.headless = next !== "0" && next !== "false";
      i += 1;
    } else if (arg === "--scene" && next) {
      defaults.sceneId = next;
      i += 1;
    } else if (arg === "--skip-ffmpeg") {
      defaults.skipFfmpeg = true;
    }
  }
  return defaults;
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}

function normalizeBaseUrl(baseUrl) {
  return String(baseUrl || "").replace(/\/+$/, "");
}

function interpolate(a, b, t) {
  return a + (b - a) * t;
}

function easingAt(name, t) {
  switch (String(name || "linear")) {
    case "easeinoutcubic":
    case "ease-in-out-cubic":
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    case "easeinoutquad":
    case "ease-in-out-quad":
      return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    case "easeoutcubic":
    case "ease-out-cubic":
      return 1 - Math.pow(1 - t, 3);
    case "linear":
    default:
      return t;
  }
}

function interpolatePose(fromPose, toPose, t) {
  if (!fromPose && !toPose) return null;
  const start = fromPose || toPose;
  const end = toPose || fromPose;
  return {
    position: start.position.map((value, index) => interpolate(value, end.position[index], t)),
    target: start.target.map((value, index) => interpolate(value, end.target[index], t)),
    fov: interpolate(start.fov, end.fov, t),
  };
}

function msToSrt(ms) {
  const totalMs = Math.max(0, Math.round(ms));
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const seconds = Math.floor((totalMs % 60000) / 1000);
  const millis = totalMs % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")},${String(
    millis
  ).padStart(3, "0")}`;
}

function buildSceneUrl(baseUrl, scene) {
  const params = new URLSearchParams();
  if (scene.mode === "trailer") {
    params.set("mode", "trailer");
  }
  if (scene.region) {
    params.set("region", scene.region);
  }
  if (scene.preset) {
    params.set("preset", scene.preset);
  }
  const query = params.toString();
  return `${baseUrl}/tools/antarctica-bedmachine-3d.html${query ? `?${query}` : ""}`;
}

function buildBaseState(scene) {
  return {
    region: scene.region ?? null,
    preset: scene.preset ?? null,
    exaggeration: scene.exaggeration ?? null,
    iceOpacity: scene.iceOpacity ?? null,
    toggles: scene.toggles ?? null,
    cameraPose: scene.cameraFrom ?? null,
    overlay: Object.prototype.hasOwnProperty.call(scene, "overlay") ? scene.overlay : undefined,
  };
}

function mergeStepState(baseState, scene, step) {
  const stepState = step?.state || {};
  return {
    region: stepState.region ?? baseState.region,
    preset: stepState.preset ?? baseState.preset,
    exaggeration: stepState.exaggeration ?? baseState.exaggeration,
    iceOpacity: stepState.iceOpacity ?? baseState.iceOpacity,
    toggles: Object.prototype.hasOwnProperty.call(stepState, "toggles") ? stepState.toggles : baseState.toggles,
    cameraPose: stepState.cameraPose ?? scene.cameraFrom ?? baseState.cameraPose,
    overlay: Object.prototype.hasOwnProperty.call(stepState, "overlay") ? stepState.overlay : baseState.overlay,
  };
}

async function waitForTrailerApi(page) {
  await page.waitForFunction(() => window.trailerApi && typeof window.trailerApi.applyState === "function", undefined, {
    timeout: 60000,
  });
  await page.evaluate(() => window.trailerApi.waitUntilReady());
}

async function scrollPanel(page, target) {
  if (!target) return;
  if (target === "top") {
    await page.evaluate(() => {
      const panel = document.querySelector(".panel");
      if (panel) panel.scrollTop = 0;
    });
    return;
  }
  if (target === "snapshot") {
    await page.evaluate(() => {
      const panel = document.querySelector(".panel");
      const metaList = document.getElementById("metaList");
      if (!panel || !metaList) return;
      panel.scrollTop = Math.max(0, metaList.offsetTop - 12);
    });
  }
}

async function applyState(page, state) {
  await page.evaluate(async (payload) => {
    await window.trailerApi.applyState(payload);
  }, state);
}

async function setCameraPose(page, pose) {
  if (!pose) return;
  await page.evaluate((payload) => {
    window.trailerApi.setCameraPose(payload);
    window.advanceTime(16);
  }, pose);
}

async function captureFrame(page, outputPath) {
  await page.screenshot({
    path: outputPath,
    type: "png",
    animations: "disabled",
    timeout: 0,
  });
}

function findFfmpeg() {
  const result = spawnSync("bash", ["-lc", "command -v ffmpeg"], {
    encoding: "utf-8",
  });
  if (result.status !== 0) return null;
  const ffmpegPath = result.stdout.trim();
  return ffmpegPath || null;
}

function runFfmpeg(ffmpegPath, args) {
  const result = spawnSync(ffmpegPath, args, {
    stdio: "inherit",
  });
  if (result.status !== 0) {
    throw new Error(`ffmpeg exited with status ${result.status}`);
  }
}

function probeDurationSeconds(mediaPath) {
  const result = spawnSync("ffprobe", [
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=nw=1:nk=1",
    mediaPath,
  ], {
    encoding: "utf-8",
  });
  if (result.status !== 0) {
    throw new Error(`ffprobe failed for ${mediaPath}`);
  }
  const value = Number(result.stdout.trim());
  if (!Number.isFinite(value)) {
    throw new Error(`ffprobe returned an invalid duration for ${mediaPath}`);
  }
  return value;
}

function writeSrt(scenes, outputPath) {
  let currentStart = 0;
  const entries = [];
  let index = 1;
  for (const scene of scenes) {
    if (!scene.voiceover) {
      currentStart += scene.durationMs;
      continue;
    }
    entries.push(
      `${index}\n${msToSrt(currentStart)} --> ${msToSrt(currentStart + scene.durationMs)}\n${String(scene.voiceover).trim()}\n`
    );
    index += 1;
    currentStart += scene.durationMs;
  }
  fs.writeFileSync(outputPath, entries.join("\n"), "utf-8");
}

function writeVoiceoverText(spec, scenes, outputPath) {
  const lines = [
    spec.meta?.title ? `Title: ${spec.meta.title}` : null,
    spec.meta?.voiceoverLanguage ? `Language: ${spec.meta.voiceoverLanguage}` : null,
    spec.meta?.voiceoverStyle ? `Style: ${spec.meta.voiceoverStyle}` : null,
    "",
    ...scenes.map((scene, index) => `${index + 1}. ${String(scene.voiceover || "").trim()}`),
    "",
  ].filter((line) => line !== null);
  fs.writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf-8");
}

async function captureScene(page, baseUrl, scene, outputDir, fps) {
  const frameCount = Math.max(1, Math.round((scene.durationMs / 1000) * fps));
  const sceneDir = path.join(outputDir, "frames", scene.id);
  ensureDir(sceneDir);

  const sceneUrl = buildSceneUrl(baseUrl, scene);
  await page.goto(sceneUrl, { waitUntil: "domcontentloaded" });
  await waitForTrailerApi(page);

  const baseState = buildBaseState(scene);
  const sceneLog = {
    id: scene.id,
    mode: scene.mode,
    url: sceneUrl,
    durationMs: scene.durationMs,
    frameCount,
    checkpoints: [],
  };

  if (!scene.steps?.length) {
    await applyState(page, baseState);
    sceneLog.checkpoints.push({
      atMs: 0,
      state: await page.evaluate(() => window.trailerApi.getState()),
    });
  }

  const steps = Array.isArray(scene.steps) ? [...scene.steps].sort((a, b) => a.atMs - b.atMs) : [];
  let appliedStepIndex = -1;

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const timeMs = Math.round((frameIndex / fps) * 1000);
    while (appliedStepIndex + 1 < steps.length && steps[appliedStepIndex + 1].atMs <= timeMs) {
      appliedStepIndex += 1;
      const step = steps[appliedStepIndex];
      const mergedState = mergeStepState(baseState, scene, step);
      await applyState(page, mergedState);
      await scrollPanel(page, step.panelScroll);
      sceneLog.checkpoints.push({
        atMs: step.atMs,
        state: await page.evaluate(() => window.trailerApi.getState()),
      });
    }

    const progress = frameCount === 1 ? 1 : frameIndex / Math.max(1, frameCount - 1);
    const eased = easingAt(scene.easing, progress);
    const pose = interpolatePose(scene.cameraFrom, scene.cameraTo, eased);
    await setCameraPose(page, pose);
    const framePath = path.join(sceneDir, `frame-${String(frameIndex).padStart(5, "0")}.png`);
    await captureFrame(page, framePath);
  }

  sceneLog.checkpoints.push({
    atMs: scene.durationMs,
    state: await page.evaluate(() => window.trailerApi.getState()),
  });
  const logPath = path.join(outputDir, "states", `${scene.id}.json`);
  ensureDir(path.dirname(logPath));
  fs.writeFileSync(logPath, JSON.stringify(sceneLog, null, 2), "utf-8");
  return sceneDir;
}

async function captureSceneRealtime(browser, baseUrl, scene, outputDir, width, height) {
  const rawVideoDir = path.join(outputDir, "raw-video", scene.id);
  ensureDir(rawVideoDir);
  const context = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: 1,
    recordVideo: {
      dir: rawVideoDir,
      size: { width, height },
    },
  });
  const page = await context.newPage();
  const sceneUrl = buildSceneUrl(baseUrl, scene);
  const sceneLog = {
    id: scene.id,
    mode: scene.mode,
    url: sceneUrl,
    durationMs: scene.durationMs,
    captureMode: "realtime-video",
    checkpoints: [],
  };

  const captureStart = Date.now();
  let recordedVideo = null;
  let rawVideoPath = null;
  try {
    recordedVideo = page.video();
    await page.goto(sceneUrl, { waitUntil: "domcontentloaded" });
    await waitForTrailerApi(page);
    const baseState = buildBaseState(scene);
    await applyState(page, baseState);
    await scrollPanel(page, scene.panelScroll);
    await page.waitForTimeout(120);
    sceneLog.checkpoints.push({
      atMs: 0,
      state: await page.evaluate(() => window.trailerApi.getState()),
    });
    sceneLog.trimStartMs = Date.now() - captureStart;

    await page.evaluate(async (payload) => {
      await window.trailerApi.animateCamera(payload);
    }, {
      from: scene.cameraFrom ?? null,
      to: scene.cameraTo ?? scene.cameraFrom ?? null,
      durationMs: scene.durationMs,
      easing: scene.easing ?? "linear",
    });
    await page.waitForTimeout(120);
    sceneLog.checkpoints.push({
      atMs: scene.durationMs,
      state: await page.evaluate(() => window.trailerApi.getState()),
    });
  } finally {
    await context.close();
  }
  if (recordedVideo) {
    rawVideoPath = await recordedVideo.path();
  }

  const logPath = path.join(outputDir, "states", `${scene.id}.json`);
  ensureDir(path.dirname(logPath));
  fs.writeFileSync(logPath, JSON.stringify(sceneLog, null, 2), "utf-8");
  return {
    rawVideoPath,
    trimStartMs: sceneLog.trimStartMs || 0,
  };
}

async function captureSegmentedScene(browser, baseUrl, scene, outputDir, width, height, ffmpegPath, fps) {
  const segmentClips = [];
  const segmentManifest = [];
  let cumulativeMs = 0;

  for (let index = 0; index < scene.segments.length; index += 1) {
    const segment = scene.segments[index];
    const segmentState = segment.state || {};
    const segmentId = `${scene.id}-${String(index + 1).padStart(2, "0")}`;
    const segmentScene = {
      ...scene,
      id: segmentId,
      durationMs: segment.durationMs,
      region: segmentState.region ?? scene.region ?? null,
      preset: segmentState.preset ?? scene.preset ?? null,
      exaggeration: segmentState.exaggeration ?? scene.exaggeration ?? null,
      iceOpacity: segmentState.iceOpacity ?? scene.iceOpacity ?? null,
      toggles: Object.prototype.hasOwnProperty.call(segmentState, "toggles") ? segmentState.toggles : scene.toggles ?? null,
      overlay: Object.prototype.hasOwnProperty.call(segmentState, "overlay") ? segmentState.overlay : scene.overlay,
      cameraFrom: segmentState.cameraPose ?? scene.cameraFrom ?? null,
      cameraTo: segmentState.cameraPose ?? scene.cameraFrom ?? scene.cameraTo ?? null,
      panelScroll: segment.panelScroll ?? null,
      steps: undefined,
      segments: undefined,
    };
    const { rawVideoPath, trimStartMs } = await captureSceneRealtime(browser, baseUrl, segmentScene, outputDir, width, height);
    const rawDurationSec = probeDurationSeconds(rawVideoPath);
    const trimStartSec = Math.max(0, rawDurationSec - segment.durationMs / 1000 - 0.12);
    const clipPath = path.join(outputDir, "clips", `${segmentId}.mp4`);
    runFfmpeg(ffmpegPath, [
      "-y",
      "-ss",
      trimStartSec.toFixed(3),
      "-i",
      rawVideoPath,
      "-t",
      (segment.durationMs / 1000).toFixed(3),
      "-vf",
      `fps=${fps}`,
      "-c:v",
      "libx264",
      "-preset",
      "medium",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      clipPath,
    ]);
    segmentClips.push(clipPath);
    segmentManifest.push({
      id: segmentId,
      atMs: cumulativeMs,
      durationMs: segment.durationMs,
      clip: clipPath,
      rawVideo: rawVideoPath,
      rawDurationSec,
      trimStartMs,
      trimStartSec,
    });
    cumulativeMs += segment.durationMs;
  }

  const sceneClipPath = path.join(outputDir, "clips", `${scene.id}.mp4`);
  renderMasterClip(ffmpegPath, segmentClips, sceneClipPath);
  fs.writeFileSync(
    path.join(outputDir, "states", `${scene.id}.json`),
    JSON.stringify(
      {
        id: scene.id,
        mode: scene.mode,
        captureMode: "segmented-realtime",
        durationMs: scene.durationMs,
        segments: segmentManifest,
      },
      null,
      2
    ),
    "utf-8"
  );
  return {
    clipPath: sceneClipPath,
    segmentClips,
  };
}

function renderSceneClip(ffmpegPath, frameDir, inputFps, outputFps, outputPath) {
  runFfmpeg(ffmpegPath, [
    "-y",
    "-framerate",
    String(inputFps),
    "-i",
    path.join(frameDir, "frame-%05d.png"),
    "-vf",
    `fps=${outputFps}`,
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    outputPath,
  ]);
}

function renderMasterClip(ffmpegPath, clipPaths, outputPath) {
  const concatFile = path.join(path.dirname(outputPath), "clips.txt");
  const concatBody = clipPaths.map((clipPath) => `file '${clipPath.replace(/'/g, "'\\''")}'`).join("\n");
  fs.writeFileSync(concatFile, `${concatBody}\n`, "utf-8");
  runFfmpeg(ffmpegPath, [
    "-y",
    "-f",
    "concat",
    "-safe",
    "0",
    "-i",
    concatFile,
    "-c",
    "copy",
    outputPath,
  ]);
}

async function main() {
  const args = parseArgs(process.argv);
  const baseUrl = normalizeBaseUrl(args.baseUrl);
  const spec = readJson(args.sceneFile);
  const selectedScenes = args.sceneId
    ? spec.scenes.filter((scene) => scene.id === args.sceneId)
    : spec.scenes;
  if (!selectedScenes.length) {
    throw new Error(args.sceneId ? `Unknown scene id: ${args.sceneId}` : "No scenes were defined.");
  }

  const width = Number(args.width || spec.meta?.width || 1920);
  const height = Number(args.height || spec.meta?.height || 1080);
  const fps = Number(args.fps || spec.meta?.fps || 30);
  const isSingleSceneRun = Boolean(args.sceneId);
  const outputStem = spec.meta?.outputName || "explorer-trailer";
  const outputDir = path.resolve(args.outputDir);
  ensureDir(outputDir);
  ensureDir(path.join(outputDir, "frames"));
  ensureDir(path.join(outputDir, "states"));
  ensureDir(path.join(outputDir, "clips"));

  const ffmpegPath = args.skipFfmpeg ? null : findFfmpeg();
  const browser = await chromium.launch({
    headless: args.headless,
    args: ["--use-gl=angle", "--use-angle=swiftshader"],
  });
  let frameCapturePage = null;

  const clipPaths = [];
  const manifest = {
    spec: args.sceneFile,
    baseUrl,
    width,
    height,
    fps,
    scenes: [],
    ffmpeg: ffmpegPath || null,
  };

  try {
    for (const scene of selectedScenes) {
      if (ffmpegPath && Array.isArray(scene.segments) && scene.segments.length) {
        const { clipPath, segmentClips } = await captureSegmentedScene(
          browser,
          baseUrl,
          scene,
          outputDir,
          width,
          height,
          ffmpegPath,
          fps
        );
        clipPaths.push(clipPath);
        manifest.scenes.push({
          id: scene.id,
          clip: clipPath,
          segmentClips,
          durationMs: scene.durationMs,
          captureMode: "segmented-realtime",
        });
        continue;
      }

      const canUseRealtimeVideo = Boolean(ffmpegPath && !Array.isArray(scene.steps));
      if (canUseRealtimeVideo) {
        const { rawVideoPath, trimStartMs } = await captureSceneRealtime(browser, baseUrl, scene, outputDir, width, height);
        const rawDurationSec = probeDurationSeconds(rawVideoPath);
        const tailPaddingSec = 0.12;
        const trimStartSec = Math.max(0, rawDurationSec - scene.durationMs / 1000 - tailPaddingSec);
        const clipPath = path.join(outputDir, "clips", `${scene.id}.mp4`);
        runFfmpeg(ffmpegPath, [
          "-y",
          "-ss",
          trimStartSec.toFixed(3),
          "-i",
          rawVideoPath,
          "-t",
          (scene.durationMs / 1000).toFixed(3),
          "-vf",
          `fps=${fps}`,
          "-c:v",
          "libx264",
          "-preset",
          "medium",
          "-crf",
          "18",
          "-pix_fmt",
          "yuv420p",
          clipPath,
        ]);
        clipPaths.push(clipPath);
        manifest.scenes.push({
          id: scene.id,
          clip: clipPath,
          rawVideo: rawVideoPath,
          rawDurationSec,
          trimStartMs,
          trimStartSec,
          durationMs: scene.durationMs,
          captureMode: "realtime-video",
        });
        continue;
      }

      if (!frameCapturePage) {
        frameCapturePage = await browser.newPage({
          viewport: { width, height },
          deviceScaleFactor: 1,
        });
      }
      const captureFps = Number(scene.captureFps || fps);
      const frameDir = await captureScene(frameCapturePage, baseUrl, scene, outputDir, captureFps);
      manifest.scenes.push({
        id: scene.id,
        frames: frameDir,
        durationMs: scene.durationMs,
        captureFps,
        captureMode: "frame-sequence",
      });
      if (ffmpegPath) {
        const clipPath = path.join(outputDir, "clips", `${scene.id}.mp4`);
        renderSceneClip(ffmpegPath, frameDir, captureFps, fps, clipPath);
        clipPaths.push(clipPath);
      }
    }
  } finally {
    if (frameCapturePage) {
      await frameCapturePage.close();
    }
    await browser.close();
  }

  if (ffmpegPath && clipPaths.length && !isSingleSceneRun) {
    const masterOutput = path.join(outputDir, `${outputStem}.mp4`);
    renderMasterClip(ffmpegPath, clipPaths, masterOutput);
    manifest.masterClip = masterOutput;
  }

  if (isSingleSceneRun) {
    const sceneOutputBase = path.join(outputDir, "states", args.sceneId);
    writeSrt(selectedScenes, `${sceneOutputBase}.en.srt`);
    writeVoiceoverText(spec, selectedScenes, `${sceneOutputBase}.voiceover.txt`);
    fs.writeFileSync(`${sceneOutputBase}.manifest.json`, JSON.stringify(manifest, null, 2), "utf-8");
    return;
  }

  writeSrt(selectedScenes, path.join(outputDir, `${outputStem}.en.srt`));
  writeVoiceoverText(spec, selectedScenes, path.join(outputDir, `${outputStem}.voiceover.txt`));
  fs.writeFileSync(path.join(outputDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf-8");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
