/**
 * Every localisation key the explorer actually asks for must resolve in BOTH locales.
 *
 * The explorer's `t()` returns the key path verbatim when a lookup misses, so a string
 * added under "en-US" but not "zh-CN" renders as `explorer.rebound.legendNote` on the
 * Chinese page instead of failing loudly. Nothing else in the suite covers
 * static/js/3d-ice-locale.js, so this walks the call sites instead of the dictionary and
 * checks both directions.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

const LOCALE_MODULE = resolve(repoRoot, "static/js/3d-ice-locale.js");
const EXPLORERS = [
  resolve(repoRoot, "static/tools/3D-interactive-cryosphere-explorer.html"),
  resolve(repoRoot, "static/zh/tools/3D-interactive-cryosphere-explorer.html"),
];
// Both pages load this one runtime module, which is where nearly every t() call lives.
const EXPLORER_RUNTIME = resolve(repoRoot, "static/tools/js/explorer-app.js");
const WORKERS = [
  resolve(repoRoot, "static/tools/antarctica-geometry-worker.js"),
  resolve(repoRoot, "static/tools/gia-rebound-worker.js"),
];
const LOCALES = ["en-US", "zh-CN"];

let localeApi = null;

function loadLocaleApi() {
  // The locale file is a browser IIFE that publishes itself on `window`, so it needs a
  // shim and runs exactly once per process (require() caches it).
  if (localeApi) return localeApi;
  globalThis.window = {
    localStorage: { getItem: () => null, setItem: () => {} },
    document: { documentElement: { dataset: {}, lang: "en" } },
    location: { pathname: "/" },
    addEventListener: () => {},
  };
  createRequire(import.meta.url)(LOCALE_MODULE);
  localeApi = globalThis.window.__3dIceLocale;
  assert.ok(localeApi, "static/js/3d-ice-locale.js should publish window.__3dIceLocale");
  return localeApi;
}

/** Literal keys passed to t("...") or errorLabel("..."); template literals are skipped. */
function collectRuntimeKeys(paths) {
  const keys = new Set();
  const pattern = /\b(?:t|errorLabel)\(\s*"((?:explorer|worker|shared|home)\.[A-Za-z0-9_.]+)"/g;
  for (const path of paths) {
    const source = readFileSync(path, "utf8");
    for (const match of source.matchAll(pattern)) keys.add(match[1]);
  }
  return keys;
}

/** Worker stage keys, which the main thread resolves as worker.progress.<key>. */
function collectWorkerStageKeys(paths) {
  const keys = new Set();
  for (const path of paths) {
    const source = readFileSync(path, "utf8");
    for (const match of source.matchAll(/progress\(\s*[^,()]+,\s*"([A-Za-z0-9_]+)"/g)) {
      keys.add(`worker.progress.${match[1]}`);
    }
    for (const match of source.matchAll(/PROGRESS_STAGE_KEY\s*=\s*"([A-Za-z0-9_]+)"/g)) {
      keys.add(`worker.progress.${match[1]}`);
    }
  }
  return keys;
}

test("every localisation key used at runtime resolves in both locales", () => {
  const api = loadLocaleApi();
  const keys = [...collectRuntimeKeys([...EXPLORERS, EXPLORER_RUNTIME]), ...collectWorkerStageKeys(WORKERS)].sort();
  assert.ok(keys.length > 100, `expected to find many keys, found ${keys.length}`);

  const missing = [];
  for (const locale of LOCALES) {
    for (const key of keys) {
      const value = api.t(locale, key);
      if (typeof value !== "string" || value === key || value.length === 0) {
        missing.push(`${locale} -> ${key}`);
      }
    }
  }
  assert.deepEqual(missing, [], `unresolved localisation keys:\n${missing.join("\n")}`);
});

test("the isostatic-rebound layer is fully localised in both locales", () => {
  const api = loadLocaleApi();
  const required = [
    "explorer.meta.isostaticReboundSection",
    "explorer.meta.reboundAssumptionsText",
    "explorer.meta.reboundMethodText",
    "explorer.meta.reboundEmergent",
    "explorer.meta.reboundSle",
    "explorer.rebound.legendNote",
    "explorer.rebound.progressNote",
    "explorer.rebound.progressNoteComplete",
    "explorer.rebound.modelNoteFlexural",
    "explorer.rebound.modelNoteLocal",
    "explorer.rebound.seaLevelNoteZero",
    "explorer.rebound.seaLevelNoteRaised",
    "explorer.status.loadingIsostaticRebound",
    "explorer.status.isostaticReboundUnavailable",
    "explorer.loading.solvingIsostaticRebound",
    "explorer.loading.isostaticReboundReady",
    "worker.progress.reboundSolvingFlexure",
  ];
  for (const locale of LOCALES) {
    for (const key of required) {
      const value = api.t(locale, key);
      assert.notEqual(value, key, `${key} is missing from ${locale}`);
      assert.ok(value.length > 0, `${key} is empty in ${locale}`);
    }
  }

  // The Chinese strings must actually be translated, not copied from English.
  for (const key of ["explorer.rebound.legendNote", "explorer.meta.reboundAssumptionsText"]) {
    assert.notEqual(api.t("zh-CN", key), api.t("en-US", key), `${key} is untranslated`);
    assert.match(api.t("zh-CN", key), /[一-鿿]/, `${key} has no Chinese text`);
  }
});

test("interpolated rebound strings substitute every placeholder", () => {
  const api = loadLocaleApi();
  const cases = [
    ["explorer.rebound.progressNote", { percent: 60, elapsed: "about 2.7 kyr", tau: 3000 }],
    ["explorer.rebound.progressNoteComplete", { tau: 3000 }],
    ["explorer.rebound.modelNoteFlexural", { lengthScale: 133 }],
    ["explorer.rebound.seaLevelNoteZero", { sle: "56.5" }],
    ["explorer.rebound.seaLevelNoteRaised", { datum: 57 }],
    ["explorer.rebound.years", { value: 1200 }],
    ["explorer.rebound.kiloyears", { value: "2.7" }],
  ];
  for (const locale of LOCALES) {
    for (const [key, vars] of cases) {
      const rendered = api.t(locale, key, vars);
      assert.doesNotMatch(rendered, /\{[a-z]+\}/i, `${key} left a placeholder unfilled in ${locale}`);
      for (const value of Object.values(vars)) {
        assert.ok(
          rendered.includes(String(value)),
          `${key} dropped the value ${value} in ${locale}: ${rendered}`
        );
      }
    }
  }
});
