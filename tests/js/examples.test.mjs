/**
 * docs/example.md quotes these figures as the expected output of the worked example, so they
 * are pinned here: a change to the solver, the decoder or the bundled terrain package that
 * moves them must update the documentation too.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { runIsostaticReboundExample } from "../../examples/isostatic-rebound.mjs";

const near = (actual, expected, tolerance, label) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} is not within ${tolerance} of ${expected}`);

test("the Antarctic worked example reproduces the documented figures", () => {
  const { dataset, stats } = runIsostaticReboundExample();

  assert.equal(dataset, "bedmachine_antarctica_v4_480");
  assert.equal(stats.model, "flexural");
  near(stats.maxUpliftMeters, 1026.8, 0.05, "maximum uplift (m)");
  near(stats.meanGroundedUpliftMeters, 553.9, 0.05, "mean grounded uplift (m)");
  near(stats.emergentAreaKm2, 3191759, 1, "emergent area (km²)");
  near(stats.marineUnderIceAfterAreaKm2, 4109375, 1, "bed still below the datum under the present ice (km²)");
  near(stats.sleMeters, 56.53, 0.005, "sea-level equivalent (m)");
});

test("local Airy isostasy bounds the flexural peak uplift from above", () => {
  const flexural = runIsostaticReboundExample().stats;
  const local = runIsostaticReboundExample({ model: "local" }).stats;

  near(local.maxUpliftMeters, 1374.3, 0.05, "local maximum uplift (m)");
  assert.ok(local.maxUpliftMeters > flexural.maxUpliftMeters);
});

test("the worked example rejects an unknown region, model or datum", () => {
  assert.throws(() => runIsostaticReboundExample({ region: "arctic" }), /Unknown region/);
  assert.throws(() => runIsostaticReboundExample({ model: "viscous" }), /Unknown model/);
  assert.throws(() => runIsostaticReboundExample({ seaLevelMeters: Number.NaN }), /sea-level datum/);
});
