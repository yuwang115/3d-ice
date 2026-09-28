/**
 * docs/example.md quotes these figures as the expected output of the worked example, so they
 * are pinned here: a change to the rebound module, the decoder or a bundled package that
 * moves them must update the documentation too.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { RESPONSE_PACKAGES, runIsostaticReboundExample } from "../../examples/isostatic-rebound.mjs";

const near = (actual, expected, tolerance, label) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} is not within ${tolerance} of ${expected}`);

test("the Antarctic worked example reproduces the documented published-response figures", () => {
  const { dataset, response, stats } = runIsostaticReboundExample();

  assert.equal(dataset, "bedmachine_antarctica_v4_480");
  assert.equal(response, "antarctica_isostatic_response_480");
  assert.equal(stats.model, "paxman2022");
  near(stats.maxUpliftMeters, 1028.6, 0.05, "maximum solid-surface uplift (m)");
  near(stats.maxTopographyChangeMeters, 940.6, 0.05, "largest rise above the ice-free sea surface (m)");
  near(stats.meanGroundedUpliftMeters, 587.2, 0.05, "mean grounded uplift (m)");
  near(stats.emergentAreaKm2, 2941863, 1, "emergent area (km²)");
  near(stats.marineUnderIceAfterAreaKm2, 4351870, 1, "bed still below the ice-free sea under the present ice (km²)");
  near(stats.sleMeters, 56.53, 0.005, "sea-level equivalent (m)");
  near(stats.eustaticSeaLevelRiseMeters, 65.3, 1e-9, "eustatic rise carried by the grids (m)");
  near(stats.components.postLgm.max, 68.3, 0.05, "post-LGM term, as in Table 1 of the paper (m)");
});

test("the Greenland worked example reproduces the documented published-response figures", () => {
  const { stats } = runIsostaticReboundExample({ region: "greenland" });

  near(stats.maxUpliftMeters, 829.0, 0.05, "maximum solid-surface uplift (m)");
  near(stats.maxTopographyChangeMeters, 784.4, 0.05, "largest rise above the ice-free sea surface (m)");
  near(stats.emergentAreaKm2, 394199, 1, "emergent area (km²)");
});

test("the idealised regional-flexure example still reproduces its figures", () => {
  const { response, stats } = runIsostaticReboundExample({ model: "flexural" });

  assert.equal(response, null);
  assert.equal(stats.model, "flexural");
  near(stats.maxUpliftMeters, 1026.8, 0.05, "maximum uplift (m)");
  near(stats.meanGroundedUpliftMeters, 553.9, 0.05, "mean grounded uplift (m)");
  near(stats.emergentAreaKm2, 3191759, 1, "emergent area (km²)");
  near(stats.marineUnderIceAfterAreaKm2, 4109375, 1, "bed still below the datum under the present ice (km²)");
  near(stats.sleMeters, 56.53, 0.005, "sea-level equivalent (m)");
});

test("local Airy isostasy bounds the flexural peak uplift from above", () => {
  const flexural = runIsostaticReboundExample({ model: "flexural" }).stats;
  const local = runIsostaticReboundExample({ model: "local" }).stats;

  near(local.maxUpliftMeters, 1374.3, 0.05, "local maximum uplift (m)");
  assert.ok(local.maxUpliftMeters > flexural.maxUpliftMeters);
});

test("every response package the example pairs with a terrain package loads on its grid", () => {
  for (const [dataset, response] of Object.entries(RESPONSE_PACKAGES)) {
    const region = dataset.includes("greenland") ? "greenland" : "antarctica";
    const result = runIsostaticReboundExample({ region, dataset });
    assert.equal(result.response, response, dataset);
    assert.ok(result.stats.maxUpliftMeters > 700, `${dataset}: ${result.stats.maxUpliftMeters}`);
  }
});

test("the worked example rejects an unknown region, model or datum", () => {
  assert.throws(() => runIsostaticReboundExample({ region: "arctic" }), /Unknown region/);
  assert.throws(() => runIsostaticReboundExample({ model: "viscous" }), /Unknown model/);
  assert.throws(
    () => runIsostaticReboundExample({ model: "flexural", seaLevelMeters: Number.NaN }),
    /sea-level datum/
  );
  assert.throws(() => runIsostaticReboundExample({ seaLevelMeters: 57 }), /fixes its own sea surface/);
});
