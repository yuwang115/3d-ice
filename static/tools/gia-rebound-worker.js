/**
 * Module worker for the isostatic-rebound layer.
 *
 * Kept separate from antarctica-geometry-worker.js so the solver can
 * be a plain ES module that is also unit-testable under node:test. The message contract
 * mirrors the geometry worker: {id, task, payload} in, {id, kind: "progress"|"result"} out.
 */

import { solveIsostaticRebound } from "./js/gia-rebound.js";

const PROGRESS_STAGE_KEY = "reboundSolvingFlexure";

self.addEventListener("message", (event) => {
  const { id, task, payload } = event.data || {};
  if (task !== "solveIsostaticRebound") {
    self.postMessage({
      id,
      kind: "result",
      ok: false,
      error: { message: `Unknown rebound worker task: ${String(task)}` },
    });
    return;
  }

  try {
    const reportProgress = Boolean(payload && payload.reportProgress);
    const result = solveIsostaticRebound({
      ...payload,
      onProgress: reportProgress
        ? (progress) => {
            self.postMessage({
              id,
              kind: "progress",
              progress,
              stageKey: PROGRESS_STAGE_KEY,
              stage: "Solving isostatic rebound...",
            });
          }
        : null,
    });
    self.postMessage({ id, kind: "result", ok: true, result }, [
      result.uplift.buffer,
      result.emergent.buffer,
    ]);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ id, kind: "result", ok: false, error: { message } });
  }
});
