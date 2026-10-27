"""E2E tests: switching dataset while a web worker is still working on the old dataset.

Two kinds of work run in workers. After the terrain loads, the research page warms its
overlay layers up in the background, building each one in the geometry worker. The
idealised isostatic-rebound models solve in a worker of their own. A dataset switch
terminates both workers, which abandons whatever they were doing. Left to timing, the switch
lands wherever it happens to, so these tests hold a worker's reply to one task. The page
then sees that task as still running, the state a slow device is in, until the switch
terminates the worker. The switch itself goes through the page's own controls.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.e2e

LOAD_TIMEOUT_MS = 90_000

# Swallows a worker's messages about the first task of the given kind, so that task never
# finishes in the page's eyes. Tasks posted to the worker that replaces it after the switch
# pass through untouched. `intercepted` records that the page listens through
# addEventListener, without which the hold would silently do nothing, and `tasks` lists every
# task posted to any worker, in order.
HOLD_FIRST_WORKER_TASK = """
(heldTask) => {
  const held = { worker: null, id: null };
  const listening = new WeakSet();
  window.__heldWorkerTask = { task: heldTask, posted: false, intercepted: false, tasks: [] };
  const postMessage = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function (message, transfer) {
    if (message?.task) window.__heldWorkerTask.tasks.push(message.task);
    if (message?.task === heldTask && !window.__heldWorkerTask.posted) {
      window.__heldWorkerTask.posted = true;
      window.__heldWorkerTask.intercepted = listening.has(this);
      held.worker = this;
      held.id = message.id;
    }
    return postMessage.call(this, message, transfer);
  };
  const addEventListener = Worker.prototype.addEventListener;
  Worker.prototype.addEventListener = function (type, listener, options) {
    if (type !== "message") return addEventListener.call(this, type, listener, options);
    listening.add(this);
    const worker = this;
    return addEventListener.call(
      this,
      type,
      function (event) {
        if (worker === held.worker && event.data?.id === held.id) return undefined;
        return listener.call(this, event);
      },
      options
    );
  };
}
"""

# Each warmup layer: the worker task that builds it, the control that shows it, and its key
# under `meshes` in render_game_to_text().
WARMUP_LAYERS = {
    "velocity": ("buildVelocity", "#showVelocity", "velocity"),
    "basalFriction": ("buildBasalFriction", "#showBasalFriction", "basalFriction"),
    "hydrology": ("buildHydrology", "#showEffectivePressure", "hydrology"),
    "oceanCurrents": ("buildOceanCurrents", "#showOceanCurrents", "oceanCurrents"),
}

# Each switch away from the default Antarctic dataset: the control, the option, and the
# render_game_to_text() condition that holds once the new dataset is in place. Bedmap3 is
# the light stand-in for the HD datasets, which, like it, skip the warmup; HD terrain renders
# too slowly in headless software WebGL for its controls to settle within a test's timeouts.
DATASET_SWITCHES = {
    "bedmap3": ("#resolutionPreset", "bedmap3", "state.dataset === 'bedmap3'"),
    "greenland": ("#regionPreset", "greenland", "state.region === 'greenland'"),
}


def _open_holding(playwright_browser, task: str):
    """A browser context whose pages hold the worker's reply to the first `task`."""
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    context.add_init_script(f"({HOLD_FIRST_WORKER_TASK})({json.dumps(task)});")
    return context


def _collect_errors(page, server: str) -> list[str]:
    """Collect the page's uncaught exceptions and console errors as they happen.

    A request to a third-party host (the analytics tag, which the session's browser cannot
    resolve, or the web fonts) logs a console error of its own when it fails. Those are
    reported at the third-party URL and left out.
    """
    errors: list[str] = []

    def on_console(message) -> None:
        url = message.location.get("url", "")
        third_party = url.startswith(("http://", "https://")) and not url.startswith(f"{server}/")
        if message.type == "error" and not third_party:
            errors.append(message.text)

    page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
    page.on("console", on_console)
    return errors


def _collect_warnings_about(page, topic: str) -> list[str]:
    """Collect the console warnings whose text mentions `topic`, ignoring case."""
    warnings: list[str] = []

    def on_console(message) -> None:
        if message.type == "warning" and topic in message.text.lower():
            warnings.append(message.text)

    page.on("console", on_console)
    return warnings


def _wait_for_state(page, condition: str) -> None:
    page.wait_for_function(
        f"""() => {{
            if (!window.render_game_to_text) return false;
            const state = JSON.parse(window.render_game_to_text());
            return {condition};
        }}""",
        timeout=LOAD_TIMEOUT_MS,
    )


def _wait_for_held_task(page) -> None:
    page.wait_for_function("() => window.__heldWorkerTask?.posted", timeout=LOAD_TIMEOUT_MS)
    assert page.evaluate("window.__heldWorkerTask.intercepted"), "the worker reply was not held"


@pytest.mark.parametrize(
    ("layer", "switch"),
    (
        ("basalFriction", "bedmap3"),
        ("basalFriction", "greenland"),
        ("velocity", "bedmap3"),
        ("hydrology", "bedmap3"),
        ("oceanCurrents", "bedmap3"),
    ),
)
def test_a_dataset_switch_abandons_the_warmup_build_quietly(
    playwright_browser, server, explorer_url, layer, switch
):
    """The abandoned build logs nothing, and the layer still loads on the new dataset."""
    task, control, mesh = WARMUP_LAYERS[layer]
    selector, option, switched = DATASET_SWITCHES[switch]

    context = _open_holding(playwright_browser, task)
    try:
        page = context.new_page()
        errors = _collect_errors(page, server)
        page.goto(explorer_url, wait_until="domcontentloaded", timeout=30_000)

        # The terrain is up and the warmup has handed this layer to the worker.
        _wait_for_held_task(page)
        assert not page.locator(control).is_checked()

        page.select_option(selector, option)
        _wait_for_state(page, f"state.ready && {switched}")

        page.locator(control).check()
        _wait_for_state(page, f"state.ready && {switched} && state.meshes.{mesh}")
        assert not errors, "\n\n".join(errors)
    finally:
        context.close()


def test_a_dataset_switch_abandons_the_rebound_solve_quietly(playwright_browser, server, explorer_url):
    """The abandoned solve logs nothing, and the new dataset's solve still runs in a worker."""
    context = _open_holding(playwright_browser, "solveIsostaticRebound")
    try:
        page = context.new_page()
        errors = _collect_errors(page, server)
        warnings = _collect_warnings_about(page, "rebound")
        page.goto(explorer_url, wait_until="domcontentloaded", timeout=30_000)
        _wait_for_state(page, "state.ready")

        # The published response loads on the main thread; regional flexure solves in the
        # worker, once the published load it interrupts has settled.
        page.locator("#showIsostaticRebound").check()
        page.locator("#reboundModel").select_option("flexural")
        _wait_for_held_task(page)

        page.select_option("#resolutionPreset", "bedmap3")
        _wait_for_state(
            page,
            "state.ready && state.dataset === 'bedmap3' && state.isostaticRebound.solved"
            " && state.isostaticRebound.solvedModel === 'flexural'"
            " && !state.isostaticRebound.pendingResolve",
        )
        assert not warnings, "\n\n".join(warnings)
        # Bedmap3 solved in a fresh worker too, rather than on the main thread from then on.
        tasks = page.evaluate("window.__heldWorkerTask.tasks")
        assert tasks.count("solveIsostaticRebound") == 2, tasks
        assert not errors, "\n\n".join(errors)
    finally:
        context.close()
