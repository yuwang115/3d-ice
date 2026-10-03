/**
 * The public edition's guide: the guided tour, the layer explainers, the rebound legend
 * note and the rebound demonstration. explorer-app.js loads this module for editions with
 * a guide and hands it an API (createGuideApi) through which it drives the explorer as a
 * user would, by setting the page's own controls and dispatching their events.
 *
 * Page hooks: #tourStartButton and #viewerTourButton start the tour; #tourCard is where
 * it plays, in the side panel, and it moves over the viewer while the panel is a phone's
 * bottom drawer or the viewer is fullscreen; every button[data-info] toggles the
 * [data-info-card] of the same name; and [data-rebound-legend] is shown while the rebound
 * layer is on. A URL parameter tour=1 (or tour=<stop id>) opens the tour on load. The first
 * time a visitor switches the rebound layer on, the ice melts away over a few seconds.
 * While a stop shows the projection, the card reads out its year and sea-level change, and
 * the last stop ends the tour in the region the visitor picks.
 */

import { TOUR_CHAPTERS, getExploreContent } from "./explore-content.js";
import {
  animationValue,
  clampChapterIndex,
  planToggleChanges,
  resolveChapterControls,
  viewShiftFor,
} from "./explore-tour.js";

const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
// Inputs that take typed text, where arrow keys and Escape belong to the input.
const TEXT_INPUT_TYPES = new Set(["", "text", "search", "email", "number", "password", "tel", "url"]);
// Values of the tour URL parameter that leave the tour closed.
const TOUR_PARAM_OFF = new Set(["0", "false", "no", "off"]);

function interpolate(template, vars) {
  return String(template).replace(/\{(\w+)\}/g, (_match, token) => (token in vars ? String(vars[token]) : ""));
}

function createElement(tag, { className = "", text = "", attrs = {} } = {}) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, value);
  return element;
}

function renderSources(sources, label) {
  if (!sources.length) return null;
  const wrapper = createElement("div", { className: "guide-sources" });
  wrapper.append(createElement("span", { className: "guide-sources__label", text: label }));
  const list = createElement("ul", { className: "guide-sources__list" });
  for (const source of sources) {
    const item = createElement("li");
    if (source.url) {
      item.append(createElement("a", { text: source.text, attrs: { href: source.url, target: "_blank", rel: "noopener" } }));
    } else {
      item.textContent = source.text;
    }
    list.append(item);
  }
  wrapper.append(list);
  return wrapper;
}

function isTypingTarget(element) {
  if (!element) return false;
  if (element.isContentEditable) return true;
  const tag = element.tagName || "";
  if (tag === "SELECT" || tag === "TEXTAREA") return true;
  return tag === "INPUT" && TEXT_INPUT_TYPES.has(String(element.getAttribute("type") || "").toLowerCase());
}

function isShown(element) {
  return Boolean(element && !element.disabled && element.getClientRects().length > 0);
}

function setToggle(api, id, checked) {
  const control = api.getControl(id);
  if (!control || control.disabled || control.checked === checked) return;
  control.checked = checked;
  control.dispatchEvent(new Event("change", { bubbles: true }));
}

function setSlider(api, id, value, { commit = true } = {}) {
  const control = api.getControl(id);
  if (!control) return;
  control.value = String(value);
  control.dispatchEvent(new Event("input", { bubbles: true }));
  if (commit) control.dispatchEvent(new Event("change", { bubbles: true }));
}

/** Choose an option of a menu, if the menu offers it. */
function setMenu(api, id, value) {
  const menu = api.getControl(id);
  const option = String(value);
  if (!menu || menu.disabled || menu.value === option) return;
  if (!Array.from(menu.options).some((candidate) => candidate.value === option)) return;
  menu.value = option;
  menu.dispatchEvent(new Event("change", { bubbles: true }));
}

// The animation driving each slider, by control id: starting one takes the slider over.
const sliderAnimations = new Map();

/**
 * Move a slider from `from` to `to` over `durationMs`, eased as `easing` says (see
 * animationValue), sending input events as a drag would and a change event at the end. It
 * stops where it is when `isCancelled()` turns true, when the user takes the slider (by
 * pointer, keyboard or assistive technology), or when another animation takes the slider
 * over. A stopped run still commits its value, so the runtime finishes the frame it was
 * drawing. Under reduced motion it jumps straight to `to`. Resolves when it ends or stops.
 */
function playSlider(api, animation, isCancelled = () => false) {
  const { control, to, durationMs } = animation;
  const slider = api.getControl(control);
  if (!slider) return Promise.resolve();
  const token = {};
  sliderAnimations.set(control, token);
  if (reducedMotionQuery.matches) {
    sliderAnimations.delete(control);
    setSlider(api, control, to);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    let startMs = null;
    let taken = false;
    const take = (event) => {
      if (event.type === "pointerdown" || event.isTrusted) taken = true;
    };
    slider.addEventListener("pointerdown", take);
    slider.addEventListener("input", take);
    const finish = ({ commitTo = null, commitCurrent = false } = {}) => {
      slider.removeEventListener("pointerdown", take);
      slider.removeEventListener("input", take);
      if (sliderAnimations.get(control) === token) sliderAnimations.delete(control);
      if (commitTo !== null) setSlider(api, control, commitTo);
      else if (commitCurrent) slider.dispatchEvent(new Event("change", { bubbles: true }));
      resolve();
    };
    const tick = (nowMs) => {
      if (sliderAnimations.get(control) !== token || taken) {
        finish();
        return;
      }
      if (isCancelled()) {
        finish({ commitCurrent: true });
        return;
      }
      startMs ??= nowMs;
      const progress = Math.min(1, (nowMs - startMs) / durationMs);
      if (progress >= 1) {
        finish({ commitTo: to });
        return;
      }
      setSlider(api, control, animationValue(animation, progress), { commit: false });
      window.requestAnimationFrame(tick);
    };
    window.requestAnimationFrame(tick);
  });
}

// ------------------------------------------------------------------ layer explainers

function mountInfoCards(content) {
  for (const button of document.querySelectorAll("button[data-info]")) {
    const id = button.dataset.info;
    const info = content.info[id];
    const card = document.querySelector(`[data-info-card="${id}"]`);
    if (!info || !card) continue;
    card.id = `info-card-${id}`;
    card.replaceChildren(
      createElement("p", { className: "info-card__title", text: info.title }),
      ...info.body.map((paragraph) => createElement("p", { text: paragraph }))
    );
    const sources = renderSources(info.sources, content.ui.sources);
    if (sources) card.append(sources);
    const close = createElement("button", { className: "info-card__close", text: content.ui.infoClose, attrs: { type: "button" } });
    card.append(close);

    button.setAttribute("aria-controls", card.id);
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-label", interpolate(content.ui.infoButton, { name: info.title }));
    button.title = info.title;
    const setOpen = (open) => {
      card.hidden = !open;
      button.setAttribute("aria-expanded", String(open));
    };
    button.addEventListener("click", () => setOpen(card.hidden));
    close.addEventListener("click", () => {
      setOpen(false);
      button.focus();
    });
  }
}

function mountReboundLegend(api, content) {
  const note = document.querySelector("[data-rebound-legend]");
  const toggle = api.getControl("showIsostaticRebound");
  if (!note || !toggle) return;
  note.textContent = content.ui.reboundLegend;
  const sync = () => {
    note.hidden = !toggle.checked;
  };
  toggle.addEventListener("change", sync);
  // A failed load unchecks and disables the toggle without a change event.
  new MutationObserver(sync).observe(toggle, { attributes: true, attributeFilter: ["disabled"] });
  sync();
}

// ------------------------------------------------------------------ rebound demonstration

const REBOUND_DEMO = Object.freeze({ control: "reboundProgress", from: 0, to: 100, durationMs: 3000 });

/**
 * The first time a visitor switches the rebound layer on themselves, the ice melts away
 * over three seconds instead of vanishing at once, so the process itself can be seen.
 * Later switches, the tour's own switching and reduced motion leave the slider where it is.
 */
function mountReboundDemo(api) {
  const toggle = api.getControl("showIsostaticRebound");
  if (!toggle || !api.getControl(REBOUND_DEMO.control)) return;
  let played = false;
  // Captured on the document, so the slider is back at 0% before the runtime's own
  // listener redraws the bed; a rebound field loaded earlier would otherwise flash at 100%.
  document.addEventListener(
    "change",
    (event) => {
      if (event.target !== toggle || !event.isTrusted || !toggle.checked || played) return;
      played = true;
      if (reducedMotionQuery.matches) return;
      const isCancelled = () => !toggle.checked;
      setSlider(api, REBOUND_DEMO.control, REBOUND_DEMO.from);
      // Wait from the next task: by then the runtime's listener has started loading the
      // rebound field, so the ice cannot melt away before the land is there to rise.
      window.setTimeout(() => {
        api
          .whenIdle(isCancelled)
          .then(() => (isCancelled() ? undefined : playSlider(api, REBOUND_DEMO, isCancelled)))
          .catch((error) => console.warn("The rebound demonstration could not play:", error));
      }, 0);
    },
    true
  );
}

// ------------------------------------------------------------------ guided tour

/**
 * The projection's year and sea-level change, for the tour card. render(view) shows them
 * for a stop whose view shows the projection, following it every frame as the years run
 * or it finishes loading; render(null) hides them.
 */
function createProjectionReadout(api, ui) {
  const element = createElement("p", { className: "tour-card__readout", attrs: { hidden: "" } });
  const year = createElement("span", { className: "tour-card__readout-year" });
  const seaLevel = createElement("span", { className: "tour-card__readout-sea-level" });
  element.append(year, seaLevel);
  const seaLevelFormat = new Intl.NumberFormat(api.locale, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    signDisplay: "exceptZero",
  });
  // Rewritten only when it changes, as the readout is redrawn every frame.
  const setText = (node, text) => {
    if (node.textContent !== text) node.textContent = text;
  };
  let frame = 0;

  function render(view) {
    if (frame) window.cancelAnimationFrame(frame);
    frame = 0;
    const shown = Boolean(view?.controls?.showIceProjection);
    // A scenario other than the stop's would put numbers on the card that its text does not give.
    const asked = shown && Object.entries(view.menus || {}).every(([id, value]) => api.getControl(id)?.value === String(value));
    const reading = asked ? api.getProjectionReadout() : null;
    element.hidden = !reading;
    if (reading) {
      setText(year, interpolate(ui.readoutYear, { year: reading.year }));
      const value = Number.isFinite(reading.seaLevelMeters)
        ? seaLevelFormat.format(reading.seaLevelMeters).replace("-", "\u2212")
        : null;
      setText(seaLevel, value === null ? "" : interpolate(ui.readoutSeaLevel, { value }));
    }
    if (shown) frame = window.requestAnimationFrame(() => render(view));
  }

  return { element, render };
}

/** The last stop's buttons for the regions it offers to start exploring in, if it offers any. */
function createStartButtons(content, onChoose) {
  const last = TOUR_CHAPTERS[TOUR_CHAPTERS.length - 1];
  const labels = content.chapters[last.id].startRegions || {};
  return (last.startRegions || []).map((region) => {
    const label = labels[region] || region;
    const name = interpolate(content.ui.startIn, { region: label });
    const button = createElement("button", {
      className: "tour-card__nav tour-card__nav--start",
      text: label,
      attrs: { type: "button", hidden: "", "aria-label": name, title: name, "data-region": region },
    });
    button.addEventListener("click", () => onChoose(region));
    return button;
  });
}

function createTour(api, content) {
  const card = document.getElementById("tourCard");
  const panelLauncher = document.getElementById("tourStartButton");
  const toolbarLauncher = document.getElementById("viewerTourButton");
  const launchers = [panelLauncher, toolbarLauncher].filter(Boolean);
  if (!card) return null;

  const ui = content.ui;
  const heading = createElement("h2", { className: "tour-card__title", attrs: { tabindex: "-1" } });
  const counter = createElement("p", { className: "tour-card__counter" });
  const body = createElement("div", { className: "tour-card__body" });
  // One short announcement per stop; the full text is there to read, not to be re-read.
  const announcer = createElement("p", { className: "visually-hidden", attrs: { "aria-live": "polite" } });
  const status = createElement("p", { className: "tour-card__status", attrs: { role: "status" } });
  const collapseButton = createElement("button", { className: "tour-card__icon", attrs: { type: "button" } });
  const closeButton = createElement("button", {
    className: "tour-card__icon",
    text: "×",
    attrs: { type: "button", "aria-label": ui.close, title: ui.close },
  });
  const backButton = createElement("button", { className: "tour-card__nav", text: ui.back, attrs: { type: "button" } });
  const replayButton = createElement("button", {
    className: "tour-card__nav tour-card__nav--quiet",
    text: ui.replay,
    attrs: { type: "button", hidden: "" },
  });
  const nextButton = createElement("button", { className: "tour-card__nav tour-card__nav--primary", attrs: { type: "button" } });
  // The last stop can end the tour with a choice of region to explore, in place of Finish.
  const startButtons = createStartButtons(content, (region) => startExploring(region));
  const readout = createProjectionReadout(api, ui);
  const stops = createElement("ol", { className: "tour-card__stops", attrs: { "aria-label": ui.stopsLabel } });
  const stopButtons = TOUR_CHAPTERS.map((chapter, index) => {
    const title = content.chapters[chapter.id].title;
    const button = createElement("button", {
      className: "tour-card__stop",
      attrs: { type: "button", "aria-label": interpolate(ui.goToStop, { number: index + 1, title }), title },
    });
    button.addEventListener("click", () => goTo(index));
    const item = createElement("li");
    item.append(button);
    stops.append(item);
    return button;
  });

  const header = createElement("div", { className: "tour-card__header" });
  const headerButtons = createElement("div", { className: "tour-card__header-buttons" });
  headerButtons.append(collapseButton, closeButton);
  header.append(counter, headerButtons);
  const text = createElement("div", { className: "tour-card__live", attrs: { id: `${card.id}-text` } });
  text.append(heading, body);
  collapseButton.setAttribute("aria-controls", text.id);
  const actions = createElement("div", { className: "tour-card__actions" });
  actions.append(backButton, replayButton, nextButton, ...startButtons);
  const footer = createElement("div", { className: "tour-card__footer" });
  footer.append(readout.element, actions, stops);
  card.replaceChildren(header, text, announcer, status, footer);
  card.setAttribute("aria-label", ui.tourLabel);

  let index = -1;
  let open = false;
  let collapsed = false;
  let applyGeneration = 0;
  let opener = null;

  // The card reads beside the view, in the panel. A phone's panel is a bottom drawer that
  // would cover the view, and a fullscreen viewer hides the panel, so in those cases the
  // card floats over the viewer instead.
  const panelHost = card.parentElement;
  const panelNextSibling = card.nextSibling;
  const viewerHost = document.getElementById("viewerShell");
  function placeCard() {
    const fullscreen = (document.fullscreenElement || document.webkitFullscreenElement) === viewerHost;
    const floating = Boolean(viewerHost) && (document.body.classList.contains("mobile-drawer") || fullscreen);
    const host = floating ? viewerHost : panelHost;
    if (card.parentElement !== host) {
      if (floating) host.append(card);
      else host.insertBefore(card, panelNextSibling);
    }
    card.classList.toggle("tour-card--floating", floating);
  }

  // The margins of the viewer the floating card and the toolbar cover, for framing a view.
  function coveredMargins() {
    if (!card.classList.contains("tour-card--floating") || !viewerHost) return {};
    const shell = viewerHost.getBoundingClientRect();
    const toolbar = viewerHost.querySelector(".viewer-toolbar")?.getBoundingClientRect();
    return {
      top: toolbar ? Math.max(0, toolbar.bottom - shell.top) : 0,
      bottom: Math.max(0, shell.bottom - card.getBoundingClientRect().top),
    };
  }

  // The view shift that keeps the view's target clear of the floating card.
  function currentViewShift() {
    if (!viewerHost) return [0, 0];
    const { width, height } = viewerHost.getBoundingClientRect();
    return viewShiftFor({ viewport: { width, height }, uncovered: coveredMargins() });
  }

  // Re-centre the view when the card changes the space it leaves: collapsed, expanded,
  // moved or closed. A stop still flying in will settle with its own framing instead.
  function refitView() {
    if (api.isFlying()) return;
    const pose = api.getCameraPose();
    if (!pose) return;
    const shift = open ? currentViewShift() : [0, 0];
    if (Math.abs(shift[0] - pose.shift[0]) < 1e-3 && Math.abs(shift[1] - pose.shift[1]) < 1e-3) return;
    api.flyTo({ ...pose, shift }, { durationMs: 450 });
  }

  function revealCard() {
    if (!card.classList.contains("tour-card--floating")) card.scrollIntoView({ block: "nearest" });
  }

  function setCollapsed(nextCollapsed) {
    collapsed = nextCollapsed;
    card.classList.toggle("is-collapsed", collapsed);
    const label = collapsed ? ui.expand : ui.collapse;
    collapseButton.textContent = collapsed ? "+" : "−";
    collapseButton.setAttribute("aria-label", label);
    collapseButton.title = label;
    if (open) refitView();
  }

  function syncLaunchers() {
    for (const launcher of launchers) {
      const idleLabel = launcher === toolbarLauncher ? ui.toolbarButton : index > 0 ? ui.resume : ui.start;
      launcher.textContent = open ? ui.close : idleLabel;
    }
  }

  function renderReadout() {
    readout.render(open && index >= 0 ? TOUR_CHAPTERS[index].view : null);
  }

  function renderStop() {
    const chapter = TOUR_CHAPTERS[index];
    const copy = content.chapters[chapter.id];
    counter.textContent = interpolate(ui.counter, { current: index + 1, total: TOUR_CHAPTERS.length });
    heading.textContent = copy.title;
    body.replaceChildren(...copy.body.map((paragraph) => createElement("p", { text: paragraph })));
    const sources = renderSources(copy.sources, ui.sources);
    if (sources) body.append(sources);
    announcer.textContent = `${counter.textContent}: ${copy.title}`;
    const last = index === TOUR_CHAPTERS.length - 1;
    const choosing = last && startButtons.length > 0;
    // Keep focus in the card when the button holding it is about to disappear. Next takes
    // it, shown first so that it can; on the last stop the heading does, as a choice of
    // region ends the tour, which an Enter key held down on Next must not do.
    const leaving = [
      ...(index === 0 ? [backButton] : []),
      ...(chapter.view.animate ? [] : [replayButton]),
      ...(choosing ? [nextButton] : startButtons),
    ];
    if (!choosing) nextButton.hidden = false;
    if (leaving.includes(document.activeElement)) (choosing ? heading : nextButton).focus({ preventScroll: true });
    backButton.disabled = index === 0;
    replayButton.hidden = !chapter.view.animate;
    nextButton.hidden = choosing;
    nextButton.textContent = last ? ui.finish : ui.next;
    for (const button of startButtons) button.hidden = !choosing;
    stopButtons.forEach((button, stopIndex) => {
      if (stopIndex === index) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    card.dataset.stop = chapter.id;
    renderReadout();
  }

  function renderStatus(message = "") {
    status.textContent = message;
    card.classList.toggle("has-status", Boolean(message));
  }

  function currentToggles(desired) {
    return Object.fromEntries(Object.keys(desired).map((id) => [id, Boolean(api.getControl(id)?.checked)]));
  }

  async function applyView(view, desired, isCancelled) {
    // Layers go off before a region switch, so the new region does not load them only
    // to have them hidden again; the rest go on once its terrain is in place.
    const offs = planToggleChanges(currentToggles(desired), desired).filter((change) => !change.checked);
    for (const change of offs) setToggle(api, change.id, false);
    // The projection opens on today's ice, so the years a stop ran through go back with it.
    const projectionYear = api.getControl("projectionYear");
    if (projectionYear && offs.some((change) => change.id === "showIceProjection")) {
      setSlider(api, "projectionYear", projectionYear.min);
    }
    if (view.region) await api.setRegion(view.region, isCancelled);
    // Its toggle stays disabled until the projection knows which scenarios it has.
    if (desired.showIceProjection) await api.whenProjectionScenariosKnown(isCancelled);
    if (isCancelled()) return false;
    for (const [id, value] of Object.entries(view.sliders || {})) setSlider(api, id, value);
    // Before the toggles, so that a layer switched on loads the option the stop asks for.
    for (const [id, value] of Object.entries(view.menus || {})) setMenu(api, id, value);
    for (const { id, checked } of planToggleChanges(currentToggles(desired), desired)) setToggle(api, id, checked);
    if (view.animate) {
      setSlider(api, view.animate.control, view.animate.from);
    }
    if (desired.showIsostaticRebound && view.animate?.control !== "reboundProgress") {
      setSlider(api, "reboundProgress", 100);
    }
    // On a phone the card's height, known now that its text is in, decides the room left.
    const pose =
      view.camera === "default"
        ? { ...api.getDefaultCameraPose(), shift: currentViewShift() }
        : api.getLookAtPose(view.camera, { uncovered: coveredMargins() });
    await api.flyTo(pose);
    return !isCancelled();
  }

  // Sweeping the projection's years would have its readout in the panel speak every year,
  // so it stays quiet meanwhile, as it does when the projection's own Play runs. A replay
  // can start its sweep before the one it replaces stops, on its next frame, so the sweeps
  // are counted.
  let projectionSweeps = 0;
  async function playAnimation(animation, isCancelled) {
    const quiet = animation.control === "projectionYear";
    if (quiet && projectionSweeps++ === 0) api.quietProjectionReadout(true);
    try {
      await playSlider(api, animation, isCancelled);
    } finally {
      if (quiet && --projectionSweeps === 0) api.quietProjectionReadout(false);
    }
  }

  async function goTo(nextIndex) {
    const target = clampChapterIndex(nextIndex, TOUR_CHAPTERS.length);
    if (target < 0) return;
    const generation = ++applyGeneration;
    const isCancelled = () => generation !== applyGeneration || !open;
    index = target;
    api.cancelFlight();
    renderStop();
    renderStatus(ui.loading);
    revealCard();
    const view = TOUR_CHAPTERS[index].view;
    try {
      const desired = resolveChapterControls(view.controls);
      const arrived = await applyView(view, desired, isCancelled);
      if (!arrived || isCancelled()) return;
      renderStatus();
      await api.whenIdle(isCancelled);
      if (isCancelled()) return;
      // A layer whose data failed to load is switched back off by the runtime, and a menu
      // without the option asked for keeps its own.
      const missing = [
        ...Object.entries(desired).filter(([id, on]) => on && api.getControl(id) && !api.getControl(id).checked),
        ...Object.entries(view.menus || {}).filter(([id, value]) => api.getControl(id) && api.getControl(id).value !== String(value)),
      ];
      if (missing.length) {
        renderStatus(ui.loadFailed);
        return;
      }
      if (view.animate) await playAnimation(view.animate, isCancelled);
    } catch (error) {
      if (isCancelled()) return;
      console.warn("Tour stop could not be shown:", error);
      renderStatus(ui.loadFailed);
    }
  }

  function setOpen(nextOpen) {
    open = nextOpen;
    placeCard();
    card.hidden = !open;
    document.body.classList.toggle("tour-open", open);
    if (!open) {
      // Bumping the generation also stops a slider animation the stop had started.
      applyGeneration += 1;
      api.cancelFlight();
      refitView();
    }
    renderReadout();
    syncLaunchers();
  }

  function start(startIndex = 0, launcher = null) {
    opener = launcher;
    setOpen(true);
    api.closeMobilePanel();
    goTo(startIndex);
    heading.focus({ preventScroll: true });
  }

  function close() {
    if (!open) return;
    setOpen(false);
    const returnTo = [opener, panelLauncher, toolbarLauncher].find(isShown);
    returnTo?.focus({ preventScroll: true });
  }

  // Finishing starts the next tour from the top rather than resuming at the end.
  function finish() {
    close();
    index = -1;
    syncLaunchers();
  }

  // Ends the tour at the opening view of the region the visitor chose to explore.
  function startExploring(regionKey) {
    finish();
    if (regionKey === api.getRegion()) {
      api.flyTo(api.getDefaultCameraPose());
      return;
    }
    // Loading the region puts the camera at its opening view.
    api.setRegion(regionKey).catch((error) => console.warn("The region could not be opened:", error));
  }

  collapseButton.addEventListener("click", () => setCollapsed(!collapsed));
  closeButton.addEventListener("click", close);
  // The runtime switches the drawer layout on resize; follow it on the next frame.
  const replaceCard = () => {
    placeCard();
    if (open) refitView();
  };
  window.addEventListener("resize", () => window.requestAnimationFrame(replaceCard));
  document.addEventListener("fullscreenchange", replaceCard);
  document.addEventListener("webkitfullscreenchange", replaceCard);
  backButton.addEventListener("click", () => goTo(index - 1));
  nextButton.addEventListener("click", () => {
    if (index < TOUR_CHAPTERS.length - 1) goTo(index + 1);
    else finish();
  });
  replayButton.addEventListener("click", () => goTo(index));
  for (const launcher of launchers) {
    launcher.disabled = false;
    launcher.setAttribute("aria-controls", card.id);
    launcher.addEventListener("click", () => (open ? close() : start(Math.max(0, index), launcher)));
  }
  window.addEventListener("keydown", (event) => {
    if (!open || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (isTypingTarget(document.activeElement)) return;
    const arrow = event.key === "ArrowRight" || event.key === "ArrowLeft";
    // A held arrow would otherwise race through the stops, loading each on the way.
    if (arrow && event.repeat) return;
    if (event.key === "Escape") close();
    else if (event.key === "ArrowRight" && index < TOUR_CHAPTERS.length - 1) goTo(index + 1);
    else if (event.key === "ArrowLeft" && index > 0) goTo(index - 1);
    else return;
    event.preventDefault();
  });

  setCollapsed(false);
  setOpen(false);
  return { start, close, goTo };
}

function startIndexFromUrl() {
  const requested = new URLSearchParams(window.location.search).get("tour");
  if (!requested || TOUR_PARAM_OFF.has(requested.toLowerCase())) return -1;
  const byId = TOUR_CHAPTERS.findIndex((chapter) => chapter.id === requested);
  return byId >= 0 ? byId : 0;
}

export function mountExploreGuide(api) {
  const content = getExploreContent(api.locale);
  mountInfoCards(content);
  mountReboundLegend(api, content);
  mountReboundDemo(api);
  const tour = createTour(api, content);
  const startIndex = startIndexFromUrl();
  if (tour && startIndex >= 0) tour.start(startIndex);
  document.documentElement.dataset.guide = "ready";
  return tour;
}
