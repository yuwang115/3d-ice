/*
 * Behaviour for the 3D ICE home pages (/ and /zh/): the language switcher, the theme toggle
 * and the images that follow it, the data tower (3d-ice-data-tower.js, loaded on demand), the
 * scroll hand-off from the hero showcase and the feedback form's messages. Loaded with
 * `defer`, after 3d-ice-locale.js.
 *
 * The theme itself is applied by a small inline script in <head> before the first paint.
 * It lives on <html class="dark"> and in localStorage["wc-color-theme"], which the
 * explorer pages read too, so a choice made here carries over to them.
 */
(() => {
  "use strict";

  const THEME_KEY = "wc-color-theme";
  const SHOWCASE_SCROLL_MESSAGE = "3d-ice:showcase-scroll";
  // Next to this script on both hosts: 3d-ice.com, and yuwang.blog, which serves the bundle.
  const TOWER_MODULE = "/js/3d-ice-data-tower.js";
  const root = document.documentElement;

  const isDark = () => root.classList.contains("dark");

  // ------------------------------------------------------------------ language switcher

  function initLocale() {
    const locale = window.__3dIceLocale;
    if (!locale) return;
    locale.initPage({ locale: root.dataset.locale || "en-US", switchers: ["#homeLocaleSwitcher"] });
  }

  // ------------------------------------------------------------------ theme

  function syncThemedImages() {
    const dark = isDark();
    for (const image of document.querySelectorAll("img[data-light-src][data-dark-src]")) {
      const source = dark ? image.dataset.darkSrc : image.dataset.lightSrc;
      if (image.getAttribute("src") !== source) image.setAttribute("src", source);
    }
  }

  function storeTheme(dark) {
    try {
      localStorage.setItem(THEME_KEY, dark ? "dark" : "light");
    } catch {
      // Storage can be unavailable (private browsing); the theme still applies to this page.
    }
  }

  function initTheme() {
    const toggle = document.getElementById("themeToggle");
    // Every change of the class, from the toggle or from another tab, lands here.
    const syncTheme = () => {
      root.style.colorScheme = isDark() ? "dark" : "light";
      toggle?.setAttribute("aria-pressed", String(isDark()));
      syncThemedImages();
    };
    toggle?.addEventListener("click", () => {
      root.classList.toggle("dark", !isDark());
      storeTheme(isDark());
    });
    new MutationObserver(syncTheme).observe(root, { attributes: true, attributeFilter: ["class"] });
    window.addEventListener("storage", (event) => {
      if (event.key === THEME_KEY && (event.newValue === "dark" || event.newValue === "light")) {
        root.classList.toggle("dark", event.newValue === "dark");
      }
    });
    syncTheme();
  }

  // ------------------------------------------------------------------ data tower

  /**
   * The data tower's script loads once its section is about a screen away, so a visitor who
   * never scrolls to it downloads neither the script nor the layer images. Without it, or if
   * it fails, the layers stay a list of links.
   */
  function initDataTower() {
    const section = document.querySelector("[data-tower]");
    if (!section || typeof IntersectionObserver !== "function" || !window.CanvasRenderingContext2D) return;
    let started = false;
    const start = () => {
      if (started) return;
      started = true;
      import(TOWER_MODULE)
        .then(({ mountDataTower }) => mountDataTower(section))
        .catch((error) => console.error("3D ICE home: the data tower failed to start", error));
    };
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        start();
      },
      { rootMargin: "100% 0px" }
    );
    observer.observe(section);
  }

  // ------------------------------------------------------------------ hero showcase

  /** The showcase explorer forwards wheel scrolling it does not use, so the page still scrolls. */
  function initShowcaseScroll() {
    const showcases = () => [...document.querySelectorAll('iframe[src*="3D-interactive-cryosphere-explorer.html"]')];
    window.addEventListener("message", (event) => {
      const data = event.data || {};
      if (data.type !== SHOWCASE_SCROLL_MESSAGE || event.origin !== window.location.origin) return;
      if (!showcases().some((frame) => frame.contentWindow === event.source)) return;
      // "instant", not "auto": the page scrolls smoothly for in-page links, and a smooth scroll
      // per wheel tick would cancel the one before it.
      window.scrollBy({ left: Number(data.deltaX) || 0, top: Number(data.deltaY) || 0, behavior: "instant" });
    });
  }

  // ------------------------------------------------------------------ feedback form

  function restoreDraft(form, draft) {
    if (!draft) return;
    for (const [name, value] of draft) {
      const field = form.elements.namedItem(name);
      if (field && field.type !== "hidden" && typeof value === "string") field.value = value;
    }
  }

  /**
   * Basin submits the form and renders its success and error panels; this keeps the button,
   * the status line and keyboard focus in step, and keeps a visitor's message if sending fails.
   */
  function initFeedbackForm() {
    const form = document.getElementById("faq-feedback-3d-ice-form");
    const submit = form?.querySelector(".tool-feedback-submit");
    const status = document.getElementById("faq-feedback-3d-ice-status");
    const success = document.getElementById("faq-feedback-3d-ice-success");
    const failure = document.getElementById("faq-feedback-3d-ice-error");
    if (!form || !submit || !status || !success || !failure) return;
    const labels = form.dataset;
    const toolSelect = form.querySelector("[data-feedback-tool-select]");
    const sourceSection = form.querySelector("[data-feedback-source-section]");
    const pageUrl = form.querySelector("[data-feedback-page-url]");
    const message = form.querySelector("[data-feedback-message]");
    const reset = document.querySelector('[data-feedback-reset="faq-feedback-3d-ice-form"]');
    const isThisForm = (event) => event.detail && event.detail.form === form;
    let draft = null;

    // Basin 2.10.1's failure path reads an undeclared `error` and throws before it reports the
    // failure, which would leave the form stuck on "Sending...". Declaring the name lets it finish.
    if (!("error" in window)) window.error = null;

    const fillContext = () => {
      if (pageUrl) pageUrl.value = window.location.href;
      if (sourceSection) sourceSection.value = "tools-feedback";
      if (toolSelect && !toolSelect.value && labels.defaultTool) toolSelect.value = labels.defaultTool;
    };
    const setSubmit = (busy) => {
      submit.disabled = busy;
      submit.textContent = busy ? labels.labelSending : labels.labelSubmit;
    };

    fillContext();
    toolSelect?.addEventListener("change", fillContext);
    // Captured before Basin handles the submission and clears the form.
    form.addEventListener("submit", () => {
      draft = [...new FormData(form).entries()];
    }, true);
    reset?.addEventListener("click", () => {
      form.style.display = "";
      success.style.display = "none";
      failure.style.display = "none";
      status.textContent = "";
      setSubmit(false);
      form.reset();
      fillContext();
      message?.focus();
    });
    document.addEventListener("basinjsFormSubmitted", (event) => {
      if (!isThisForm(event)) return;
      failure.style.display = "none";
      setSubmit(true);
      status.textContent = labels.statusSending;
    });
    document.addEventListener("basinjsFormSuccess", (event) => {
      if (!isThisForm(event)) return;
      draft = null;
      form.reset();
      fillContext();
      setSubmit(false);
      status.textContent = labels.statusSent;
      form.style.display = "none";
      success.style.display = "block";
      window.requestAnimationFrame(() => success.focus());
    });
    document.addEventListener("basinjsFormError", (event) => {
      if (!isThisForm(event)) return;
      failure.style.display = "block";
      setSubmit(false);
      status.textContent = labels.statusFailed;
      // Basin hides and clears the form after this event; bring it back with the message intact.
      window.setTimeout(() => {
        form.style.display = "";
        restoreDraft(form, draft);
        fillContext();
      }, 0);
    });
  }

  // ------------------------------------------------------------------ start

  // Each part starts on its own, so a missing element cannot take the others down with it.
  for (const init of [initLocale, initTheme, initDataTower, initShowcaseScroll, initFeedbackForm]) {
    try {
      init();
    } catch (error) {
      console.error(`3D ICE home: ${init.name} failed`, error);
    }
  }
})();
