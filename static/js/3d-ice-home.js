/*
 * Behaviour for the 3D ICE home pages (/ and /zh/): the language switcher, the theme toggle
 * and the images that follow it, the preview videos, the scroll hand-off from the hero
 * showcase and the feedback form's messages. Loaded with `defer`, after 3d-ice-locale.js.
 *
 * The theme itself is applied by a small inline script in <head> before the first paint.
 * It lives on <html class="dark"> and in localStorage["wc-color-theme"], which the
 * explorer pages read too, so a choice made here carries over to them.
 */
(() => {
  "use strict";

  const THEME_KEY = "wc-color-theme";
  const SHOWCASE_SCROLL_MESSAGE = "3d-ice:showcase-scroll";
  const root = document.documentElement;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

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

  // ------------------------------------------------------------------ preview videos

  function preparePreview(link) {
    const video = link.querySelector("video");
    if (!video) return null;
    // Muted inline playback is what lets browsers start a video without a gesture.
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    return video;
  }

  function playPreview(preview) {
    const { link, video } = preview;
    link.classList.add("is-video-active");
    const started = video.play();
    if (started && typeof started.then === "function") {
      started
        .then(() => link.classList.add("is-video-playing"))
        .catch(() => {
          // Refused (Low Power Mode, an unsupported format): leave the poster rather than retry.
          preview.refused = true;
          link.classList.remove("is-video-active");
        });
    } else {
      link.classList.add("is-video-playing");
    }
  }

  function pausePreview(link, video) {
    link.classList.remove("is-video-active", "is-video-playing");
    video.pause();
  }

  /** Reduced motion and Data Saver keep the previews on their posters. */
  function previewsMayPlay() {
    return !reducedMotion.matches && !navigator.connection?.saveData;
  }

  /**
   * Preview loops play while they are on screen and pause off it, so six videos never decode
   * at once. They are marked preload="none" and load on their first play, so a visitor who
   * never scrolls to them downloads only the posters. The links work either way.
   */
  function initPreviews() {
    const previews = [...document.querySelectorAll(".explorer-preview-link--video[data-video-preview]")]
      .map((link) => ({ link, video: preparePreview(link), refused: false }))
      .filter((preview) => preview.video);
    if (!previews.length) return;

    const visible = new Set();
    const update = () => {
      const mayPlay = previewsMayPlay();
      for (const preview of previews) {
        const { link, video } = preview;
        const active = link.classList.contains("is-video-active");
        const shouldPlay = mayPlay && visible.has(link) && !preview.refused;
        if (shouldPlay && video.paused && !active) playPreview(preview);
        else if (!shouldPlay && (!video.paused || active)) pausePreview(link, video);
      }
    };

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) visible.add(entry.target);
            else visible.delete(entry.target);
          }
          update();
        },
        { threshold: 0.25 }
      );
      previews.forEach(({ link }) => observer.observe(link));
    } else {
      previews.forEach(({ link }) => visible.add(link));
      update();
    }
    reducedMotion.addEventListener("change", update);
    window.addEventListener("pagehide", () => previews.forEach(({ link, video }) => pausePreview(link, video)));
    // Coming back with the Back button restores the page from the cache, paused.
    window.addEventListener("pageshow", () => {
      previews.forEach((preview) => {
        preview.refused = false;
      });
      update();
    });
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
  for (const init of [initLocale, initTheme, initPreviews, initShowcaseScroll, initFeedbackForm]) {
    try {
      init();
    } catch (error) {
      console.error(`3D ICE home: ${init.name} failed`, error);
    }
  }
})();
