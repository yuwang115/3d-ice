/*
 * Analytics for 3d-ice.com: Google Analytics, plus a cookieless GoatCounter count kept as an
 * independent check on it. Every page loads this file without `defer` in <head>, before any
 * script that reports an event.
 *
 * The trackers load only on the production host and in a top-level window. Local previews,
 * test runs, the copy of the explorer that yuwang.blog serves from the compatibility bundle,
 * and the explorer inside the home page's showcase iframe are therefore never counted.
 * `window.gtag` is defined everywhere, so the pages' own events (outbound clicks, the 404
 * page's page_not_found) can always be queued; away from 3d-ice.com nothing sends them.
 */
(() => {
  "use strict";

  const PRODUCTION_HOSTS = ["3d-ice.com"];
  // The 3D ICE Google tag. Its one destination is the 3D ICE GA4 stream, G-YGK0V9S67F, which
  // has no tag of its own: Google answers gtag/js?id=G-YGK0V9S67F with a 404.
  const GA_TAG_ID = "GT-WPQPVBWP";
  // The GoatCounter site code, as in https://<code>.goatcounter.com. Empty turns it off.
  const GOATCOUNTER_CODE = "yuwang";
  // count.js v5, pinned by its published Subresource Integrity hash
  // (https://www.goatcounter.com/help/countjs-versions).
  const GOATCOUNTER_SCRIPT = "https://gc.zgo.at/count.v5.js";
  const GOATCOUNTER_INTEGRITY = "sha384-atnOLvQb9t+jTSipvd75X2yginT4PjVbqDdlJAmxMm+wYElFmeR6EmLP5bYeoRVQ";

  window.dataLayer = window.dataLayer || [];
  window.gtag =
    window.gtag ||
    function gtag() {
      // gtag.js expects the arguments object itself, not an array copy of it.
      window.dataLayer.push(arguments);
    };

  if (window.top !== window.self || !PRODUCTION_HOSTS.includes(window.location.hostname)) return;

  const appendScript = (src, attributes = {}) => {
    const script = document.createElement("script");
    script.async = true;
    script.src = src;
    for (const [name, value] of Object.entries(attributes)) script.setAttribute(name, value);
    document.head.appendChild(script);
  };

  appendScript(`https://www.googletagmanager.com/gtag/js?id=${GA_TAG_ID}`);
  window.gtag("js", new Date());
  window.gtag("set", { cookie_flags: "SameSite=None;Secure" });
  window.gtag("config", GA_TAG_ID, {});

  if (GOATCOUNTER_CODE) {
    appendScript(GOATCOUNTER_SCRIPT, {
      "data-goatcounter": `https://${GOATCOUNTER_CODE}.goatcounter.com/count`,
      integrity: GOATCOUNTER_INTEGRITY,
      crossorigin: "anonymous",
    });
  }
})();
