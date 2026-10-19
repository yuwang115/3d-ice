/**
 * The Data Snapshot's source lines. A line used to hold one link, so a data set and the
 * article describing it could not both be credited: the basal-friction line linked only the
 * preprints, never the CC BY 4.0 Zenodo data sets it draws, and the hydrology line named the
 * GRL paper but linked the data.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { escapeHtml, renderSourceLine, renderSourceValue } from "../../static/tools/js/source-links.js";

test("escapeHtml escapes the characters that could break out of text or an attribute", () => {
  assert.equal(escapeHtml(`<a href="x">Tom's & co</a>`), "&lt;a href=&quot;x&quot;&gt;Tom&#39;s &amp; co&lt;/a&gt;");
  assert.equal(escapeHtml(42), "42");
});

test("a plain string is text, and a link without a URL is text too", () => {
  assert.equal(renderSourceValue("Greenland <basins>"), "Greenland &lt;basins&gt;");
  assert.equal(renderSourceValue({ text: "No link" }), "No link");
  assert.equal(renderSourceValue(null), "");
});

test("a link opens in a new tab without handing over the page", () => {
  assert.equal(
    renderSourceValue({ text: "BedMachine & co", url: "https://example.org/?a=1&b=2" }),
    '<a class="meta-link" href="https://example.org/?a=1&amp;b=2" target="_blank" rel="noopener noreferrer">BedMachine &amp; co</a>'
  );
});

test("an array runs its parts together, so one line can credit a data set and its article", () => {
  const html = renderSourceValue([
    { text: "Data set (2026)", url: "https://doi.org/10.1/data" },
    "; ",
    { text: "article (2025)", url: "https://doi.org/10.1/article" },
  ]);
  assert.equal((html.match(/<a /g) || []).length, 2);
  assert.ok(html.includes('href="https://doi.org/10.1/data"'));
  assert.ok(html.includes('>Data set (2026)</a>; <a '));
  assert.ok(html.endsWith(">article (2025)</a>"));
});

test("a source line labels its source and appends a suffix", () => {
  assert.equal(
    renderSourceLine("Melt <drivers>", "RISE", " (15 cells)"),
    '<li class="meta-compact"><strong>Melt &lt;drivers&gt;:</strong> RISE (15 cells)</li>'
  );
});
