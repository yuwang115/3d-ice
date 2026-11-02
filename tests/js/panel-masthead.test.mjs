import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { DEFAULT_REGION, editionHref } from "../../static/tools/js/panel-masthead.js";

describe("editionHref", () => {
  it("links to the other edition's page as it is for the default region", () => {
    assert.equal(DEFAULT_REGION, "antarctica");
    assert.equal(editionHref("../explore/", "antarctica"), "../explore/");
    assert.equal(
      editionHref("../tools/3D-interactive-cryosphere-explorer.html", DEFAULT_REGION),
      "../tools/3D-interactive-cryosphere-explorer.html",
    );
  });

  it("names any other region, so the reader lands on the same ice sheet", () => {
    assert.equal(editionHref("../explore/", "greenland"), "../explore/?region=greenland");
  });

  it("replaces the query and drops the hash it was given, whatever the region", () => {
    assert.equal(editionHref("../explore/?region=greenland", "antarctica"), "../explore/");
    assert.equal(editionHref("../explore/?tour=1#stop-2", "greenland"), "../explore/?region=greenland");
  });

  it("leaves out a region it does not have and escapes one it does", () => {
    assert.equal(editionHref("../explore/", ""), "../explore/");
    assert.equal(editionHref("../explore/", undefined), "../explore/");
    assert.equal(editionHref("../explore/", "a b&c"), "../explore/?region=a%20b%26c");
    assert.equal(editionHref(undefined, "greenland"), "?region=greenland");
  });
});
