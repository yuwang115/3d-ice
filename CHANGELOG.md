# Changelog

All notable changes to 3D ICE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] — Unreleased

### Added

- Isostatic-rebound layer: the ice-free equilibrium bed under regional flexure (an elastic
  lithosphere over a relaxed asthenosphere) or local Airy isostasy, solved spectrally in a
  module worker from whichever terrain package is loaded and validated against the analytic
  point-load solution.
- Bedmap3 v1.0 as an Antarctic terrain alternative (10 km Balanced, 4 km HD), with its
  velocity, basal-friction and hydrology overlays re-gridded onto Bedmap3's native grid.
- Greenland QRF 2025 subglacial topography as a hybrid terrain alternative (3 km, 1 km).
- Places and geographic features: searchable research stations and place names for both
  regions, and refined drainage basins in search.
- Selectable flowline profiles showing ice surface, ice base and bed along each flowline.
- Chinese (zh-CN) explorer and landing page.
- `static/tools/js/data-contract.js`, the package decoder shared by the explorer page, the
  geometry worker and the tests, and a contract test that decodes every committed package
  and checks it against the statistics recorded when it was prepared.
- Documentation of the architecture, the data contract and two worked examples in `docs/`,
  and the runnable example `examples/isostatic-rebound.mjs`.
- A test that regenerates the six Bedmap3 overlay packages from the committed inputs and
  requires the payloads to match the committed ones byte for byte.
- JOSS paper draft, `CITATION.cff`, `codemeta.json`, contribution guide and code of conduct.
- Landing-page light/dark toggle, latest-updates section and 404 page.

### Changed

- The English and Chinese explorer pages load one shared runtime,
  `static/tools/js/explorer-app.js`, and stylesheet, `static/tools/css/explorer.css`,
  instead of each carrying its own copy of the application.
- The geometry worker is now an ES module worker that imports the shared decoder. The
  explorer needs Chrome or Edge 89+, Firefox 114+ or Safari 15+.
- Terrain fields are decoded with the quantization their package declares rather than an
  assumed unit scale. Every shipped terrain package already used unit scale, so rendering
  is unchanged.
- Subglacial-hydrology packages record their quantization on the `effective_pressure` field
  as well as in the legacy package-level keys.
- 3d-ice.com is served independently of yuwang.blog, with its own analytics property.
- CI runs the JavaScript jobs on Node.js 24.

### Fixed

- Flow lights could be switched off by the normalised `flowGlow` flag.
- The geometry worker's decoder could not read multi-byte fields stored at unaligned
  offsets; it now uses the page's decoder, which can.
- The browser test suite deadlocked once the static server's log filled an undrained pipe.

## [0.1.2] — 2026-03-21

### Changed

- The desktop showcase no longer orbits automatically.
- Site metadata for the 3d-ice.com domain.

## [0.1.1] — 2026-03-21

### Added

- Standalone GitHub Pages site with its own landing page, and manual release publishing.

### Fixed

- Desktop interaction in the landing-page demo.

## [0.1.0] — 2026-03-21

First release as a standalone repository: the interactive explorer for Antarctica and
Greenland with BedMachine terrain, ice velocity, basal friction, Antarctic subglacial
hydrology, WAOM2 and Copernicus ocean streamlines, RISE ice-shelf basal melt and thermal
driving, and drainage-basin boundaries, together with the preparation scripts and the
compatibility-bundle release workflow.

[0.2.0]: https://github.com/yuwang115/3d-ice/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.2
[0.1.1]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.1
[0.1.0]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.0
