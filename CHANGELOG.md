# Changelog

All notable changes to 3D ICE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The published total isostatic response of Paxman, Austermann & Hollyday (2022), grids v3
  (NSF Arctic Data Center, doi:10.18739/A22Z12R8C, CC BY 4.0), as the isostatic-rebound
  layer's default Earth response: laterally variable elastic thickness, the post-LGM
  disequilibrium still to come, and water loading under a sea surface raised by both ice
  sheets' meltwater plus the residual geoid change. Six `*_isostatic_response_*` packages,
  one per BedMachine Antarctica v4, Bedmap3 and BedMachine Greenland v6 terrain package,
  sampled at the terrain nodes by `scripts/prepare_isostatic_response.py`, which verifies
  each download's MD5 and the published identities `T = R − G` and
  `T = ice unloading + post-LGM + water loading`. The QRF Greenland terrain borrows the
  BedMachine v6 response and says so.
- `summarisePublishedResponse` in `static/tools/js/gia-rebound.js`, and a `--model
  paxman2022` default in `examples/isostatic-rebound.mjs`.

### Changed

- The ELRA and Airy responses are labelled idealised what-ifs; the sea-level datum slider
  applies to them only. The worked example's headline figure is now 2.94 million km² of
  newly emergent Antarctic bed (relative to the ice-free sea surface) instead of the
  3.19 million km² ELRA gives at today's datum.

### Fixed

- The rebound caveats no longer attribute the Amundsen Sea Embayment's 41 mm/yr uplift to
  the Last Glacial Maximum; it is a response to recent ice loss over a weak mantle
  (Barletta et al. 2018). The "full equilibrium after 15–20 kyr" note now says that this
  is a property of the single ELRA relaxation time, and that full re-equilibration takes of
  order 100 kyr.

## [0.2.0] — 2026-09-27

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
- Documentation of the architecture, the data contract, the data pipeline (with a source
  for every upstream input) and two worked examples in `docs/`, and the runnable example
  `examples/isostatic-rebound.mjs`.
- A test that regenerates the six Bedmap3 overlay packages from the committed inputs and
  requires the payloads to match the committed ones byte for byte.
- JOSS paper draft by Yu Wang and Yucheng Lin, with matching `CITATION.cff` and
  `codemeta.json`, a contribution guide and a code of conduct.
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
- Broken source links: the WAOM v1.0 paper in the ocean-current package metadata, and the
  two basal-friction preprints in the explorer, the landing pages and the README.

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

[0.2.0]: https://github.com/yuwang115/3d-ice/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.2
[0.1.1]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.1
[0.1.0]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.0
