# Changelog

All notable changes to 3D ICE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Ice-sheet projections to 2300 in both editions. A year slider and Play button morph the
  Antarctic ice from 2015 to 2300 under three ISMIP6 scenarios with UKESM1-0-LL forcing: low
  emissions (SSP1-2.6, `expAE10`), high emissions (SSP5-8.5, `expAE05`) and high emissions with
  ice-shelf collapse (`expAE14`). Each is the equal-weight mean change of the eight ice sheet
  models that ran all three (Seroussi et al. 2024; model output CC BY 4.0), applied to today's
  BedMachine v4 ice, so the first frame is the ice already on screen. Thinning is scaled to
  BedMachine's thickness so that each cell keeps the share of ice the models keep, which spares
  the ice shelves the holes that adding the change outright would leave. The ice is coloured by
  its thickness change, a readout and chart give the models' mean sea-level contribution with
  their range, and the ice flowlines stay on, ride the projected surface and speed up and
  brighten with the mean change in ice speed. They come on with the projection and go off
  with it unless changed by hand, and a switch in the projection's controls shows or hides
  them. The packages and the two scripts that build them
  next to the 2D archive on an HPC system are in the repository (`docs/data-pipeline.md`).

- A public edition of the explorer at `/explore/` and `/zh/explore/`, for visitors,
  classrooms and outreach stands. It offers each region's default terrain, see-through ice,
  vertical exaggeration, place search, animated ice flowlines coloured by speed, ocean
  currents under a single toggle, sea level and the published ice-free rebound. It adds a
  nine-stop guided tour, with camera flights, layer changes and animated sliders, and a
  plain-language explainer, with sources, beside every layer. One stop plays Antarctica's
  future under high emissions (`expAE05`) from 2015 to 2300, its card reading out the year
  and the sea-level change from Antarctica as they pass, and the last stop ends the tour in
  Antarctica or Greenland, whichever the visitor picks to explore first. `?tour=1` or
  `?tour=<stop id>` opens the tour on load. On desktop the tour plays in the side panel
  next to the view; on phones the card floats over the viewer and can be collapsed. The
  first time a visitor switches "Remove the ice" on, the ice melts away over three seconds,
  from today's ice to the fully rebounded land, instead of vanishing at once.
- Edition profiles (`static/tools/js/editions.js`). Every explorer page runs the same
  runtime, and `<html data-edition>` decides which datasets and layers it offers. A
  restricted layer also loses its package URLs, so the public pages never fetch a
  research-only package. Controls a page leaves out are replaced by detached stand-ins in a
  fixed state, so no preset can switch those layers on.
- Camera flights and geographic framing in the runtime: an eased flight between poses that
  any camera input cancels, and a pose that frames a disc of a given diameter around a
  latitude and longitude in the part of the viewer the page leaves uncovered. Browser
  EPSG:3031 and EPSG:3413 projections (`static/tools/js/polar-projection.js`) reproduce
  every stored position in the place catalogues.
- Unit tests for the profiles, the projections, the tour logic and the copy, and browser
  tests for the public edition.
- The compatibility bundle carries the public edition and the Chinese research page, the home
  page's stylesheets, script and typefaces, and the two home pages under `home/`, so a host
  that mounts it at its site root serves both editions at 3d-ice.com's paths. yuwang.blog now
  builds its 3D ICE page from the bundled home pages instead of keeping a hand-made copy. The
  language switcher knows that copy's addresses (`/tools/3d-ice/`, `/zh/tools/3d-ice/`), and
  `tests/test_site_embed.py` pins what the copy relies on. The bundle manifest's `files` now
  lists paths from the bundle root (`tools/css/explorer.css`), not from `tools/`.
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
- `tests/js/gia-point-load.test.mjs`, which checks the flexural solver against the analytic
  point-load Kelvin-function solution (Brotchie & Silvester 1969): within 10⁻⁴ of the peak
  beyond half a flexural length, and within the grid's Nyquist truncation bound under the
  load. The 0.2.0 notes, the README and the paper cited this validation, but the test suite
  did not contain it.

### Changed

- The research pages' link to the guided tour is relative, so it opens the tour on whichever
  site serves the page; it used to send yuwang.blog's visitors to 3d-ice.com.
- The home pages are rebuilt around the two editions. The hero leads with the guided tour
  and a way into the research edition, both in the first screen of a laptop, and a new "Two
  ways to explore" section introduces each edition, with a folded comparison table. Every
  preview names the editions that offer it and opens in one of them. The update cards
  announce the public edition and describe the published rebound response. The research
  edition keeps its URL, features and behaviour, and the research pages link to the tour.
- The home pages are now readable HTML with one stylesheet and one script, in the typefaces
  the site already bundled (Space Grotesk and Playfair Display). Both are variable fonts, now
  declared with their weight ranges, so bold text is real rather than synthesised and six
  duplicate font files are gone. The 188 KB framework stylesheet, the Inter font and two
  scripts that did nothing are gone too. Preview videos load only when scrolled into view and
  stay on their posters with reduced motion or Data Saver. The page has a skip link, in-page
  navigation, a visible focus ring on every control, including the preview videos, and text
  and buttons at 4.5:1 contrast or better. The 404 page uses the same typefaces, and its
  explorer button opens the public edition.
- The ELRA and Airy responses are labelled idealised what-ifs; the sea-level datum slider
  applies to them only. The worked example's headline figure is now 2.94 million km² of
  newly emergent Antarctic bed (relative to the ice-free sea surface) instead of the
  3.19 million km² ELRA gives at today's datum.
- The ice and ocean flow animation is quieter: particles are about half the size, sparser,
  tinted by the line colour rather than near-white, and travel roughly half as fast; the
  pulse along each line is dimmer.
- Sidebar order: the research pages gather "Show future projection (ISMIP6, to 2300)" and
  "Show ice-free isostatic rebound", the two interactive scenarios, in a section of their own
  (Interactive Scenarios) below View Controls, and "Animate ice & ocean flow" moves down beside
  "Wireframe mode".

### Fixed

- The home pages' source list includes the Paxman et al. (2022) response, which the rebound
  layer shows by default, and a test keeps the list in step with the sources the explorer
  cites. Section labels are drawn small and in the accent colour as designed, rather than as
  body text, and the update cards no longer touch.
- A feedback message that fails to send no longer leaves the form stuck on "Sending..." with
  the message wiped: the form library's failure path threw before reporting the failure. The
  page works around it, shows the error and restores the message.
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
