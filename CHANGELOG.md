# Changelog

All notable changes to 3D ICE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.1] — Unreleased

Corrections from a pre-announcement audit of 0.3.0.

### Fixed

- The Antarctic ocean streamlines now follow the modelled currents. Version 0.3.0 rotated
  WAOM2's native velocity components by the ROMS grid angle, which on WAOM2's
  EPSG:3031-aligned grid equals the longitude, and then stepped the streamlines with those
  east/north components as if they were x/y: every traced vector was turned by the local
  longitude, so the Antarctic Circumpolar Current ran westward in the Pacific sector. The
  packer now rotates by the grid angle minus the longitude, the identity on WAOM2's grid. The
  combined package and its two components were rebuilt within the same segment budget (8,150
  longer streamlines, 1.5 million segments), and a test checks that the circumpolar current
  flows east in every well-sampled sector. The two unused seeding variants (`..._yr5_annual`
  and `..._cavity_margin50km`) were removed rather than rebuilt. The home page's Southern
  Ocean preview and Figure 1(a) of the paper were re-rendered.
- The vertical exaggeration labels show the true factor. Terrain is drawn 13.7 times taller
  than horizontal distances before the slider multiplies it, so the public edition's default
  is 66× and the research edition's 55×, and the tour no longer says the heights are
  stretched "about five times" or that 1× shows true proportions.
- The subglacial channel layer keeps channels whatever their direction. GlaDS discharge is
  signed by the orientation of each mesh edge, and the packer kept only positive values, which
  dropped 35–42 % of the channel network. It now filters and reduces on the magnitude, and the
  hydrology packages and their Bedmap3 copies were rebuilt: 9,068 channel edges at 10 km and
  29,793 at 4 km, from 5,898 and 17,238. The default minimum discharge now matches the
  packages (0.001 m³/s).
- Sea-level equivalents are converted at freshwater density, as Gregory et al. (2019) define
  them: 58.1 m for Antarctica on BedMachine v4 and 7.46 m for Greenland, instead of 56.5 m and
  7.26 m at seawater density. The research edition's notes now compare them with 57.9 ± 0.9 m
  for BedMachine Antarctica v1 (Morlighem et al. 2020) and 7.42 ± 0.05 m of ice above
  flotation for BedMachine Greenland v3 (Morlighem et al. 2017), rather than attributing the
  first to version 4 and calling the second a total-volume figure.
- The idealised regional-flexure note had the forebulge backwards: when the ice is removed,
  the bulge it had raised beyond the margin subsides.
- The Data Snapshot's mean ice thickness and ice-speed statistics cover ice-covered cells only.
  Antarctica reads 1,940 m and a median speed of 7 m/yr rather than 590 m and 0 m/yr, and
  Greenland's maximum speed is its fastest ice rather than a fjord pixel. Until the velocity
  layer has been built, the speed statistics read "Available on demand".
- The effective-pressure and channel-discharge legends place their labels at their values,
  mark the saturated top of each scale with ≥, and say that over-pressurised bed (N < 0) is
  not drawn. Unit labels read MPa and m³/s.
- Public copy. Antarctica's ice covers about 14 million km² (areas are now corrected for the
  map's scale factor), the ice resting on land is on average more than 2 km thick, most rather
  than almost all of West Antarctica rests on bed below sea level, nearly two-thirds of the
  grounded ice moves less than 10 m a year, the full rebound takes tens of thousands of years,
  and the bedrock below sea level under the ice is buried by ice rather than water.
- On phones and other narrow screens, both editions open on the whole ice sheet instead of a
  cropped close-up.
- Subglacial hydrology is drawn only beneath grounded ice. Effective pressure was also drawn
  on 522 floating and 359 ice-free cells of the 10 km grid, on the sea floor beneath the
  shelves and on bare rock, and channel ribbons ran on past the grounding line; a channel now
  ends half-way between its last grounded cell and the next, where the grounded ice ends.
- No ocean streamline runs inside an ice shelf: segment ends in a cavity must lie below
  today's ice base as well as above the bed. About 2,000 of 59,000 sub-shelf segments did not.
- The ice over Lake Vostok stays on its lake. BedMachine's bed there is the lake floor, so
  switching the projection on dropped the ice surface by 254 m (median; up to 837 m) with no
  change shown, and the rebound scenario did the same.
- During a partial rebound, grounded marine ice that has thinned enough to float lifts off
  its bed, as floating ice already did, instead of staying pinned far below the sea surface.
- The projection's "Grounded ice on the map" readout uses scale-corrected cell areas, 12.28
  rather than 12.01 million km² for Antarctica in 2015, as the rebound statistics do.
- The RISE source line counts the ice-shelf cells of the package in view (95,490 at 4 km),
  rather than always 15,269.
- The Chinese Data Snapshot is in Chinese: grid spacing, projection, ranges, medians, counts
  and years. Its coordinates name their axes and projection, for example
  "x = 977 km, y = 783 km (EPSG:3031)", in both languages.
- Chinese terminology: 地壳均衡 throughout, 格陵兰冰盖 rather than 冰原, 冰层不透明度 for the
  ice-opacity slider, and 冰流 kept for ice streams, with the flow of ice now 冰的流动 and its
  speed 冰速.
- The projection readout and chart say "Sea-level change from Antarctica", as the tour card
  does, since the value can be negative.
- The home pages describe the Southern Ocean preview as annual-mean circulation, not
  "ever-changing", and the Greenland one as currents on the shelf and offshore, not in deep
  fjords, which the 6 km Copernicus grid does not resolve.
- The ocean legend, the public explainer and the Data Snapshot say what the streamlines'
  four depth bands are: where they were seeded, at 3–18 %, 18–40 %, 40–70 % and 70–95 % of
  the local water depth.

### Changed

- Data credits. A Data Snapshot line can credit a data set and its article together. The
  explorer and the home pages now link the AISEFI and GrISEFI basal shear stress data sets
  (CC BY 4.0) under their titles, the RISE data (Australian Antarctic Data Centre) and the
  GlaDS data beside their articles, the Paxman et al. (2022) article and the grid files v3
  each under its own label, and the Greenland basins (Mouginot & Rignot, 2019), which the
  explorer did not credit before. The Antarctic basins are named as the MEaSUREs Antarctic
  Boundaries v2 refined basins, not "IMBIE", and the Greenland velocity as the ITS_LIVE V02.1
  summary mosaic (2014–2024), with the credit ITS_LIVE asks for.
- The JOSS paper cites the BedMachine Antarctica v4, BedMachine Greenland v6, NSIDC-0754,
  Bedmap3 and RISE data sets beside their articles, Payne et al. (2021) for the CMIP6 forcing,
  and renders every reference correctly in the journal's style (preprints marked, name
  particles in front, data sets with their publishers).
- On phones, the collapsed control drawer's button says what the drawer holds ("Layers and
  places") and opens it; it reads "Done" only while the drawer is open. Tour text that runs on
  below the card's edge fades out there.
- The README describes Bedmap3's overlays as nearest-cell copies of the BedMachine-aligned
  packages and gives the map scale factor's effect correctly (a nominal cell covers about
  5.7 % more ground at the pole and 8 % less at 60°S). It notes that BedMachine Greenland v7
  was released in October 2026; 3D ICE still uses v6. The stale v0.1 README PDF was removed.

- Phones, small windows and Save-Data connections load a light Antarctic ocean package (12 MB
  rather than 62 MB), and there the research edition no longer fetches the ocean package in
  the background.
- `DATA_LICENSES.md` lists the licence or terms of every data package and colour table (the
  cmocean maps under the MIT License, GMT's relief table under the LGPL). The MIT License covers
  the code and scripts only, as the README and the home pages' footers now say. The Copernicus
  Marine credit ("Generated using E.U. Copernicus Marine Service Information") and product
  DOI, COMNAP's educational and non-commercial term, and the permission under which the WAOM2
  streamlines are redistributed are stated where the data are credited; the paper's
  acknowledgements carry the Copernicus Marine sentence.
- The home pages present the 2300 projections in both edition cards, the comparison table,
  the structured data and a dated update, and describe the subglacial layer as modelled. The
  link-preview image is a clean capture without interface overlays, and the Chinese public
  page has a Chinese social description.
- The E2E suite blocks Google Analytics, so test runs no longer count as visits.

## [0.3.0] — 2026-10-06

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
- The JOSS paper describes the public edition and the ISMIP6 2300 projections, cites the
  ISMIP6 papers and datasets the ISMIP6 data terms ask for, acknowledges CMIP6, ESGF, ISMIP6
  and Ghub, and extends the AI-usage disclosure to October 2026. Figure 1(c) shows the
  projected ice in 2300 under high emissions instead of the idealised ice-free bed.

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
- `docs/data-pipeline.md` gave the ISMIP6 2300 archive on NCI Gadi as 17 TB; it is 8 TB.
  `docs/architecture.md` said the browser tests play the public tour in both locales; they
  walk it in English and check its Chinese copy.
- The paper no longer says that every data layer links to its source (Greenland's basin
  boundaries have no link) or that a URL parameter selects the terrain dataset (none does).
- The paper's PDF printed the citation of Paxman et al. (2022) and its grids as a broken
  parenthetical; it now reads "Paxman et al. (2022, 2026)".

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

[0.3.1]: https://github.com/yuwang115/3d-ice/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yuwang115/3d-ice/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yuwang115/3d-ice/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.2
[0.1.1]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.1
[0.1.0]: https://github.com/yuwang115/3d-ice/releases/tag/v0.1.0
