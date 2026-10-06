---
title: '3D ICE: An Interactive Browser-Based Cryosphere Explorer for Antarctica and Greenland'
tags:
  - cryosphere
  - Antarctica
  - Greenland
  - ice sheet
  - WebGL
  - visualization
  - glaciology
  - scientific communication
authors:
  - name: Yu Wang
    orcid: 0000-0001-9070-6004
    corresponding: true
    affiliation: "1, 2"
  - name: Yucheng Lin
    orcid: 0000-0002-5556-4490
    affiliation: 3
affiliations:
  - name: Climate Systems Engineering initiative, Institute for Climate and Sustainable Growth, University of Chicago, Chicago, Illinois, United States
    index: 1
    ror: 024mw5h28
  - name: Institute for Marine and Antarctic Studies, University of Tasmania, Hobart, Tasmania, Australia
    index: 2
    ror: 01nfmeh72
  - name: School of Energy and Environment, City University of Hong Kong, Hong Kong SAR, China
    index: 3
    ror: 03q8dnn23
date: 28 September 2026
bibliography: paper.bib
---

# Summary

3D ICE is an open-source platform for exploring research-grade representations
of the Antarctic and Greenland ice sheets in a web browser
(\autoref{fig:overview}). It combines terrain, ice velocity, ocean circulation,
ice-shelf basal melting, basal friction, subglacial hydrology, and drainage
basins in layered three-dimensional scenes. Users can rotate and zoom each ice
sheet, switch datasets and resolutions, search polar places, show the ice-free
bed after isostatic rebound, play multi-model projections of Antarctic ice to
2300, and follow links to the source data. A research edition offers every
dataset and layer; a public edition for classrooms and outreach pairs a subset
with a guided tour and plain-language explanations.

An offline Python pipeline converts heterogeneous NetCDF, HDF5, GeoTIFF, and
vector products into compact binary packages with provenance metadata, which a
static JavaScript application built with Three.js [@threejs] and WebGL renders
without plugins or server-side computation, on desktop and mobile devices.

![The 3D ICE explorer. (a) The interface, with Antarctic ice-surface speed [@mouginot2019] and surface-layer WAOM2 ocean streamlines [@richter2022] over BedMachine Antarctica v4 [@morlighem2020]. (b) Greenland ice-surface speed from ITS_LIVE [@gardner2024] over BedMachine Greenland v6 [@morlighem2017]. (c) The ice-free Antarctic bed after complete isostatic rebound under the idealised regional-flexure response. Vertical scales are exaggerated.\label{fig:overview}](docs/images/explorer-overview.jpg)

# Statement of need

Modern cryosphere science generates large, gridded datasets in specialized
formats and polar stereographic coordinate systems, and ice-sheet projections
add terabyte-scale multi-model archives. Inspecting several products together
commonly requires a desktop GIS, a numerical computing environment, or
purpose-built scripts. These workflows suit quantitative analysis but require
installation, data transfer, reprojection, and preprocessing before a user can
see an ice sheet in context.

The barrier falls hardest on researchers in adjacent disciplines, educators who
want to teach with real polar data, and scientists preparing interactive
research communication, who need to see quickly how ice geometry, flow, ocean
forcing, and basal processes relate, with a path back to the source data. 3D ICE
provides a curated, zero-install view, not a replacement for GIS or numerical
analysis; its research purpose is to make heterogeneous cryosphere products
easier to compare, explain, and select for analysis.

# State of the field

Quantarctica [@matsuoka2021] and its Greenland counterpart QGreenland
[@moon2023qgreenland] are comprehensive data packages and mapping environments
for desktop QGIS. They
suit local geospatial analysis and span more disciplines than 3D ICE, which
trades general GIS operations for an immediately shareable, curated 3D view with
one interaction model for both ice sheets.

NASA Worldview offers web access to more than a thousand global
satellite-imagery layers, with polar views, temporal comparison, animation, and
data download [@nasa_worldview], but focuses on two-dimensional, often
near-real-time imagery. 3D ICE instead combines terrain with subsurface and
model-derived fields such as basal friction, hydrology, basal melt,
depth-dependent ocean circulation, and projected ice geometry.

General 3D libraries such as CesiumJS [@cesiumjs], with its high-precision
WGS84 globe and scalable data formats, and Three.js [@threejs], the lower-level
graphics that 3D ICE uses, supply no cryosphere data model, polar-grid
conversion, provenance records, layer semantics, or data curation. Contributing
these product-specific transformations to a general rendering engine would not
address the research need, so 3D ICE reuses established graphics infrastructure
and concentrates its scholarly contribution in the reproducible polar-data
pipeline, the data contract, and domain-specific interaction design.

# Software design

3D ICE has two stages joined by a specified data contract. The offline stage reads
authoritative source products, reconciles their coordinate conventions, and
resamples fields onto terrain-aligned polar grids with NumPy [@harris2020].
Floating-point grids are quantized to signed 16-bit arrays for browser delivery,
a controlled trade of precision for transfer size: a paired `.meta.json` file
stores scale, offset, fill value, units, grid geometry, statistics, source
citation, and processing provenance, so decoding is deterministic and the loss
of precision inspectable. One decoder module reads this metadata in the page,
in the geometry worker, and in a test that checks every committed package
against the statistics recorded before quantization, to within half a
quantization step.

Expensive transformations run before deployment, so the application needs only
static hosting; the trade-off is that updating a source product means
regenerating its package. Antarctic ocean streamlines, for example, are
integrated through multi-depth WAOM2 velocity fields with seeds balanced by
region and depth [@richter2022; @dias2023]. The projection layer plays the
equal-weight mean of eight ISMIP6 Antarctica 2300 models, one per modelling
group, under three UKESM1-0-LL scenarios: low emissions, high emissions, and
high emissions with ice-shelf collapse
[@seroussi2024; @nowicki2024ismip6; @nowicki2020; @barthel2020;
@jourdain2020]. Scripts beside the 8 TB archive on a supercomputer regrid each
model conservatively. A packer applies the models' mean thickness change since
2015 to today's BedMachine ice at 5-year keyframes, scaling thinning to
BedMachine's thickness so that ice shelves do not break into spurious holes,
and stores their mean speed change. The browser interpolates between keyframes,
rebuilds the ice by flotation on the static bed, and moves the flowlines,
coloured by today's speed plus that change, onto the projected surface. The averaged geometry implies more sea-level rise than the mean of the
models' published contributions (1.93 against 1.46 m by 2300 under high
emissions), partly because flotation is non-linear where the models disagree
about which basins collapse, so the explorer reports the published mean and
range [@seroussi2024data].

Web Workers build overlay geometry off the main thread, and Balanced and HD
packages trade resolution against memory. Logic that needs neither page nor
scene, such as package decoding, place search, the rebound solver, and
projection playback, lives in modules that run unchanged under Node's test
runner. Both editions, in English and Chinese, are thin shells around one
runtime: an edition profile removes restricted datasets and layers, with their
package URLs, before anything is fetched, and pins the controls a page omits to
fixed states, so the public pages can neither request nor show a research-only
layer.

The isostatic-rebound layer shows which parts of today's sub-sea-level bed would
emerge once glacial isostatic adjustment is complete. By default it displays the
published response of @paxman2022 [@paxman2026data]: elastic-plate flexure with
laterally variable elastic thickness [@swain2021; @steffen2018], remaining
post-LGM disequilibrium, and water loading under a sea surface raised by both
ice sheets' meltwater. For comparison it solves an elastic-lithosphere,
relaxed-asthenosphere steady state [@lemeur1996; @lingle1985] spectrally
[@bueler2007] and, as an upper bound on peak uplift, local Airy isostasy
[@turcotte2002]. Sea-level-equivalent figures follow @gregory2019. The
interface states each response's assumptions, including that the present bed is
not in balance with the present load
[@whitehouse2019] and that response times are orders of magnitude shorter over
the low-viscosity mantle beneath parts of West Antarctica [@barletta2018].

Unit tests cover quantization, coordinate sampling, metadata statistics,
search, projection playback, edition profiles, the tour and its copy in both
languages, and the rebound solver against the analytic Kelvin-function
point-load solution [@brotchie1969]; integration tests regenerate derived
packages and compare them with the committed files; and Playwright tests load
both editions, check that the public edition requests no research-only package,
and walk its tour stop by stop. All run in continuous integration and locally.

# Research impact statement

Y. Wang uses 3D ICE to compare candidate bed, velocity, and basal-friction
products before configuring ice-flow and subglacial-hydrology simulations, and
has used its views in a research funding proposal. It has been
demonstrated in talks at the Antarctic Research Centre, Victoria University of
Wellington (February 2026), the ISMIP7 Workshop in Copenhagen (March 2026,
presented by C. Zhao), the Asia Early Career Polar Forum in Zhuhai (June 2026),
and the School of Oceanography, Shanghai Jiao Tong University (July 2026).

Y. Lin has used 3D ICE in teaching GE1301 Climate Change and Extreme Weather at
the City University of Hong Kong, to show students the geography, ice flow, and
surrounding ocean of Antarctica and Greenland, and is scheduled to demonstrate it
to the public at InnoCarnival 2026 (Hong Kong Science Park, October–November
2026). F. Boeira Dias, who provided the WAOM2 output that 3D ICE renders, has
described using the explorer to visualise ocean circulation that is difficult to
interpret in standard two-dimensional plots [@aapp2026;
@spatialsource2026]. From its launch on 22 March to 27 September 2026, the public
site recorded 578 active users and about 1,200 page views.

3D ICE integrates representative products for bed geometry
[@morlighem2020; @pritchard2025; @morlighem2017; @palmer2025], ice velocity
[@mouginot2019; @gardner2018; @gardner2024], ocean circulation [@richter2022;
@dias2023; @cmems2025], ice-shelf basal melting [@galtonfenzi2025], basal
friction [@jager2026antarctica; @jager2026greenland; @jager2026aisefi;
@jager2026grisefi], subglacial hydrology [@werder2013; @ehrenfeucht2025;
@ehrenfeucht2024], drainage boundaries [@rignot2011; @mouginot2017boundaries;
@mouginot2019basins], and polar places [@scar_gazetteer; @comnap2024].

The repository supplies a specified data contract, a documented preparation
pipeline, tagged releases with a changelog and a compatibility bundle
[@wang2026software], contribution and support pathways, and worked examples that
run from a clean clone: one reproduces the explorer's Antarctic rebound
figures, including 2.94 million km² of today's sub-sea-level bed emerging above
the ice-free sea surface, and another regenerates six derived data packages
whose payloads match the committed files byte for byte.

# AI usage disclosure

Generative AI was used extensively in developing 3D ICE. The first prototype of
the explorer, and much of its subsequent application code, was generated with
the OpenAI Codex coding agent (GPT-5-family models, February to August 2026)
from Y. Wang's natural-language specifications, data, and review, as the
project's public coverage describes [@aapp2026]. Anthropic's Claude Code
(Claude Opus 4.6 in March and April 2026; Claude Opus 5 and Claude Opus 5.5 in
September and October 2026) was used for later work: parts of the interface,
including the public edition and the home pages; test infrastructure; the
isostatic-rebound and projection layers and their data pipelines; the
refactoring of the runtime into shared, tested modules; the repository
documentation; reference checking; and the drafting and copy-editing of this
paper. Y. Wang selected the datasets and scientific
methods and set the architecture. Every AI-generated step was checked by hand
before it was kept: Y. Wang read each change, ran the explorer to confirm its
behaviour, validated each scientific layer against its source product, and
checked every reference against its source. The automated checks described
above back this review. The authors accept responsibility for the
accuracy, originality, licensing, and integrity of the submitted work.

# Author contributions

Y. Wang conceived, designed, and developed 3D ICE and wrote the paper. Y. Lin
helped debug and evaluate the explorer, advised on the isostatic-rebound and
sea-level calculations and on the interface design for public audiences, and
joined discussions of its development.

# Acknowledgements

3D ICE has received no dedicated funding. It was begun at the Institute for
Marine and Antarctic Studies, University of Tasmania, while Y. Wang
held an Australian Antarctic Program Partnership top-up scholarship and a
Tasmanian Graduate Research Scholarship, and continued at the Climate Systems
Engineering initiative, University of Chicago. The authors gratefully acknowledge
the providers of the data that make 3D ICE possible: the National Snow and Ice
Data Center, the UK Polar Data Centre, ITS_LIVE, the Copernicus Marine Service,
the Australian Antarctic Division, the NSF Arctic Data Center, SCAR, COMNAP, and
the authors of the QRF, WAOM2, GlaDS, and basal-friction products. They also
acknowledge the World Climate Research Programme for coordinating CMIP6, the
climate and ice-sheet modelling groups, the Earth System Grid Federation, and
ISMIP6 and Ghub for the projections.

# References
