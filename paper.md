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
date: 24 September 2026
bibliography: paper.bib
---

# Summary

3D ICE is an open-source platform for exploring research-grade representations
of the Antarctic and Greenland ice sheets in a web browser
(\autoref{fig:overview}). It combines terrain, ice velocity, ocean circulation,
ice-shelf basal melting, basal friction, subglacial hydrology, and drainage
basins in layered three-dimensional scenes. Users can rotate and zoom each ice
sheet, switch datasets and quality levels, search polar places and geographic
features, compute the ice-free bed after isostatic rebound, and follow links to
the original data products.

The software has two stages. An offline Python pipeline converts heterogeneous
NetCDF, HDF5, GeoTIFF, and vector products into compact binary arrays paired
with machine-readable provenance metadata. A static JavaScript application,
built with Three.js [@threejs] and WebGL, decodes these packages and renders
them without plugins or server-side computation. This separation makes the
scientific transformations reproducible while keeping the public interface
simple to deploy, share, and use on desktop and mobile devices.

![The 3D ICE explorer. (a) The interface, with Antarctic ice-surface speed [@mouginot2019] and surface-layer WAOM2 ocean streamlines [@richter2022] over BedMachine Antarctica v4 [@morlighem2020]. (b) Greenland ice-surface speed from ITS_LIVE [@gardner2024] over BedMachine Greenland v6 [@morlighem2017]. (c) The ice-free Antarctic bed that the explorer computes after complete isostatic rebound under regional flexure. Vertical scales are exaggerated.\label{fig:overview}](docs/images/explorer-overview.jpg)

# Statement of need

Modern cryosphere science generates large, gridded datasets in specialized
formats and polar stereographic coordinate systems. Inspecting several products
together commonly requires a desktop GIS, a numerical computing environment,
or purpose-built scripts. These workflows are appropriate for quantitative
analysis but impose installation, data-transfer, projection, and preprocessing
requirements before a user can obtain a contextual view of an ice sheet.

That barrier particularly affects researchers in adjacent disciplines,
educators who want to use real polar data in teaching, and scientists preparing
interactive research communication. They need a rapid way to see relationships
among ice geometry, flow, ocean forcing, and basal processes while retaining a
path back to the source datasets. 3D ICE addresses this need with a curated,
zero-install view rather than attempting to replace GIS or numerical analysis.
Its research purpose is to make heterogeneous cryosphere products easier to
compare, explain, and select for subsequent analysis.

# State of the field

Quantarctica is a comprehensive Antarctic data package, analysis environment,
and visualization platform centred on a desktop QGIS workflow
[@matsuoka2021], and QGreenland is its counterpart QGIS data package for
Greenland [@moon2023qgreenland]. Both are well suited to local geospatial
analysis and span a broader range of disciplines than 3D ICE. In contrast, 3D ICE trades
general GIS operations for an immediately shareable, curated 3D experience that
uses a common interaction model for both Antarctica and Greenland.

NASA Worldview provides rapid web access to more than a thousand global
satellite-imagery layers, including polar views, temporal comparison, animation,
and data download [@nasa_worldview]. Its focus is two-dimensional, often
near-real-time Earth observation imagery. 3D ICE instead combines terrain with
subsurface and model-derived fields such as basal friction, hydrology, basal
melt, and depth-dependent ocean circulation.

General 3D geospatial libraries such as CesiumJS [@cesiumjs] provide a
high-precision WGS84 globe, scalable data formats, and rendering primitives from
which developers can build web applications. Three.js provides the lower-level
graphics foundation used by 3D ICE [@threejs]. Neither library supplies a
cryosphere data model, polar-grid conversion, provenance records, layer
semantics, or data curation. Contributing these product-specific transformations
to a general rendering engine would therefore not address the research need.
3D ICE reuses established graphics infrastructure while concentrating its
scholarly contribution in the reproducible polar-data pipeline, explicit data
contract, and domain-specific interaction design.

# Software design

3D ICE uses a two-stage, contract-oriented architecture. The offline stage reads
authoritative source products, reconciles their coordinate conventions, and
resamples fields onto terrain-aligned polar grids with NumPy [@harris2020].
Floating-point grids are quantized to signed 16-bit arrays for browser delivery.
This introduces a controlled precision-versus-transfer-size trade-off: scale,
offset, fill value, units, grid geometry, statistics, source citation, and
processing provenance are stored in a paired `.meta.json` file so that decoding
is deterministic and the loss of precision is inspectable. The metadata schema,
specified in the repository, is the boundary between scientific preparation and
visualization. A single decoder module reads it in the page, in the geometry
worker, and in a test that decodes every committed package and checks it against
the statistics the pipeline recorded before quantizing: every quantized field
reproduces them to within half a quantization step.

Computationally expensive transformations are performed before deployment.
For example, Antarctic ocean-current packages are generated by advecting
particles through multi-depth WAOM2 velocity fields, balancing seeds by region
and depth, and controlling streamline density [@richter2022; @dias2023]. This
offline choice leaves the public application needing only static hosting, with
no database, application server, or restricted compute resources.
The trade-off is that integrating or updating a source product requires
regenerating and versioning its web package.

The online stage fetches binary arrays, reconstructs terrain and overlays, and
renders them with Three.js. Geometry construction runs in Web Workers so that
large grids do not block interface updates. Balanced and HD packages make the
resolution-versus-memory choice explicit across mobile and desktop devices. The
English and Chinese pages are thin shells around one shared runtime; logic that
needs neither the page nor the scene, such as package decoding, place search,
label styling, and refined-basin validation, lives in separate modules that run
unchanged under Node's test runner. URL parameters select the region, terrain
dataset, and display mode, and every data layer links to its source product.

One layer is computed rather than loaded. The isostatic-rebound layer solves for
the equilibrium deflection of the solid Earth after the present ice load is
removed, showing which parts of today's sub-sea-level bed would emerge once
glacial isostatic adjustment is complete. It
offers an elastic-lithosphere, relaxed-asthenosphere steady state
[@lemeur1996; @lingle1985] solved spectrally [@bueler2007] and, as an upper
bound on peak uplift, local Airy isostasy [@turcotte2002]; the spectral solver is
verified against the analytic point-load solution in Kelvin functions
[@brotchie1969]. Sea-level-equivalent figures follow the terminology and ocean
area of @gregory2019. Computed in the browser from the loaded terrain, the layer
adds no data dependency. The interface
reports its assumptions alongside its numbers, including that the present bed is
not in balance with the present load [@whitehouse2019] and that low mantle
viscosity beneath parts of West Antarctica shortens the response time by orders
of magnitude [@barletta2018].

The domain contract is checked at several levels: unit tests cover
quantization, coordinate sampling, metadata statistics, search, the rebound
solver, and bounded geometry; the contract test covers every committed package;
integration tests regenerate derived packages from committed inputs and compare
them with the committed files; compatibility tests build a distributable static
bundle; and Playwright tests load the explorer and exercise browser
interactions. These checks run in continuous integration and locally, without
the production website.

# Research impact statement

Y. Wang uses 3D ICE to compare candidate bed, velocity, and basal-friction
products before configuring ice-flow and subglacial-hydrology simulations, and
has used views generated with it in a research funding proposal. It has been
demonstrated in talks at the Antarctic Research Centre, Victoria University of
Wellington (February 2026), the ISMIP7 Workshop in Copenhagen (March 2026,
presented by C. Zhao), the Asia Early Career Polar Forum in Zhuhai (June 2026),
and the School of Oceanography, Shanghai Jiao Tong University (July 2026).

F. Boeira Dias, who provided the WAOM2 output that 3D ICE renders, has described
using the explorer to visualise ocean circulation that is difficult to interpret
in standard two-dimensional plots, in coverage of the project by the Australian
Antarctic Program Partnership and independent geospatial media [@aapp2026;
@spatialsource2026]. Y. Lin is scheduled to demonstrate it to the public at
InnoCarnival 2026 (Hong Kong Science Park, October–November 2026). Between 15 April and
[END DATE] 2026 the public site recorded [N] active users in [M] countries and
regions[, TEACHING-USE EVIDENCE].

The public application and versioned compatibility bundle [@wang2026software]
integrate representative products for bed geometry [@morlighem2020;
@pritchard2025; @morlighem2017; @palmer2025], ice velocity [@mouginot2019;
@gardner2018; @gardner2024], ocean circulation [@richter2022; @dias2023;
@cmems2025], ice-shelf basal melting [@galtonfenzi2025], basal friction
[@jager2026antarctica; @jager2026greenland;
@jager2026aisefi; @jager2026grisefi], subglacial hydrology
[@werder2013; @ehrenfeucht2025; @ehrenfeucht2024], drainage boundaries
[@rignot2011; @mouginot2017boundaries; @mouginot2019basins], and polar places
[@scar_gazetteer; @comnap2024].

The repository supplies a specified data contract, a documented preparation
pipeline, tagged releases with a changelog, contribution and support pathways,
and worked examples that run from a clean clone. One reproduces the rebound figures the
explorer reports for Antarctica, including 3.19 million km² of today's
sub-sea-level bed emerging after complete rebound; another regenerates six
derived data packages whose payloads match the committed files byte for byte.

# AI usage disclosure

Generative AI was used extensively in developing 3D ICE. The first prototype of
the explorer, and much of its subsequent application code, was generated with
the OpenAI Codex coding agent (GPT-5-family models, February to August 2026)
from Y. Wang's natural-language specifications, data, and review, as
described in the project's public coverage [@aapp2026]. Anthropic's Claude Code
(Claude Opus 4.6 in March and April 2026; Claude Opus 5 and Claude Opus 5.5 in
September 2026) was used for later work: parts of the interface, test
infrastructure, the refactoring that moved the runtime into shared and tested
modules, the repository documentation, reference checking, and the drafting and
copy-editing of this paper. Y. Wang selected the datasets and scientific
methods, set the architecture, specified and reviewed every change, validated
the scientific layers against their source products and the rebound solver
against analytic solutions, ran the test suites and the JOSS build, and checked
every reference against its source. The authors accept responsibility for the
accuracy, originality, licensing, and integrity of the submitted work.

# Author contributions

Y. Wang conceived, designed, and developed 3D ICE and wrote the paper. Y. Lin
helped debug and evaluate the explorer, advised on the isostatic-rebound and
sea-level calculations and on the interface design for public audiences, and
joined discussions of its development.

# Acknowledgements

3D ICE has received no dedicated funding. It was begun at the Institute for
Marine and Antarctic Studies (IMAS), University of Tasmania, while Y. Wang
held an Australian Antarctic Program Partnership (AAPP) top-up scholarship and a
Tasmanian Graduate Research Scholarship, and continued at the Climate Systems
Engineering initiative, University of Chicago. The authors gratefully acknowledge
the providers of the data that make 3D ICE possible: the National Snow and Ice
Data Center (NSIDC) for BedMachine and MEaSUREs datasets, the UK Polar Data
Centre for Bedmap3, the ITS_LIVE project for ice velocity mosaics, the
Copernicus Marine Service for Arctic ocean analysis products, the Australian
Antarctic Division for the RISE basal melt compilation, the authors of the QRF
Greenland topography and of the WAOM2, GlaDS, and basal-friction model outputs,
and SCAR and COMNAP for the place-name and station catalogues.

# References
