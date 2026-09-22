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
    affiliation: 1
affiliations:
  - name: Institute for Marine and Antarctic Studies, University of Tasmania, Hobart, Tasmania, Australia
    index: 1
date: 17 August 2026
bibliography: paper.bib
---

# Summary

3D ICE (Interactive 3D Cryosphere Explorer) is an open-source platform for
exploring research-grade representations of the Antarctic and Greenland ice
sheets in a web browser. It combines terrain, ice velocity, ocean circulation,
ice-shelf basal melting, basal conditions, subglacial hydrology, and drainage
basins in layered three-dimensional scenes. Users can rotate and zoom each ice
sheet, switch datasets and quality levels, search polar places and geographic
features, and follow links to the original data products.

The software has two stages. An offline Python pipeline converts heterogeneous
NetCDF, HDF5, GeoTIFF, and vector products into compact binary arrays paired
with machine-readable provenance metadata. A static JavaScript application,
built with Three.js [@threejs] and WebGL, decodes these packages and renders
them without plugins or server-side computation. This separation makes the
scientific transformations reproducible while keeping the public interface
simple to deploy, share, and use on desktop and mobile devices.

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
[@matsuoka2021]. It is well suited to local geospatial analysis and contains a
broader range of Antarctic disciplines than 3D ICE. In contrast, 3D ICE trades
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
resamples fields onto terrain-aligned polar grids. Floating-point grids are
quantized to signed 16-bit arrays for browser delivery. This introduces a
controlled precision-versus-transfer-size trade-off: scale, offset, fill value,
units, grid geometry, statistics, source citation, and processing provenance are
stored in a paired `.meta.json` file so that decoding is deterministic and the
loss of precision is inspectable. The metadata schema is the boundary between
scientific preparation and visualization, rather than implicit assumptions in
the renderer.

Computationally expensive transformations are performed before deployment.
For example, Antarctic ocean-current packages are generated by advecting
particles through multi-depth WAOM2 velocity fields, balancing seeds by region
and depth, and controlling streamline density [@richter2022; @dias2023]. This
offline choice allows the public application to use static hosting: it does not
need a database, application server, or access to restricted compute resources.
The trade-off is that integrating or updating a source product requires
regenerating and versioning its web package.

The online stage fetches binary arrays, reconstructs terrain and overlays, and
renders them with Three.js. Geometry construction runs in a Web Worker so that
large grids do not block interface updates. Balanced and HD packages make the
resolution-versus-memory choice explicit across mobile and desktop devices.
Pure JavaScript modules isolate polar search, label styling, and refined-basin
validation from the page interface, while URL state, English/Chinese
localization, and direct source links make views reproducible and shareable.

The domain contract is checked at several levels: unit tests cover
quantization, coordinate sampling, metadata statistics, search, and bounded
geometry; integration tests exercise prepared datasets; compatibility tests
build a distributable static bundle; and Playwright tests load the explorer and
exercise browser interactions. These checks run in continuous integration and
make the web application locally verifiable without relying on the production
website.

# Research impact statement

At the current public-release stage, evidence for 3D ICE is based on
reproducible research materials and community readiness rather than downstream
publication citations. The public application and versioned compatibility
bundle [@wang2026software] integrate representative products for bed geometry
[@morlighem2020; @morlighem2017], ice velocity [@mouginot2019; @gardner2018;
@gardner2025], ocean circulation [@richter2022; @dias2023; @cmems2025],
ice-shelf basal melting [@galtonfenzi2025], subglacial hydrology
[@werder2013; @ehrenfeucht2024], and drainage boundaries [@rignot2011]. Each
visual layer retains a citation or link to its source, allowing an exploratory
view to lead into a reproducible analysis workflow.

One layer is derived rather than ingested. The isostatic-rebound view solves the
equilibrium deflection of the solid Earth after the present ice load is removed,
so a user can see which parts of the bed that lie below sea level today would
emerge once the ice is gone and glacial isostatic adjustment has run to
completion. It offers an elastic-lithosphere/relaxing-asthenosphere steady state
[@lemeur1996; @lingle1985] solved spectrally [@bueler2007] and, as an upper
bound on peak uplift, local Airy isostasy [@turcotte2002]; the spectral solver
is verified against the analytic point-load Kelvin-function solution
[@brotchie1969]. Sea-level-equivalent figures follow the terminology and ocean
area of @gregory2019. Because this layer is computed in the browser from the
bed, surface, thickness and mask fields of whichever terrain package is already
loaded, it adds no data dependency and can be reproduced from a clean clone. Its
assumptions are reported alongside its numbers in the interface, including the
fact that the present bed is not in balance with the present load
[@whitehouse2019] and that mantle viscosity beneath parts of West Antarctica is
low enough to shorten the response time by orders of magnitude [@barletta2018].

The repository supplies English and Chinese interfaces, cross-device quality
levels, a documented preparation environment, tagged releases, contribution
and support pathways, and automated verification from data functions through
browser behaviour. Together these materials provide a concrete basis for reuse
in research communication, teaching, and dataset discovery. External use,
presentations, and feedback will be documented as that evidence becomes
available rather than inferred from intended audiences.

# AI usage disclosure

OpenAI Codex using GPT-5 (accessed August 2026) assisted with CI and metadata
review, test scaffolding, documentation editing, literature discovery, and the
drafting and copy-editing of portions of this paper. The author made the
scientific and architectural decisions; reviewed and edited all AI-assisted
code and prose; executed the automated tests and JOSS build; checked citations
against primary sources; and accepts responsibility for the accuracy,
originality, licensing, and integrity of the submitted work.

# Acknowledgements

This work was supported by the Institute for Marine and Antarctic Studies (IMAS)
at the University of Tasmania and the Australian Antarctic Program Partnership
(AAPP). The author gratefully acknowledges the data providers whose products
make 3D ICE possible: the National Snow and Ice Data Center (NSIDC) for
BedMachine and MEaSUREs datasets, the ITS_LIVE project for ice velocity
mosaics, the Copernicus Marine Service for Arctic ocean analysis products, the
Australian Antarctic Division for the RISE basal melt compilation, and the
authors of the WAOM2 and GlaDS datasets for making their model outputs openly
available.

# References
