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
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Institute for Marine and Antarctic Studies, University of Tasmania, Hobart, Tasmania, Australia
    index: 1
date: 17 April 2026
bibliography: paper.bib
---

# Summary

3D ICE (Interactive 3D Cryosphere Explorer) is a browser-based platform that
transforms authoritative cryosphere datasets into layered, interactive 3D
visualizations of the Antarctic and Greenland ice sheets. The software consists
of two components: an offline Python data preparation pipeline that converts
heterogeneous scientific data products (NetCDF4 / HDF5) into optimized,
web-ready binary packages with full provenance metadata; and an online
JavaScript runtime built on Three.js [@threejs] and WebGL that renders these
packages as explorable 3D scenes directly in the browser, requiring no
installation, plugins, or server-side computation.

3D ICE currently integrates seven major data products spanning bed topography
[@morlighem2020; @morlighem2017], ice velocity [@mouginot2019; @gardner2018;
@gardner2025], ocean circulation [@richter2022; @dias2023; @cmems2025],
ice-shelf basal melt rates [@galtonfenzi2025], basal friction, subglacial
hydrology [@werder2013; @ehrenfeucht2024], and drainage basin boundaries
[@rignot2011]. Users can rotate, zoom, and layer these datasets in real time
while maintaining direct links to the underlying source data for further
research.

# Statement of Need

Modern cryosphere science generates large, gridded datasets in specialized
formats such as NetCDF4 and HDF5. Visualizing these datasets typically requires
desktop GIS software (e.g., QGIS with the Quantarctica data package), numerical
computing environments (MATLAB, Python with matplotlib), or bespoke analysis
scripts. Each of these tools demands installation, domain expertise, and
familiarity with polar stereographic projections and ice-sheet coordinate
systems. This creates a significant barrier for three important audiences:
educators who wish to incorporate real ice-sheet data into teaching;
science communicators who need compelling visuals for public engagement; and
researchers in adjacent disciplines who want a quick, contextual overview of
cryosphere conditions without setting up a full analysis environment.

Existing web-based tools address parts of this problem but leave gaps.
NASA Worldview provides global satellite imagery in two dimensions but offers no
3D rendering or cryosphere-specific overlays. The Quantarctica project assembles
a comprehensive Antarctic GIS dataset, but it is tied to desktop QGIS.
The NSIDC IceBridge Data Portal and the Australian Antarctic Data Centre focus
on data access and archival rather than interactive exploration. None of these
tools allow users to combine bed topography, surface velocity, ocean currents,
basal melt, and subglacial hydrology into a single, layered 3D view—and none
run entirely in the browser with zero installation.

3D ICE fills this gap by coupling a reproducible data preparation pipeline with
an accessible, zero-install 3D visualization runtime. The platform is designed
for three use cases: (1) research communication, where scientists can share
interactive views of their datasets via URL; (2) teaching, where educators can
walk students through ice-sheet structure and dynamics using real data; and
(3) public engagement, where the 3D visual immediacy of ice sheets lowers the
barrier to understanding cryosphere change.

# Architecture and Implementation

## Data Preparation Pipeline

The offline pipeline consists of twelve Python scripts (approximately 7,100
lines) that read authoritative source products and emit paired `.bin` /
`.meta.json` files. Each script performs four operations: (1) reading the source
NetCDF4 or HDF5 file via h5py or netCDF4-python; (2) resampling the data onto
a target grid aligned with BedMachine coordinates, using nearest-neighbor or
bilinear interpolation; (3) quantizing floating-point values to int16 with
configurable scale and offset, mapping invalid or missing data to a sentinel
fill value; and (4) writing the binary payload alongside a JSON metadata file
that records source provenance, grid geometry, quantization parameters, and
per-field statistics.

The most algorithmically complex component is the ocean current streamline
generator (`prepare_antarctica_ocean_currents.py`, 2,551 lines), which
implements Lagrangian particle advection through a 3D velocity field. The
generator seeds particles using a spatially balanced, depth-stratified strategy
with configurable sector weighting, traces streamlines through multi-depth
velocity layers with adaptive step sizing, and applies spatial binning to
balance visual density. This approach produces visually coherent streamlines
that convey both the horizontal circulation patterns and the vertical structure
of Antarctic shelf and cavity currents derived from WAOM2 [@richter2022;
@dias2023].

Each `.meta.json` file serves as a machine-readable provenance record, enabling
downstream users to trace any rendered pixel back to its source variable,
spatial resolution, quantization scheme, and original dataset DOI.

## Browser Runtime

The runtime is a self-contained HTML application (approximately 10,400 lines)
that loads binary data packages via `fetch()` and `ArrayBuffer`, constructs
Three.js geometries from the quantized grids, and renders layered 3D scenes
with orbit controls. Heavy geometry construction is offloaded to a dedicated
Web Worker to avoid blocking the main thread during initial loading. The runtime
supports theme-aware rendering (light and dark modes), bilingual localization
(English and Chinese), cross-device optimization with selectable quality presets
(Balanced and HD), and URL-based deep linking to specific regions and layer
configurations.

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
