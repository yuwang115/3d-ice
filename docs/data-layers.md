# Data layers

What each layer of the explorer shows, where it comes from and how it is prepared. Licences
and terms for every data package are in [DATA_LICENSES.md](../DATA_LICENSES.md); how to
rebuild each package is in [data-pipeline.md](data-pipeline.md).


| Region | Layers in the experience | Representative source products |
| --- | --- | --- |
| Antarctica | Bed topography, surface elevation, thickness, mask, grounding line, refined basins, velocity, basal friction, subglacial hydrology, ocean streamlines, basal melt, thermal driving, isostatic rebound, ice-sheet projections to 2300, observed surface elevation change 1995–2026, observed sea ice 1978–2026 | [BedMachine Antarctica v4](https://nsidc.org/data/NSIDC-0756/versions/4), [Bedmap3 v1.0](https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2) (CC BY 4.0), [MEaSUREs Antarctic Boundaries v2, refined basins](https://nsidc.org/data/NSIDC-0709/versions/2), [MEaSUREs Phase-Based Antarctica Velocity v1](https://nsidc.org/data/NSIDC-0754/versions/1), [AISEFI basal shear stress](https://doi.org/10.5281/zenodo.18508904) (CC BY 4.0; method: [Jager et al. 2026, preprint](https://essopenarchive.org/doi/full/10.22541/essoar.177099457.70593031/v1)), [GlaDS Antarctic subglacial hydrology](https://doi.org/10.5281/zenodo.12738170) (CC BY 4.0; [Ehrenfeucht et al. 2025](https://doi.org/10.1029/2024GL111386)), [WAOM2](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1027704/full), [RISE data](https://doi.org/10.26179/q0rh-gt74) (CC BY 4.0; [Galton-Fenzi et al. 2025](https://doi.org/10.5194/tc-19-6507-2025)), [Paxman et al. (2022)](https://doi.org/10.1038/s41598-022-15440-y) isostatic response, [grid files v3](https://doi.org/10.18739/A22Z12R8C) (CC BY 4.0), [ISMIP6 Antarctica 2300 projections](https://doi.org/10.5281/zenodo.13135599) (CC BY 4.0; [Seroussi et al. 2024](https://doi.org/10.1029/2024EF004561)), [CPOM surface elevation change](https://www.cpom.ucl.ac.uk/landice/) (CC BY 4.0; [Shepherd et al. 2019](https://doi.org/10.1029/2019GL082182)), [NOAA/NSIDC sea-ice concentration CDR v6](https://doi.org/10.7265/b18j-z797) |
| Greenland | Bed topography, surface elevation, thickness, mask, grounding line, basin boundaries, velocity, basal friction, ocean streamlines, isostatic rebound, observed sea ice 1978–2026 | [BedMachine Greenland v6](https://nsidc.org/data/idbmg4/versions/6) (v7 was released in October 2026), [QRF Greenland subglacial topography (2025)](https://doi.org/10.1017/jog.2025.10071), [ITS_LIVE V02.1 summary velocity mosaic, 2014–2024](https://its-live.jpl.nasa.gov/) (velocity data generated using auto-RIFT and provided by the NASA MEaSUREs ITS_LIVE project; archived as [NSIDC-0776 v2](https://nsidc.org/data/NSIDC-0776/versions/2)), [Greenland basins](https://doi.org/10.7280/D1WT11) (Mouginot & Rignot 2019, CC BY 4.0), [GrISEFI basal shear stress](https://doi.org/10.5281/zenodo.18508850) (CC BY 4.0; method: [Jager et al. 2026, preprint](https://essopenarchive.org/doi/full/10.22541/essoar.177099472.28419248/v1)), [Copernicus Marine Arctic Ocean Physics](https://doi.org/10.48670/moi-00001) (generated using E.U. Copernicus Marine Service Information), [Paxman et al. (2022)](https://doi.org/10.1038/s41598-022-15440-y) isostatic response, [grid files v3](https://doi.org/10.18739/A22Z12R8C) (CC BY 4.0), [NOAA/NSIDC sea-ice concentration CDR v6](https://doi.org/10.7265/b18j-z797) |

Bedmap3 is available as a 10 km Balanced or 4 km HD Antarctica terrain alternative. Both modes support velocity, flowlines, basal friction, effective pressure, subglacial channels, refined basins, and WAOM2 ocean streamlines. Its velocity, basal-friction and hydrology layers are nearest-cell copies of the BedMachine-aligned 10 km and 4 km packages, moved onto Bedmap3's grid, whose origin is offset by 250 m (subglacial-channel endpoints are reprojected); the projected WAOM2 streamlines are clipped against the active terrain at runtime. RISE basal melt and thermal-driving fields remain exclusive to BedMachine v4.

Greenland QRF subglacial topography (2025) is available as 3 km Balanced and 1 km HD terrain alternatives. The 300 m QRF GeoTIFF is sampled at pixel centres. The package replaces bed elevation only for grounded ice where the QRF prediction is valid, keeps BedMachine Greenland v6 surface elevation and mask, derives internally consistent thickness from those two fields, and falls back to BedMachine values over QRF gaps, ocean, and floating ice. Velocity, flowlines, basal friction, basins, and ocean streamlines reuse their existing BedMachine-aligned grids. The upstream data repository does not state a standalone data licence; see [DATA_LICENSES.md](../DATA_LICENSES.md).

## Ice-sheet projections (ISMIP6, to 2300)

The projection layer plays Antarctica forward from 2015 to 2300 under three ISMIP6
scenarios, all forced by UKESM1-0-LL: low emissions (SSP1-2.6), high emissions (SSP5-8.5),
and high emissions with ice-shelf collapse driven by surface melt. Each scenario is the
equal-weight mean of the eight ice sheet models that ran all three, out of the sixteen
groups in Seroussi et al. (2024). Every model's change in thickness and in depth-averaged
ice speed since 2015 is resampled conservatively onto the explorer's 10 km grid. The mean
change is then applied to today's BedMachine v4 ice, so the first frame is the ice already on
screen. Thickening is added as it is. Thinning is scaled by BedMachine thickness over the
models' own 2015 thickness, so each cell keeps the share of its ice that the models keep;
where the models average less than 10 m of ice, it is gone. Adding the thinning outright
would riddle the ice shelves with holes, because the models' shelves start out thicker than
BedMachine's in some cells and thinner in others.

The browser rebuilds the surface and base by flotation on the BedMachine bed. It colours the
ice by its thickness change and keeps the flowlines on: they ride the projected surface,
recolour and speed up with today's speed plus the mean change, and drop out where the ice
is gone. Switching the projection on switches the flowlines on, and switching it off
switches them off again unless they were changed in between. On the research pages *Ice flow →
Flowlines* is the one switch for them, and the projection, the isostatic rebound and the
observed records share the panel's *Change over time* group, one at a time. Each switches off
the layers drawn against today's ice or bed that it replaces, says so on their rows, and puts
them back, with the sea plane and flowlines as they were, when the reader returns to the
present day or moves straight to another; switching one of those layers on instead ends it,
and nothing else comes back. The public page keeps a switch at the end of the projection's
controls, the same layer as its *Ice flow* toggle.

The sea-level number and chart are the mean of the models' own published results, with
their range. Because flotation is not linear, the averaged ice on the map implies more
sea-level rise than that mean: by 2300 it is 0.15 m against 0.05 m (SSP1-2.6), 1.93 m
against 1.46 m (SSP5-8.5) and 3.12 m against 2.43 m (with collapse). The packages record
both figures. The packages,
the pipeline ([data-pipeline.md](data-pipeline.md)) and the browser module
(`static/tools/js/ice-projection.js`) are tested in `tests/test_prepare_ismip6_projection.py`,
`tests/test_ismip6_2300_hpc_scripts.py`, `tests/js/ice-projection.test.mjs` and
`tests/e2e/test_ice_projection.py`.

## Observed elevation change (CPOM, 1995–2026)

The research edition shows CPOM's record of Antarctic surface elevation change from
multi-mission radar altimetry (ERS-2, Envisat and CryoSat-2; product version 1.0, CC BY 4.0;
Shepherd et al. 2019) as a colour on today's ice surface, under *Change over time*. *Change
since 1995* plays the 81 cumulative 140-day epochs on the Balanced dataset; *Rate, 1995–2026*
shows the linear trend on the Balanced and HD datasets. CPOM's 5 km cells are averaged by
area onto each explorer cell they cover by at least half. The product covers grounded ice
only, so ice shelves, islands and ice rises have no colour. Changes within twice their
uncertainty are faded, a fade that can be switched off, and the polar cap that CPOM
extrapolates always is: south of 81.5°S until CryoSat-2 data arrive in late 2010, and south
of 88°S throughout. The readout quotes
CPOM's own mass change for the ice sheet and the IMBIE regions, not sums over the resampled
grid. Clicking the ice, or the basin list, picks one of CPOM's 18 IMBIE drainage basins
(numbered as in CPOM's `basin_id`) or a region: the map outlines it, and a chart plots its mass change since
1995 with its 1σ band, in both views. The layer is exclusive with the velocity colour and, as one of the
features of *Change over time*, with the projection, the observed sea ice and the rebound. The
packer (`scripts/prepare_cpom_land_ice.py`) and the browser module
(`static/tools/js/surface-change.js`) are tested in `tests/test_prepare_cpom_land_ice.py`,
`tests/js/surface-change.test.mjs` and `tests/e2e/test_surface_change.py`.

## Observed sea ice (NOAA/NSIDC, 1978–2026)

The research edition shows the NOAA/NSIDC climate data record of sea-ice concentration from
passive microwave radiometers (G02202 Version 6; Meier et al. 2026) on the sea surface around
Antarctica and Greenland, every month from November 1978, under *Change over time*. A month
slider shows the seasons and a year slider the same month across the years; Play runs the record
month by month. Water with less than 15 % ice is left clear, and an orange line marks the month's
1981–2010 median ice edge. The readout quotes the extent of the whole hemisphere from the same
record (the Greenland grid holds only part of the Arctic), its departure from the 1981–2010
average and its rank among the years, and a chart plots the month's extent in every year with its
least-squares trend. NSIDC's 25 km grids are resampled onto a 25 km grid in each region's own
projection and clipped to the terrain's ocean; near the coast, where NSIDC masks the cells as land,
the concentration comes from the nearest observed cells. As one of the features of *Change over
time*, the layer is exclusive with the projection, the observed elevation change and the rebound.
The packer (`scripts/prepare_nsidc_sea_ice.py`) and the browser module
(`static/tools/js/sea-ice.js`) are tested in `tests/test_prepare_nsidc_sea_ice.py`,
`tests/js/sea-ice.test.mjs` and `tests/e2e/test_sea_ice.py`.

## Isostatic rebound (ice-free equilibrium)

The isostatic-rebound layer answers "what would the bed look like with the ice gone and
rebound complete?" By default it shows the **published** total isostatic response of
Paxman, Austermann & Hollyday (2022), loaded from packages sampled at each terrain grid's
nodes. Two **idealised** responses, solved in the browser from the loaded terrain, stay
available as what-ifs.

**Published response (default).** Paxman et al. compute the fully re-equilibrated response
to removing both ice sheets: flexure of an elastic plate whose effective elastic thickness
varies laterally (Swain & Kirby 2021 for Antarctica, Steffen et al. 2018 for Greenland),
plus the post-LGM disequilibrium still to come (the mean of 24 self-gravitating
viscoelastic Earth models driven by ICE-6G_C) and the load of the seawater that floods the
rebounded bed. The sea surface rises by the 65.3 m eustatic contribution of both ice sheets
plus the residual post-LGM geoid change, 73–91 m in all above the grounded Antarctic ice and
43–51 m above Greenland's. Version 3 of their grid files ([doi:10.18739/A22Z12R8C](https://doi.org/10.18739/A22Z12R8C),
CC BY 4.0) is computed on the same BedMachine Antarctica v4, Bedmap3 and BedMachine
Greenland v6 grids the terrain packages are point-sampled from, so
`scripts/prepare_isostatic_response.py` samples it at exactly the terrain nodes, with no
interpolation, and writes one `*_isostatic_response_*` package per terrain package. Each
stores three fields at 0.1 m: the topography change `T = R − G` (the published total
response), the solid-surface displacement `R` and the Earth-model spread `σ`. The script
checks each download against the repository's MD5 and checks that the fields satisfy the
published identities `T = R − G` and `T = ice unloading + post-LGM + water loading`
(residuals below 1e-4 m).

The explorer draws the bed at `bed + T`, so heights and emergence read directly against the
ice-free sea surface however much that surface varies, and it reports uplift as `R`, as the
solver does. The sea-level datum slider is disabled in this mode because the model fixes its
own sea surface. On the 10 km BedMachine Antarctica v4 grid the peak solid-surface uplift is
1028.6 m, the peak rise above the ice-free sea surface 940.6 m, and 2.94 million km² of
today's sub-sea-level bed emerges; on the 3 km BedMachine Greenland v6 grid the figures are
829.0 m, 784.4 m and 0.394 million km². The QRF Greenland terrain has no published grid of
its own and borrows the BedMachine v6 response on the same grid; computing the flexure for
the QRF ice load instead changes it by about 10 m RMS, well inside the model spread.

**Idealised responses (what-ifs).** The solver below computes the equilibrium vertical
displacement of the solid Earth after the present ice load is removed, from the thin-plate
flexure equation

```
D grad^4 u + rho_m g u = sigma_now - sigma_after
```

solved in the spectral domain, where `u` is uplift (positive up), `D` is flexural
rigidity and `sigma` is the vertical stress the overburden applies to the bed. Two Earth
responses are offered: **regional flexure** (an elastic lithosphere over a fluid
asthenosphere, the ELRA steady state, `D = 1e25 N m`, flexural length scale 133 km) and
**local Airy isostasy** (`D = 0`), which bounds the peak uplift from above.

The present-day load is case-split by mask, which matters: grounded ice contributes
`rho_i g H`; a subglacial lake adds its own fresh-water column; and floating ice
contributes exactly the load of the seawater it displaces, so removing an ice shelf
produces no rebound at all. After deglaciation each column is either dry or flooded to
the chosen sea-level datum, but only where it still drains to the open ocean — basins that
rebound into closed hollows carry no marine water. Both the flooded depth and the flooded
footprint depend on the uplift, so the system is non-linear and is closed by Picard
iteration, which contracts at `rho_w / rho_m ~ 0.31` and converges to centimetre residuals
in about eight iterations.

**Numerical choices.** The deflection is band-limited near the flexural length scale, so
the transform is taken on a ~16–20 km grid and bicubically upsampled with half-cell
registration; against a native-resolution solve this moves the peak uplift by under 0.1 %.
The water load is evaluated against each coarse cell's sub-cell bathymetry rather than its
mean bed, which removes a Jensen bias worth roughly 3 m RMS of uplift and half a percent
of the emergent-area figure. Areas are integrated with the polar-stereographic point scale
factor: a nominally equal grid cell covers about 5.7 % more ground at the pole and about 8 %
less at 60°S. The test suite
checks the solver against the analytic point-load Kelvin-function solution: it agrees to
within 10⁻⁴ of the peak beyond half a flexural length, and to about 0.1 % under the load
itself, where the solve grid's Nyquist limit cuts off the kernel. Local isostasy reproduces
the closed-form Airy uplift exactly.

**What the idealised responses are not.** They are steady states, so they say where the bed
ends up and not how it gets there: neither is a transient GIA simulation nor a sea-level
projection. They assume the present bed is in balance with the present load, which it is
not: part of the post-LGM rebound is still to come (Paxman et al. put it at up to +68 m of
bed elevation under the Ross and Weddell embayments, while the collapsing Laurentide
forebulge lowers Greenland by up to 25 m), and the Amundsen Sea Embayment is rising at up to
41 mm/yr in response to recent ice loss over a low-viscosity mantle (Barletta et al. 2018).
They omit the sea-level equation, geoid change and rotational feedback, and replace real
lateral Earth structure with a single rigidity and mantle density. Against the published
response, regional flexure differs by 68 m RMS over grounded Antarctic ice and 69 m over
Greenland, mostly because of its uniform rigidity (`D = 1e25 N m` is an elastic thickness of
about 104 km, stiffer than any of Paxman et al.'s cases) and the missing post-LGM term. The
UI states these assumptions alongside the figures.

The scenario slider advances ice thinning and bed relaxation together, which is an
illustrative coupling rather than a simulated deglaciation path; for the published response
it has no time axis, since that response has no single relaxation time. Because every other
overlay is baked onto the present-day bed or ice surface, enabling this layer clears them.

**Cost.** The published response is one fetch per terrain package (2.7 MB on the Balanced
grids, 17–25 MB on the HD grids), decoded and summarised once and then cached. The
idealised solve runs once per (region, dataset, Earth response, sea-level datum) in a
dedicated module worker, and takes 0.2–0.9 s across the shipped packages; there is a
main-thread fallback for browsers without module workers. Moving the scenario slider
afterwards only rewrites vertex heights and colours from the cached uplift field, holding
~13 ms frames on the 10 km Antarctic grid.

**Where the code lives.** `scripts/prepare_isostatic_response.py` (published packages),
`static/tools/js/gia-rebound.js` (`summarisePublishedResponse` and the idealised physics),
`static/tools/js/gia-grid.js` (coarsening, upsampling, connectivity, area weighting),
`static/tools/js/fft2d.js` (transform) and `static/tools/gia-rebound-worker.js` (module
worker). Unit tests: `tests/test_prepare_isostatic_response.py` and
`tests/js/gia-rebound.test.mjs` (`npm run test:gia-rebound`). Browser tests:
`tests/e2e/test_isostatic_rebound.py`.

Key references: Paxman, Austermann & Hollyday (2022) for the published response, with
Swain & Kirby (2021) and Steffen et al. (2018) for the elastic-thickness models it uses;
Turcotte & Schubert (2002) for plate flexure; Le Meur & Huybrechts (1996) for the ELRA
formulation and parameter defaults; Lingle & Clark (1985) and Bueler et al. (2007) for the
deformable-Earth response and its spectral solution; Brotchie & Silvester (1969) for the
Kelvin-function validation case; Whitehouse et al. (2019) and Barletta et al. (2018) for
present-day Antarctic uplift and lateral viscosity structure.
