# Data contract

Everything the browser draws, apart from the idealised isostatic-rebound solves, comes from
a **data package**: a pair of files in `static/tools/data/` with the same base name.

| File | Content |
| --- | --- |
| `<name>.bin` | Little-endian binary payload: the package's stored fields, concatenated in declared order |
| `<name>.meta.json` | UTF-8 JSON: grid geometry, the location and encoding of every field, statistics, and provenance |

The Python scripts in `scripts/` write both files. `static/tools/js/data-contract.js` is the
browser decoder, and `tests/js/data-contract.test.mjs` decodes every committed package with
it and checks the result against the rules below. The browser never needs to know which
script produced a package: the metadata is the whole interface.

## Package families

- **Gridded packages** (terrain, ice velocity, basal friction, subglacial hydrology, RISE,
  isostatic response, surface elevation change rate) carry a `grid` block, and their gridded
  fields hold one value per grid cell.
- **Streamline packages** (ocean currents) set `"geometry_type": "streamlines_3d"` and have
  no grid; their fields hold one value per streamline segment.
- **Sparse time-series packages** (ice-sheet projections, observed surface elevation change,
  observed sea ice) set `"geometry_type": "sparse_grid_time_series"`. They carry a `grid`, but
  only a mask covers it; their other fields hold one value per domain cell, or per domain cell
  and keyframe (or calendar month). The observed records add a `time_series_kind`
  (`"observed_surface_change"` or `"observed_sea_ice_concentration"`), which names their fields.

## Grid

```json
"grid": { "nx": 667, "ny": 667, "x0_m": -3333000, "y0_m": 3333000, "dx_m": 10000, "dy_m": -10000 }
```

Coordinates are metres in the region's polar stereographic projection: EPSG:3031 for
Antarctica and EPSG:3413 for Greenland. `x0_m` and `y0_m` are the projected coordinates of
the first sample, taken from the source product's coordinate arrays, so sample
(`row`, `col`) lies at (`x0_m + col · dx_m`, `y0_m + row · dy_m`). `dy_m` is negative:
rows run from north to south in projected *y*. Gridded fields are stored row-major, `ny`
rows of `nx` values.

## Fields

`fields` is an array. An entry with a `dtype` is a **stored field**:

| Key | Meaning |
| --- | --- |
| `name` | Field name, unique within the package |
| `dtype` | `uint8`, `int16`, `uint16`, `int32` or `float32` |
| `byte_offset`, `byte_length` | Location of the field in the `.bin` payload |
| `unit` | Physical unit of the decoded values (optional) |
| `stats_<unit>` | `min`, `max` and `mean` of the valid source values, computed before quantization |
| `quantiles_<unit>` | Selected quantiles of the same values (optional) |
| `flags` | Code-to-class map for categorical `uint8` fields such as masks |
| `scale`, `offset`, `fill_value` | Field-level quantization of an `int16` field (see below) |

An entry without a `dtype` is a **summary** of a derived quantity that is not stored, for
example ice `speed` computed from `vx` and `vy`; it carries statistics only.

These layout rules hold for every committed package and are enforced by the contract test:

1. Stored fields tile the payload exactly: the first starts at byte 0, each starts where the
   previous one ends, and the last ends at the end of the file.
2. `byte_length` is a whole number of values of the field's `dtype`.
3. In gridded packages, every `int16` field and every mask has `nx · ny` values.
4. Offsets need not be aligned to the value size. The RISE packages place `int16` and
   `uint16` fields after a one-byte-per-cell mask, at odd offsets, so a decoder must read
   such fields through a `DataView` rather than a typed-array view.

## Quantization

`int16` fields store quantized codes. A code decodes to `code × scale + offset`, and the
fill code decodes to missing (`NaN`). The decoder takes `scale`, `offset` and the fill code
from the first place that defines them:

1. the field's own `scale`, `offset` and `fill_value`;
2. the package's `quantization` block: `scale`, `offset` and `int16_fill_value`;
3. the defaults `1`, `0` and `-32768`.

The preparation scripts clip source values to the code range ±32767, round half to even
(`numpy.rint`), and reserve −32768 for fill. A decoded value is therefore within half a
quantization step of its source value, and the decoded field reproduces the recorded
`min`, `max` and `mean` to within `scale / 2`. The contract test checks exactly that for
every `int16` field of every committed package, and checks that no recorded range exceeds
what the codes can represent, so no package clips.

| Package family | Quantized fields | Step |
| --- | --- | --- |
| Terrain (BedMachine, Bedmap3, QRF) | `bed`, `surface`, `thickness` | 1 m |
| Ice velocity | `vx`, `vy` | 1 m/yr |
| Subglacial hydrology | `effective_pressure` | 1000 Pa |
| RISE | `zice`; `ismr`; `tstar_zice` | 1 m; 0.001 m/yr; 0.0001 °C |
| Isostatic response | `topography_change`, `solid_surface_change`, `standard_deviation` | 0.1 m |
| Surface elevation change rate | `dhdt`, `dhdt_uncertainty` | 0.001 m/yr |
| Observed surface elevation change | `surface_change`, `final_uncertainty` | 0.01 m |

The observed sea-ice packages store whole percent in `uint8` fields, with `255` for no data.

`float32` fields are stored unquantized and mark missing values with `NaN`. Categorical
`uint8` fields carry their class codes in `flags`, which are authoritative for each
package. The terrain masks share `0` ocean, `1` ice-free land, `2` grounded ice and
`3` floating ice (Bedmap3 defines slightly broader classes, such as floating or
transiently grounded ice), and BedMachine Antarctica adds `4`, Lake Vostok.

## Family-specific fields

**Subglacial hydrology.** `effective_pressure` is stored only where the source value is
positive; cells with zero or negative effective pressure, or no data, hold the fill code.
Its package-level keys `effective_pressure_scale_pa_per_int16` and
`effective_pressure_offset_pa` duplicate the field-level quantization and are what the
geometry worker currently reads. `channel_col1`, `channel_row1`, `channel_col2` and
`channel_row2` (`uint16`) are the grid indices of each channel segment's two ends, and
`channel_discharge` (`float32`, m³/s) is its discharge after the filter recorded in
`channel_filter`.

**Isostatic response.** One package per terrain package, on exactly its grid, holding point
samples of the published grids of Paxman, Austermann & Hollyday (2022, grid files v3) at
the same source nodes the terrain was sampled from; `resampling.max_node_offset_m` records
how far they sit from a published node (0 m for Antarctica). `topography_change` (`T`) is the
published total isostatic response, the change in bed elevation relative to the sea
surface, so present bed + `T` is the fully rebounded bed above the ice-free sea surface;
`solid_surface_change` (`R`) is the solid-surface displacement, and the two differ by the
sea-surface change `G = R − T`; `standard_deviation` is the spread across the published
Earth-model suite. The three published components (ice unloading, post-LGM disequilibrium,
water loading), which sum to `T`, and `G` are not stored; `grounded_ice_summary_m` records
their statistics over the terrain's grounded-ice nodes. `earth_model` records the densities,
the elastic-thickness model and the 65.3 m eustatic term, `validation` the residuals of
`T = R − G` and of the component sum, and `source_dataset` the DOI, the object URL and the
MD5 the download was checked against. `source_package` names the terrain package whose grid
it shares; the QRF Greenland terrain has no package of its own and the explorer pairs it
with the BedMachine Greenland v6 one.

**Ice-sheet projections.** The fields are, in this order, `thickness` (`int16`, m,
`frames.count × domain.cell_count` values, frame-major: keyframe *k*, domain cell *i* is
value `k · cell_count + i`), an optional `speed_change` (`int16`, m/yr, frame-major like
`thickness`), `bed` (`int16`, m, one value per domain cell) and `domain_mask` (`uint8`,
`nx · ny`, `1` for domain cells). Domain cells are the mask's non-zero cells in row-major
order, and are the cells with ice in at least one keyframe. The `int16` fields come first so
that they start on even offsets. `frames.years` lists the keyframe years in increasing order;
`physical_constants.density_ratio` is ρ<sub>ice</sub>/ρ<sub>seawater</sub>, from which the browser rebuilds
`base = max(bed, −H · ratio)` and `surface = base + H`. `series` holds annual diagnostics:
`sea_level_contribution_m` always, the model spread `sea_level_contribution_min_m` and
`sea_level_contribution_max_m` for an ensemble package, or grounded and floating area for a
single-model one. The committed packages (`ismip6_2300_mean8_ae{05,10,14}_480`) are ISMIP6
Antarctica 2300 multi-model means: `thickness` is BedMachine v4 thickness plus the mean
change of eight models, `bed` is the BedMachine bed, `speed_change` the mean change in
depth-averaged ice speed since 2015 that the flowlines follow, and `series` the mean, minimum
and maximum of the models' published sea-level contribution. `experiment`, `models`,
`sources`, `method` and `validation` record the scenario, the models averaged, the citations
(CC BY 4.0), how the mean was formed, and how much of the mean volume change the packed frames
keep and the sea level their geometry implies. Rule 3 above does not apply to this family; the
contract test checks instead that the mask, bed and frame sizes agree with `domain` and
`frames`.

**Observed surface elevation change.** CPOM's Antarctic record from multi-mission radar
altimetry (product version 1.0, CC BY 4.0), resampled from its own 5 km EPSG:3031 grid by
an area-weighted mean onto the cells it covers by at least half, on explorer ice only.
`cpom_antarctica_dhdt_{480,741}` are gridded: `dhdt` and `dhdt_uncertainty` are the
1995–2026 linear trend and its 1σ uncertainty, and `basin_id` (`uint8`) the IMBIE drainage
basin (1–18, numbered and cited as in CPOM's `basin_id`) under each node, named in the field's `flags` as CPOM names
it ("West G-H"). `cpom_antarctica_dh_480` is a sparse time series whose
fields are, in this order, `surface_change` (`int16`, m, frame-major like the projections'
`thickness`), `final_uncertainty` (`int16`, m, one value per domain cell: the 1σ uncertainty
of the last epoch, because CPOM's per-epoch uncertainty is zero for most cells before 2002)
and `domain_mask`. Its frames are the 140-day epochs at their centre years, with their
windows in `frames.start_years` and `frames.end_years`; the change is CPOM's, relative to its
1995 reference level, and the first epoch is not zero. `series` carries CPOM's cumulative
mass change and its uncertainty (Gt) at every epoch, which the explorer quotes: under
`regions` for the ice sheet and the IMBIE regions, and under `basins`, keyed by `basin_id`,
for the 18 basins with their `code` and `region`; `validation.basin_sum_residual_gt` records
how closely each region's series is the sum of its basins'; `pole_hole` the radii of the extrapolated polar caps (81.5°S and
88°S) and the first year with CryoSat-2 data south of 81.5°S; `source_files` the name, SHA-256,
`tracking_id` and creation date of each CPOM file. Field-level `scale`, `offset` and
`fill_value` define the quantization.

**Observed sea-ice concentration.** `nsidc_sea_ice_antarctica` and `nsidc_sea_ice_greenland`
hold the monthly means of the NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice
Concentration, Version 6 (G02202, v06r00; Meier et al. 2026), from November 1978. Unlike every
other gridded package, their grid is not a terrain grid: it is a 25 km grid of its own in the
region's projection (`crs`, EPSG:3031 or EPSG:3413), with cell centres at odd multiples of
12.5 km, covering every terrain grid of the region. The explorer draws it as a texture on the
sea surface and clips it to the ocean (mask class `0`) of whichever terrain is shown, so one
package serves all of a region's terrains. Each cell centre is projected into NSIDC's grid
(`source_grid`: polar stereographic on the Hughes 1980 ellipsoid, EPSG:3412 or EPSG:3411) and
takes the bilinear mean of the NSIDC ocean cells around it; cells that a terrain lookup can read
but that NSIDC's land mask leaves without an ocean neighbour take the mean of their filled
neighbours (`resampling`, `validation.filled_cells`). The fields are, in this order,
`concentration` (`uint8`, whole percent 0–100, frame-major like the projections' `thickness`,
`255` throughout a month without data), `reference_ice_frequency` (`uint8`, percent of the
1981–2010 years in which the cell held at least 15 % ice, one frame per calendar month, January
first; its 50 % contour is the month's median ice edge) and `domain_mask` (the cells that hold
ice in at least one month and that a lookup on any terrain's ocean can read; every other cell is
open water throughout). The frames are consecutive months at mid-month: `frames.months` labels
them `YYYY-MM`, `frames.empty_months` lists those without data (December 1987 and January 1988
in both hemispheres, July 1984 in the Arctic), and `frames.sensors` gives the sensor of each run
of months (SMMR, SSM/I, SSMIS, AMSR2). `series` carries the extent and area of the whole
hemisphere every month, summed on NSIDC's grid with each cell's area from the projection's
scale factor (`null` for an empty month). The extent counts the ocean cells that the files' QA
flag marks as holding at least 15 % ice on at least half of the month's days, which is close to
the Sea Ice Index's monthly extent (an average of daily extents); the area weights the same
cells by their monthly mean concentration. `mean_field_extent_mkm2` is the area of the cells
whose monthly mean concentration reaches 15 %, the ice the map draws, which is larger. Under
`monthly` are each calendar month's 1981–2010 average extent and least-squares trend from
1979, in million km² per year and in percent of that average per decade. `validation` records how closely the projection
reproduces NSIDC's cell positions and how this extent compares with the Sea Ice Index's;
`source_files` the name, MD5 (checked against NSIDC's manifest), sensor and creation date of
every monthly file. Rule 3 above does not apply to this family.

**Streamlines.** Each segment stores its end points as `x0_ps_m`, `y0_ps_m`, `depth0_m`
and `x1_ps_m`, `y1_ps_m`, `depth1_m` (projected metres and depth in metres), the potential
temperature and salinity at each end (`theta0_c`, `sal0_psu`, `theta1_c`, `sal1_psu`), and a
`terminal_flag` (`uint8`) that is 1 on the last segment of each streamline. Segments are
stored streamline by streamline. `streamline_count`, `segment_count` and the seeding
configuration in `sampling` describe how the lines were generated.

## Provenance

Every package names the file it was built from (`source_file`, `source_files` or
`source_dataset`) and the product version (`product_version`), and, where the product
defines them, its reference, source URL and licence. Packages derived from another package
add `source_package` (the metadata and binary they were read from, and that package's grid)
and `resampling` (the method and the reason). Coverage counts, display hints
(`visualization`) and product-specific processing notes (`hybridization`,
`channel_filter`, `inference_note`) sit alongside.

The research edition's Data Snapshot summarises these records and links the source products
of the datasets in view; the public edition's layer explainers cite them.

## Packages the runtime does not load

The combined Antarctic ocean-current package lists its inputs in
`sampling.component_datasets`: `..._cavity_margin80km` and `..._remote_open_ocean`. They
are kept so that `combine_antarctica_ocean_current_datasets.py` can rebuild the combined
package without repeating the streamline integration. Neither is loaded by the explorer.
The phone-sized `..._combined_light` package, which the explorer does load on small
screens, is merged from lighter components that are not committed.

## Adding a package

1. Write the `.bin` payload and `.meta.json` following the rules above, from a script in
   `scripts/` that records the source file, product version and reference.
2. Register the package URLs for the relevant region and dataset in
   `static/tools/js/explorer-app.js`.
3. Run `npm run test:data-contract`. The contract test discovers every package in
   `static/tools/data/`, so the new one is checked without further changes; so is the
   schema test, `tests/test_metadata_schema.py`.
