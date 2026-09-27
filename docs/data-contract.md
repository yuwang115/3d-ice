# Data contract

Everything the browser draws, apart from the computed isostatic-rebound layer, comes from a
**data package**: a pair of files in `static/tools/data/` with the same base name.

| File | Content |
| --- | --- |
| `<name>.bin` | Little-endian binary payload: the package's stored fields, concatenated in declared order |
| `<name>.meta.json` | UTF-8 JSON: grid geometry, the location and encoding of every field, statistics, and provenance |

The Python scripts in `scripts/` write both files. `static/tools/js/data-contract.js` is the
browser decoder, and `tests/js/data-contract.test.mjs` decodes every committed package with
it and checks the result against the rules below. The browser never needs to know which
script produced a package: the metadata is the whole interface.

## Package families

- **Gridded packages** (terrain, ice velocity, basal friction, subglacial hydrology, RISE)
  carry a `grid` block, and their gridded fields hold one value per grid cell.
- **Streamline packages** (ocean currents) set `"geometry_type": "streamlines_3d"` and have
  no grid; their fields hold one value per streamline segment.

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

The explorer's metadata panel shows these records, and each data layer links back to its
source product.

## Packages the runtime does not load

The combined Antarctic ocean-current package lists its inputs in
`sampling.component_datasets`: `..._cavity_margin80km` and `..._remote_open_ocean`. They
are kept so that `combine_antarctica_ocean_current_datasets.py` can rebuild the combined
package without repeating the streamline integration. Two further Antarctic streamline
packages, `..._yr5_annual` and `..._cavity_margin50km`, are alternative seeding
configurations. None of these four is loaded by the explorer.

## Adding a package

1. Write the `.bin` payload and `.meta.json` following the rules above, from a script in
   `scripts/` that records the source file, product version and reference.
2. Register the package URLs for the relevant region and dataset in
   `static/tools/js/explorer-app.js`.
3. Run `npm run test:data-contract`. The contract test discovers every package in
   `static/tools/data/`, so the new one is checked without further changes; so is the
   schema test, `tests/test_metadata_schema.py`.
