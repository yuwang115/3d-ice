# Regenerating the data packages

The prepared packages in `static/tools/data/` are committed, so the explorer, the tests and
the [worked examples](example.md) need no upstream data. This page is for rebuilding
packages from their source products. Each package's `.meta.json` records the source file,
product version and processing settings it was built from; the format is described in
[data-contract.md](data-contract.md).

Set up the Python environment first (Python 3.10 or newer):

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Every script writes into `static/tools/data/` by default and **overwrites the committed
package of the same name**. Pass `--output-dir` (or `--data-dir`) to write elsewhere and
compare before replacing anything. Run the tests afterwards: the contract test checks every
package in the directory.

## Steps that need only the repository

These steps transform packages that are already committed. They reproduce the committed
payloads byte for byte, and the metadata exactly apart from the last digits of some
floating-point statistics, which depend on the NumPy version.

| Command | Rebuilds |
| --- | --- |
| `python scripts/prepare_bedmap3_antarctica_overlays.py` | The six Bedmap3 overlay packages ([example 2](example.md#2-regenerating-bedmap3-overlay-packages-from-the-repository)) |
| `python scripts/combine_antarctica_ocean_current_datasets.py` | The combined Antarctic ocean-current package, from its two component packages |
| `npm run prepare:refined-basin-search` | `*_refined_basins_search.json`, from the committed basin files |

## Steps that start from upstream products

| Packages | Source product and file | Where to get it | Command |
| --- | --- | --- | --- |
| `bedmachine_antarctica_v4_480` (10 km), `_741` (4 km) | BedMachine Antarctica v4, `BedMachineAntarctica_V4.nc` | [NSIDC-0756 v4](https://nsidc.org/data/NSIDC-0756/versions/4) (NASA Earthdata login) | `python scripts/prepare_bedmachine_antarctica.py --input BedMachineAntarctica_V4.nc`; HD: add `--step 8 --basename bedmachine_antarctica_v4_741` |
| `bedmap3_antarctica_10km`, `_4km` | Bedmap3 v1.0, `bm3_bed.tif`, `bm3_surface.tif`, `bm3_thickness.tif`, `bm3_masks.tif` | [UK Polar Data Centre](https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2) (CC BY 4.0) | `python scripts/prepare_bedmap3_antarctica.py --input-dir <dir>`; HD: add `--resolution-m 4000 --basename bedmap3_antarctica_4km` |
| `bedmachine_greenland_v6_3km`, `_1km` | BedMachine Greenland v6, `BedMachineGreenland-v6.nc` | [NSIDC IDBMG4 v6](https://nsidc.org/data/idbmg4/versions/6) (NASA Earthdata login) | `python scripts/prepare_bedmachine_greenland.py --input BedMachineGreenland-v6.nc`; HD: add `--resolution-m 1000 --basename bedmachine_greenland_v6_1km` |
| `greenland_qrf_2025_3km`, `_1km` | QRF Greenland subglacial topography, `QRF_greenland_ice_predictions_300m.tif` | [charliekirkwood/greenlandice at 5fba8ad](https://raw.githubusercontent.com/charliekirkwood/greenlandice/5fba8ad8332752ed3b780f15ac5f70580fe0acaf/QRF_greenland_ice_predictions_300m.tif) (no data licence stated upstream) | `python scripts/prepare_qrf_greenland.py --input QRF_greenland_ice_predictions_300m.tif` (needs the BedMachine Greenland packages) |
| `greenland_ice_velocity_3km`, `_1km` | ITS_LIVE v2.1 velocity mosaic, `ITS_LIVE_velocity_120m_RGI05A_0000_V02.1.nc` | [ITS_LIVE](http://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/production/ITS_LIVE_velocity_120m_RGI05A_0000_V02.1.nc) (public) | `python scripts/prepare_greenland_velocity.py --input <file>` (writes both resolutions) |
| `antarctic_ice_velocity_phase_v01_480`, `_741` | MEaSUREs phase-based Antarctic ice velocity v1 | [NSIDC-0754 v1](https://nsidc.org/data/NSIDC-0754/versions/1) (NASA Earthdata login) | `python scripts/prepare_antarctica_velocity.py --input <file>` (writes both resolutions) |
| `antarctica_subglacial_hydrology_480`, `_741` | GlaDS Antarctic subglacial hydrology, `Antarctica_SubglacialHydrology.nc` | [Zenodo 12738170](https://zenodo.org/records/12738170) (public) | `python scripts/prepare_subglacial_hydrology.py --input Antarctica_SubglacialHydrology.nc` (writes both resolutions) |
| `rise_antarctica_480`, `_741` | RISE multi-model mean, `RISE_MultiModelMean_Antarctica_v01.nc` | [Australian Antarctic Data Centre](https://data.aad.gov.au/metadata/RISE) (CC BY 4.0) | `python scripts/prepare_rise_antarctica.py --input RISE_MultiModelMean_Antarctica_v01.nc` (writes both resolutions) |
| `antarctica_basal_friction_*`, `greenland_basal_friction_*` | Ensemble-median basal shear stress (variable `taub`), `ens_med.nc` | [AISEFI](https://doi.org/10.5281/zenodo.18508904) (Antarctica) and [GrISEFI](https://doi.org/10.5281/zenodo.18508850) (Greenland) on Zenodo (CC BY 4.0), described in Jager et al. (2026) | `python scripts/prepare_basal_friction.py --input ens_med.nc --region antarctica`, or `--region greenland` with the Greenland file |
| `antarctica_ocean_currents_waom2_yr5_annual_*` | WAOM2 year-5 annual mean, `ocean_avg_yr5_annual.nc` | Output of the WAOM2 simulations of [Dias et al. (2023)](https://doi.org/10.3389/fmars.2023.1027704), whose data availability statement says the authors will provide the data without undue reservation; model described by [Richter et al. (2022)](https://gmd.copernicus.org/articles/15/617/2022/) | `python scripts/prepare_antarctica_ocean_currents.py --input ocean_avg_yr5_annual.nc --streamline-class <class>` with the settings in the package's `sampling` block, then the combine step above |
| `greenland_ocean_currents_cmems_202508` | Copernicus Marine Arctic Ocean physics, dataset `cmems_mod_arc_phy_anfc_6km_detided_P1M-m` | [Copernicus Marine](https://data.marine.copernicus.eu/product/ARCTIC_ANALYSISFORECAST_PHY_002_001/description) (free registration) | `python scripts/prepare_greenland_ocean_currents.py --input <file>` with the settings in the package's `sampling` block |
| `greenland_basins_ps_v1_4_2.json` | Greenland drainage basins, `Greenland_Basins_PS_v1.4.2.shp` | See the package's `source_shapefile` | `python scripts/prepare_greenland_basins.py --input Greenland_Basins_PS_v1.4.2.shp` |
| `*_research_stations.json`, `*_geographic_names.json` | COMNAP facilities list, INTERACT stations, SCAR Composite Gazetteer, Natural Earth, Greenland Place Names Register | Online catalogues listed in each file's `sources` | `python scripts/prepare_polar_features.py --as-of <date>` (network access) |

Notes:

- The basal-friction packages record their input as `taub_med.nc`, a local copy of the
  dataset's `ens_med.nc`.
- The ocean-current scripts integrate streamlines from random seeds. The seed, seeding
  strategy and every other setting are recorded in the package's `sampling` block, which is
  what a rebuild must pass to reproduce a package.
- `scripts/plot_antarctica_depth_averaged_ocean_speed.py` is a diagnostic plot of the WAOM2
  field, not a package builder.
