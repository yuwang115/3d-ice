# Data licences

The [MIT License](LICENSE) covers the 3D ICE source code and the scripts that prepare its
data packages. It does not cover the data packages in `static/tools/data/`, or images and
videos made from them. Each package is a resampled, quantized derivative of a third-party
dataset and remains under that dataset's licence and terms. Cite the original producers when
you reuse the data; the references are in [paper.bib](paper.bib) and in each package's
`.meta.json`.

## Openly licensed (CC BY 4.0)

| Data | Packages |
| --- | --- |
| Bedmap3 v1.0 gridded products ([doi:10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2](https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2)) | `bedmap3_antarctica_*` |
| Ehrenfeucht (2024), Antarctic subglacial hydrology ([doi:10.5281/zenodo.12738170](https://doi.org/10.5281/zenodo.12738170)) | `antarctica_subglacial_hydrology_*`, `bedmap3_antarctica_subglacial_hydrology_*` |
| Jager (2026), AISEFI and GrISEFI basal shear stress ([doi:10.5281/zenodo.18508904](https://doi.org/10.5281/zenodo.18508904), [doi:10.5281/zenodo.18508850](https://doi.org/10.5281/zenodo.18508850)) | `*_basal_friction_*` |
| Galton-Fenzi (2025), RISE ice-shelf basal melting ([doi:10.26179/q0rh-gt74](https://doi.org/10.26179/q0rh-gt74)) | `rise_antarctica_*` |
| Mouginot & Rignot (2019), Greenland glacier catchments ([doi:10.7280/D1WT11](https://doi.org/10.7280/D1WT11)) | `greenland_basins_ps_v1_4_2.json`, `greenland_refined_basins_search.json` |
| Paxman, Austermann & Hollyday, isostatic response grid files v3 ([doi:10.18739/A22Z12R8C](https://doi.org/10.18739/A22Z12R8C)) | `*_isostatic_response_*` |
| Nowicki & ISMIP6 Team (2024), ISMIP6 Antarctica 2300 projections ([doi:10.5281/zenodo.13135599](https://doi.org/10.5281/zenodo.13135599)), with Seroussi & Pelle (2024) ([doi:10.5281/zenodo.10528582](https://doi.org/10.5281/zenodo.10528582)) | `ismip6_2300_mean8_*` |
| SCAR Composite Gazetteer of Antarctica ([placenames.aq](https://placenames.aq/)) | `antarctica_geographic_names.json` (with a few Natural Earth and GEBCO labels, below) |

## NASA open data (cite the data set)

| Data | Packages |
| --- | --- |
| MEaSUREs BedMachine Antarctica, Version 4 ([doi:10.5067/POJQI54A45HX](https://doi.org/10.5067/POJQI54A45HX)) | `bedmachine_antarctica_v4_*` |
| IceBridge BedMachine Greenland, Version 6 ([doi:10.5067/6B6B225B8V2D](https://doi.org/10.5067/6B6B225B8V2D)) | `bedmachine_greenland_v6_*` |
| MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 ([doi:10.5067/PZ3NJ5RXRH10](https://doi.org/10.5067/PZ3NJ5RXRH10)) | `antarctic_ice_velocity_phase_v01_*`, `bedmap3_antarctica_velocity_*` |
| MEaSUREs Antarctic Boundaries, Version 2 ([doi:10.5067/AXE4121732AD](https://doi.org/10.5067/AXE4121732AD)) | `imbie_refined_basins_v2.json`, `antarctica_refined_basins_search.json` |
| ITS_LIVE velocity mosaic V02.1 (NASA MEaSUREs ITS_LIVE; CC0 on the ITS_LIVE archive) | `greenland_ice_velocity_*` |

## Other terms

- **Copernicus Marine Service licence.** The Greenland ocean currents
  (`greenland_ocean_currents_cmems_*`) were generated using E.U. Copernicus Marine Service
  Information; [doi:10.48670/moi-00001](https://doi.org/10.48670/moi-00001).
- **COMNAP Antarctic Facilities List** (November 2024 edition), in
  `antarctica_research_stations.json`: © COMNAP. The data may be used for educational and
  non-commercial purposes only; the reference for their use is "COMNAP 2025". Ask the COMNAP
  Secretariat about other uses.
- **WAOM2 ocean output** (Boeira Dias et al., 2023; model described by Richter et al., 2022),
  in `antarctica_ocean_currents_waom2_*`: not openly licensed. The derived streamlines are
  redistributed with the written permission of F. Boeira Dias. Ask the authors before reusing
  them.
- **QRF Greenland bed** (Palmer et al., 2025), in `greenland_qrf_2025_*`: the upstream
  repository ([charliekirkwood/greenlandice](https://github.com/charliekirkwood/greenlandice))
  states no licence. It is shown here with attribution to the article; contact the authors
  before reusing it.
- **Natural Earth** labels are in the public domain.
- **Greenland Place Names Register** (NunaGIS), in `greenland_geographic_names.json`, the
  **INTERACT** station directory, in `greenland_research_stations.json`, and the **IHO-IOC
  GEBCO Gazetteer of Undersea Feature Names**, for undersea feature labels, are used under those
  services' published terms. GEBCO asks to be cited as "IHO-IOC GEBCO Gazetteer of Undersea
  Feature Names, www.gebco.net".

## Colour tables

- `cmocean_dense.rgb` and `cmocean_matter.rgb` are the cmocean colour maps of Thyng et al.
  (2016), [doi:10.5670/oceanog.2016.66](https://doi.org/10.5670/oceanog.2016.66), under the MIT
  License: "Copyright (c) 2015 Kristen M. Thyng. Permission is hereby granted, free of charge,
  to any person obtaining a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
  Software, and to permit persons to whom the Software is furnished to do so, subject to the
  following conditions: The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software." The full text, with its warranty
  disclaimer, is at [github.com/matplotlib/cmocean](https://github.com/matplotlib/cmocean/blob/main/LICENSE.txt).
- `GMT_relief.rgb` is the relief colour table of the Generic Mapping Tools, designed by
  P. Wessel and F. Martinez (Wessel et al., 2019,
  [doi:10.1029/2019GC008515](https://doi.org/10.1029/2019GC008515)), distributed with GMT under
  the GNU Lesser General Public License, version 3 or later.
- The `glads_*.rgb` tables are generated by `scripts/build_hydrology_colormaps.py` and are
  covered by the MIT License with the rest of the code.
