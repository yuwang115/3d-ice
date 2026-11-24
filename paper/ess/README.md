# Figures of the Earth and Space Science report

This folder holds the scripts and data that make Figures 1–5 of *3D ICE: A Browser-Based
Three-Dimensional Explorer for Antarctic and Greenland Ice-Sheet Data and Projections*
(Y. Wang and Y. Lin), a Technical Report: Methods submitted to AGU's *Earth and Space Science*.
The manuscript is not in the repository.

| Figure | Made by | From |
| --- | --- | --- |
| 1 | `capture_figure1a.py`, `composite_panel_a.py`, `check_composite.py` | A capture of the research edition, put into panel (a) of `docs/images/explorer-overview.jpg`; panels (b) and (c) are kept byte for byte |
| 2 | `make_figure2.py` | The numbers of this release, written into the script |
| 3 | `capture_figure3.py`, `make_figure3.py` | Two captures of the research edition |
| 4 | `capture_figure4.py`, `make_figure4.py` | Six captures of the public edition, and `data/thickness_change_ramp.json` |
| 5 | `make_figure5.py` | `data/series_*.json` |

`agu_style.py` holds the print size (170 mm wide), the 8 pt Arial labels and the checks shared
by Figures 3–5; every script stops if a label is too small, leaves the figure or overlaps
another.

`data/` is the output of `examples/ismip6-projection.mjs` at this release for the three
ISMIP6 experiments (`series_*`, every 5 years from 2015 to 2300; `basins_*`, by drainage basin
in 2015, 2100, 2200 and 2300) and the explorer's thickness-change colour scale. `make_data.sh`
writes it again; at v0.4.1 the result is byte-identical. `make_figure5.py --table` also writes
the per-basin table, `basin_table.md`.

## Requirements

- Node.js 22.12 or newer (see `docs/development.md`).
- Python 3.10 or newer with matplotlib 3.10.8 and Pillow, and the Arial font. The figures
  were drawn with Python 3.13.2, matplotlib 3.10.8, Pillow 12.1.1 and NumPy 2.4.4 on macOS;
  other matplotlib versions move pixels.
- For the captures, Playwright for Python and its Chromium
  (`python -m playwright install chromium`); Playwright 1.62.0 was used.
- For Figure 1, `jpegtran` from libjpeg-turbo (3.1.3 was used).

## Steps

From the repository root, serve `static/` with any static file server, for example:

```sh
python3 -m http.server 8144 --directory static
```

Then, in a second shell:

```sh
RESEARCH=http://127.0.0.1:8144/tools/3D-interactive-cryosphere-explorer.html
FIG=paper/ess/figures

# Figure 1
python3 $FIG/capture_figure1a.py $RESEARCH $FIG/raw/figure1a_capture.png
mkdir -p /tmp/figure1
python3 $FIG/composite_panel_a.py $FIG/raw/figure1a_capture.png docs/images/explorer-overview.jpg $FIG/figure1.jpg /tmp/figure1
python3 $FIG/check_composite.py docs/images/explorer-overview.jpg $FIG/figure1.jpg /tmp/figure1

# Figure 2
python3 $FIG/make_figure2.py

# Figure 3
python3 $FIG/capture_figure3.py $RESEARCH $FIG/raw
python3 $FIG/make_figure3.py

# Figure 4
python3 $FIG/capture_figure4.py http://127.0.0.1:8144 $FIG/raw
python3 $FIG/make_figure4.py

# Figure 5
python3 $FIG/make_figure5.py
```

The figures and captures are written into `figures/` and `figures/raw/`, which git ignores.

## What to expect

Figures 2 and 5 come out pixel-identical to the manuscript's with the versions above. Figures
1, 3 and 4 depend on captures, which depend on the browser and the graphics stack, and the
animated flowlines and ocean streamlines are caught at a different moment each time. At
v0.4.1, new captures differed from those in the manuscript only there. From the same captures,
Figures 1, 3 and 4 come out identical.
