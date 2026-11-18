# Worked examples

All three examples run from a clean clone with the data that ships in the repository; none
needs a download.

## 1. The ice-free bed after isostatic rebound

**Question.** If the Antarctic ice sheet were removed and the solid Earth allowed to reach
isostatic equilibrium, how much of the bed that lies below sea level today would end up
above it, and how much would remain marine?

The answer matters when reading bed topography under a retreating ice sheet: the bed is not
fixed, and unloading raises it by hundreds of metres. 3D ICE answers the question with its
isostatic-rebound layer, whose default Earth response is the published one of Paxman,
Austermann & Hollyday (2022), shipped as a package on each terrain grid; idealised responses
solved from the terrain package are available for comparison (both routes are described in
[data-layers.md](data-layers.md#isostatic-rebound-ice-free-equilibrium)).

### In the browser

1. Start the local preview (`python3 -m http.server 4173 --directory static`) and open
   <http://127.0.0.1:4173/tools/3D-interactive-cryosphere-explorer.html?region=antarctica>,
   which loads BedMachine Antarctica v4 on the Balanced 10 km grid, the default dataset.
2. Under **Change over time**, tick **Ice-free rebound**. The defaults are the
   headline scenario: **Earth response** is the published response of Paxman et al. (2022)
   and **Deglaciation & rebound** is at 100 %. The sea-level datum slider is disabled: the
   published model fixes its own ice-free sea surface.
3. Open **Data and sources** at the foot of the panel: its **Isostatic Rebound (modelled)**
   section lists the figures below, and newly emergent land is highlighted on the rebounded
   bed.

### From the command line

The same decoder and rebound module run under Node:

```bash
node examples/isostatic-rebound.mjs
```

Expected output:

```text
Package                                                     bedmachine_antarctica_v4_480 (BedMachine Antarctica v4, 667 × 667 cells at 10 km)
Earth response                                              published: Paxman et al. (2022), grids v3 (antarctica_isostatic_response_480)
Sea surface                                                 ice-free: 65.3 m eustatic plus the residual post-LGM geoid
Maximum solid-surface uplift (R)                            1028.6 m
Largest rise above the ice-free sea (T)                     940.6 m
Mean uplift under grounded ice (R)                          587.2 m
Components under grounded ice (mean)                        ice unloading 513.7 m, post-LGM 14.4 m, water loading -28.8 m
Model spread (1 sigma) under grounded ice                   35.8 m mean, 234.5 m max
Bed above sea level today                                   6,634,908 km²
Bed above the ice-free sea surface after rebound            9,560,313 km²
Newly emergent land                                         2,941,863 km²
Still below the ice-free sea surface under the present ice  4,351,870 km²
Sea-level equivalent                                        58.05 m
```

Options: `--region greenland` uses BedMachine Greenland v6, `--dataset` selects any other
terrain package, for example `bedmachine_antarctica_v4_741` (the 4 km HD grid) or
`bedmap3_antarctica_10km`, and `--model flexural` or `--model local` solves an idealised
response instead, which alone accepts `--sea-level <metres>`.

### Reading the result

- **About 2.94 million km² of land emerges.** That is bed below sea level today that ends
  up above the ice-free sea surface once rebound is complete. Emergence is judged on
  `bed + T > 0`, where `T` is the published total isostatic response, the change in bed
  elevation relative to the sea surface.
- **About 4.35 million km² of the bed under today's ice stays below the ice-free sea
  surface,** even at full re-equilibration.
- **Peak solid-surface uplift is 1028.6 m** (at 78.5° S, 51° E, in the East Antarctic
  interior between Dome Fuji and Dome A), and the largest rise relative to the sea surface is
  940.6 m. The two
  differ by the sea-surface change `G`: the 65.3 m eustatic rise from both ice sheets plus
  the residual post-LGM geoid change, 73–91 m in all over the grounded ice.
- **The components** are the paper's Table 1 terms on this grid. Ice unloading dominates;
  the post-LGM disequilibrium still to come adds 14.4 m on average and up to 68.3 m under
  the Ross and Weddell embayments; water loading takes back 28.8 m on average and up to
  419 m in West Antarctica's deep marine basins, which stay flooded.
- **The Earth-model spread** (four elastic-thickness models, 24 viscoelastic models)
  averages 35.8 m under the grounded ice and peaks where ice thickness changes steeply.
- **The 4 km HD grid agrees to within 0.2 %** (2,936,401 km² emergent), because both packages
  sample the same 500 m published grid.
- **The sea-level equivalent, 58.1 m,** is the ice volume above flotation converted to
  mass with an ice density of 917 kg m⁻³ and spread over an ocean area of 3.625 × 10¹⁴ m²
  at freshwater density (1000 kg m⁻³), as Gregory et al. (2019) define sea-level
  equivalent (362.5 Gt per mm). Morlighem et al. (2020) give 57.9 ± 0.9 m for BedMachine
  Antarctica v1 the same way. Converting at seawater density instead would give a figure
  2.6 % lower.

**Against the idealised response.** `node examples/isostatic-rebound.mjs --model flexural`
solves an elastic plate of uniform rigidity (`D = 1e25 N m`) at today's datum instead. It
peaks at 1026.8 m, almost the same value but in a different place (the Aurora Subglacial
Basin), and emerges 3.19 million km², 8.5 % more, because it adds no meltwater to the ocean
and no post-LGM term. Over grounded ice it differs from the published solid-surface
displacement by 68 m RMS, mostly through its uniform rigidity, which is stiffer than any of
the published elastic-thickness models and over-smooths West Antarctica.

These are steady-state figures. They say where the bed ends up, not how fast: full
re-equilibration takes of order 100 kyr, and the published response has no single
timescale. They include neither erosion and sedimentation nor thermosteric or dynamic sea
level, and the post-LGM term rests on a one-dimensional viscosity profile. The explorer
shows the same caveats next to the numbers.

## 2. Regenerating Bedmap3 overlay packages from the repository

Most preparation scripts start from upstream products of several gigabytes. One step of the
pipeline does not: `scripts/prepare_bedmap3_antarctica_overlays.py` re-grids the committed
velocity, basal-friction and hydrology packages from the BedMachine grids onto the Bedmap3
grids, whose origins differ by 250 m. Its outputs are committed too, so a reviewer can
rebuild them and compare:

```bash
out="$(mktemp -d)"
cp static/tools/data/bedmap3_antarctica_{10km,4km}.meta.json \
   static/tools/data/{antarctic_ice_velocity_phase_v01,antarctica_basal_friction,antarctica_subglacial_hydrology}_{480,741}.{meta.json,bin} \
   "$out"
python scripts/prepare_bedmap3_antarctica_overlays.py --data-dir "$out"
for f in "$out"/bedmap3_antarctica_*_{10km,4km}.*; do cmp "$f" "static/tools/data/${f##*/}" && echo "identical ${f##*/}"; done
```

With Python 3.13 and NumPy 2.4, all twelve files (six packages, each a `.bin` and a
`.meta.json`) are reported identical. Other NumPy versions can sum in a different order,
which changes the last digits of a recorded mean in the metadata; the six payloads are
identical regardless.

## 3. Where the ISMIP6 ensemble loses ice by 2300

**Question.** Under the ISMIP6 Antarctic projections to 2300, how much grounded ice does the
ensemble lose under low and high emissions, which drainage basins lose it, and how does the
sea-level rise implied by the explorer's geometry compare with what the same models
published?

The explorer's *Future projection* layer carries three ISMIP6-2300 experiments (Seroussi et
al., 2024): expAE10 (UKESM1-0-LL, SSP1-2.6), expAE05 (UKESM1-0-LL, SSP5-8.5) and expAE14
(SSP5-8.5 with ice-shelf collapse). Each package holds the equal-weight mean, over the eight
models that ran all three, of the change in ice thickness since 2015, applied to BedMachine
Antarctica v4 and stored every five years ([data-layers.md](data-layers.md) and the
packages' `method` block give the details). The browser rebuilds the surface and base from
that thickness and the static BedMachine bed by flotation.

### In the browser

1. Open the research edition and, under **Change over time**, switch on **Future projection**.
2. Choose a **Scenario** and drag the **Year** slider, or press **Play**. *Colour ice by*
   switches between thickness change since 2015 and grounded versus floating ice.
3. The chart under the slider draws the mean and range of the eight models' published
   sea-level contributions, and the readout gives the grounded area of the frame on screen.

### From the command line

```bash
node examples/ismip6-projection.mjs                     # expAE05, 2015-2300
node examples/ismip6-projection.mjs --scenario ae10     # or ae14
node examples/ismip6-projection.mjs --years 2100,2300 --top 5 --json
```

For expAE05 it prints:

```text
Year   Grounded (10^6 km²)  Floating (10^6 km²)  Implied SLE (m)  Published mean [min, max] (m)
2015       12.281                1.561             0.000        0.000 [0.000, 0.000]
2100       12.157                1.674             0.018        -0.025 [-0.217, 0.130]
2150       11.919                1.869             0.229        0.113 [-0.264, 0.500]
2200       11.648                1.971             0.689        0.453 [-0.188, 1.469]
2250       11.389                2.073             1.296        0.923 [-0.085, 2.669]
2300       11.192                2.166             1.974        1.463 [-0.084, 3.990]
```

followed by the drainage basins of MEaSUREs Antarctic Boundaries v2 (Mouginot et al., 2017),
ranked by their contribution in the last year: Thwaites (0.333 m by 2300), Pine Island
(0.186 m), the Siple Coast ice streams MacAyeal, Whillans, Kamb and Bindschadler
(0.120-0.148 m each), Cook (0.099 m) and Institute (0.098 m). Two rows close the table so that
its parts add up to the ice sheet:

```text
Ice rises and islands           0.000      0.015      0.033      0.054      0.067      0.073   -99,528
Outside every outline           0.000      0.001      0.003      0.004      0.005      0.005   -14,211
```

By 2300 expAE10 implies 0.151 m (published mean 0.052 m) and expAE14 3.190 m (published mean
2.429 m).

### Reading the result

- *Implied SLE* is the loss of ice above flotation since 2015, converted with
  ρ<sub>ice</sub> = 917 kg m⁻³ and the package's 362.5 Gt per mm. Each 10 km cell counts its
  ground area, the nominal 100 km² corrected by the polar stereographic scale factor.
  `--cell-area grid` uses nominal cells instead, which gives figures about 2.5 % lower and
  reproduces the `validation` block of each package (1.926 m for expAE05).
- The rows are per MEaSUREs polygon. The product records its ice rises and islands as one
  entry of 656 rings ("Islands"), which is not a drainage basin, so it is reported beside
  the basins rather than ranked with them. Some basins are split into several polygons
  (Ross East1–5, Wilkes Land1–2 and others); a sector total is their sum. About 25,000 km² of
  grounded ice in 2015 lies in cells whose centres fall inside no outline, almost all of them
  next to floating ice or open ocean, and is reported as its own row.
- The implied rise and the published values measure the same quantity, the loss of ice
  above flotation since 2015 with the control run not subtracted, but on different
  geometries. The implied rise exceeds the published mean in all three experiments. One
  cause is certain in sign: thickness above flotation, max(0, H − H<sub>f</sub>), is convex in
  H, so on a common baseline the mean geometry loses at least as much as the mean of the
  models. The example does not measure how much of the gap that explains. The fixed
  BedMachine bed (some of the models deform theirs), the difference between BedMachine's
  2015 ice and each model's own, and the removal of ice where the models' mean falls below
  10 m may contribute too; separating them needs each model's geometry, which the packages
  do not hold.
- The figures describe the ensemble mean only; no single model followed this path. The
  published per-model range in 2300, from −0.08 to 3.99 m under expAE05, is far wider than
  either mean.

## How these examples are kept honest

- `tests/js/examples.test.mjs` runs example 1 and pins the figures quoted above, for the
  published response and for the idealised one, so a change to the rebound module, the
  decoder, a terrain package or a response package that moves them fails CI.
- `tests/js/ismip6-example.test.mjs` runs example 3 for all three scenarios and pins the
  figures quoted above. It checks that the basins, the islands and the cells outside every
  outline add up to the ice sheet in every year, that nominal grid cells reproduce the
  packages' own validation figures, and that a package without a published series still
  reports its own geometry.
- `tests/test_prepare_isostatic_response.py` checks the script that builds the response
  packages: that it samples the same nodes as the terrain packages, enforces the published
  identities, and records provenance. The script itself verifies each 5 GB grid file against
  the MD5 the Arctic Data Center records, so rebuilding the packages needs that download.
- `tests/test_prepare_bedmap3_antarctica_overlays.py` repeats example 2. It requires every
  regenerated payload to match the committed one byte for byte, and every metadata file to
  match it exactly apart from floating-point rounding.
- `tests/js/gia-point-load.test.mjs` validates the solver itself against the analytic
  point-load solution for a thin elastic plate (Kelvin functions),
  `tests/js/gia-rebound.test.mjs` against the closed-form Airy limit, and
  `tests/e2e/test_isostatic_rebound.py` checks the layer in a real browser.
