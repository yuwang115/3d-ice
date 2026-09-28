# Worked examples

Both examples run from a clean clone with the data that ships in the repository; neither
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
the [README](../README.md#isostatic-rebound-ice-free-equilibrium)).

### In the browser

1. Start the local preview (`python3 -m http.server 4173 --directory static`) and open
   <http://127.0.0.1:4173/tools/3D-interactive-cryosphere-explorer.html?region=antarctica>,
   which loads BedMachine Antarctica v4 on the Balanced 10 km grid, the default dataset.
2. Under **View controls**, tick **Show ice-free isostatic rebound**. The defaults are the
   headline scenario: **Earth response** is the published response of Paxman et al. (2022)
   and **Deglaciation & rebound** is at 100 %. The sea-level datum slider is disabled: the
   published model fixes its own ice-free sea surface.
3. The metadata panel's **Isostatic Rebound (modelled)** section then lists the figures
   below, and newly emergent land is highlighted on the rebounded bed.

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
Sea-level equivalent                                        56.53 m
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
- **The sea-level equivalent, 56.5 m,** is the ice volume above flotation converted with
  the densities of ice (917 kg m⁻³) and seawater (1027 kg m⁻³) and an ocean area of
  3.625 × 10¹⁴ m², following Gregory et al. (2019). Converting to meltwater at freshwater
  density instead gives a figure about 3 % higher.

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

## How these examples are kept honest

- `tests/js/examples.test.mjs` runs example 1 and pins the figures quoted above, for the
  published response and for the idealised one, so a change to the rebound module, the
  decoder, a terrain package or a response package that moves them fails CI.
- `tests/test_prepare_isostatic_response.py` checks the script that builds the response
  packages: that it samples the same nodes as the terrain packages, enforces the published
  identities, and records provenance. The script itself verifies each 5 GB grid file against
  the MD5 the Arctic Data Center records, so rebuilding the packages needs that download.
- `tests/test_prepare_bedmap3_antarctica_overlays.py` repeats example 2. It requires every
  regenerated payload to match the committed one byte for byte, and every metadata file to
  match it exactly apart from floating-point rounding.
- `tests/js/gia-rebound.test.mjs` validates the solver itself against the analytic
  point-load solution for a thin elastic plate (Kelvin functions) and the closed-form Airy
  limit, and `tests/e2e/test_isostatic_rebound.py` checks the layer in a real browser.
