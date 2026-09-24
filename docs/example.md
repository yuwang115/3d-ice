# Worked examples

Both examples run from a clean clone with the data that ships in the repository; neither
needs a download.

## 1. The ice-free bed after isostatic rebound

**Question.** If the Antarctic ice sheet were removed and the solid Earth allowed to reach
isostatic equilibrium, how much of the bed that lies below sea level today would end up
above it, and how much would remain marine?

The answer matters when reading bed topography under a retreating ice sheet: the bed is not
fixed, and unloading raises it by hundreds of metres. 3D ICE answers the question with its
isostatic-rebound layer, which it computes from the loaded terrain package rather than
loading as data (the method is described in the
[README](../README.md#isostatic-rebound-ice-free-equilibrium)).

### In the browser

1. Start the local preview (`python3 -m http.server 4173 --directory static`) and open
   <http://127.0.0.1:4173/tools/3D-interactive-cryosphere-explorer.html?region=antarctica&preset=balanced>,
   which loads BedMachine Antarctica v4 on the Balanced 10 km grid.
2. Under **View controls**, tick **Show ice-free isostatic rebound**. The defaults are the
   headline scenario: **Earth response** is regional flexure, **Deglaciation & rebound** is
   at 100 %, and the sea-level datum is 0 m.
3. The metadata panel's **Isostatic Rebound (modelled)** section then lists the figures
   below, and newly emergent land is highlighted on the rebounded bed.

### From the command line

The same decoder and solver run under Node:

```bash
node examples/isostatic-rebound.mjs
```

Expected output:

```text
Package                                      bedmachine_antarctica_v4_480 (BedMachine Antarctica v4, 667 × 667 cells at 10 km)
Earth response                               flexural (D = 1e25 N m, length scale 133 km)
Sea-level datum                              0 m
Maximum equilibrium uplift                   1026.8 m
Mean uplift under grounded ice               553.9 m
Bed above sea level today                    6,634,908 km²
Bed above the datum after rebound            9,826,667 km²
Newly emergent land                          3,191,759 km²
Still below the datum under the present ice  4,109,375 km²
Closed basins below the datum                289,797 km²
Sea-level equivalent                         56.53 m
Solve                                        9 Picard iterations, residual 0.007 m, 20.0 km solve grid
```

Options: `--model local` switches to local Airy isostasy, `--sea-level <metres>` raises the
datum, `--region greenland` uses BedMachine Greenland v6, and `--dataset` selects any other
terrain package, for example `bedmachine_antarctica_v4_741` (the 4 km HD grid).

### Reading the result

- **About 3.19 million km² of land emerges.** That is bed below sea level today that ends
  up above it once rebound is complete; 99 % of it lies under the present ice.
- **About 4.11 million km² of the bed under today's ice stays below the datum,** even at
  full equilibrium. Of all the bed left below the datum, 0.29 million km² lies in closed
  basins that no longer drain to the ocean, and the solver loads those with no marine
  water.
- **Peak uplift is about 1 km** under the thickest ice. Local Airy isostasy gives 1374 m,
  which bounds the flexural value from above, because a rigid lithosphere spreads each load
  over roughly its flexural length scale of 133 km.
- **The 4 km HD grid gives the same peak uplift to within 0.1 m** (1026.9 m), because the
  deflection is solved on a 20 km grid in both cases. Its emergent area is 0.6 % smaller,
  because emergence is decided cell by cell on each package's own grid.
- **The sea-level equivalent, 56.5 m,** is the ice volume above flotation converted with
  the densities of ice (917 kg m⁻³) and seawater (1027 kg m⁻³) and an ocean area of
  3.625 × 10¹⁴ m², following Gregory et al. (2019). Converting to meltwater at freshwater
  density instead gives a figure about 3 % higher.

These are steady-state figures. They say where the bed ends up, not how fast: they are
not a transient glacial-isostatic-adjustment simulation, they assume today's bed is in
balance with today's load, and they replace lateral variations in Earth structure with a
single rigidity and mantle density. The explorer shows the same caveats next to the numbers.

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

- `tests/js/examples.test.mjs` runs example 1 and pins the figures quoted above, so a change
  to the solver, the decoder or the terrain package that moves them fails CI.
- `tests/test_prepare_bedmap3_antarctica_overlays.py` repeats example 2. It requires every
  regenerated payload to match the committed one byte for byte, and every metadata file to
  match it exactly apart from floating-point rounding.
- `tests/js/gia-rebound.test.mjs` validates the solver itself against the analytic
  point-load solution for a thin elastic plate (Kelvin functions) and the closed-form Airy
  limit, and `tests/e2e/test_isostatic_rebound.py` checks the layer in a real browser.
