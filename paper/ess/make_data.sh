#!/bin/sh
# Regenerate paper/ess/data/: the worked example's output for the three ISMIP6 experiments,
# from which Figure 5 is drawn (series every 5 years, basins in 2015, 2100, 2200 and 2300),
# and the thickness-change colour scale of Figure 4. Run from anywhere in the repository.
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$ROOT/paper/ess/data"
YEARS="$(python3 -c "print(','.join(str(year) for year in range(2015, 2301, 5)))")"
for scenario in ae10 ae05 ae14; do
  node "$ROOT/examples/ismip6-projection.mjs" --scenario "$scenario" --years "$YEARS" --json \
    > "$DATA/series_$scenario.json"
  node "$ROOT/examples/ismip6-projection.mjs" --scenario "$scenario" --years 2015,2100,2200,2300 --json \
    > "$DATA/basins_$scenario.json"
done
node "$ROOT/paper/ess/figures/sample_ramp.mjs" "$DATA/thickness_change_ramp.json"
echo "wrote $DATA"
