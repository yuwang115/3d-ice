# Development guide

Setting up, testing, rebuilding data, packaging and releasing 3D ICE. The design itself is
described in [architecture.md](architecture.md); contribution and support are in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Development environment

- **Python 3.10 or newer** for the data-preparation scripts and the Python tests. Dependencies
  are declared in `pyproject.toml`.
- **Node.js 22.12 or newer** for the JavaScript tests; CI uses Node 24. The runtime has no npm
  dependencies, and the bundle scripts also run on Node 20.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Running tests

With the environment active:

```bash
python -m pytest tests/ --ignore=tests/e2e     # Python unit and integration tests
npm run test:js                                 # JavaScript unit, data-contract and example tests
npm run bundle:compat && npm run smoke:compat   # build and check the distributable bundle

# Browser end-to-end tests (Playwright; about 25 minutes on a laptop)
python -m pip install -e ".[e2e]"
python -m playwright install chromium           # on Linux, add --with-deps as CI does
python -m pytest tests/e2e/
```

## Rebuilding the data packages

The prepared packages are committed, so nothing needs rebuilding to run the explorer or the
tests. [data-pipeline.md](data-pipeline.md) gives, for every package, its source
product, where to obtain it and the exact command, and lists the steps that rebuild packages
from the repository alone. For example:

```bash
# Rebuild the six Bedmap3 overlay packages from committed inputs (identical payloads)
python scripts/prepare_bedmap3_antarctica_overlays.py

# Rebuild BedMachine Antarctica from the NSIDC file: Balanced, then HD
python scripts/prepare_bedmachine_antarctica.py --input BedMachineAntarctica_V4.nc
python scripts/prepare_bedmachine_antarctica.py --input BedMachineAntarctica_V4.nc --step 8 --basename bedmachine_antarctica_v4_741
```

Scripts write into `static/tools/data/` by default and overwrite the committed package of the
same name.

## Worked examples

`node examples/isostatic-rebound.mjs` reproduces the isostatic-rebound figures the explorer shows
for Antarctica, with the explorer's own decoder and rebound module: the published response by
default, or an idealised solve with `--model flexural|local`. [example.md](example.md)
walks through it and through a rebuild of part of the data pipeline from the repository alone.

## Building the compatibility bundle

The bundling scripts need no npm packages: they use Node built-ins and the system `tar`, and run
on Node 20 (the release workflow's version) or newer.

```bash
npm run bundle:compat
npm run smoke:compat
```

This produces:

- `dist/3d-ice-compat.tar.gz`
- `dist/3d-ice-compat.tar.gz.sha256`
- `dist/3d-ice-compat-manifest.json`

The tarball holds the files a host serves at the same paths as 3d-ice.com, so it can be
mounted at a site root:

- `tools/`: the research edition, the runtime, data, media and vendored libraries, so legacy
  `/tools/...` URLs keep working;
- `explore/`, `zh/explore/` and `zh/tools/3D-interactive-cryosphere-explorer.html`: the
  public edition and the Chinese research page;
- `css/`, `js/` and `fonts/`: what the home page loads.

It also holds the two home pages as `home/en-US.html` and `home/zh-CN.html`, for a host that
builds its own copy of the home page. They sit under `home/` so that mounting the bundle cannot
replace the host's own home page. yuwang.blog builds its `/tools/3d-ice/` page this way, and
`tests/test_site_embed.py` pins what it relies on.

## GitHub Pages site

This repository publishes `static/` directly to GitHub Pages. That serves:

- `/` as the standalone landing page
- `/css/3d-ice-home.css` as the home-page stylesheet
- `/explore/` as the public edition, with the guided tour
- `/tools/3D-interactive-cryosphere-explorer.html` as the research edition
- `/tools/data/*`, `/tools/media/3d-ice/*`, `/tools/vendor/*`, and `/tools/3d-antarctica/` as supporting assets

The Pages workflow writes `static/.nojekyll` before deployment so the project-path asset layout is
preserved exactly as shipped.

## Release flow

1. Push to `main` to deploy `static/` to GitHub Pages.
2. Push a tag like `v0.1.0` to build and publish compatibility assets to a GitHub release.
3. Use `workflow_dispatch` when you want either workflow to run manually.

The release workflow uploads and optionally publishes:

- `dist/3d-ice-compat.tar.gz`
- `dist/3d-ice-compat.tar.gz.sha256`
- `dist/3d-ice-compat-manifest.json`

## The JOSS paper draft

Changes to `paper.md`, `paper.bib`, or the paper workflow trigger the
[Draft JOSS PDF workflow](https://github.com/yuwang115/3d-ice/actions/workflows/draft-pdf.yml).
The compiled `paper.pdf` is available from each workflow run as the
`joss-paper` artifact. To use the same Open Journals toolchain locally when
Docker is available:

```bash
docker run --rm \
  --volume "$PWD:/data" \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara -o pdf paper.md
```

## How the editions are configured

A page names its edition with `<html data-edition>`, and `static/tools/js/editions.js` turns
that into the datasets, layers and behaviours on offer. Restricting a layer also drops the
URLs of its packages, so the public pages never fetch a research-only package. The tour's
stops and all the public edition's words, in English and Chinese, are in
`static/tools/js/explore-content.js`. `?tour=1` opens the tour when the page loads, and
`?tour=<stop id>` opens it at that stop.

## Repository layout

| Path | Purpose |
| --- | --- |
| `static/index.html`, `static/zh/index.html` | Home pages (English and Chinese): the hero, *Explore* (the data tower with the two editions under it), the latest updates with a drawer holding all of them, the source data and feedback. |
| `static/js/3d-ice-data-tower.js`, `3d-ice-data-tower-geometry.js` | The home page's data tower: every dataset as one layer of a turning stack, each opening its view. Its layer images are drawn by `scripts/build_home_data_tower.py`. |
| `static/css/3d-ice-home.css`, `static/css/3d-ice-type.css`, `static/js/3d-ice-home.js` | Home-page styles, typefaces and behaviour (theme, preview videos, feedback form, language switcher). |
| `static/fonts/` | Self-hosted Space Grotesk and Playfair Display (SIL Open Font License). |
| `static/tools/3D-interactive-cryosphere-explorer.html`, `static/zh/tools/…` | Research-edition explorer pages (English and Chinese). |
| `static/explore/index.html`, `static/zh/explore/index.html` | Public-edition explorer pages (English and Chinese). |
| `static/tools/js/explorer-app.js`, `static/tools/css/explorer.css` | The explorer runtime and styles that every explorer page loads. |
| `static/tools/js/editions.js` | What each edition offers: datasets, layers and runtime behaviours. |
| `static/tools/js/explore-guide.js`, `explore-content.js`, `explore-tour.js`, `static/tools/css/explore.css` | The public edition's guided tour and layer explainers: interface, copy, camera and tour logic, and styles. |
| `static/tools/js/` (other modules) | Data-package decoder, polar projections, vertical scale, layer statistics, overlay cell rules, source links, isostatic-rebound solver, and place and feature search. |
| `static/tools/*-worker.js` | Web Workers for overlay geometry and the rebound solve. |
| `static/js/3d-ice-locale.js` | English and Chinese interface strings. |
| `static/tools/data/` | Prepared data packages (`.bin` + `.meta.json`) and feature catalogues. |
| `static/tools/media/3d-ice/` | Preview stills and loop videos, and in `tower/` the data tower's layer images. |
| `static/tools/vendor/three/` | Vendored Three.js r161. |
| `scripts/` | Data preparation, bundle and release utilities, and trailer capture. |
| `tests/` | Python, JavaScript and browser tests. |
| `docs/`, `examples/` | Design documentation and runnable examples. |
| `dist/` | Generated compatibility bundle (not committed). |
| `.github/workflows/` | CI, the JOSS paper draft, GitHub Pages deployment and release automation. |

## What this repository is for

- Publish `static/` directly to GitHub Pages as a standalone site.
- Preserve the legacy `/tools/...` public paths used by the main personal site.
- Ship a compatibility bundle for downstream deployment into another repo.
- Keep runtime assets, prepared data products, and data-preparation tooling together.

This repo is intentionally focused on the standalone 3D ICE experience and related distribution
artifacts. It is the source of truth for:

- the browser runtime
- the standalone landing page
- prepared data bundles
- preview media
- compatibility packaging for downstream deployment

If you are looking for the broader personal site that consumes the compatibility bundle, this repo
is the upstream asset and runtime source rather than the final umbrella website.
