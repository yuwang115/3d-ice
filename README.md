# 3D ICE

Standalone source repository for the 3D ICE runtime, data bundles, preview media, and data preparation scripts.

## Repo scope

- `static/tools/` contains the browser runtime and all assets needed to preserve the legacy `/tools/...` public paths.
- `static/index.html` contains the standalone landing page for the custom domain root.
- `scripts/` contains data preparation utilities and trailer capture tooling related to 3D ICE.
- `dist/` is generated and contains the compatibility bundle consumed by the main personal site repo.

## Standalone GitHub Pages site

This repo can now publish `static/` directly to GitHub Pages. That serves:

- `/` as the standalone landing page
- `/tools/3D-interactive-cryosphere-explorer.html` as the main runtime
- `/tools/data/*`, `/tools/media/3d-ice/*`, `/tools/vendor/*`, and `/tools/3d-antarctica/` as supporting assets

The runtime auto-detects project-path prefixes, so it can work both at a site root and under a GitHub Pages project path such as `/3d-ice/tools/...`.

## Compatibility bundle

Build the bundle consumed by the main personal site repo (expected as a sibling checkout at `../yuwang115.github.io`):

```bash
npm run bundle:compat
```

This produces:

- `dist/3d-ice-compat.tar.gz`
- `dist/3d-ice-compat.tar.gz.sha256`
- `dist/3d-ice-compat-manifest.json`

The tarball always expands to a top-level `tools/` directory so the main site can continue serving the legacy `/tools/...` URLs unchanged.

## Release flow

- Push a tag like `v0.1.0` to publish a GitHub release with the compatibility assets attached.
- `workflow_dispatch` can also build the bundle on demand and upload it as a workflow artifact.
