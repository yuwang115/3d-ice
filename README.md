# 3D ICE

Standalone source repository for the 3D ICE runtime, data bundles, preview media, and data preparation scripts.

## Repo scope

- `static/tools/` contains the browser runtime and all assets needed to preserve the legacy `/tools/...` public paths.
- `scripts/` contains data preparation utilities and trailer capture tooling related to 3D ICE.
- `dist/` is generated and contains the compatibility bundle consumed by the main personal site repo.

## Compatibility bundle

Build the bundle consumed by `/Users/eddie/Documents/yuwang115.github.io`:

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
