# Contributing to 3D ICE

Thank you for your interest in contributing to 3D ICE! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/yuwang115/3d-ice/issues) to report bugs or request features.
- Include steps to reproduce the issue, the browser/OS you used, and any console errors.
- For dataset-related issues, specify which data layer is affected.

### Suggesting Enhancements

- Open an issue describing the enhancement and its use case.
- For new dataset integrations, include a link to the source data and its license.

### Submitting Changes

1. Fork the repository.
2. Create a feature branch from `main` (`git checkout -b feature/my-change`).
3. Make your changes and test locally.
4. Commit with a descriptive message following the format: `type: description` (e.g., `feat: add Ross Ice Shelf overlay`).
5. Push to your fork and open a Pull Request against `main`.

### Development Setup

**Prerequisites:**

- Python 3.10+ (data-preparation scripts and Python tests)
- Node.js 22.12+ (JavaScript tests; CI uses Node 24)
- A WebGL-capable browser

**Running locally:**

```bash
# Serve the static site
python3 -m http.server 4173 --directory static

# Create the Python environment used by the scripts and tests
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Rebuilding a data package needs its source product; see
[docs/data-pipeline.md](docs/data-pipeline.md) for where to obtain each one and the exact
command.

**Running tests:**

```bash
# Python tests
python -m pytest tests/ --ignore=tests/e2e -v

# JavaScript unit, data-contract and example tests
npm run test:js

# Bundle smoke test
npm run bundle:compat && npm run smoke:compat

# Optional browser end-to-end tests
python -m pip install -e ".[e2e]"
python -m playwright install chromium
python -m pytest tests/e2e/ -v
```

### Code Style

- JavaScript: ES modules served as static files, with no build step and no npm runtime
  dependencies.
- Logic with no DOM or scene dependency belongs in its own module under `static/tools/js/`
  with a `node:test` suite; `static/tools/js/explorer-app.js` is the orchestration layer that
  both locale pages share.
- User-visible strings go in `static/js/3d-ice-locale.js` in both locales; the locale test
  fails if a key the runtime uses is missing from either.
- New data packages follow [docs/data-contract.md](docs/data-contract.md); the contract test
  checks every package in `static/tools/data/` automatically.
- Python: Follow PEP 8. Include docstrings for functions that process scientific data.

## Support and Governance

3D ICE is maintained by Yu Wang, who reviews issues and pull requests and decides what is
merged and released. Questions, bug reports and feature requests go through
[GitHub Issues](https://github.com/yuwang115/3d-ice/issues); support is provided on a
best-effort basis without a guaranteed response time. Releases are tagged on GitHub and
summarised in [CHANGELOG.md](CHANGELOG.md). A new data layer needs a source product whose
licence allows derived packages to be redistributed.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Questions?

Open an issue or email [wangyu@uchicago.edu](mailto:wangyu@uchicago.edu).
