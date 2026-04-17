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
- Node.js 20+
- Python 3.10+ (for data preparation scripts)
- A WebGL-capable browser

**Running locally:**
```bash
# Serve the static site
npx serve static

# Run data preparation (requires source NetCDF files)
pip install h5py numpy scipy netCDF4
python scripts/prepare_bedmachine_antarctica.py
```

**Running tests:**
```bash
# Python tests
pip install -r requirements-dev.txt
pytest tests/

# Bundle smoke test
npm run smoke:compat
```

### Code Style

- JavaScript: ES modules, no external build dependencies for the runtime.
- Python: Follow PEP 8. Include docstrings for functions that process scientific data.
- Keep the runtime HTML self-contained for easy deployment.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Questions?

Open an issue or email [yu.wang0@utas.edu.au](mailto:yu.wang0@utas.edu.au).
