# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v1.2.0 (2026-04-12)

### Feat

- **cli**: add --version and centralize __version__

### Fix

- **test**: avoid tomllib for Python 3.10 ty compatibility

## v1.1.0 (2026-04-12)

### Feat

- **release**: set version 1.0.0 and run commitizen in CI
- **cli**: add postprocess command for offline image/video overlay
- custom overlay image, scale, and chroma-key helper
- **overlay**: terminal lambda tuning, roi horizontal smoothing, defaults

### Fix

- **ci**: skip release when commits are not bump-eligible (e.g. docs/ci)
- satisfy ruff and ty checks across source and scripts
