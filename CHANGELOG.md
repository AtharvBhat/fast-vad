# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-03-25

### Added

- `FeatureExtractor.feature_engineer()`: returns a `(num_frames, 24)` float32 array of log-energy features plus first and second-order deltas.

### Changed

- Passing a numpy array with the wrong dtype now raises a clear `TypeError` (e.g. `expected a numpy array with dtype float32, but got dtype float64`) instead of an opaque PyO3 type error.

## [0.2.0] - 2026-03-21

### Added

- Added __repr__ and Display traits for python and rust crates

### Changed

-  `VAD.detect()` and `VAD.detect_frames()` now return numpy arrays. This is a breaking change, Hence the version bump.

## [0.1.1] - 2026-03-09

### Added

- Added `fast_vad.__version__` to the Python module.

### Changed

- Switched Python wheel builds to `abi3` with a Python 3.11 compatibility baseline.

## [0.1.0]

### Added

- Initial public release.
- Rust crate publication to crates.io.
- Python package publication to PyPI.
