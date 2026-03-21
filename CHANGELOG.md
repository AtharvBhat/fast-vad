# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-03-21

### Added

- Added __repr__ and Display traits for python and rust crates
-  `detect()` and `detect_frames()` now return numpy arrays. This is a breaking change, Hence the version bump.

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
