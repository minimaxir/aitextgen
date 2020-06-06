# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- CHANGELOG.md

## [0.2.0] - 2020-06-01

### Added

- Progress bar for loading a dataset.
- `progress_bar_refresh_rate` parameter for `train()` and `TokenDataset()`.

### Changed

- Set numpy data store for `TokenDataset`.
- Set single-text files to be loaded delimited as newlines.

### Removed

- `shuffle` and `seed` parameters for TokenDataset.

## [0.1.1] - 2020-05-17

### Changed

- Set `generate()` defaults to `max_length=256` and `temperature=0.7`.
- Added to docs notes about GPT-2 maximum length of 1024.

## [0.1] - 2020-05-17

### Added

- Everything!
