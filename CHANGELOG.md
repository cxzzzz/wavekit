# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## Unreleased

## 0.7.0a1 - 2026-06-19

### Added
- Programmable Pattern API (async body + declarative builder unified runtime)
- Unknown-mask waveform loading (`load_unknown_mask`, `load_matched_unknown_masks`)
- Pattern result status filters (`filter_ok`, `filter_failed`, `filter_status`)

### Changed
- Unify declarative/programmable pattern runtime into single engine
- Remove legacy pattern engine and tick compatibility
- Simplify reader backend hook docstrings; mark unknown-mask API experimental
- Unify value-change loading across readers

## v0.6.1 - 2026-05-23

### Fixed
- Fix wheel packaging for Cython reader extensions so installed wheels expose `wavekit.readers.value_change` and FSDB extension modules at their runtime import paths.

## v0.6.0 - 2026-05-23

### Added
- Add `FstReader` for loading FST waveform files through the same reader APIs as VCD and FSDB.
- Add `Channel`-based FIFO consumption to Pattern matching for ordered request/response pairing and per-ID routing.
- Add relative time access helpers for waveform analysis.
- Add Chinese README documentation.

### Changed
- Refactor the pattern API around tick, channel, capture mode, and require semantics.
- Improve VCD reader error reporting for empty value-change data and unsupported sub-range access.

### Fixed
- Fix FSDB array signal value parsing and reader resource handling.
- Restrict pattern trigger optimization to `wait()` steps.
