# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed
- Refactor Pattern execution to a start-major synchronous runtime and remove async/await from programmable bodies.
- Move pattern execution to module-level `match(...)` / `collect(...)` entry points and keep `Pattern` as a declarative builder.
- Replace the old batch result surface with `MatchRecord` / `MatchRecords` and structured `MatchStatus` objects.
- Replace the legacy `scope.py`, `signal.py`, and `pattern_parser.py` modules with
  immutable hierarchy types and the unified query matcher.
- Standardize matched-reader results on typed `CaptureKey` tuples and remove
  ordinary exact-name captures from public keys.
- Expose immutable reader roots through `Reader.top_scopes` and use
  `signal_path` / `clock_path` for matched loading APIs.
- Add canonical `/regex/` query syntax while retaining `@regex` as a compatibility
  spelling.

### Fixed
- Guard Pattern zero-time declarative loops against infinite same-cycle execution.
- Avoid double-evaluating the first unguarded declarative `wait(...)` trigger.
- Preserve non-integer Python capture values in `MatchRecords.captures`.
- Reject invalid Pattern timeout and dynamic integer values instead of coercing them.
- Check programmable `ctx.value/cycle/time(..., offset=...)` bounds before reading waveform arrays.
- Make `Waveform.compress()` preserve value-change points and the final sample.

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
