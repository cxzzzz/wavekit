# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added
- Add exact-path `Reader.get_signal()` and `Reader.get_scope()` hierarchy lookups.
- Allow declarative `Pattern.consume()` to create a private channel when
  `channel` is omitted.
- Add `Waveform.changed()` for value changes and `Waveform.any_edge()` for
  either edge of a one-bit waveform.

## v0.7.2 - 2026-08-25

### Added
- Extend `Reader.eval()` to support richer waveform expressions, including
  function calls, nested calls, and complete matcher-path expressions such as
  `rising_edge(falling_edge(tb.dut.**./valid/))`.
- Add relational Waveform operators `<`, `<=`, `>`, and `>=`.

## v0.7.1 - 2026-08-23

### Changed
- Improve repeated recursive query-path matching performance for `**` and `$$`
  by caching reusable hierarchy candidates.
- Refactor recursive capture restoration and matcher range-selection handling
  while keeping ordinary query-path behavior unchanged.
- Suppress the NPI startup banner by default when reading FSDB files.
  Pass `FsdbReader(..., quiet=False)` to re-enable it.
- Remove the pynpi bootstrap; FSDB runtime initialization now calls
  `npi_init` directly through the Cython reader.

## v0.7.0 - 2026-07-26

### Breaking Changes
- Redesign Pattern imports and execution. Pattern APIs now live under
  `wavekit.pattern`, and execution uses module-level `match(...)` /
  `collect(...)` instead of `Pattern().match()` / `Pattern().timeout(...)`.

- Replace `MatchResult` with `MatchRecords` / `MatchRecord`. Results now support
  row access with `records[i]`, slicing, structured start/end points, status
  objects, and capture columns.

- Replace enum-style Pattern statuses and `valid` helpers. Use
  `MatchStatus.OK()`, `MatchStatus.Timeout(message)`,
  `MatchStatus.RequireViolated(message)`, plus `ok`, `failed`,
  `filter_ok()`, `filter_failed()`, and `filter_status(...)`.

- Update Pattern timing and blocking semantics. `wait(...)` observes events,
  `consume(...)` claims events exclusively for channel-based pairing, and
  successful blocking steps continue in the same cycle. Use `delay(1)` for
  next-cycle behavior.

- Rename matched-reader parameters from `pattern` / `clock_pattern` to
  `signal_path` / `clock_path`.

- Change matched-reader keys from plain captured values to typed `CaptureKey`
  tuples containing capture objects such as `BraceCapture` and `RegexCapture`.

- Replace legacy `wavekit.scope`, `wavekit.signal`, and
  `wavekit.readers.pattern_parser` modules with the unified hierarchy/query
  model under `wavekit.readers`.

### Added
- Add programmable Pattern execution with normal Python functions through
  `match(body)` for checks and `collect(body)` for extraction.

- Add Pattern failure diagnostics with `require_message=` and
  `timeout_message=`.

- Add experimental unknown-mask loading APIs:
  `load_unknown_mask(...)` and `load_matched_unknown_masks(...)`.

- Add unified query matcher support for canonical `/regex/` syntax, `*` / `**`
  wildcards, and typed capture objects.

### Changed
- Unify reader hierarchy, signal resolution, query matching, and matched loading
  across VCD, FST, and FSDB readers.

- Improve range handling for native ranges, non-zero ranges, scalar bit selects,
  packed ranges, and FSDB leaf signals.

- Improve `Waveform.__repr__` / `__str__`.

- Make `Waveform.compress()` preserve value-change points and the final sample.

### Fixed
- Fix Pattern zero-time declarative loops so they cannot run forever in the same
  cycle.

- Avoid double-evaluating the first unguarded declarative `wait(...)` trigger.

- Improve VCD/FST/FSDB reader behavior for empty signals, invalid ranges,
  composite signals, and X/Z handling.

- Validate `xz_value` consistently.

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
