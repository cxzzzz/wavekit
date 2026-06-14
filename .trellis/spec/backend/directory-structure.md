# Directory Structure

> How backend code is organized in this project.

---

## Overview

wavekit is a Python library for digital waveform analysis. The codebase follows a
clear separation between:

- **Core domain types** (`waveform.py`, `signal.py`, `scope.py`)
- **File readers** (`readers/` directory with VCD and FSDB implementations)
- **Pattern matching engine** (`pattern/` directory)

---

## Directory Layout

```
src/wavekit/
├── __init__.py              # Public API exports
├── waveform.py              # Core Waveform class (numpy-backed time series)
├── signal.py                # Signal metadata descriptor
├── scope.py                 # Hierarchical scope tree traversal
└── readers/
    ├── base.py              # Abstract Reader base class
    ├── pattern_parser.py    # Brace/regex pattern expansion
    ├── expr_parser.py       # Expression parsing for eval()
    ├── vcd/
    │   └── reader.py        # VcdReader implementation
    ├── fsdb/
    │   └── reader.py        # FsdbReader implementation (requires Verdi runtime)
    └── fst/
        └── reader.py        # FstReader implementation (requires pylibfst package)
    pattern/
    ├── __init__.py          # Pattern, MatchResult, MatchStatus exports
    ├── dsl.py               # Pattern builder DSL (wait, capture, etc.)
    ├── steps.py             # Read-only declarative AST node definitions
    ├── compiler.py          # Declarative AST to async PatternRuntime program compiler
    ├── runtime.py           # Unified runtime for declarative and programmable patterns
    ├── errors.py            # PatternError
    └── result.py            # MatchResult struct-of-arrays output
tests/
├── test_waveform.py         # Waveform arithmetic, bit ops, filtering
├── test_pattern.py          # Pattern matching engine tests
├── test_vcdreader.py        # VCD reader tests
└── test_examples.py         # Integration tests with example waveforms
```

---

## Module Organization

### Core Types (`src/wavekit/`)

| File | Purpose |
|------|---------|
| `waveform.py` | Clock-synchronised numpy-backed time series with arithmetic/bit ops |
| `signal.py` | Immutable metadata descriptor (name, width, signed, range) |
| `scope.py` | Hierarchical scope tree for signal discovery |

### Readers (`src/wavekit/readers/`)

- **`base.py`** — Abstract `Reader` class with shared logic for:
  - `load_waveform()` — single signal loading
  - `load_matched_waveforms()` — batch pattern matching
  - `eval()` — expression evaluation with inline signal paths
  - `get_matched_signals()` — pattern resolution without loading

- **`vcd/reader.py`** — VCD format implementation using `vcdvcd` library
- **`fsdb/reader.py`** — FSDB format implementation using Verdi's NPI library
- **`fst/reader.py`** — FST format implementation using required `pylibfst` package

### Reader Dependency Policy

- Readers backed by normal Python package dependencies should be imported directly
  from the public API. Example: `FstReader` depends on `pylibfst`, which is declared
  in `pyproject.toml` as a normal dependency.
- Readers backed by external vendor runtimes may use a runtime availability stub to
  keep unrelated features importable. Example: `FsdbReader` depends on Verdi's
  `libNPI.so`, so `wavekit.__init__` exposes a stub when that runtime is missing.

### Pattern Engine (`src/wavekit/pattern/`)

- **`dsl.py`** — Fluent builder API (`Pattern().wait().capture()`) and public execution entry points
- **`steps.py`** — Read-only declarative pattern AST nodes; do not store per-instance runtime state here
- **`compiler.py`** — Internal declarative `Step` AST to async program compiler
- **`runtime.py`** — Unified cycle-major runtime for programmable patterns and compiled declarative patterns
- **`errors.py`** — Pattern-specific exception type
- **`result.py`** — Struct-of-arrays output with filter/iteration

Declarative and programmable patterns must share `PatternRuntime` as the only
production execution backend. Do not add a second matching engine; translate new
declarative constructs through `compiler.py` and execute them through `runtime.py`.

Pattern time movement is explicit. Do not add `tick` compatibility back to
declarative steps or runtime ops; use `.delay(1)` for next-cycle continuation.
Blocking guards belong on public wait/consume/delay APIs in both modes:
`Pattern().wait(..., require=...)`, `Pattern().consume(..., channel=..., require=...)`,
`Pattern().delay(..., require=...)`, `await ctx.wait(..., require=...)`,
`await ctx.consume(..., channel=..., require=...)`, and `await ctx.delay(..., require=...)`.
Plain `wait` is observational and non-consuming in both declarative and
programmable APIs; exclusive/FIFO event ownership must be spelled explicitly
with `consume(cond, channel)`, where `channel` can be a `Channel`, a hashable key,
or a dynamic callable returning either.

---

## Naming Conventions

### Files
- Lowercase with underscores: `waveform.py`, `pattern_parser.py`
- Test files prefixed with `test_`: `test_waveform.py`

### Classes
- PascalCase: `Waveform`, `VcdReader`, `MatchResult`
- Abstract bases: `Reader`, `Scope`

### Functions/Methods
- snake_case: `load_waveform()`, `rising_edge()`, `filter_valid()`
- Private helpers prefixed with `_`: `_eval_bool()`, `_collect_waveforms()`

### Variables
- snake_case for locals: `width`, `signed`, `clock`
- Single underscore for unused: `_`

---

## Examples

**Adding a new reader type:**
```
src/wavekit/readers/
└── <format>/
    └── reader.py        # Inherit from Reader, implement abstract methods
```

Reference: `src/wavekit/readers/vcd/reader.py:VcdReader`

**Adding a new pattern step:**
1. Add step class to `src/wavekit/pattern/steps.py`
2. Add builder method to `src/wavekit/pattern/dsl.py`
3. Add declarative compilation handling to `src/wavekit/pattern/compiler.py`
4. Add runtime support to `src/wavekit/pattern/runtime.py` only if the existing ops cannot express it
