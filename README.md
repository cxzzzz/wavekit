# wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

**Wavekit** is a fundamental Python library for digital waveform analysis. By seamlessly converting VCD and FSDB data into Numpy arrays, it empowers engineers to perform high-performance signal processing, protocol analysis, and automated verification with ease.

## ✨ Features

- **High Performance & Easy Loading**: Cython-optimized VCD/FSDB parsers with Numpy-backed storage for speed and memory efficiency, plus flexible batch signal extraction via brace expansion, integer ranges, and regular expressions.
- **Rich Analysis Tools**: Numpy-like API for arithmetic, masking, bit-field manipulation, edge detection, and time/cycle slicing — compose complex signal queries in just a few lines.
- **Pattern Matching**: NFA-based temporal pattern engine that scans waveforms in a single pass to extract protocol transactions, measure latencies, and detect timing violations.

## 📦 Installation

```bash
pip install wavekit
```

**Note**: To read FSDB files, the Verdi runtime library (`libNPI.so`) must be available at runtime. Configure via:
- `WAVEKIT_NPI_LIB` — direct path to `libNPI.so`
- `VERDI_HOME` — Verdi installation directory (searches `$VERDI_HOME/share/NPI/lib/...`)
- `LD_LIBRARY_PATH` — system library search path

## 🚀 Quick Start

> The examples below use placeholder filenames such as `sim.vcd`. Replace them with the path to your own VCD or FSDB file, and adjust signal paths to match your design hierarchy.

### 1. Batch Signal Extraction

Use brace expansion or regular expressions to load multiple related signals in one call.

```python
from wavekit import VcdReader

with VcdReader("jtag.vcd") as f:
    # Brace expansion: load J_state and J_next in one call
    # Returns: { ('state',): Waveform, ('next',): Waveform }
    waves = f.load_matched_waveforms(
        "tb.u0.J_{state,next}[3:0]",
        clock_pattern="tb.tck",
    )

    # Regex mode (@ prefix): capture groups become dict keys
    waves = f.load_matched_waveforms(
        r"tb.u0.@J_([a-z]+)",
        clock_pattern="tb.tck",
    )
```

---

### 2. Signal Analysis

Waveforms support Numpy-style arithmetic, masking, and edge detection out of the box.

```python
import numpy as np
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    clock = "fifo_tb.clk"
    depth = 8

    w_ptr = f.load_waveform("fifo_tb.s_fifo.w_ptr[2:0]", clock=clock)
    r_ptr = f.load_waveform("fifo_tb.s_fifo.r_ptr[2:0]", clock=clock)
    wr_en = f.load_waveform("fifo_tb.s_fifo.wr_en",      clock=clock)

    occupancy = (w_ptr + depth - r_ptr) % depth
    print(f"Average occupancy: {np.mean(occupancy.value):.2f}")

    # Filter to cycles where a write is active
    write_occ = occupancy.mask(wr_en == 1)

    # Detect write bursts
    burst_cycles = wr_en.rising_edge()
```

---

### 3. Expression Evaluation

Compute waveform expressions directly from signal path strings without loading each signal manually.

```python
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    # Single mode: paths must each match exactly one signal
    occupancy = f.eval(
        "fifo_tb.s_fifo.w_ptr[2:0] - fifo_tb.s_fifo.r_ptr[2:0]",
        clock="fifo_tb.clk",
    )

    # Zip mode: brace patterns expand per key, evaluated once per match
    # Returns: { (0,): Waveform, (1,): Waveform, (2,): Waveform, (3,): Waveform }
    occupancies = f.eval(
        "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
        clock="tb.clk",
        mode="zip",
    )
```

---

### 4. Pattern Matching

Describe a temporal sequence of events; the engine finds all matching transactions in one pass.

**AXI-lite Read Latency**

```python
from wavekit import VcdReader, Pattern

with VcdReader("axi_tb.vcd") as f:
    clk     = "tb.clk"
    arvalid = f.load_waveform("tb.dut.arvalid",     clock=clk)
    arready = f.load_waveform("tb.dut.arready",     clock=clk)
    rvalid  = f.load_waveform("tb.dut.rvalid",      clock=clk)
    rready  = f.load_waveform("tb.dut.rready",      clock=clk)
    rdata   = f.load_waveform("tb.dut.rdata[31:0]", clock=clk)

result = (
    Pattern()
    .wait(arvalid & arready)   # AR handshake → start
    .wait(rvalid  & rready)    # R  handshake → end
    .capture("rdata", rdata)
    .timeout(256)
    .match()
)

valid = result.filter_valid()
print(f"Read latencies (cycles): {valid.duration.value}")
print(f"Read data: {valid.captures['rdata'].value}")
```

**AXI Write Burst (multi-beat)**

```python
beat = Pattern().wait(wvalid & wready).capture("beats[]", wdata)

result = (
    Pattern()
    .wait(awvalid & awready)   # AW handshake → burst start
    .loop(beat, until=wlast)   # collect beats until wlast
    .timeout(512)
    .match()
)

for i, inst in enumerate(result.filter_valid()):
    print(f"Burst {i}: {len(inst.captures['beats'])} beats")
```

**Stall Detection**

```python
stall = valid & (ready == 0)

result = (
    Pattern()
    .wait(stall.rising_edge())             # stall begins
    .loop(Pattern().delay(1), when=stall)  # wait while stalling
    .match()
)

stalls = result.filter_valid()
print(f"Stall durations: {stalls.duration.value} cycles")
```

---

## 📖 API Reference

### Reader

| Method | Description |
|--------|-------------|
| `VcdReader(file)` / `FsdbReader(file)` | Open a waveform file. Use as a context manager. `FsdbReader` requires Verdi runtime (`WAVEKIT_NPI_LIB`, `VERDI_HOME`, or `LD_LIBRARY_PATH`). |
| `reader.load_waveform(signal, clock, ...)` | Load one signal sampled on every clock edge. Returns `Waveform`. |
| `reader.load_matched_waveforms(pattern, clock_pattern, ...)` | Batch-load signals matching a brace/regex pattern. Returns `dict[tuple, Waveform]`. |
| `reader.eval(expr, clock, mode='single'\|'zip', ...)` | Evaluate an arithmetic expression with embedded signal paths. |
| `reader.get_matched_signals(pattern)` | Resolve a pattern to signal paths without loading data. |
| `reader.top_scope_list()` | Return root `Scope` nodes of the signal hierarchy. |

**Pattern syntax** used in signal paths:

| Syntax | Example | Effect |
|--------|---------|--------|
| `{a,b,c}` | `sig_{read,write}` | Enumerate named variants |
| `{N..M}` | `fifo_{0..3}.ptr` | Integer range |
| `{N..M..step}` | `lane_{0..6..2}` | Stepped range |
| `@<regex>` | `@([a-z]+)_valid` | Regex with capture groups |
| `$ModName` | `tb.$fifo_unit.ptr` | Match a direct-child scope by module/definition name (FSDB only) |
| `$$ModName` | `tb.$$fifo_unit.ptr` | Match any-depth descendant scope by module/definition name (FSDB only) |

---

### Waveform

A `Waveform` wraps three parallel numpy arrays (`.value`, `.clock`, `.time`). All operations return a new `Waveform`.

**Arithmetic & comparison**: `+`, `-`, `*`, `//`, `%`, `**`, `/`, `&`, `|`, `^`, `~`, `==`, `!=`, `<<`, `>>`

**Filtering & slicing**

| Method | Description |
|--------|-------------|
| `wave.mask(mask)` | Keep samples where a boolean Waveform or array is True |
| `wave.filter(fn)` | Keep samples where `fn(value)` is True |
| `wave.cycle_slice(begin, end)` | Trim to clock cycle range `[begin, end)` |
| `wave.time_slice(begin, end)` | Trim to simulation time range |
| `wave.slice(begin_idx, end_idx)` | Trim by array index |
| `wave.take(indices)` | Select samples at given indices |

**Transformation**

| Method | Description |
|--------|-------------|
| `wave.map(fn, width, signed)` | Element-wise transform |
| `wave.unique_consecutive()` | Remove consecutive duplicates |
| `wave.downsample(chunk, fn)` | Aggregate into chunks |
| `wave.as_signed()` / `wave.as_unsigned()` | Reinterpret signedness |

**Bit manipulation**

| Method / Syntax | Description |
|-----------------|-------------|
| `wave[high:low]` | Extract bit field (Verilog convention, returns unsigned) |
| `wave[n]` | Extract single bit |
| `wave.split_bits(n)` | Split into n-bit groups (LSB first) |
| `Waveform.concatenate([w0, w1, ...])` | Concatenate (w0 = LSB) |
| `wave.bit_count()` | Population count |

**Edge detection** (1-bit only)

| Method | Description |
|--------|-------------|
| `wave.rising_edge()` | True at 0→1 transitions |
| `wave.falling_edge()` | True at 1→0 transitions |

---

### Pattern

| Method | Description |
|--------|-------------|
| `.wait(cond, guard=None, channel=None)` | Block until `cond` is True. `guard` is checked each waiting cycle. `channel` enforces FIFO ordering among concurrent instances. |
| `.delay(n, guard=None)` | Advance `n` cycles. `delay(0)` is a no-op. |
| `.capture(name, signal)` | Record signal value at current cycle. `name[]` appends to a list. |
| `.require(cond)` | Assert condition; fail with `REQUIRE_VIOLATED` if False. |
| `.loop(body, *, until=None, when=None)` | `until`: do-while (exit when True after body). `when`: while (exit when False before body). |
| `.repeat(body, n)` | Execute body exactly `n` times. `n` may be a callable. |
| `.branch(cond, true_body, false_body)` | Conditional branch. |
| `.timeout(max_cycles)` | Terminate unfinished instances with `TIMEOUT`. |
| `.match(start_cycle=None, end_cycle=None)` | Run the engine; return `MatchResult`. |

**`MatchResult`**

| Field | Description |
|-------|-------------|
| `.start` / `.end` | Start and end cycle of each match (both inclusive). |
| `.duration` | `end - start + 1` cycles. |
| `.status` | `MatchStatus.OK`, `TIMEOUT`, or `REQUIRE_VIOLATED`. |
| `.captures` | `dict[str, Waveform]` of captured values. |
| `.filter_valid()` | Return only `OK` matches. |

---

## 🛠️ Development

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Setup

```bash
git clone https://github.com/cxzzzz/wavekit.git
cd wavekit
poetry install
```

### Testing

Tests are located in the `tests/` directory and run with [pytest](https://pytest.org/).

```bash
# Run all tests
poetry run pytest

# Run a specific test file
poetry run pytest tests/test_pattern.py

# Run with verbose output
poetry run pytest -v
```

### Linting & Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check for lint errors
poetry run ruff check .

# Check formatting (no changes)
poetry run ruff format --check .

# Auto-fix formatting
poetry run ruff format .
```

### Type Checking

```bash
poetry run mypy .
```

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss a bug or feature request before submitting a pull request. When contributing code, make sure all tests pass and the linter reports no errors:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
