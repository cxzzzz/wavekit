# wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Downloads](https://img.shields.io/pypi/dm/wavekit.svg)](https://pypi.org/project/wavekit/)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

English | [中文](README_ZH.md)

**Wavekit** is a fundamental Python library for digital waveform analysis. By seamlessly converting VCD, FST, and FSDB data into Numpy arrays, it empowers engineers to perform high-performance signal processing, protocol analysis, and automated verification with ease.

> 🤖 **AI Integration**: [wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) — MCP server for AI-assisted waveform analysis. Let AI load signals and run pattern matching — no manual coding required.

## ✨ Features

- **Flexible Signal Extraction**: Flexible batch signal extraction via brace expansion, integer ranges, and regular expressions — load groups of related signals in one call.
- **Rich Analysis Tools**: Numpy-like API for arithmetic, masking, bit-field manipulation, edge detection, and time/cycle slicing — compose complex signal queries in just a few lines.
- **Pattern Matching**: Unified temporal pattern runtime that scans waveforms in a single pass to extract protocol transactions, measure latencies, and detect timing violations.
- **High-Performance Parsing & Storage**: VCD, FST, and FSDB readers with Numpy-backed storage for fast loading and memory efficiency, handling large simulation files with ease.

## 📦 Installation

```bash
pip install wavekit
```

**Note**: To read FSDB files, the Verdi runtime library (`libNPI.so`) must be available at runtime. Configure via:
- `WAVEKIT_NPI_LIB` — direct path to `libNPI.so`
- `VERDI_HOME` — Verdi installation directory (searches `$VERDI_HOME/share/NPI/lib/...`)
- `LD_LIBRARY_PATH` — system library search path

## 🚀 Quick Start

> The examples below use placeholder filenames such as `sim.vcd`. Replace them with the path to your own VCD, FST, or FSDB file, and adjust signal paths to match your design hierarchy.

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

`Pattern` scans a waveform and extracts all matching transactions — a request/response pair, a burst, a stall interval, or any other repeating timing pattern.

There are two ways to describe a pattern:

- **Declarative** — chain steps like `.wait()`, `.consume()`, `.capture()`, `.loop()`. Best for fixed transaction flows.
- **Programmable** — pass a handler function to `Pattern(...)`. Best for dynamic branches, per-ID routing, and other complex flows.

#### Declarative examples

**AXI-Lite read latency**

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
        Pattern(timeout=256)
        .wait(arvalid & arready)   # AR handshake → transaction starts
        .wait(rvalid  & rready)    # R  handshake → transaction ends
        .capture("rdata", rdata)
        .match()
    )

    ok = result.filter_ok()
    print(f"Read latencies (cycles): {ok.duration.value}")
    print(f"Read data: {ok.captures['rdata'].value}")
```

**AXI write burst (multi-beat)**

```python
beat = Pattern().consume(wvalid & wready, channel="w").capture("beats", wdata, mode="list")

result = (
    Pattern()
    .wait(awvalid & awready)   # AW handshake → burst starts
    .loop(beat, until=wlast)   # collect each beat until wlast
    .timeout(512)
    .match()
)

for i, inst in enumerate(result.filter_ok()):
    print(f"Burst {i}: {len(inst.captures['beats'])} beats")
```

**Stall detection**

```python
stall = valid & (ready == 0)

result = (
    Pattern()
    .wait(stall.rising_edge())             # stall begins
    .loop(Pattern().delay(1), when=stall)  # keep waiting until stall ends
    .match()
)

stalls = result.filter_ok()
print(f"Stall durations: {stalls.duration.value} cycles")
```

#### Programmable example

**Out-of-order AXI reads by ID**

When R beats from different IDs interleave on the bus, match each AR to its response beats by `arid` and collect results as Python dicts.

```python
arfire = arvalid & arready   # precompute outside the handler
rfire = rvalid & rready

async def read_burst(ctx):
    if ctx.value(arfire):
        my_id = ctx.value(arid)
        beats = []

        while True:
            await ctx.consume(
                lambda: ctx.value(rfire) and ctx.value(rid) == my_id,
                channel=("r", my_id),
            )
            beats.append(int(ctx.value(rdata)))
            if ctx.value(rlast):
                break

        return {"arid": my_id, "beats": beats}
    return None

records = Pattern(read_burst, timeout=64).collect()
```

Some tips for programmable patterns:

- Precompute fixed waveform expressions (like `fire = valid & ready`) outside
  the handler function so they aren't rebuilt every cycle.
- Start the handler with `if ctx.value(fire): ...` and `return None`
  otherwise — this tells the runtime which cycles begin a transaction.

---

## 📖 API Reference

### Reader

| Method | Description |
|--------|-------------|
| `VcdReader(file)` / `FstReader(file)` / `FsdbReader(file)` | Open a waveform file. Use as a context manager. `FsdbReader` requires Verdi runtime (`WAVEKIT_NPI_LIB`, `VERDI_HOME`, or `LD_LIBRARY_PATH`). |
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

**Relative time access**

| Method | Description |
|--------|-------------|
| `wave.relative(offset, pad, pad_value)` | Shift by *offset* cycles (positive = future, negative = past) |
| `wave.ahead(n, pad, pad_value)` | Look *n* cycles into the future (shorthand for `relative(n)`) |
| `wave.back(n, pad, pad_value)` | Look *n* cycles into the past (shorthand for `relative(-n)`) |

`pad` controls boundary handling: `'repeat'` (default) pads with the first/last value, `'value'` pads with a given `pad_value`.

```python
# Rising edge detection
rising = (wave == 0) & wave.ahead()

# Compare current vs 3 cycles ago
changed = wave != wave.back(3)
```

---

### Pattern

**Construction**

| API | Description |
|-----|-------------|
| `Pattern(timeout=..., max_active=...)` | Create a Declarative Pattern. Add steps with builder methods, then call `.match()`. |
| `Pattern(async_fn, timeout=..., max_active=...)` | Create a Programmable Pattern. The async function receives `ctx`. |
| `.match(start_cycle=None, end_cycle=None)` | Run the pattern and return `MatchResult`. In a Programmable Pattern, return `ctx.OK` for success and `None` to skip. |
| `.collect(start_cycle=None, end_cycle=None)` | Programmable Pattern only. Collect each non-`None` Python return value. |

**Declarative Steps**

| Method | Description |
|--------|-------------|
| `.wait(cond, *, require=None)` | Block until `cond` is True without consuming the event. Resumes in the same cycle when already true; use `.delay(1)` for next-cycle behavior. `require` is checked each waiting cycle (failure → `REQUIRE_VIOLATED`). |
| `.consume(cond, channel, *, require=None)` | Block until `cond` is True and this instance can exclusively consume from `channel`. Resumes in the same cycle on success. Use this for FIFO request/response pairing and per-key routing. |
| `.delay(n, *, require=None)` | Advance `n` cycles. `delay(0)` is a no-op. `require` must hold every cycle. |
| `.capture(name, signal, *, mode='last')` | Record signal value at current cycle. `mode='last'` (default) overwrites; `'first'` keeps the first write; `'list'` appends to a list. |
| `.require(cond)` | Assert condition; fail with `REQUIRE_VIOLATED` if False. |
| `.loop(body, *, until=None, when=None)` | `until`: do-while (exit when True after body). `when`: while (exit when False before body). |
| `.repeat(body, n)` | Execute body exactly `n` times. `n` may be a callable. |
| `.branch(cond, true_body, false_body)` | Conditional branch. |
| `.timeout(max_cycles)` | Terminate unfinished instances with `TIMEOUT`. |

The same time and ownership operations are available inside Programmable
Patterns as `await ctx.wait(...)`, `await ctx.consume(...)`, and
`await ctx.delay(...)`.

**Programmable Context**

| API | Description |
|-----|-------------|
| `ctx.value(waveform, offset=0)` | Read a scalar value at the current sample plus optional offset. |
| `ctx.cycle(waveform, offset=0)` | Read the cycle number at the current sample plus optional offset. |
| `ctx.time(waveform, offset=0)` | Read the timestamp at the current sample plus optional offset. |
| `await ctx.wait(cond, require=None)` | Observe cycles until `cond` is true; does not consume the event. |
| `await ctx.consume(cond, channel, require=None)` | Wait for `cond` and exclusively consume from `channel`. |
| `await ctx.delay(n, require=None)` | Advance `n` cycles. |
| `ctx.capture(name, value, mode='last')` | Record a capture for programmable `.match()`. |
| `ctx.OK` | Return from programmable `.match()` to record a successful match. |

**Dynamic callbacks**

Declarative callbacks use `callable(index, captures)`. `index` is the absolute
waveform sample index; it is not rebased when `match(start_cycle=...)` is used.

**Channels and consume vs. wait**

When multiple in-flight instances are waiting for the same kind of event, plain
`wait()` won't pair each request with its own response — every instance sees
every event. `consume()` solves this: it hands the event to exactly one instance
per cycle, in FIFO order.

A `Channel` is the FIFO queue that `consume()` uses. Pass a `Channel` object, a
hashable key, or a dynamic callback to `consume(..., channel=...)`. All
instances sharing the same channel key consume from the same queue.

```python
from collections import defaultdict
from wavekit import Channel, Pattern

# Multi-bank cache: each bank has its own response port, so two banks
# can return data in the *same* cycle. A per-bank Channel lets each in-flight
# read consume from its own bank while preserving FIFO order within that bank.
banks = defaultdict(Channel)

result = (
    Pattern()
    .wait(req_valid)
    .capture('bank', req_addr & 1)
    .consume(
        lambda i, cap: bank_valid[cap['bank']].value[i],
        channel=lambda i, cap: banks[cap['bank']],
    )
    .capture('rdata',
        lambda i, cap: bank_data[cap['bank']].value[i])
    .match()
)
```

**`MatchResult`**

| Field | Description |
|-------|-------------|
| `.start` / `.end` | Start and end cycle of each match (both inclusive). |
| `.duration` | `end - start + 1` cycles. |
| `.status` | `MatchStatus.OK`, `TIMEOUT`, or `REQUIRE_VIOLATED`. |
| `.captures` | `dict[str, Waveform]` of captured values. |
| `.ok` | Boolean Waveform where `status == MatchStatus.OK`. |
| `.failed` | Boolean Waveform where `status != MatchStatus.OK`. |
| `.filter_ok()` | Return only `OK` matches. |
| `.filter_status(status)` | Return only matches with the given `MatchStatus` or integer status. |
| `.filter_failed()` | Return only non-OK matches. |

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
