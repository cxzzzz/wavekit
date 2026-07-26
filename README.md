# wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Downloads](https://pepy.tech/badge/wavekit)](https://pepy.tech/project/wavekit)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

English | [中文](README_ZH.md)

**wavekit** is a high-performance Python library for digital waveform analysis. It loads VCD, FST, and FSDB signals into NumPy-backed `Waveform` objects, and builds on them with efficient signal processing, protocol analysis, and automated verification APIs.

> 🤖 **AI integration**: [wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) provides an MCP server for AI-assisted waveform analysis: load signals and run pattern matching from AI tools without hand-written scripts.

## Features

- **Batch signal extraction**: use brace expansion, integer ranges, regex, and wildcards to load related signals in one call.
- **Rich analysis API**: use NumPy-style arithmetic, masking, bit slicing, edge detection, and time/cycle slicing to build signal queries in a few lines.
- **Temporal pattern matching**: use declarative and programmable Pattern APIs to extract protocol transactions, measure handshake latency, and detect timeout or hang failures.
- **High-performance waveform processing**: read VCD, FST, and FSDB files into compact NumPy-backed arrays for fast loading and efficient memory use.

## Installation

```bash
pip install wavekit
```

**Note**: To read FSDB files, the Verdi runtime library (`libNPI.so`) must be available at runtime. Configure via:
- `WAVEKIT_NPI_LIB`: direct path to `libNPI.so`
- `VERDI_HOME`: Verdi installation directory (searches `$VERDI_HOME/share/NPI/lib/...`)
- `LD_LIBRARY_PATH`: system library search path

## Quick start

> The examples below use placeholder filenames such as `sim.vcd`. Replace them with the path to your own VCD, FST, or FSDB file, and adjust signal paths to match your design hierarchy.

### 1. Batch Signal Extraction

Use brace expansion or regular expressions to load multiple related signals in one call.

```python
from wavekit import VcdReader

with VcdReader("jtag.vcd") as f:
    # Keys contain BraceCapture objects whose groups are ("state",) / ("next",).
    waves = f.load_matched_waveforms(
        "tb.u0.J_{state,next}[3:0]",
        clock_path="tb.tck",
    )

    # /regex/ is the canonical regex syntax; groups are stored by RegexCapture.
    waves = f.load_matched_waveforms(
        r"tb.u0./J_([a-z]+)/",
        clock_path="tb.tck",
    )
```

---

### 2. Signal Analysis

Waveforms support NumPy-style arithmetic, masking, and edge detection.

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

To inspect unknown/high-impedance source bits without changing the ordinary
two-state value model, load an unsigned unknown mask alongside the value
waveform.  Each mask bit is `1` where the source sample contained `X` or `Z`.

> **Experimental**: The `load_unknown_mask` / `load_matched_unknown_masks`
> APIs are experimental and may change in a future release.

```python
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    clock = "fifo_tb.clk"
    data = f.load_waveform("fifo_tb.s_fifo.data[7:0]", clock=clock, xz_value=0)
    unknown = f.load_unknown_mask("fifo_tb.s_fifo.data[7:0]", clock=clock)

    # Keep only samples whose source bits were fully known.
    known_data = data.mask(unknown == 0)
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

    # Zip mode: brace patterns expand per typed CaptureKey and evaluate per match.
    occupancies = f.eval(
        "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
        clock="tb.clk",
        mode="zip",
    )
```

---

### 4. Pattern Matching

`Pattern` scans a waveform and extracts matching transactions, such as a request/response pair, a burst, a stall interval, or another repeating timing pattern.

Choose the form that matches the transaction shape:

- **Declarative**: use `.wait()`, `.consume()`, `.capture()`, and `.loop()` steps to describe fixed flows.
- **Programmable**: use a normal Python function when the flow depends on waveform values, such as dynamic branches or per-ID routing.

In declarative patterns, the first blocking step selects transaction start cycles; later blocking steps describe timing within each matched transaction.

`match(pattern)` returns `MatchRecords`, while `collect(body)` returns a Python `list` of extracted values.

#### Declarative examples

**AXI-Lite read latency**

```python
from wavekit import VcdReader
from wavekit.pattern import Pattern, match, collect

with VcdReader("axi_tb.vcd") as f:
    clk     = "tb.clk"
    arvalid = f.load_waveform("tb.dut.arvalid",     clock=clk)
    arready = f.load_waveform("tb.dut.arready",     clock=clk)
    rvalid  = f.load_waveform("tb.dut.rvalid",      clock=clk)
    rready  = f.load_waveform("tb.dut.rready",      clock=clk)
    rdata   = f.load_waveform("tb.dut.rdata[31:0]", clock=clk)

    pattern = (
        Pattern()
        .wait(arvalid & arready)   # AR handshake → transaction starts
        .wait(rvalid  & rready)    # R  handshake → transaction ends
        .capture("rdata", rdata)
    )
    result = match(pattern)

    ok = result.filter_ok()
    latency = ok.end.clock - ok.start.clock
    print(f"Read latencies (cycles): {latency}")
    print(f"Read data: {ok.captures['rdata'].value}")
```

**AXI write burst (multi-beat)**

```python
beat = Pattern().consume(wvalid & wready, channel="w").capture("beats", wdata, mode="list")

pattern = (
    Pattern()
    .wait(awvalid & awready)   # AW handshake → burst starts
    .loop(beat, until=wlast)   # collect each beat until wlast
)
result = match(pattern)

for i, inst in enumerate(result.filter_ok()):
    print(f"Burst {i}: {len(inst.captures['beats'])} beats")
```

**Stall detection**

```python
stall = valid & (ready == 0)

pattern = (
    Pattern()
    .wait(stall.rising_edge())             # stall begins
    .loop(Pattern().delay(1), when=stall)  # keep waiting until stall ends
)
result = match(pattern)

stalls = result.filter_ok()
stall_cycles = stalls.duration.value - 1
print(f"Stall durations: {stall_cycles} cycles")
```

#### Programmable example

**DMA-style command stream**

Use programmable control flow when a command's opcode changes the following timing shape.

```python
cmd_fire = cmd_valid & cmd_ready   # precompute outside the handler
w_fire = w_valid & w_ready
rsp_fire = rsp_valid & rsp_ready
r_fire = r_valid & r_ready

OP_READ = 0
OP_WRITE = 1

def read_dma_cmd(ctx):
    if not ctx.value(cmd_fire):
        return None

    op = int(ctx.value(cmd_op))
    addr = int(ctx.value(cmd_addr))
    length = int(ctx.value(cmd_len))

    if op == OP_WRITE:
        data = []
        for _ in range(length):
            ctx.consume(w_fire, channel="wdata")
            data.append(int(ctx.value(w_data)))

        ctx.consume(rsp_fire, channel="rsp")
        return {
            "op": "write",
            "addr": addr,
            "data": data,
            "status": int(ctx.value(rsp_status)),
        }

    if op == OP_READ:
        ctx.consume(rsp_fire, channel="rsp")
        data = []
        for _ in range(length):
            ctx.consume(r_fire, channel="rdata")
            data.append(int(ctx.value(r_data)))

        return {"op": "read", "addr": addr, "data": data}

    ctx.require(False, message=f"unknown DMA op {op}")
    return None

commands = collect(read_dma_cmd)
print(f"Captured {len(commands)} commands")
```

Some tips for programmable patterns:

- Precompute fixed waveform expressions, such as `fire = valid & ready`, outside the handler so they are not rebuilt every cycle.
- Start the handler with `if ctx.value(fire): ...` to test whether the current cycle starts a transaction, and `return None` otherwise.
- Use `ctx.try_consume(...)` for non-blocking polling or arbitration between candidate channels. For a linear burst, `ctx.consume(...)` is more direct.
- Add `timeout=<cycles>` only when a blocking step needs a wait bound. In `match()`, timeout becomes `MatchStatus.Timeout(...)`; in `collect()`, it raises `PatternError`.

---

## API reference

### Reader

| Method | Description |
|--------|-------------|
| `VcdReader(file)` / `FstReader(file)` / `FsdbReader(file)` | Open a waveform file. Use as a context manager. `FsdbReader` requires Verdi runtime (`WAVEKIT_NPI_LIB`, `VERDI_HOME`, or `LD_LIBRARY_PATH`). |
| `reader.load_waveform(signal, clock, ...)` | Load one signal sampled on every clock edge. Returns `Waveform`. |
| `reader.load_unknown_mask(signal, clock, ...)` | **Experimental.** Load X/Z bit presence as an unsigned mask `Waveform`. |
| `reader.load_matched_waveforms(signal_path, clock_path, ...)` | Batch-load matching signals. Returns `dict[CaptureKey, Waveform]`. |
| `reader.load_matched_unknown_masks(signal_path, clock_path, ...)` | **Experimental.** Batch-load X/Z masks for matched signals. Returns `dict[CaptureKey, Waveform]`. |
| `reader.eval(expr, clock, mode='single'\|'zip', ...)` | Evaluate an arithmetic expression with embedded signal paths. |
| `reader.get_matched_signals(path)` | Resolve a query to `Signal` objects without loading data. |
| `reader.get_matched_scopes(path)` | Resolve a query to `Scope` objects. |
| `reader.top_scopes` | Immutable tuple of root `Scope` nodes. |
| `has_fsdb_support()` | Report whether the Verdi FSDB runtime is currently available. |

### Signal path patterns

| Syntax | Example | Effect | Capture in result key |
|--------|---------|--------|-----------------------|
| Plain name | `tb.dut.valid` | Exact-name match | None |
| `{a,b,c}` | `sig_{read,write}` | Enumerate named variants | `BraceCapture(path=..., groups=...)` |
| `{N..M}` | `fifo_{0..3}.ptr` | Integer range | `BraceCapture(path=..., groups=...)` |
| `{N..M..step}` | `lane_{0..6..2}` | Stepped range | `BraceCapture(path=..., groups=...)` |
| `/<regex>/` | `/([a-z]+)_valid/` | Canonical regex syntax with capture groups | `RegexCapture(path=..., groups=...)` |
| `@<regex>` | `@([a-z]+)_valid` | Legacy-compatible regex syntax | `RegexCapture(path=..., groups=...)` |
| `*` / `**` | `tb.*.valid` / `tb.**.valid` | Single-level / recursive wildcard | `WildcardCapture(path=...)` |
| `$ModName` | `tb.$fifo_unit.ptr` | Match a direct-child scope by module/definition name (FSDB only) | `ExactCapture(path=..., definition=...)` |
| `$$ModName` | `tb.$$fifo_unit.ptr` | Match any-depth descendant scope by module/definition name (FSDB only) | `ExactCapture(path=..., definition=...)` |

`$` and `$$` are path-step modifiers: they can combine with exact names, brace
expansion, regex, and a trailing range selector in the same query.
For example: `tb.$/fifo_(in|out)/.data_{0..3}[7:0]`.

Matched-reader and hierarchy query APIs return dictionaries keyed by
`CaptureKey = tuple[Capture, ...]`. Exact-name components are omitted, so a
fully exact query uses key `()`.

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
| `wave.unique_consecutive()` | Remove consecutive duplicate values |
| `wave.compress()` | Compact a waveform while preserving value changes and the final sample |
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
| `Pattern()` | Create a declarative Pattern. Add steps with builder methods; execution options live on `match()` / `collect()`. |
| `match(pattern_or_body, *, axis=None, timeout=None, timeout_message=None, start_cycle=None, end_cycle=None)` | Run a declarative Pattern or programmable check body and return `MatchRecords`. Check bodies return `ctx.OK` or `None`. |
| `collect(body, *, axis=None, timeout=None, timeout_message=None, start_cycle=None, end_cycle=None)` | Run a programmable extraction body and collect each non-`None` Python return value. |

**Declarative Steps**

| Method | Description |
|--------|-------------|
| `.wait(cond, *, require=None, require_message=None)` | Block until `cond` is True without consuming the event. Resumes in the same cycle when already true; use `.delay(1)` for next-cycle behavior. `require` is checked each waiting cycle (failure → `MatchStatus.RequireViolated`). |
| `.consume(cond, channel, *, require=None, require_message=None)` | Block until `cond` is True and this instance can exclusively consume from `channel`. Resumes in the same cycle on success. Use this for request/response pairing and per-key routing. |
| `.delay(n, *, require=None, require_message=None)` | Advance `n` cycles. `delay(0)` is a no-op. `require` must hold every cycle. |
| `.capture(name, signal, *, mode='last')` | Record signal value at current cycle. `mode='last'` (default) overwrites; `'first'` keeps the first write; `'list'` appends to a list. |
| `.require(cond)` | Assert condition; fail with `MatchStatus.RequireViolated` if False. |
| `.loop(body, *, until=None, when=None)` | `until`: do-while (exit when True after body). `when`: while (exit when False before body). |
| `.repeat(body, n)` | Execute body exactly `n` times. `n` may be a callable. |
| `.branch(cond, true_body, false_body)` | Conditional branch. |

The same time and ownership operations are available inside Programmable
patterns as `ctx.wait(...)`, `ctx.consume(...)`, and `ctx.delay(...)`.

**Programmable Context**

| API | Description |
|-----|-------------|
| `ctx.value(waveform, offset=0)` | Read a scalar value at the current sample plus optional offset. |
| `ctx.cycle(waveform, offset=0)` | Read the cycle number at the current sample plus optional offset. |
| `ctx.time(waveform, offset=0)` | Read the timestamp at the current sample plus optional offset. |
| `ctx.wait(cond, require=None, require_message=None)` | Observe cycles until `cond` is true; does not consume the event. |
| `ctx.consume(cond, channel, require=None, require_message=None)` | Wait for `cond` and exclusively consume from `channel`. |
| `ctx.try_consume(cond, channel)` | Poll `channel` without blocking. Returns `True` only when both the condition and channel are available. |
| `ctx.delay(n, require=None, require_message=None)` | Advance `n` cycles. |
| `ctx.capture(name, value, mode='last')` | Record a capture for programmable `match()`. |
| `ctx.OK` | Return from programmable `match()` to record a successful match. |

**Dynamic callbacks**

Callback arguments depend on whether the callback is used in the declarative API or the programmable API:

- In declarative APIs such as `Pattern().wait(...)` and `Pattern().consume(...)`, callbacks receive `(index, captures)`.
- In programmable functions, callbacks passed to methods such as `ctx.wait(...)` and `ctx.consume(...)` take no arguments. If they need the current index, captures, or signal values, close over `ctx`.

**Channels and consume vs. wait**

`wait()` is observational: every matching instance can see the same event.
`consume()` adds ownership: only one instance can claim a given `(channel, cycle)`.
On the same channel, matches with earlier start cycles claim available events first.

A `Channel` is an identity token for consume ownership. Pass a `Channel` object,
a hashable key, or a dynamic callback to `consume(..., channel=...)`. All
instances sharing the same channel key compete for the same logical channel.

```python
from collections import defaultdict
from wavekit.pattern import Channel, Pattern, match

# Multi-bank cache: each bank has its own response port, so two banks can return
# data in the same cycle. Per-bank channels let independent requests consume
# independent response streams.
banks = defaultdict(Channel)

pattern = (
    Pattern()
    .wait(req_valid)
    .capture('bank', req_addr & 1)
    .consume(
        lambda i, cap: bank_valid[cap['bank']].value[i],
        channel=lambda i, cap: banks[cap['bank']],
    )
    .capture('rdata',
        lambda i, cap: bank_data[cap['bank']].value[i])
)
result = match(pattern)
```

**`MatchRecords`**

| Field | Description |
|-------|-------------|
| `.start` / `.end` | Point Waveforms. `.value` is the waveform-array sample index, `.clock` is the absolute cycle, and `.time` is the simulation timestamp. End is inclusive. |
| `.duration` | `end.value - start.value + 1` sampled cycles. |
| `.status` | `MatchStatus.OK()`, `MatchStatus.Timeout(...)`, or `MatchStatus.RequireViolated(...)`. |
| `.captures` | `dict[str, Waveform]` of captured values. |
| `.ok` | Boolean Waveform where `status == MatchStatus.OK()`. |
| `.failed` | Boolean Waveform where `status != MatchStatus.OK()`. |
| `.filter_ok()` | Return only `OK` matches. |
| `.filter_status(status_class)` | Return only matches with the given status class, such as `MatchStatus.Timeout`. |
| `.filter_failed()` | Return only non-OK matches. |

`MatchRecords[i]` returns a single `MatchRecord`, and slices return another
`MatchRecords` batch.

---

## Development

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

## Contributing

Issues and pull requests are welcome. Before sending a pull request, run the tests and format checks:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
