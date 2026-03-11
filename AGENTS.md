# wavekit — AI Agent Reference

This file is a concise, structured guide for AI agents using the wavekit library
to analyse hardware simulation waveforms (VCD / FSDB).

---

## What wavekit does

Parse VCD or FSDB waveform files, extract digital signals as numpy arrays,
perform clock-synchronised time-series analysis, and extract protocol
transactions using a temporal pattern matching engine.

---

## Core workflow

```
1. Open a Reader  ->  2. Load signals as Waveform objects  ->  3. Operate on Waveforms  ->  4. Extract numpy results
```

```python
from wavekit import VcdReader

with VcdReader("sim.vcd") as r:
    data  = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
    valid = r.load_waveform("tb.dut.valid",      clock="tb.clk")

valid_data = data.mask(valid == 1)     # keep cycles where valid is high
print(valid_data.value)                # numpy array of integer values
```

---

## Reader — loading signals

### `VcdReader(file: str)` / `FsdbReader(file: str)`

Open a waveform file.  Use as a context manager (`with`) to ensure the file is
closed.  `FsdbReader` requires the `VERDI_HOME` environment variable to be set.

---

### `reader.load_waveform(signal, clock, ...) -> Waveform`

Load one signal, sampled on every clock edge.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `signal` | `str` | required | Full dotted path, e.g. `"tb.dut.data[7:0]"`. Range suffix optional. |
| `clock` | `str` | required | Full dotted path of the clock, e.g. `"tb.clk"`. |
| `xz_value` | `int` | `0` | Value substituted for X/Z states. |
| `signed` | `bool` | `False` | Interpret values as two's-complement signed. |
| `sample_on_posedge` | `bool` | `False` | `False` = sample on negedge (default); `True` = posedge. |
| `begin_time` | `int\|None` | `None` | Start of time window (inclusive, file time units). Mutually exclusive with `begin_cycle`. |
| `end_time` | `int\|None` | `None` | End of time window (exclusive). Mutually exclusive with `end_cycle`. |
| `begin_cycle` | `int\|None` | `None` | Start of window as absolute clock cycle number (inclusive). Mutually exclusive with `begin_time`. |
| `end_cycle` | `int\|None` | `None` | End of window as absolute clock cycle number (exclusive). Mutually exclusive with `end_time`. |

**Clock cycle semantics**: the `.clock` array in every `Waveform` holds **absolute** cycle numbers counted from the start of simulation (cycle 0 = first sampling edge in the file). The clock signal is always loaded in full so cycle numbers are consistent across multiple `load_waveform` calls, regardless of `begin_time`/`begin_cycle`.

---

### `reader.load_matched_waveforms(pattern, clock_pattern, ...) -> dict[tuple, Waveform]`

Batch-load all signals matching a pattern.  Returns a dict keyed by the
captured pattern values.

**Clock assignment:**
- If `clock_pattern` matches **one** signal -> that clock is shared by all.
- If `clock_pattern` matches **multiple** signals -> keys must match signal keys
  exactly (per-signal clock).

```python
# Single clock broadcast
waves = r.load_matched_waveforms("tb.dut.fifo_{0..3}.w_ptr[2:0]", "tb.clk")
# -> { (0,): Waveform, (1,): Waveform, (2,): Waveform, (3,): Waveform }
```

---

### `reader.get_matched_signals(pattern) -> dict[tuple, str]`

Resolve a pattern to signal paths without loading data.  Useful to inspect
what a pattern would match before committing to a load.

---

### `reader.eval(expr, clock, mode='single'|'zip', ...) -> Waveform | dict`

Evaluate a Python arithmetic expression where signal paths are embedded inline.

- **`mode='single'`** (default): every path must match exactly one signal;
  returns a single `Waveform`.
- **`mode='zip'`**: paths with brace/regex patterns expand per key; returns
  `dict[tuple, Waveform]`.  Single-match paths are broadcast.

```python
# single mode
occ = r.eval("tb.dut.w_ptr[2:0] - tb.dut.r_ptr[2:0]", clock="tb.clk")

# zip mode -- evaluates once per matched fifo index
occs = r.eval(
    "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
    clock="tb.clk",
    mode="zip",
)
# -> { (0,): Waveform, (1,): Waveform, ... }
```

---

### `reader.top_scope_list() -> list[Scope]`

Return the root `Scope` nodes of the hierarchy.  Each `Scope` has:
- `.name` -- local scope name
- `.signal_list` -- signals at this level
- `.child_scope_list` -- child scopes
- `.full_name()` -- fully-qualified dotted name

---

## Pattern syntax

Used in `load_matched_waveforms`, `get_matched_signals`, and `eval`.

| Syntax | Example | Keys produced |
|--------|---------|---------------|
| `{a,b,c}` | `sig_{read,write}` | `('read',)`, `('write',)` |
| `{N..M}` | `fifo_{0..3}.ptr` | `(0,)`, `(1,)`, `(2,)`, `(3,)` |
| `{N..M..step}` | `lane_{0..6..2}` | `(0,)`, `(2,)`, `(4,)`, `(6,)` |
| `@<regex>` | `@([a-z]+)_valid` | `(capture_group,)` per match |
| `$ModName` | `tb.$fifo_unit.ptr` | `(scope_path,)` — match direct-child scope by module/definition name (**FSDB only**) |
| `$$ModName` | `tb.$$fifo_unit.ptr` | `(scope_path,)` — match any-depth descendant scope by module/definition name (**FSDB only**) |

Multiple `{...}` in one path produce a compound tuple key, e.g.
`u{0,1}.ch{0..1}` -> keys `('0', 0)`, `('0', 1)`, `('1', 0)`, `('1', 1)`.

---

## Waveform — operations

A `Waveform` wraps three parallel numpy arrays: `.value`, `.clock`, `.time`.
Every operation returns a **new** `Waveform`; none mutate in place.

### Key properties

| Property | Type | Description |
|----------|------|-------------|
| `.value` | `ndarray` | Signal values (int64 / uint64 / object for >64-bit) |
| `.clock` | `ndarray` | Clock edge counter per sample — **absolute** cycle number from start of simulation (cycle 0 = first sampling edge in file) |
| `.time` | `ndarray` | Simulation timestamp per sample |
| `.width` | `int\|None` | Bit-width of the signal |
| `.signed` | `bool` | Whether values are two's-complement signed |
| `.name` | `str` | Full signal path string |
| `.data` | `np.recarray` | All three arrays as `("time","clock","value")` |

### Filtering

| Method | Description |
|--------|-------------|
| `wave.mask(mask)` | Keep samples where bool array or 1-bit Waveform is True |
| `wave.filter(fn)` | Keep samples where scalar `fn(value)` returns True |
| `wave.vectorized_filter(fn)` | Same but `fn` receives the whole array |
| `wave.time_slice(begin, end)` | Trim to simulation time range (binary search) |
| `wave.cycle_slice(begin, end)` | Trim to absolute clock cycle range (binary search on `.clock`) |
| `wave.slice(begin_idx, end_idx)` | Trim by array index range |
| `wave.take(indices)` | Select samples at integer index positions |

### Transformation

| Method | Description |
|--------|-------------|
| `wave.map(fn, width, signed)` | Apply scalar `fn` element-wise |
| `wave.vectorized_map(fn, width, signed)` | Apply vectorized `fn` to entire array |
| `wave.unique_consecutive()` | Remove consecutive duplicate values (alias: `.compress()`) |
| `wave.downsample(chunk, fn)` | Aggregate into chunks (default: mean) |
| `wave.as_signed()` / `.as_unsigned()` | Reinterpret signedness |

### Bit manipulation

| Syntax / Method | Description |
|-----------------|-------------|
| `wave[7:0]` | Extract bits 7 down to 0 (little-endian, high:low), always unsigned |
| `wave[n]` | Extract single bit n |
| `wave.split_bits(n)` | Split into equal n-bit groups (LSB first) |
| `wave.split_bits([n1,n2,...])` | Split into explicit-width groups (LSB first) |
| `Waveform.concatenate([w0,w1,...])` | Join waveforms (w0=LSB, last=MSB), all must be unsigned |
| `wave.bit_count()` | Population count per sample -> uint64 |

### Edge detection (1-bit waveforms only)

| Method | Description |
|--------|-------------|
| `wave.rising_edge()` | True at 0->1 transitions |
| `wave.falling_edge()` | True at 1->0 transitions |

### Arithmetic operators

`+`, `-`, `*`, `//`, `%`, `**`, `/` work between two `Waveform`s or a
`Waveform` and a scalar (`int` or `float`).

**Rules:**
- Both `Waveform` operands must have the **same signedness**; mixing raises `ValueError`.
- Width is inferred automatically:
  - `+`  -> `max(w1, w2) + 1`
  - `-`, `//`, `%` -> `max(w1, w2)`
  - `*`  -> `w1 + w2`
  - `/`  -> `None` (float result, no width)
- Width inference is capped at 64 bits for integer types.

### Bitwise / comparison operators

`&`, `|`, `^`, `~`, `<<`, `>>`, `==`, `!=` operate on integer-typed waveforms.

- For `&`, `|`, `^`: both waveforms must have the **same width**.
- `==` and `!=` return a 1-bit (`width=1`) unsigned `Waveform`.
- `~` requires `width` to be known.

---

## Pattern Matching

The `Pattern` API extracts all matching protocol transactions from waveforms in a
single NFA-based scan. It is useful for latency measurement, protocol
compliance checking, and temporal data extraction.

### How it works

- A `Pattern` is a sequence of **steps** describing what to wait for and what to
  capture.
- Calling `.match()` runs the engine over all loaded waveforms and returns a
  `MatchResult` (struct-of-arrays, one entry per matched instance).
- If the **first step is `wait`**, its condition is used as the trigger: one
  instance is forked each cycle the condition is True.
- If the **first step is not `wait`** (e.g. `capture`, `delay`), an instance
  is forked **every cycle** (useful for paired-sample extraction).

### Step reference

| Step | Blocking? | Description |
|------|-----------|-------------|
| `.wait(cond, guard=None, channel=None)` | yes | Block until `cond` is True. `guard` is checked each waiting cycle (not the match cycle); violation → `REQUIRE_VIOLATED`. `channel` enforces FIFO ordering among concurrent instances on the same named channel. |
| `.delay(n, guard=None)` | yes (n≥1) / epsilon (n=0) | Advance exactly `n` cycles. `delay(0)` is a no-op. |
| `.capture(name, signal)` | no | Record signal value at current cycle into `captures[name]`. Use `name[]` to append to a list (inside loop/repeat). `signal` can be a Waveform or `callable(index, captures)`. |
| `.require(cond)` | no | Assert condition; terminate with `REQUIRE_VIOLATED` if False. |
| `.loop(body, *, until=None, when=None)` | — | Exactly one of `until`/`when` required. `until`: do-while — run body first, exit when True. `when`: while — check before each iteration, exit when False. |
| `.repeat(body, n)` | — | Run `body` exactly `n` times. `n` may be `callable(index, captures) -> int`. |
| `.branch(cond, true_body, false_body)` | — | Epsilon conditional branch. |
| `.timeout(max_cycles)` | — | Per-instance timeout; incomplete instances become `TIMEOUT`. Instances still active at end of waveform are always reported as `TIMEOUT` regardless. |

### `match()` parameters

```python
result = pattern.match(start_cycle=None, end_cycle=None)
```

- `start_cycle` / `end_cycle`: limit the scan window (same convention as
  `load_waveform`'s `begin_cycle`/`end_cycle` — start inclusive, end exclusive).

### `MatchResult` fields

All fields are `Waveform` objects whose `.clock` axis is `start_cycle`, so they
live in the same coordinate system as ordinary signal waveforms.

| Field | Type | Description |
|-------|------|-------------|
| `.start` | `Waveform[int64]` | Start cycle of each match (inclusive). |
| `.end` | `Waveform[int64]` | End cycle of each match (inclusive, last active cycle). |
| `.duration` | `Waveform[int64]` | `end - start + 1` (number of cycles). |
| `.status` | `Waveform[uint8]` | `MatchStatus.OK=0`, `TIMEOUT=1`, `REQUIRE_VIOLATED=2`. |
| `.captures` | `dict[str, Waveform]` | Named captures. List captures (`name[]`) have `object` dtype where each element is a Python list. |
| `.valid` | property | Boolean 1-bit Waveform: `status == OK`. |
| `.filter_valid()` | method | Return new `MatchResult` with only `OK` instances. |

**`end` is inclusive**: to extract a waveform slice for a match use
`wf.cycle_slice(start, end + 1)`.

### Dynamic conditions and captures

Any `cond` or `signal` argument can be a **callable** `(index, captures) -> value`
instead of a static `Waveform`. `index` is the current waveform array index;
`captures` is the instance's capture dict so far.

```python
# Capture the length field, then repeat that many times
Pattern()
.wait(start_valid)
.capture("len", length_field)
.repeat(Pattern().wait(data_valid).capture("data[]", data),
        n=lambda idx, cap: int(cap["len"]))
.match()
```

### Channel ordering

When multiple concurrent instances wait on the same `channel` name, they
consume events in FIFO order (oldest instance first). This is how
request/response pairing is implemented without explicit demultiplexing.

---

## Key constraints to remember

1. **Signal path format**: always `top.module.submodule.signal_name` with
   optional `[high:low]` range suffix.
2. **Signedness must match** when operating on two Waveforms; use
   `.as_signed()` / `.as_unsigned()` to convert first if needed.
3. **Bit-slicing is little-endian**: `wave[7:0]` means bits 7 down to 0
   (high index first, matching Verilog convention).
4. **`take()` vs `mask()`**: `take` needs integer indices; `mask` needs a
   boolean array or 1-bit Waveform.
5. **Width > 64**: stored as Python `object` arrays; arithmetic still works but
   is slower.
6. **Time units**: `begin_time` / `end_time` are in the file's native simulator
   time unit (no automatic conversion).  Use `begin_cycle` / `end_cycle` for
   clock-cycle-based windowing (mutually exclusive with time parameters).
7. **Absolute cycle numbers**: `.clock` values are always absolute from simulation
   start, so two waveforms loaded with different `begin_time` windows can still
   be compared by `.clock` value for alignment.
8. **Pattern matching — all waveforms must share the same clock axis**: pass
   waveforms loaded with the same `clock` signal to all pattern steps.
9. **`MatchResult.end` is inclusive**: use `cycle_slice(start, end + 1)` to
   extract the corresponding waveform window.

---

## Complete minimal example

```python
from wavekit import VcdReader
import numpy as np

with VcdReader("sim.vcd") as r:
    clk  = "tb.clk"

    # Load raw signals
    wptr = r.load_waveform("tb.dut.w_ptr[3:0]", clock=clk)
    rptr = r.load_waveform("tb.dut.r_ptr[3:0]", clock=clk)
    wr   = r.load_waveform("tb.dut.wr_en",       clock=clk)

    # Compute occupancy
    depth = 16
    occ = (wptr + depth - rptr) % depth        # Waveform arithmetic

    # Only active write cycles
    active_occ = occ.mask(wr == 1)

    print("Mean occupancy during writes:", np.mean(active_occ.value))
    print("Max occupancy:", np.max(occ.value))

    # Detect write bursts (rising edge of wr_en)
    burst_starts = wr.rising_edge()
    burst_indices = np.where(burst_starts.value)[0]
    print("Burst start timestamps:", burst_starts.time[burst_indices])
```
