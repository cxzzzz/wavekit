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

### `VcdReader(file: str)` / `FstReader(file: str)` / `FsdbReader(file: str)`

Open a waveform file.  Use as a context manager (`with`) to ensure the file is
closed.  `FsdbReader` requires the Verdi runtime library (`libNPI.so`):

- `WAVEKIT_NPI_LIB` — direct path to `libNPI.so`
- `VERDI_HOME` — Verdi installation directory (searches `$VERDI_HOME/share/NPI/lib/...`)
- `LD_LIBRARY_PATH` — system library search path

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

### `reader.load_matched_waveforms(signal_path, clock_path, ...) -> dict[CaptureKey, Waveform]`

Batch-load all signals matching a pattern. Returns a dict keyed by tuples of
`Capture` objects. Ordinary exact-name path components are omitted from keys;
brace, regex, wildcard, and module-definition matches are retained.

**Clock assignment:**
- If `clock_path` matches **one** signal -> that clock is shared by all.
- If `clock_path` matches **multiple** signals -> keys must match signal keys
  exactly (per-signal clock).

```python
# Single clock broadcast
waves = r.load_matched_waveforms("tb.dut.fifo_{0..3}.w_ptr[2:0]", "tb.clk")
# -> keys contain BraceCapture(groups=("0",)), ..., BraceCapture(groups=("3",))
```

### `reader.load_unknown_mask(signal, clock, ...) -> Waveform`

Load X/Z presence as a normal unsigned `Waveform`.  Each returned value is a
bitmask; bit `1` means the corresponding source bit was selected by
`include_x` and/or `include_z`.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `signal` | `str` | required | Full dotted path, with optional range suffix. |
| `clock` | `str` | required | Full dotted path of the sampling clock. |
| `include_x` | `bool` | `True` | Mark source `X`/`x` bits. |
| `include_z` | `bool` | `True` | Mark source `Z`/`z` bits. |
| `sample_on_posedge` / windows | | same as `load_waveform` | Align masks exactly with value waveforms. |

When both flags are `False`, the returned mask is all zero.  The returned mask
is always unsigned, has the requested signal width after range selection, and is
named `unknown_mask(<signal>)`.

```python
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk", xz_value=0)
mask = r.load_unknown_mask("tb.dut.data[7:0]", clock="tb.clk")
known_data = data.mask(mask == 0)
```

### `reader.load_matched_unknown_masks(signal_path, clock_path, ...) -> dict[CaptureKey, Waveform]`

Batch-load unknown masks with the same pattern and clock assignment rules as
`load_matched_waveforms`.  Returned keys match `get_matched_signals(signal_path)`.

---

### `reader.get_matched_signals(path) -> dict[CaptureKey, Signal]`

Resolve a query to `Signal` objects without loading data. Exact-name components
are omitted from keys; binding matchers return typed `Capture` objects.

### `reader.get_matched_scopes(path) -> dict[CaptureKey, Scope]`

Resolve a query to `Scope` objects. The same CaptureKey rules apply; a terminal
range selector is not valid for scope queries.

---

### `reader.eval(expr, clock, mode='single'|'zip', ...) -> Waveform | dict`

Evaluate a Python arithmetic expression where signal paths are embedded inline.

- **`mode='single'`** (default): every path must match exactly one signal;
  returns a single `Waveform`.
- **`mode='zip'`**: paths with brace/regex patterns expand per key; returns
  `dict[CaptureKey, Waveform]`. Single-match paths are broadcast.

```python
# single mode
occ = r.eval("tb.dut.w_ptr[2:0] - tb.dut.r_ptr[2:0]", clock="tb.clk")

# zip mode -- evaluates once per matched fifo index
occs = r.eval(
    "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
    clock="tb.clk",
    mode="zip",
)
# -> keys contain BraceCapture(groups=("0",)), ..., BraceCapture(groups=("3",))
```

---

### `reader.top_scopes -> tuple[Scope, ...]`

Return the immutable tuple of root `Scope` nodes. Each hierarchy node has:
- `.base_name` -- local name without a signal range
- `.name` -- local name including the selected/native signal range
- `.children` -- immutable tuple of direct `Scope`/`Signal` children
- `.parent` -- parent node, or `None` for top-level scopes
- `.full_name` -- fully-qualified dotted name property

---

## Pattern syntax

Used in `load_matched_waveforms`, `load_matched_unknown_masks`,
`get_matched_signals`, and `eval`.

| Syntax | Example | Capture retained in key |
|--------|---------|-------------------------|
| `{a,b,c}` | `sig_{read,write}` | `BraceCapture(groups=('read',))`, etc. |
| `{N..M}` | `fifo_{0..3}.ptr` | `BraceCapture(groups=('0',))`, etc. |
| `{N..M..step}` | `lane_{0..6..2}` | `BraceCapture(groups=('0',))`, etc. |
| `/<regex>/` | `/([a-z]+)_valid/` | Canonical regex; returns `RegexCapture` |
| `@<regex>` | `@([a-z]+)_valid` | Legacy-compatible regex spelling |
| `$ModName` | `tb.$fifo_unit.ptr` | `ExactCapture(path=..., definition='fifo_unit')` (**FSDB only**) |
| `*` / `**` | `tb.*.valid` / `tb.**.valid` | Single-level / recursive wildcard |
| `$$ModName` | `tb.$$fifo_unit.ptr` | Same capture type for any-depth matches (**FSDB only**) |

Multiple `{...}` in one path produce a compound tuple key, e.g.
`u{0,1}.ch{0..1}` -> keys containing two `BraceCapture` objects, one for each brace component.

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

Pattern APIs live under `wavekit.pattern`, not top-level `wavekit`:

```python
from wavekit.pattern import Channel, MatchStatus, Pattern, collect, match
```

Use Pattern for declarative step construction, and module-level `match()` /
`collect()` for execution. The runtime is synchronous and start-major: it starts
one candidate per eligible cycle and runs that candidate to completion before
trying the next start cycle. Matches may still overlap in time.

### Declarative usage

```python
pattern = (
    Pattern()
    .wait(req_valid & req_ready)
    .consume(rsp_valid & rsp_ready, channel='response')
    .capture('rsp_data', rsp_data)
)
result = match(pattern)
```

If the first step is an unguarded `wait`, that condition is the trigger. If the
first step is not `wait`, the pattern is attempted at every scanned cycle.

### Programmable usage

Programmable bodies are normal functions, not `async` functions. Use synchronous
context calls and return `ctx.OK` for a successful check row, or `None` to skip
the current start cycle.

```python
fire = valid & ready

def tx(ctx):
    if ctx.value(fire):
        ctx.consume(lambda: ctx.value(done), channel='done')
        return ctx.OK
    return None

result = match(tx, timeout=64)
```

For extraction, use `collect(body)`. It records every non-`None` Python return
value and raises `PatternError` on timeout or require failure. `collect(pattern)`
is intentionally unsupported.

### Step reference

| Step | Blocking? | Description |
|------|-----------|-------------|
| `.wait(cond, require=None, require_message=None)` | yes | Block until `cond` is true without consuming the event. |
| `.consume(cond, channel, require=None, require_message=None)` | yes | Block until `cond` is true and exclusively consume `(channel, cycle)`. |
| `.delay(n, require=None, require_message=None)` | yes for n≥1 / epsilon for n=0 | Advance exactly `n` cycles. |
| `.capture(name, signal, mode='last')` | no | Record a value. `mode='list'` appends to a Python list. |
| `.require(cond, message=None)` | no | Assert a condition; failure produces `MatchStatus.RequireViolated(message)`. |
| `.loop(body, *, until=None, when=None)` | — | `until`: do-while. `when`: while. |
| `.repeat(body, n)` | — | Run `body` exactly `n` times. |
| `.branch(cond, true_body, false_body)` | — | Epsilon conditional branch. |

Declarative dynamic callbacks use `callable(index, captures)`. Programmable
conditions passed to `ctx.wait()` / `ctx.consume()` are zero-argument callables
that close over `ctx`. In both cases `index` / `ctx.index` is the current
waveform-array sample index, not a cycle number and not rebased by
`match(start_cycle=...)`.

### `MatchRecords` fields

`match()` returns `MatchRecords`, a row-aligned batch. It can be iterated to get
`MatchRecord` rows with `MatchPoint` start/end values.

| Field | Type | Description |
|-------|------|-------------|
| `.start` | `Waveform[int64]` | Start point: `.value` is sample index, `.clock` is absolute cycle, `.time` is simulation timestamp. |
| `.end` | `Waveform[int64]` | End point, inclusive; same `.value` / `.clock` / `.time` meaning as `.start`. |
| `.duration` | `Waveform[int64]` | `end.value - start.value + 1` sampled cycles. |
| `.status` | `Waveform[object]` | `MatchStatus.OK()`, `MatchStatus.Timeout(...)`, or `MatchStatus.RequireViolated(...)`. |
| `.captures` | `dict[str, Waveform]` | Named captures aligned to result rows. |
| `.ok` / `.failed` | `Waveform[bool]` | Boolean result-row masks. |

Use `filter_ok()`, `filter_failed()`, and `filter_status(status_or_class)`. For
example, `result.filter_status(MatchStatus.Timeout)` keeps all timeout rows.

**`end` is inclusive**: to extract a waveform slice for a match use
`wf.cycle_slice(start.clock, end.clock + 1)`.

### Channel ordering with consume

`consume(cond, channel)` claims only the current `(logical_channel, cycle)` when
`cond` is true and that event is free. It does not reserve a channel while
waiting. A successful consume remains committed even if the candidate later
fails or times out.

```python
def match_id(idx, cap):
    return bool(rvalid.value[idx] & rready.value[idx]) and int(rid.value[idx]) == int(cap['arid'])

pattern = (
    Pattern()
    .wait(arvalid & arready)
    .capture('arid', arid)
    .consume(match_id, channel=lambda idx, cap: f'read_{cap["arid"]}')
    .capture('rdata', rdata)
)
result = match(pattern)
```

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
9. **`MatchRecords.end` is inclusive**: use `cycle_slice(start.clock, end.clock + 1)` to
   extract the corresponding waveform window.
10. **Pattern time movement is explicit**: `wait` / `consume` resume in the same
    cycle when already true; insert `delay(1)` for next-cycle behavior.

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

<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

This project is managed by Trellis. The working knowledge you need lives under `.trellis/`:

- `.trellis/workflow.md` — development phases, when to create tasks, skill routing
- `.trellis/spec/` — package- and layer-scoped coding guidelines (read before writing code in a given layer)
- `.trellis/workspace/` — per-developer journals and session traces
- `.trellis/tasks/` — active and archived tasks (PRDs, research, jsonl context)

If a Trellis command is available on your platform (e.g. `/trellis:finish-work`, `/trellis:continue`), prefer it over manual steps. Not every platform exposes every command.

If you're using Codex or another agent-capable tool, additional project-scoped helpers may live in:
- `.agents/skills/` — reusable Trellis skills
- `.codex/agents/` — optional custom subagents

Managed by Trellis. Edits outside this block are preserved; edits inside may be overwritten by a future `trellis update`.

<!-- TRELLIS:END -->
