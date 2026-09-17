# Signal query

There are two ways to locate a signal: direct access when the exact path is
known, and batch queries when names follow a family structure. Once you
have a `Signal`, load it as a waveform to use it in analysis.

## Direct access

`reader[path]` returns the `Signal` or `Scope` at an exact path, and raises
`KeyError` when nothing matches:

```python
signal = reader['tb.dut.data']
scope = reader['tb.dut']
selected = reader['tb.dut.data[31:16]']  # trailing range selector
```

Scopes and composite signals (structs, arrays) support the same lookup with
relative paths:

```python
tb = reader['tb']
signal = tb['dut']['data']        # chain through nested nodes
signal = tb['dut.data']           # equivalent relative dotted path
member = tb['pkt']['valid']       # struct member through a composite signal
```

On a signal, an integer or slice key selects bits, with the same semantics
as Verilog:

```python
bit = signal[7]        # one bit
field = signal[31:16]  # a range
```

## Load a waveform

`reader[path]` returns a `Signal`, not a waveform. To get waveform data,
enter a clock domain, then use `Signal.w` or `load_waveform()`:

```python
with VcdReader('simulation.vcd') as reader:
    with reader.clock_domain('tb.clk'):
        valid = reader['tb.dut.valid'].w
        data = reader.load_waveform('tb.dut.data[7:0]')
```

To load a waveform outside any clock domain, or to sample a different clock
or window just for this call, pass `clock`, edge, and window parameters
explicitly:

```python
data = reader.load_waveform(
    'tb.dut.data[7:0]', clock='tb.clk',
    sample_on_posedge=True,
    start_cycle=100, end_cycle=200,
)
```

By default, wavekit samples on the falling edge to avoid errors caused by
sampling during a signal transition; pass `sample_on_posedge=True` to sample
on rising edges instead. Use `start_time`/`end_time` for a simulation-time
window, or `start_cycle`/`end_cycle` for an absolute clock-cycle window —
the two cannot be mixed.

Waveforms used together in the same calculation or pattern match must share
the same clock source, sampling edge, and window, which is why a shared
clock domain is usually the better fit over repeating the same parameters
on every call.

## Load a mask

Loading a waveform replaces X/Z states with `xz_value` (zero by default). To
preserve that information — for example to exclude unknown values, or to
track which bits were X/Z — use `Signal.m` or `load_unknown_mask()`:

```python
with reader.clock_domain(clock='tb.clk'):
    value = reader['tb.data[7:0]'].w
    unknown = reader['tb.data[7:0]'].m   # X/Z presence mask
    # or: unknown = reader.load_unknown_mask('tb.data[7:0]')
    known_value = value.mask(unknown == 0)
```

`Signal.m` is the default-parameter spelling of `Signal.unknown_mask()`. The
mask has one bit per selected source bit, marking whether that bit was X/Z
in the source file.

## Batch queries

When the names follow a family structure — repeated instances, numbered
lanes, shared prefixes — match them with one query.

A query path consists of dot-separated components. Each component is either a
fixed name for an exact match or contains a matching expression, such as a
brace, regex, or wildcard expression. A matching expression produces a
capture describing what it matched.

`get_matched_signals()` returns a dictionary with one entry per matched
signal. Each key is a tuple of captures, ordered according to the matching
expressions in the query path. Fixed path components do not contribute to
the key; a query without matching expressions uses the empty tuple `()`.

For example, the query below matches two dimensions: the FIFO index and the
signal type:

```python
with VcdReader('simulation.vcd') as reader:
    signals = reader.get_matched_signals('tb.fifo_{0..3}.{wr,rd}_en')

    for key, signal in signals.items():
        print(key, signal.full_name)
```

The output is:

```text
(BraceCapture(groups=('0',)), BraceCapture(groups=('wr',))) tb.fifo_0.wr_en
(BraceCapture(groups=('0',)), BraceCapture(groups=('rd',))) tb.fifo_0.rd_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('wr',))) tb.fifo_1.wr_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('rd',))) tb.fifo_1.rd_en
...
```

To load the matched signals as waveforms in one call, use
`load_matched_waveforms()`:

```python
with VcdReader('simulation.vcd') as reader:
    with reader.clock_domain('tb.clk'):
        waves = reader.load_matched_waveforms('tb.fifo_{0..3}.{wr,rd}_en')
```

### Query syntax

Use the following syntax to construct query paths:

| Syntax | Example | Captured key component |
| --- | --- | --- |
| Exact path | `tb.dut.valid` | No capture |
| Brace list | `sig_{read,write}` | `BraceCapture` |
| Integer range | `fifo_{0..3}.ptr` | One `BraceCapture` per index |
| Stepped range | `lane_{0..6..2}.valid` | `BraceCapture` for `0`, `2`, `4`, `6` |
| Canonical regex | `tb.u0./J_([a-z]+)/` | `RegexCapture` |
| Legacy regex | `@([a-z]+)_valid` | `RegexCapture` |
| Single wildcard | `tb.*.valid` | `WildcardCapture` |
| Recursive wildcard | `tb.**.valid` | `WildcardCapture` |
| Direct module definition | `tb.$fifo_unit.ptr` | `ExactCapture` (FSDB) |
| Recursive module definition | `tb.$$fifo_unit.ptr` | `ExactCapture` (FSDB) |

`get_matched_scopes()`, `get_matched_nodes()`, `load_matched_unknown_masks()`,
and `Reader.eval()` all support the same syntax.

If `clock_path` matches one signal, that clock is shared by every result. If
it matches multiple signals, wavekit selects, for each signal, the matched
clock whose capture key is the longest prefix of the signal key.

`$` and `$$` are available only for FSDB module-definition matching.

## Evaluate expressions

`Reader.eval()` is convenient for one-off calculations that fit in a single
expression.

In `single` mode (the default), every path must resolve to exactly one
signal:

```python
occupancy = reader.eval(
    '(tb.dut.w_ptr - tb.dut.r_ptr + 8) % 8',
    clock='tb.clk',
)
```

Waveform operations can also be called inside expressions:

```python
byte_count = reader.eval(
    'bit_count(tb.axi.wstrb[7:0] * (tb.axi.wvalid & tb.axi.wready))',
    clock='tb.clk',
)
```

In `zip` mode, matching paths expand together by their capture tuples; a
path that matches only one signal is broadcast to every group:

```python
occupancies = reader.eval(
    'tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]',
    clock='tb.clk',
    mode='zip',
)
```
