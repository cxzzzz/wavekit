# Signal query

Waveform files contain a hierarchy of scopes and signals. A signal query lets
you resolve one path, select a range, or load a family of related signals without
writing one call for every concrete name.

## Exact paths

Use dotted hierarchy paths for exact lookups. A trailing range selector is
optional and follows Verilog-style indexing and range notation, such as `[0]`
or `[31:16]`. When omitted, the reader loads the signal's native range:

```python
with VcdReader('simulation.vcd') as reader:
    data = reader.load_waveform(
        'tb.dut.data[31:16]',
        clock='tb.clk',
    )
```

Use `get_matched_signals()` or `get_matched_scopes()` when you want hierarchy
metadata without loading samples.

## Batch loading

When a design contains a family of signals with related names, use
`load_matched_waveforms()` to load them in one operation.

A query path consists of dot-separated components. Each component is either a
fixed name for an exact match or contains a matching expression, such as a
brace, regex, wildcard, or module-definition expression. A matching expression
produces a capture describing what it matched.

The result is a dictionary with one entry per matched signal. Each key is a
tuple of captures, ordered according to the matching expressions in the query
path. Fixed path components do not contribute to the key, so a query without
matching expressions uses the empty tuple `()`.

For example, the query below matches two dimensions: the FIFO index and the
signal type:

```python
with VcdReader('simulation.vcd') as reader:
    waves = reader.load_matched_waveforms(
        'tb.fifo_{0..3}.{wr,rd}_en',
        clock_path='tb.clk',
    )

    for key, wave in waves.items():
        print(key, wave.signal.full_name)
```

The output is:

```text
(BraceCapture(groups=('0',)), BraceCapture(groups=('wr',))) tb.fifo_0.wr_en
(BraceCapture(groups=('0',)), BraceCapture(groups=('rd',))) tb.fifo_0.rd_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('wr',))) tb.fifo_1.wr_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('rd',))) tb.fifo_1.rd_en
...
```

The first capture identifies the FIFO index and the second identifies the
signal type.

### Query syntax

Use the following syntax to construct query paths:

| Syntax | Example | Captured key component |
| --- | --- | --- |
| Exact path | `tb.dut.valid` | None; exact components are omitted |
| Brace list | `sig_{read,write}` | `BraceCapture` |
| Integer range | `fifo_{0..3}.ptr` | One `BraceCapture` per index |
| Stepped range | `lane_{0..6..2}.valid` | `BraceCapture` for `0`, `2`, `4`, `6` |
| Canonical regex | `tb.u0./J_([a-z]+)/` | `RegexCapture` |
| Legacy regex | `@([a-z]+)_valid` | `RegexCapture` |
| Single wildcard | `tb.*.valid` | `WildcardCapture` |
| Recursive wildcard | `tb.**.valid` | `WildcardCapture` |
| Direct module definition | `tb.$fifo_unit.ptr` | `ExactCapture` (FSDB) |
| Recursive module definition | `tb.$$fifo_unit.ptr` | `ExactCapture` (FSDB) |

The same query syntax is also used by `get_matched_signals()`,
`get_matched_scopes()`, and `Reader.eval()`.

`$` and `$$` are available only for FSDB module-definition matching.

`load_matched_unknown_masks()` follows the same query and result-key rules as
`load_matched_waveforms()`. Use `get_matched_scopes()` for scope queries; a
terminal range selector is valid for signal queries, not for scope queries.

If `clock_path` matches one signal, that clock is shared by every result. If it
matches multiple clocks, wavekit selects the matched clock whose capture key is
the longest prefix of the signal key.

## Evaluate expressions

`Reader.eval()` is convenient for simple, one-off calculations that fit in a
single expression.

In `single` mode, every path must resolve to exactly one signal. This is the
default mode:

```python
occupancy = reader.eval(
    'tb.dut.w_ptr[2:0] - tb.dut.r_ptr[2:0]',
    clock='tb.clk',
)
```

Waveform operations can also be called from expressions:

```python
byte_count = reader.eval(
    'bit_count(tb.axi.wstrb[7:0] * (tb.axi.wvalid & tb.axi.wready))',
    clock='tb.clk',
)
```

In `zip` mode, matching paths expand together by their capture tuples; a path
that matches only one signal is broadcast:

```python
occupancies = reader.eval(
    'tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]',
    clock='tb.clk',
    mode='zip',
)
```
