# Reader

Wavekit has three readers: `VcdReader` for VCD files, `FstReader` for FST
files, and `FsdbReader` for FSDB files. They share the same interface. Use
readers as context managers so their resources are released automatically
when the block exits:

```python
from wavekit import FsdbReader, FstReader, VcdReader

with VcdReader('simulation.vcd') as reader:
    signal = reader['tb.dut.data']
    print(signal.name, signal.width)    # data 8

with FstReader('simulation.fst') as reader:
    signal = reader['tb.dut.data']
    print(signal.name, signal.width)    # data 8

with FsdbReader('simulation.fsdb') as reader:
    signal = reader['tb.dut.data']
    print(signal.name, signal.width)    # data 8
```

`FsdbReader` requires the Verdi NPI runtime (`libNPI.so`). See the
[FSDB installation and runtime setup](../getting-started/installation.md)
before opening an FSDB file.

## Hierarchy

Once a reader opens a file, it exposes a tree of `Scope` and `Signal` nodes,
rooted at `reader.top_scopes`.

A `Scope` corresponds to a module instance or a generate block; a `Signal`
corresponds to a concrete signal, which may be a plain scalar or a
composite type such as struct, array, or union. Each `Signal` carries basic
metadata such as `width` and `composite_type`.

```python
from wavekit import Scope, Signal

top = reader.top_scopes[0]
for child in top.children:
    if isinstance(child, Scope):
        print('Scope', child.name)
    elif isinstance(child, Signal):
        print('Signal', child.name, child.width)
```

See [Signal query](signal-query.md) for how to locate a `Signal` or `Scope`
by name.

## Waveform

Sampling a signal on a clock produces a `Waveform`, three equal-length
arrays:

- `.value` — the sampled value at each edge;
- `.cycle` — absolute clock-cycle number, with cycle 0 at the first sampling
  edge in the file;
- `.time` — the simulation timestamp of each sample.

```python
with reader.clock_domain(clock='tb.clk'):
    data = reader['tb.dut.data[7:0]'].w
    valid = reader['tb.dut.valid'].w
    ready = reader['tb.dut.ready'].w

print(data.cycle[:5])   # [0 1 2 3 4]
print(data.value[:5])   # [0 3 3 7 12]

# Operate on the whole Waveform at once; the result is a Waveform too
fire = valid & ready   # whether the handshake fired each cycle
```

Operations on a `Waveform` (filtering, bitwise, arithmetic, etc.) return a
new `Waveform` and keep these three arrays aligned, so results can always be
traced back to the original cycle or timestamp.

See [Signal query](signal-query.md) for how to load a signal as a `Waveform`,
and [Waveform analysis](waveform-analysis.md) for the operations it supports.
