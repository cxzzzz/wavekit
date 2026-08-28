# Reader

Wavekit has three readers: `VcdReader` for VCD files, `FstReader` for FST
files, and `FsdbReader` for FSDB files. They use the same clock-sampled loading
interface. Use readers as context managers so their resources are released
automatically when the block exits:

```python
from wavekit import FsdbReader, FstReader, VcdReader

with VcdReader('simulation.vcd') as reader:
    data = reader.load_waveform('tb.dut.data[7:0]', clock='tb.clk')

with FstReader('simulation.fst') as reader:
    data = reader.load_waveform('tb.dut.data[7:0]', clock='tb.clk')

with FsdbReader('simulation.fsdb') as reader:
    data = reader.load_waveform('tb.dut.data[7:0]', clock='tb.clk')
```

`FsdbReader` uses the same API, but requires the Verdi NPI runtime (`libNPI.so`). See
the [FSDB installation and runtime setup](../getting-started/installation.md#fsdb-support)
before opening an FSDB file.

## Sampling and windows

`load_waveform(signal, clock, ...)` samples the selected signal on every edge of
the clock. By default, wavekit samples on the falling edge to reduce errors
caused by sampling during signal transitions. Pass
`sample_on_posedge=True` to sample on rising edges. Use `begin_time` and
`end_time` for timestamp windows, or `begin_cycle` and `end_cycle` for absolute
clock-cycle windows. Time and cycle windows cannot be mixed.

Waveforms used together for arithmetic, masking, or pattern matching must
use the same clock source, sampling edge, and time or cycle window.

The `load_unknown_mask()` and `load_matched_unknown_masks()` methods expose X/Z
source bits as unsigned masks. Use these masks to track X/Z states without
losing information when unknown values are replaced with `xz_value=0`.
