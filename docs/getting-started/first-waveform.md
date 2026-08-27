# First waveform: FIFO occupancy

This tutorial uses the complete FIFO occupancy example in
[`example/fifo_occupancy/`](../../example/fifo_occupancy/). Run the commands from
the repository root.

Generate the waveform first:

```console
make -C example/fifo_occupancy sim
```

## Load the waveform

Load the FIFO's write and read pointers from the VCD, calculate its occupancy,
and report summary statistics:

```python
import numpy as np

from wavekit import VcdReader

with VcdReader('example/fifo_occupancy/fifo_tb.vcd') as reader:
    clock = 'fifo_tb.s_fifo.clk'
    depth = 8

    write_pointer = reader.load_waveform(
        'fifo_tb.s_fifo.w_ptr', clock=clock
    )
    read_pointer = reader.load_waveform(
        'fifo_tb.s_fifo.r_ptr', clock=clock
    )

    occupancy = (write_pointer + depth - read_pointer) % depth

    print(f'Average occupancy: {np.mean(occupancy.value):.2f}')
    print(f'Maximum occupancy: {np.max(occupancy.value)}')
```

`occupancy` is also a `Waveform`, so it can be used in further waveform
analysis. Its `.value` attribute is a NumPy array, so it can be passed directly
to NumPy functions or combined with other NumPy arrays.

For this fixture, the script prints:

```text
Average occupancy: 4.64
Maximum occupancy: 7
```

## Run the complete example

The command below regenerates the waveform and runs `occupancy.py`:

```console
make -C example/fifo_occupancy all
```

The Makefile compiles `fifo_tb.sv` and `fifo.sv`, runs the simulation, and then
runs `occupancy.py`. See the [examples index](../examples.md) for more examples.

## Next steps

- Learn the [reader options and format-specific setup](../guides/reader.md).
- Explore [Waveform operations](../guides/waveform-analysis.md).
- Use [signal queries](../guides/signal-query.md) when a design has repeated
  hierarchy paths.
