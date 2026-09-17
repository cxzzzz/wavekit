# Waveform analysis

Operations on a `Waveform` return new `Waveform` objects, keeping the
`.cycle` and `.time` axes aligned when filtering or transforming samples, so
results can always be traced back to the original simulation.

## Arithmetic and bitwise operations

Waveforms support element-wise arithmetic, comparison, bitwise, and shift
operations with other waveforms or numeric scalars:

```python
from wavekit import VcdReader

with VcdReader('simulation.vcd') as reader:
    with reader.clock_domain(clock='tb.clk'):
        write_pointer = reader['tb.fifo.w_ptr'].w
        read_pointer = reader['tb.fifo.r_ptr'].w

    depth = 8
    occupancy = (write_pointer + depth - read_pointer) % depth
```

Loaded waveforms are interpreted as unsigned by default. Pass `signed=True` to
`Signal.waveform()` or `load_waveform()` when a signal should be interpreted
as signed. Two waveform operands must have the same signedness. Convert an
existing waveform with `.as_signed()` or `.as_unsigned()` when necessary.

## Filter and slice

Use `mask()` with a boolean NumPy array or a one-bit waveform. Use `filter()`
when the condition is easiest to express as a scalar callback:

```python
known = data.filter(lambda value: value != 0)
active = data.mask(valid == 1)

first_cycles = data.cycle_slice(0, 100)
window = data.time_slice(start_time=10_000, end_time=20_000)
```

By default, `cycle_slice(start, end)` and `time_slice(start, end)` use a
half-open range: the start is included and the end is excluded. Their bounds are
absolute cycle numbers or simulation timestamps, not array indices. Use
`slice()` or `take()` when you need array-index operations.

## Relative access

Use `relative()`, `ahead()`, and `back()` when a calculation needs neighboring
samples while retaining the original axes:

```python
changed = data != data.back(1)
future_valid = valid.ahead(1)
```

The default boundary behavior repeats the first or last value. Use the `pad` and
`pad_value` arguments when a different boundary policy is required.

## Bit operations and composition

Bit selection follows Verilog-style indexing:

```python
low_byte = data[7:0]
ready = valid[0]
```

`split_bits()` accepts either an integer for equal-width groups or a list of
explicit widths. Groups are returned from least significant to most significant:

```python
byte_fields = data.split_bits(8)
# byte_fields[0] contains bits [7:0], byte_fields[1] contains bits [15:8], and so on.

low, middle, high = data.split_bits([4, 8, 20])
# low contains bits [3:0], middle bits [11:4], and high bits [31:12].
```

Use `Waveform.concatenate()` to combine same-length fields into a wider
waveform. The first field occupies the least-significant bits:

```python
reconstructed = Waveform.concatenate([low, middle, high])
```

Use `Waveform.merge()` to combine corresponding samples with a custom function:

```python
# Compute a majority vote across three 1-bit signals at each sample.
majority = Waveform.merge(
    [a, b, c],
    lambda values: int(sum(values) >= 2),
    width=1,
    signed=False,
)
```

## Transitions and reduction

Use `changed()` to detect value changes in signals of any width:

```python
occupancy_changes = occupancy.changed()
change_cycles = occupancy_changes.cycle[occupancy_changes.value]
```

For one-bit signals, use `rising_edge()` and `falling_edge()` to locate the
start and end of an active interval:

```python
starts = valid.rising_edge()
stops = valid.falling_edge()

start_times = starts.time[starts.value]
stop_times = stops.time[stops.value]
```

Use `any_edge()` when the transition direction does not matter.

Use `unique_consecutive()` to keep the first sample of each consecutive run, or
`compress()` to also preserve the final sample of the waveform. Use
`downsample()` to aggregate consecutive chunks:

```python
one_per_run = data.unique_consecutive()
compact = data.compress()
summary = data.downsample(100)  # average each 100-sample chunk
```

## Extract NumPy results

Use `.value` when handing data to NumPy or another analysis library:

```python
import numpy as np

print(f'Average active value: {np.mean(active.value):.2f}')
print(f'Maximum active value: {np.max(active.value)}')
```

Keep the `Waveform` itself until you no longer need cycle or timestamp context.
