# 波形分析

`Waveform` 由三个相互对齐的数组组成：

- `.value`：采样得到的信号值；
- `.clock`：绝对时钟周期号，从文件中的第一个采样边沿开始计为周期 0；
- `.time`：波形文件中的时间戳，单位沿用文件的原生时间单位。

Waveform 操作会返回新的 `Waveform` 对象。对采样点进行过滤或变换时，返回结果会保留 clock 和 time 轴，因此仍然可以与原始仿真中的周期和时间对应。

## 加载并操作波形

```python
from wavekit import VcdReader

with VcdReader('simulation.vcd') as reader:
    write_pointer = reader.load_waveform(
        'tb.fifo.w_ptr',
        clock='tb.clk',
    )
    read_pointer = reader.load_waveform(
        'tb.fifo.r_ptr',
        clock='tb.clk',
    )

    depth = 8
    occupancy = (write_pointer + depth - read_pointer) % depth
```

`Waveform` 对象支持与其他 `Waveform` 或数值标量进行逐元素算术、比较、位运算和移位运算。

加载的波形默认按无符号数解释。如果信号应按有符号数解释，可以在 `load_waveform()` 中传入 `signed=True`。两个 Waveform 操作数必须具有相同的符号性；必要时，可以使用 `.as_signed()` 或 `.as_unsigned()` 转换已有波形。

## 过滤和切片

当条件适合用布尔 NumPy 数组或单比特 Waveform 表示时，使用 `mask()`。如果条件更适合用逐个样本调用的函数表示，可以使用 `filter()`：

```python
known = data.filter(lambda value: value != 0)
active = data.mask(valid == 1)

first_cycles = data.cycle_slice(0, 100)
window = data.time_slice(begin=10_000, end=20_000)
```

默认情况下，`cycle_slice(begin, end)` 和 `time_slice(begin, end)` 使用半开区间，包含起点但不包含终点。它们的边界分别是绝对周期号和仿真时间戳，不是数组下标。需要按数组下标操作时，使用 `slice()` 或 `take()`。

## 访问相邻样本

需要访问相邻采样点，同时保留原有坐标轴时，可以使用 `relative()`、`ahead()` 和 `back()`：

```python
changed = data != data.back(1)
future_valid = valid.ahead(1)
```

默认的边界行为是重复第一个或最后一个值。如果需要其他边界处理方式，可以使用 `pad` 和 `pad_value` 参数。

## 位操作和组合

位选择遵循 Verilog 风格的索引方式：

```python
low_byte = data[7:0]
ready = valid[0]
```

`split_bits()` 接受一个整数或一个宽度列表。传入整数时，信号会被拆成等宽的分组；传入列表时，可以指定每个分组的宽度。返回的分组从最低有效位到最高有效位排列：

```python
byte_fields = data.split_bits(8)
# byte_fields[0] 是 bits [7:0]，byte_fields[1] 是 bits [15:8]，以此类推。

low, middle, high = data.split_bits([4, 8, 20])
# low 是 bits [3:0]，middle 是 bits [11:4]，high 是 bits [31:12]。
```

使用 `Waveform.concatenate()` 将长度相同的字段组合成更宽的波形。列表中的第一个字段放在最低有效位：

```python
reconstructed = Waveform.concatenate([low, middle, high])
```

使用 `Waveform.merge()` 将对应位置的样本交给自定义函数处理：

```python
# 对三个单比特信号逐采样点计算多数表决。
majority = Waveform.merge(
    [a, b, c],
    lambda values: int(sum(values) >= 2),
    width=1,
    signed=False,
)
```

## 边沿与归约

使用 `changed()` 检测任意位宽信号的值变化：

```python
occupancy_changes = occupancy.changed()
change_cycles = occupancy_changes.clock[occupancy_changes.value]
```

对于单比特信号，可以使用 `rising_edge()` 和 `falling_edge()` 找到有效区间的起止位置：

```python
starts = valid.rising_edge()
stops = valid.falling_edge()

start_times = starts.time[starts.value]
stop_times = stops.time[stops.value]
```

如果不需要区分上升沿和下降沿，可以使用 `any_edge()`。

使用 `unique_consecutive()` 保留每个连续区间的第一个采样点；使用 `compress()` 时，还会保留波形的最后一个采样点。需要将长波形聚合为连续的数据块时，可以使用 `downsample()`：

```python
one_per_run = data.unique_consecutive()
compact = data.compress()
summary = data.downsample(100)  # 每 100 个采样点求一个平均值
```

## 保留未知值信息

普通方式加载时，X/Z 状态会被 `xz_value` 替换，默认值为 0。需要保留这些状态时，可以同时加载对应的掩码：

```python
value = reader.load_waveform('tb.data[7:0]', clock='tb.clk', xz_value=0)
unknown = reader.load_unknown_mask('tb.data[7:0]', clock='tb.clk')
known_value = value.mask(unknown == 0)
```

掩码的每一位对应一个被选中的源信号位。无论值波形是否为有符号数，掩码都按无符号数返回。

## 提取 NumPy 结果

需要将数据交给 NumPy 或其他分析库时，使用 `.value`：

```python
import numpy as np

print(f'Average active value: {np.mean(active.value):.2f}')
print(f'Maximum active value: {np.max(active.value)}')
```

在还需要周期或时间信息时，保留 `Waveform` 对象，不要过早取出 `.value`。
