# Reader

wavekit 提供三种 Reader，分别读取 VCD、FST 和 FSDB 文件。它们使用相同的时钟采样加载接口。使用 `with` 管理 Reader，代码块结束时会自动释放资源：

```python
from wavekit import FstReader, VcdReader

with VcdReader('simulation.vcd') as reader:
    data = reader.load_waveform('tb.dut.data[7:0]', clock='tb.clk')

with FstReader('simulation.fst') as reader:
    data = reader.load_waveform('tb.dut.data[7:0]', clock='tb.clk')
```

`FsdbReader` 还需要 Verdi NPI 运行时（`libNPI.so`）。打开 FSDB 文件前，请先阅读 [FSDB 安装和运行时配置](../getting-started/installation.md)。

## 采样和窗口

`load_waveform(signal, clock, ...)` 会在时钟边沿采样指定信号。默认使用下降沿，以减少信号变化带来的采样错误。传入 `sample_on_posedge=True` 可以改为在上升沿采样。使用 `begin_time` 和 `end_time` 指定时间窗口，或使用 `begin_cycle` 和 `end_cycle` 指定绝对时钟周期窗口。时间窗口和周期窗口不能混用。

同一计算、掩码操作或模式匹配中使用的波形，必须使用同源时钟、相同的采样边沿，以及相同的时间窗口或周期窗口。

普通方式加载时，X/Z 状态会被 `xz_value` 替换，默认值为 0。如果需要保留这些状态，可以加载对应的掩码：

```python
value = reader.load_waveform('tb.data[7:0]', clock='tb.clk', xz_value=0)
unknown = reader.load_unknown_mask('tb.data[7:0]', clock='tb.clk')
known_value = value.mask(unknown == 0)
```

`load_unknown_mask()` 和 `load_matched_unknown_masks()` 会将源信号中的 X/Z 状态编码为无符号掩码。这样即使未知值被替换为 `xz_value=0`，仍然可以跟踪 X/Z 状态。
