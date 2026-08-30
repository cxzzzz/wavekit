# 第一个波形：FIFO 水位

本教程使用仓库中的完整 FIFO 水位示例：[`example/fifo_occupancy/`](https://github.com/cxzzzz/wavekit/tree/main/example/fifo_occupancy/)。命令请从仓库根目录执行。

先生成 VCD 波形：

```console
make -C example/fifo_occupancy sim
```

## 加载波形

从 VCD 中加载 FIFO 的写指针和读指针，计算 FIFO 水位并输出统计量：

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

`occupancy` 仍然是一个 `Waveform`，可以继续参与波形分析。它的 `.value` 属性是 NumPy 数组，可以直接传给 NumPy 函数，也可以和其他 NumPy 数组配合使用。

对这个测试用例，脚本输出：

```text
Average occupancy: 4.64
Maximum occupancy: 7
```

## 运行完整示例

下面的命令会重新生成波形并运行 `occupancy.py`：

```console
make -C example/fifo_occupancy all
```

Makefile 会编译 `fifo_tb.sv` 和 `fifo.sv`，运行仿真，然后运行 `occupancy.py`。更多示例请参阅[示例索引](../examples.md)。

## 下一步

- 了解 [Reader 选项和格式相关配置](../guides/reader.md)。
- 探索 [Waveform 操作](../guides/waveform-analysis.md)。
- 如果设计中有重复的层次路径，可以使用[信号查询](../guides/signal-query.md)。
