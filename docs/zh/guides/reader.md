# Reader

wavekit 提供三种 Reader，分别读取 VCD、FST 和 FSDB 文件。它们使用相同的接口。推荐使用 `with` 管理 Reader，代码块结束时会自动释放资源：

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

`FsdbReader` 需要 Verdi NPI 运行时（`libNPI.so`）。打开 FSDB 文件前，请先阅读 [FSDB 安装和运行时配置](../getting-started/installation.md)。

## 层级结构

Reader 打开文件后，会解析出一棵由 `Scope` 和 `Signal` 组成的层级树，`reader.top_scopes` 是这棵树的入口。

`Scope` 对应模块实例或 generate block 之类的作用域，`Signal` 对应具体的信号——可以是普通标量，也可以是 struct、array、union 等复合类型。`Signal` 上带有 `width`（位宽）、`composite_type`（信号类型）等基本信息。

```python
from wavekit import Scope, Signal

top = reader.top_scopes[0]
for child in top.children:
    if isinstance(child, Scope):
        print('Scope', child.name)
    elif isinstance(child, Signal):
        print('Signal', child.name, child.width)
```

关于如何访问某个 `Signal`/`Scope`，见 [信号查询](signal-query.md)。

## 波形

信号按时钟采样后得到的就是波形 `Waveform`，它是三个等长数组的组合：

- `.value` —— 每个采样点的信号值；
- `.cycle` —— 绝对时钟周期号，文件中第一个采样边沿记为周期 0；
- `.time` —— 每个采样点对应的仿真时间戳。

```python
with reader.clock_domain(clock='tb.clk'):
    data = reader['tb.dut.data[7:0]'].w
    valid = reader['tb.dut.valid'].w
    ready = reader['tb.dut.ready'].w

print(data.cycle[:5])   # [0 1 2 3 4]
print(data.value[:5])   # [0 3 3 7 12]

# 可以直接对整个 Waveform 做运算，结果也是一个 Waveform
fire = valid & ready   # 每个周期握手是否成立
```

`Waveform` 上的操作（过滤、位运算、算术运算等）都会返回新的 `Waveform`，并同步保持这三个数组的对齐关系，方便随时追溯到原始的周期或时间点。

关于如何加载 `Waveform`，见 [信号查询](signal-query.md)；关于`Waveform` 支持哪些操作，见 [波形分析](waveform-analysis.md)。
