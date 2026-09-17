# 信号查询

获取信号有两种方式：知道确切路径时直接访问，名字有规律时用批量查询。拿到 `Signal` 后，需要加载对应波形才能参与分析。

## 直接访问

`reader[path]` 返回指定路径对应的 `Signal` 或 `Scope`：

```python
signal = reader['tb.dut.data']
scope = reader['tb.dut']
selected = reader['tb.dut.data[31:16]']  # 末尾可以带位选择
```

scope 和复合信号（struct、array）支持同样的查找方式，可以继续用相对路径：

```python
tb = reader['tb']
signal = tb['dut']['data']        # 一级一级往下取
signal = tb['dut.data']           # 也可以一次写完
member = tb['pkt']['valid']       # struct 成员同样适用
```

用整数或切片下标对信号取位，语义和 Verilog 一致：

```python
bit = signal[7]        # 取一位
field = signal[31:16]  # 取多位
```

## 加载波形

`reader[path]` 拿到的是 `Signal`，不是波形`Waveform`。要获取波形数据，先进入一个时钟域，然后使用 `Signal.w`/`load_waveform()` ：

```python
with VcdReader('simulation.vcd') as reader:
    with reader.clock_domain('tb.clk'):
        valid = reader['tb.dut.valid'].w
        data = reader.load_waveform('tb.dut.data[7:0]')
```

如果想在时钟域外获取波形，或者想临时指定不同的时钟/窗口，就需要显式传 `clock`、边沿、窗口等参数：

```python
data = reader.load_waveform(
    'tb.dut.data[7:0]', clock='tb.clk',
    sample_on_posedge=True,
    start_cycle=100, end_cycle=200,
)
```

采样默认发生在下降沿，以避免信号变化瞬间带来的采样错误；可以使用 `sample_on_posedge=True`改为上升沿采样。可以用 `start_time`/`end_time` 按仿真时间指定采样窗口，也可以用`start_cycle`/`end_cycle` 按绝对时钟周期指定，但两者不能混用。

用于同一次计算或模式匹配的波形，必须使用同一个时钟源、相同的采样边沿，以及相同的采样窗口，所以更推荐使用统一的时钟域，而不是每次都重复传参。

## 加载掩码

加载波形会把 X/Z 状态替换成 `xz_value`（默认为 0）。如果需要保留这些状态（比如需要排除 X 态，或者需要追踪哪些位是 X/Z 时），可以用 `Signal.m` 或 `load_unknown_mask()` 加载对应的掩码：

```python
with reader.clock_domain(clock='tb.clk'):
    value = reader['tb.data[7:0]'].w
    unknown = reader['tb.data[7:0]'].m   # X/Z 存在性掩码
    # 或者：unknown = reader.load_unknown_mask('tb.data[7:0]')
    known_value = value.mask(unknown == 0)
```

`Signal.m` 是 `Signal.unknown_mask()` 的默认参数写法。掩码的每一位对应源信号的一位，标记该位在源文件里是否为 X/Z。

## 批量查询

如果信号名字有规律——同一模块复制了多份、信号带编号、名字共享前缀——可以用一条
查询把它们一起匹配出来。

查询路径由点号分隔，每一级要么是固定名称（精确匹配），要么是一个匹配表达式，
比如大括号、正则或通配符。每个匹配表达式都会在结果 key 里留下
一个 capture，描述它具体匹配到的内容。

`get_matched_signals()` 返回一个字典：每个匹配到的信号一条记录，key 是按查询路径
顺序排好的 capture tuple（精确匹配的key不会被包含在capture tuple中）。

下面这条查询同时匹配两个维度——FIFO 编号和信号类型：

```python
with VcdReader('simulation.vcd') as reader:
    signals = reader.get_matched_signals('tb.fifo_{0..3}.{wr,rd}_en')

    for key, signal in signals.items():
        print(key, signal.full_name)
```

输出：

```text
(BraceCapture(groups=('0',)), BraceCapture(groups=('wr',))) tb.fifo_0.wr_en
(BraceCapture(groups=('0',)), BraceCapture(groups=('rd',))) tb.fifo_0.rd_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('wr',))) tb.fifo_1.wr_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('rd',))) tb.fifo_1.rd_en
...
```

要把匹配成功的信号一次加载成波形，可以用 `load_matched_waveforms()`:

```python
with VcdReader('simulation.vcd') as reader:
    with reader.clock_domain('tb.clk'):
      waves = reader.load_matched_waveforms('tb.fifo_{0..3}.{wr,rd}_en')
```

### 查询语法

查询路径支持以下语法：

| 语法 | 示例 | 捕获的 key 组件 |
| --- | --- | --- |
| 精确路径 | `tb.dut.valid` | 无 capture |
| 大括号列表 | `sig_{read,write}` | `BraceCapture` |
| 整数范围 | `fifo_{0..3}.ptr` | 每个索引一个 `BraceCapture` |
| 步进范围 | `lane_{0..6..2}.valid` | `BraceCapture` 为 `0`、`2`、`4`、`6` |
| 规范正则 | `tb.u0./J_([a-z]+)/` | `RegexCapture` |
| 兼容正则 | `@([a-z]+)_valid` | `RegexCapture` |
| 单层通配符 | `tb.*.valid` | `WildcardCapture` |
| 递归通配符 | `tb.**.valid` | `WildcardCapture` |
| 直接模块定义 | `tb.$fifo_unit.ptr` | `ExactCapture`（FSDB） |
| 递归模块定义 | `tb.$$fifo_unit.ptr` | `ExactCapture`（FSDB） |

`get_matched_scopes()`、`get_matched_nodes()`、`load_matched_unknown_masks()` 和
`Reader.eval()` 都支持上述语法。

`clock_path` 只匹配到一个信号时，所有结果共用它作为时钟；匹配到多个信号时，
每个信号会选取 key 是其最长前缀的一个时钟。

`$` 和 `$$` 只有 FSDB 的模块定义匹配能用。

## 计算表达式

对于只需要写一行的简单计算，使用`Reader.eval()` 比较方便。

`single` 模式（默认）下，表达式里的每条路径都必须对应唯一信号：

```python
occupancy = reader.eval(
    '(tb.dut.w_ptr - tb.dut.r_ptr + 8) % 8',
    clock='tb.clk',
)
```

表达式里也能调波形操作函数：

```python
byte_count = reader.eval(
    'bit_count(tb.axi.wstrb[7:0] * (tb.axi.wvalid & tb.axi.wready))',
    clock='tb.clk',
)
```

`zip` 模式下，带匹配的路径按各自的 capture tuple 分组展开；只匹配到一个信号的路径
会广播到所有组：

```python
occupancies = reader.eval(
    'tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]',
    clock='tb.clk',
    mode='zip',
)
```
