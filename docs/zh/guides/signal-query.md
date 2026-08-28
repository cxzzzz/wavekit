# 信号查询

波形文件中的层次结构由 scope 和 signal 组成。信号查询可以解析单条路径、选择信号范围，也可以一次加载一组名称有规律的信号，不必为每个具体名称单独调用 API。

## 精确路径

使用点号分隔的层次路径进行精确查询。末尾的范围选择器是可选的，使用 Verilog 风格的索引和范围表示，例如 `[0]` 或 `[31:16]`。省略范围时，Reader 会使用信号在文件中保存的原始范围：

```python
with VcdReader('simulation.vcd') as reader:
    data = reader.load_waveform(
        'tb.dut.data[31:16]',
        clock='tb.clk',
    )
```

如果只需要查看单个节点的层次信息，而不需要加载采样值，可以使用 `get_signal()` 或 `get_scope()`：

```python
signal = reader.get_signal('tb.dut.data')
scope = reader.get_scope('tb.dut')
```

## 批量加载

当设计中有一组名称有规律的信号时，可以使用 `load_matched_waveforms()` 一次加载它们。

查询路径由点号分隔的多个组件组成。每个组件可以是用于精确匹配的固定名称，也可以包含大括号、正则、通配符或模块定义表达式。每个匹配表达式都会在返回结果的 key 中对应一个 capture。

返回值是一个字典，每个匹配到的信号对应一个条目。字典的 key 是由 capture 组成的 tuple，顺序与查询路径中的匹配表达式一致。固定路径组件不会进入 key，因此不含匹配表达式的查询使用空 tuple `()`。

例如，下面的查询会同时匹配两个维度：FIFO 索引和信号类型：

```python
with VcdReader('simulation.vcd') as reader:
    waves = reader.load_matched_waveforms(
        'tb.fifo_{0..3}.{wr,rd}_en',
        clock_path='tb.clk',
    )

    for key, wave in waves.items():
        print(key, wave.signal.full_name)
```

输出如下：

```text
(BraceCapture(groups=('0',)), BraceCapture(groups=('wr',))) tb.fifo_0.wr_en
(BraceCapture(groups=('0',)), BraceCapture(groups=('rd',))) tb.fifo_0.rd_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('wr',))) tb.fifo_1.wr_en
(BraceCapture(groups=('1',)), BraceCapture(groups=('rd',))) tb.fifo_1.rd_en
...
```

第一个 capture 表示 FIFO 索引，第二个 capture 表示信号类型。

### 查询语法

使用以下语法构造查询路径：

| 语法 | 示例 | key 中保存的 capture |
| --- | --- | --- |
| 精确路径 | `tb.dut.valid` | 无 capture |
| 大括号列表 | `sig_{read,write}` | `BraceCapture` |
| 整数范围 | `fifo_{0..3}.ptr` | 每个索引对应一个 `BraceCapture` |
| 步进范围 | `lane_{0..6..2}.valid` | `BraceCapture`，值为 `0`、`2`、`4`、`6` |
| 标准正则 | `tb.u0./J_([a-z]+)/` | `RegexCapture` |
| 旧版正则 | `@([a-z]+)_valid` | `RegexCapture` |
| 单层通配符 | `tb.*.valid` | `WildcardCapture` |
| 递归通配符 | `tb.**.valid` | `WildcardCapture` |
| 直接匹配模块定义 | `tb.$fifo_unit.ptr` | `ExactCapture`（仅 FSDB） |
| 递归匹配模块定义 | `tb.$$fifo_unit.ptr` | `ExactCapture`（仅 FSDB） |

`get_matched_signals()`、`get_matched_scopes()` 和 `Reader.eval()` 也使用同一套查询语法。

`$` 和 `$$` 仅用于 FSDB 的模块定义匹配。

`load_matched_unknown_masks()` 使用与 `load_matched_waveforms()` 相同的查询和结果 key 规则。scope 查询使用 `get_matched_scopes()`；末尾范围选择器只能用于 signal 查询，不能用于 scope 查询。

如果 `clock_path` 只匹配到一个信号，该时钟会用于所有结果。如果匹配到多个时钟，wavekit 会根据每个信号的 key，选择与其匹配且前缀最长的时钟。

## 计算表达式

对于只需要写一行的简单计算，`Reader.eval()` 比较方便。

在 `single` 模式下，每条路径都必须精确匹配一个信号。这是默认模式：

```python
occupancy = reader.eval(
    'tb.dut.w_ptr[2:0] - tb.dut.r_ptr[2:0]',
    clock='tb.clk',
)
```

表达式中也可以调用 Waveform 操作：

```python
byte_count = reader.eval(
    'bit_count(tb.axi.wstrb[7:0] * (tb.axi.wvalid & tb.axi.wready))',
    clock='tb.clk',
)
```

在 `zip` 模式下，wavekit 会按 capture tuple 将带匹配表达式的路径配对展开；只匹配到一个信号的路径会广播到每一组：

```python
occupancies = reader.eval(
    'tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]',
    clock='tb.clk',
    mode='zip',
)
```
