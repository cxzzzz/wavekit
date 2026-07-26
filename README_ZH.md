# wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Downloads](https://pepy.tech/badge/wavekit)](https://pepy.tech/project/wavekit)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

**wavekit** 是一个面向数字电路波形分析的高性能 Python 库。它把 VCD / FST / FSDB 信号加载成基于 NumPy 的 `Waveform` 对象，并在此基础上提供高效的信号处理、协议分析和自动化验证能力。

> 🤖 **AI 集成**：[wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) 提供了 MCP 服务器，支持 AI 辅助波形分析——自动加载信号、运行模式匹配，无需手写脚本。

## 特性

- **批量信号提取**：支持大括号展开、整数范围、正则表达式等匹配方式，可以一次加载一组相关信号。
- **波形分析 API**：提供类 NumPy 的接口，支持算术运算、掩码过滤、位域截取、边沿检测、时间/时钟切片等常用操作。
- **时序模式匹配**：内置声明式和编程式 Pattern 引擎，可从波形中提取协议事务、测量握手延迟、检测 timeout 或挂死等异常。
- **高性能波形处理**：支持 VCD、FST 与 FSDB，采样数据存储在紧凑的 NumPy 数组中，适合处理较大的仿真波形。

## 安装

```bash
pip install wavekit
```

**FSDB 支持说明**：读取 FSDB 文件需要 Verdi 运行时库（`libNPI.so`）在运行时可访问。可通过以下任一方式配置：

- `WAVEKIT_NPI_LIB`：直接指定 `libNPI.so` 的路径
- `VERDI_HOME`：Verdi 安装目录（库文件会在 `$VERDI_HOME/share/NPI/lib/...` 下自动查找）
- `LD_LIBRARY_PATH`：系统库搜索路径

## 快速上手

> 以下示例中的文件名（如 `sim.vcd`）和信号路径均为占位符，请替换为你实际的 VCD / FST / FSDB 文件及设计层级路径。

### 1. 批量提取信号

通过大括号展开或正则表达式，一次加载多个相关信号。

```python
from wavekit import VcdReader

with VcdReader("jtag.vcd") as f:
    # key 中是 BraceCapture，其 groups 分别为 ("state",) / ("next",)。
    waves = f.load_matched_waveforms(
        "tb.u0.J_{state,next}[3:0]",
        clock_path="tb.tck",
    )

    # /regex/ 是正式正则语法，捕获组保存在 RegexCapture.groups 中。
    waves = f.load_matched_waveforms(
        r"tb.u0./J_([a-z]+)/",
        clock_path="tb.tck",
    )
```

### 2. 信号分析

Waveform 支持 NumPy 风格的算术运算、掩码过滤和边沿检测。

```python
import numpy as np
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    clock = "fifo_tb.clk"
    depth = 8

    w_ptr = f.load_waveform("fifo_tb.s_fifo.w_ptr[2:0]", clock=clock)
    r_ptr = f.load_waveform("fifo_tb.s_fifo.r_ptr[2:0]", clock=clock)
    wr_en = f.load_waveform("fifo_tb.s_fifo.wr_en", clock=clock)

    # 计算 FIFO 实时占用量
    occupancy = (w_ptr + depth - r_ptr) % depth
    print(f"平均占用: {np.mean(occupancy.value):.2f}")

    # 只保留写使能有效时的占用量
    write_occ = occupancy.mask(wr_en == 1)

    # 检测写突发（wr_en 的上升沿）
    burst_cycles = wr_en.rising_edge()
```

加载未知态（X/Z）掩码，在不改变两态数值模型的前提下检测源信号中的未知位。掩码中每一位为 `1` 表示对应源位为 `X` 或 `Z`。

> **实验性功能**：`load_unknown_mask` / `load_matched_unknown_masks` 为实验性 API，未来版本可能调整。

```python
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    clock = "fifo_tb.clk"
    data = f.load_waveform("fifo_tb.s_fifo.data[7:0]", clock=clock, xz_value=0)
    unknown = f.load_unknown_mask("fifo_tb.s_fifo.data[7:0]", clock=clock)

    # 只保留源位完全已知的采样
    known_data = data.mask(unknown == 0)
```

### 3. 表达式求值

直接通过信号路径字符串书写表达式，无需逐个手动加载信号。

```python
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    # single 模式：每个路径必须精确匹配一个信号
    occupancy = f.eval(
        "fifo_tb.s_fifo.w_ptr[2:0] - fifo_tb.s_fifo.r_ptr[2:0]",
        clock="fifo_tb.clk",
    )

    # zip 模式：大括号模式按类型化 CaptureKey 逐组展开，每组求值一次
    occupancies = f.eval(
        "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
        clock="tb.clk",
        mode="zip",
    )
```

### 4. 时序模式匹配

`Pattern` 用来在波形里扫描并提取匹配的事务，比如一次请求/响应、一个 burst、一段 stall，或者其他重复出现的时序过程。

根据事务特点选择写法：

- **声明式**：用 `.wait()`、`.consume()`、`.capture()`、`.loop()` 等步骤描述事务，适合固定流程。
- **编程式**：用普通 Python 函数描述事务，适合依赖波形值的动态流程，比如动态分支或按 ID 路由。

`match(pattern)` 返回 `MatchRecords`，而 `collect(body)` 返回 Python `list`，用于保存提取出的值。

#### 声明式示例

**AXI-Lite 读延迟测量**

```python
from wavekit import VcdReader
from wavekit.pattern import Pattern, match, collect

with VcdReader("axi_tb.vcd") as f:
    clk     = "tb.clk"
    arvalid = f.load_waveform("tb.dut.arvalid",     clock=clk)
    arready = f.load_waveform("tb.dut.arready",     clock=clk)
    rvalid  = f.load_waveform("tb.dut.rvalid",      clock=clk)
    rready  = f.load_waveform("tb.dut.rready",      clock=clk)
    rdata   = f.load_waveform("tb.dut.rdata[31:0]", clock=clk)

    pattern = (
        Pattern()
        .wait(arvalid & arready)   # AR 握手 → 事务开始
        .wait(rvalid  & rready)    # R  握手 → 事务结束
        .capture("rdata", rdata)
    )
    result = match(pattern)

    ok = result.filter_ok()
    print(f"读延迟（周期）: {ok.duration.value}")
    print(f"读数据: {ok.captures['rdata'].value}")
```

**AXI 写突发（多拍数据）**

```python
beat = Pattern().consume(wvalid & wready, channel="w").capture("beats", wdata, mode="list")

pattern = (
    Pattern()
    .wait(awvalid & awready)   # AW 握手 → 突发开始
    .loop(beat, until=wlast)   # 收集每拍数据，直到 wlast
)
result = match(pattern)

for i, inst in enumerate(result.filter_ok()):
    print(f"突发 {i}: {len(inst.captures['beats'])} 拍")
```

**Stall 检测**

```python
stall = valid & (ready == 0)

pattern = (
    Pattern()
    .wait(stall.rising_edge())             # stall 开始
    .loop(Pattern().delay(1), when=stall)  # 持续等待，直到 stall 结束
)
result = match(pattern)

stalls = result.filter_ok()
print(f"Stall 持续时间: {stalls.duration.value} 周期")
```

#### 编程式示例

**DMA 风格命令流**

当命令的 opcode 会改变后续时序形状时，用编程式控制流更自然。

```python
cmd_fire = cmd_valid & cmd_ready   # 在函数外预先算好
w_fire = w_valid & w_ready
rsp_fire = rsp_valid & rsp_ready
r_fire = r_valid & r_ready

OP_READ = 0
OP_WRITE = 1

def read_dma_cmd(ctx):
    if not ctx.value(cmd_fire):
        return None

    op = int(ctx.value(cmd_op))
    addr = int(ctx.value(cmd_addr))
    length = int(ctx.value(cmd_len))

    if op == OP_WRITE:
        data = []
        for _ in range(length):
            ctx.consume(w_fire, channel="wdata")
            data.append(int(ctx.value(w_data)))

        ctx.consume(rsp_fire, channel="rsp")
        return {
            "op": "write",
            "addr": addr,
            "data": data,
            "status": int(ctx.value(rsp_status)),
        }

    if op == OP_READ:
        ctx.consume(rsp_fire, channel="rsp")
        data = []
        for _ in range(length):
            ctx.consume(r_fire, channel="rdata")
            data.append(int(ctx.value(r_data)))

        return {"op": "read", "addr": addr, "data": data}

    ctx.require(False, message=f"unknown DMA op {op}")
    return None

commands = collect(read_dma_cmd)
print(f"捕获到 {len(commands)} 个 command")
```

一些编程式 Pattern 的使用建议：

- 固定的波形表达式（比如 `fire = valid & ready`）在函数外先算好，避免每周期反复构造。
- 函数开头用 `if ctx.value(fire): ...` 判断当前周期是不是事务起点，不是起点就 `return None`。
- 如果你需要的是“非阻塞轮询”或多个候选通道之间的仲裁，可以用 `ctx.try_consume(...)`。像这种线性的 burst 例子，`ctx.consume(...)` 更直接。
- 如果某个阻塞步骤必须限制等待时间，再加 `timeout=<cycles>`。
  在 `match()` 里 timeout 会变成 `MatchStatus.Timeout(...)`；在 `collect()`
  里会抛出 `PatternError`。


## API 参考

### Reader

| 方法 | 说明 |
|------|------|
| `VcdReader(file)` / `FstReader(file)` / `FsdbReader(file)` | 打开波形文件。建议作为上下文管理器使用。`FsdbReader` 需要 Verdi 运行时环境（通过 `WAVEKIT_NPI_LIB`、`VERDI_HOME` 或 `LD_LIBRARY_PATH` 配置）。 |
| `reader.load_waveform(signal, clock, ...)` | 加载单个信号，按时钟边沿采样，返回 `Waveform`。 |
| `reader.load_unknown_mask(signal, clock, ...)` | **实验性**。加载 X/Z 位存在性为无符号掩码 `Waveform`。 |
| `reader.load_matched_waveforms(signal_path, clock_path, ...)` | 批量加载匹配信号，返回 `dict[CaptureKey, Waveform]`。 |
| `reader.load_matched_unknown_masks(signal_path, clock_path, ...)` | **实验性**。批量加载匹配信号的 X/Z 掩码，返回 `dict[CaptureKey, Waveform]`。 |
| `reader.eval(expr, clock, mode='single'\|'zip', ...)` | 对包含信号路径的算术表达式直接求值。 |
| `reader.get_matched_signals(path)` | 将查询解析为 `Signal` 对象，不加载数据。 |
| `reader.get_matched_scopes(path)` | 将查询解析为 `Scope` 对象。 |
| `reader.top_scopes` | 根 `Scope` 节点组成的不可变 tuple。 |
| `has_fsdb_support()` | 检查当前是否可以使用 Verdi FSDB 运行时。 |

### 信号路径模式

| 语法 | 示例 | 作用 | 结果 key 中的 Capture |
|------|------|------|-----------------------|
| 普通名称 | `tb.dut.valid` | 精确名称匹配 | 无 |
| `{a,b,c}` | `sig_{read,write}` | 枚举命名变体 | `BraceCapture(path=..., groups=...)` |
| `{N..M}` | `fifo_{0..3}.ptr` | 整数范围 | `BraceCapture(path=..., groups=...)` |
| `{N..M..step}` | `lane_{0..6..2}` | 带步长的范围 | `BraceCapture(path=..., groups=...)` |
| `/<regex>/` | `/([a-z]+)_valid/` | 正式正则语法，保留捕获组 | `RegexCapture(path=..., groups=...)` |
| `@<regex>` | `@([a-z]+)_valid` | 兼容旧代码的正则语法 | `RegexCapture(path=..., groups=...)` |
| `*` / `**` | `tb.*.valid` / `tb.**.valid` | 单层 / 递归通配 | `WildcardCapture(path=...)` |
| `$ModName` | `tb.$fifo_unit.ptr` | 按模块名匹配直接子层级（仅 FSDB） | `ExactCapture(path=..., definition=...)` |
| `$$ModName` | `tb.$$fifo_unit.ptr` | 按模块名匹配任意深度后代（仅 FSDB） | `ExactCapture(path=..., definition=...)` |

`$` 和 `$$` 是路径段级修饰符：它们可以与普通名称、brace、regex，
以及末尾的 range 选择继续组合。
例如：`tb.$/fifo_(in|out)/.data_{0..3}[7:0]`。

匹配与层级查询 API 返回以 `CaptureKey = tuple[Capture, ...]` 为 key 的字典。普通精确路径段不会进入 key，因此纯精确查询使用 `()`。

### Waveform

`Waveform` 内部封装了三个平行的 NumPy 数组（`.value`、`.clock`、`.time`）。所有操作均返回新的 `Waveform` 实例。

**算术与比较**：`+`、`-`、`*`、`//`、`%`、`**`、`/`、`&`、`|`、`^`、`~`、`==`、`!=`、`<<`、`>>`

**过滤与切片**

| 方法 | 说明 |
|------|------|
| `wave.mask(mask)` | 保留布尔 Waveform 或数组为 True 的采样点 |
| `wave.filter(fn)` | 保留满足 `fn(value)` 为 True 的采样点 |
| `wave.cycle_slice(begin, end)` | 按时钟周期范围裁剪 `[begin, end)` |
| `wave.time_slice(begin, end)` | 按仿真时间范围裁剪 |
| `wave.slice(begin_idx, end_idx)` | 按数组下标裁剪 |
| `wave.take(indices)` | 按给定下标选取采样点 |

**变换**

| 方法 | 说明 |
|------|------|
| `wave.map(fn, width, signed)` | 逐元素变换 |
| `wave.unique_consecutive()` | 去除连续的重复值 |
| `wave.downsample(chunk, fn)` | 按块聚合降采样 |
| `wave.as_signed()` / `wave.as_unsigned()` | 重新解释有符号/无符号 |

**位操作**

| 方法 / 语法 | 说明 |
|-------------|------|
| `wave[high:low]` | 截取位域（Verilog 风格，返回无符号） |
| `wave[n]` | 截取单比特 |
| `wave.split_bits(n)` | 按 n 位分组拆分（低位在前） |
| `Waveform.concatenate([w0, w1, ...])` | 拼接（w0 为最低位） |
| `wave.bit_count()` | population count |

**边沿检测**（仅支持 1-bit 信号）

| 方法 | 说明 |
|------|------|
| `wave.rising_edge()` | 0→1 跳变时为 True |
| `wave.falling_edge()` | 1→0 跳变时为 True |

**相对时间访问**

| 方法 | 说明 |
|------|------|
| `wave.relative(offset, pad, pad_value)` | 按周期偏移（正数为未来，负数为过去） |
| `wave.ahead(n, pad, pad_value)` | 向前看 n 个周期（`relative(n)` 的简写） |
| `wave.back(n, pad, pad_value)` | 向后看 n 个周期（`relative(-n)` 的简写） |

`pad` 控制边界处理方式：`'repeat'`（默认）用首/尾值填充，`'value'` 用指定的 `pad_value` 填充。

```python
# 上升沿检测的另一种写法
rising = (wave == 0) & wave.ahead()

# 与 3 个周期前的值比较
changed = wave != wave.back(3)
```

### Pattern

**构造方式**

| API | 说明 |
|-----|------|
| `Pattern()` | 创建声明式 Pattern。继续调用 `.wait()`、`.capture()` 等方法添加步骤；执行参数放在 `match()` / `collect()`。 |
| `match(pattern_or_body, *, axis=None, timeout=None, timeout_message=None, start_cycle=None, end_cycle=None)` | 运行声明式 Pattern 或编程式检查函数，返回 `MatchRecords`。检查函数中 `return ctx.OK` 表示匹配成功，`return None` 表示跳过。 |
| `collect(body, *, axis=None, timeout=None, timeout_message=None, start_cycle=None, end_cycle=None)` | 运行编程式提取函数，收集所有非 `None` Python 对象。 |

**声明式步骤**

| 方法 | 说明 |
|------|------|
| `.wait(cond, *, require=None, require_message=None)` | 等到 `cond` 为真，但不占用这个事件。如果当前周期已经满足条件，会在同一周期继续；如果想等到下一周期，显式写 `.delay(1)`。`require` 会在等待期间每周期检查，失败则标记为 `MatchStatus.RequireViolated`。 |
| `.consume(cond, channel, *, require=None, require_message=None)` | 等到 `cond` 为真，并从 `channel` 独占消费这个事件。适合把请求和响应按 FIFO 顺序配对，或按 key 分流。 |
| `.delay(n, *, require=None, require_message=None)` | 前进 n 个周期。`delay(0)` 不做任何事。`require` 在延迟期间必须一直为真。 |
| `.capture(name, signal, *, mode='last')` | 在当前周期记录信号值。`mode='last'` 默认覆盖旧值；`'first'` 只保留第一次；`'list'` 追加到列表。 |
| `.require(cond)` | 检查当前周期必须满足 `cond`，否则标记为 `MatchStatus.RequireViolated`。 |
| `.loop(body, *, until=None, when=None)` | 循环执行 `body`。`until` 是先执行再判断退出；`when` 是先判断，不满足就不进入循环。 |
| `.repeat(body, n)` | 把 `body` 重复执行 n 次。n 可以是可调用对象。 |
| `.branch(cond, true_body, false_body)` | 条件分支。 |

编程式函数里也可以用同样的等待、消费和延迟操作：
`ctx.wait(...)`、`ctx.consume(...)`、`ctx.delay(...)`。

**编程式上下文**

| API | 说明 |
|-----|------|
| `ctx.value(waveform, offset=0)` | 读取当前采样点的值，可用 `offset` 读前后几个采样点。 |
| `ctx.cycle(waveform, offset=0)` | 读取当前采样点的周期号，可带 `offset`。 |
| `ctx.time(waveform, offset=0)` | 读取当前采样点的时间戳，可带 `offset`。 |
| `ctx.wait(cond, require=None, require_message=None)` | 等到条件为真；只观察，不消费事件。 |
| `ctx.consume(cond, channel, require=None, require_message=None)` | 等到条件为真，并从 `channel` 独占消费这个事件。 |
| `ctx.try_consume(cond, channel)` | 非阻塞地轮询 `channel`。只有条件和通道都可用时才返回 `True`。 |
| `ctx.delay(n, require=None, require_message=None)` | 前进 n 个周期。 |
| `ctx.capture(name, value, mode='last')` | 在编程式 `match()` 中记录捕获值。 |
| `ctx.OK` | 在编程式 `match()` 中返回，表示这次匹配成功。 |

**动态回调**

回调函数的参数取决于它用在声明式 API 还是编程式 API 中：

- 在 `Pattern().wait(...)`、`Pattern().consume(...)` 这类声明式 API 中，回调接收 `(index, captures)`。
- 在编程式函数中，传给 `ctx.wait(...)`、`ctx.consume(...)` 等方法的回调不接收参数。需要当前下标、
  captures 或信号值时，直接在闭包里使用 `ctx`。

**Channel 与 consume 的关系**

`wait()` 只是观察事件，所以多个匹配实例可以看到同一个响应。
`consume()` 会记录所有权：同一个 `(channel, cycle)` 只能被一个实例占用，
因此可以按 FIFO 顺序把请求和响应配对。

`Channel` 是 `consume()` 的逻辑占用键。可以传 `Channel` 对象、
hashable key，或者动态回调给 `consume(..., channel=...)`。
共享同一个 channel key 的实例会竞争同一个逻辑通道。

```python
from collections import defaultdict
from wavekit.pattern import Channel, Pattern, match

# 多 Bank Cache：每个 bank 有独立的响应端口，多个 bank 可以在同一周期返回数据。
# 按 bank 分 Channel 可以让每个在飞读请求各自消费对应 bank 的响应，
# 同时保留同一 bank 内的 FIFO 顺序。
banks = defaultdict(Channel)

pattern = (
    Pattern()
    .wait(req_valid)
    .capture('bank', req_addr & 1)
    .consume(
        lambda i, cap: bank_valid[cap['bank']].value[i],
        channel=lambda i, cap: banks[cap['bank']],
    )
    .capture('rdata',
        lambda i, cap: bank_data[cap['bank']].value[i])
)
result = match(pattern)
```

**`MatchRecords`**

| 字段 | 说明 |
|------|------|
| `.start` / `.end` | 点位 Waveform。`.value` 是波形数组采样下标，`.clock` 是绝对 cycle，`.time` 是仿真时间戳。end 为 inclusive。 |
| `.duration` | `end.value - start.value + 1` 个采样周期。 |
| `.status` | `MatchStatus.OK()`、`MatchStatus.Timeout(...)` 或 `MatchStatus.RequireViolated(...)`。 |
| `.captures` | 捕获结果，类型为 `dict[str, Waveform]`。 |
| `.ok` | 布尔 Waveform，表示 `status == MatchStatus.OK()`。 |
| `.failed` | 布尔 Waveform，表示 `status != MatchStatus.OK()`。 |
| `.filter_ok()` | 只保留状态为 `OK` 的匹配实例。 |
| `.filter_status(status)` | 只保留指定 status 对象或 status class 的匹配实例。 |
| `.filter_failed()` | 只保留非 OK 的匹配实例。 |

`MatchRecords[i]` 返回单个 `MatchRecord`，切片则返回新的 `MatchRecords`。

## 开发

本项目使用 [Poetry](https://python-poetry.org/) 管理依赖与打包。

### 环境搭建

```bash
git clone https://github.com/cxzzzz/wavekit.git
cd wavekit
poetry install
```

### 测试

测试用例位于 `tests/` 目录，使用 [pytest](https://pytest.org/) 运行。

```bash
# 运行全部测试
poetry run pytest

# 运行指定文件
poetry run pytest tests/test_pattern.py

# 详细输出
poetry run pytest -v
```

### 代码检查与格式化

使用 [Ruff](https://github.com/astral-sh/ruff) 进行代码检查与格式化。

```bash
# 检查代码规范
poetry run ruff check .

# 检查格式（不修改文件）
poetry run ruff format --check .

# 自动修复格式
poetry run ruff format .
```

### 类型检查

```bash
poetry run mypy .
```

## 参与贡献

欢迎提交 Issue 和 PR！提交 PR 前请先运行测试和格式检查：

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](./LICENSE) 文件。
