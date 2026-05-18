# wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Downloads](https://img.shields.io/pypi/dm/wavekit.svg)](https://pypi.org/project/wavekit/)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

**wavekit** 是一个面向数字电路波形分析的 Python 基础库。它将 VCD / FSDB 仿真数据无缝转换为 Numpy 数组，让工程师能够高效地进行信号处理、协议分析和自动化验证。

> 🤖 **AI 集成**：配套的 [wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) 提供了 MCP 服务器，支持 AI 辅助波形分析——自动加载信号、运行模式匹配，无需手写脚本。

## 特性

- **灵活的信号提取**：支持大括号展开、整数范围、正则表达式等多种匹配模式，一次加载一批相关信号，省去逐个手写的繁琐。
- **丰富的分析能力**：提供类 Numpy 的 API，支持算术运算、掩码过滤、位域截取、边沿检测、时间/时钟切片等操作，几行代码即可组合出复杂的信号查询逻辑。
- **强大的时序模式匹配**：内置基于 NFA 的时序模式引擎，单次扫描即可从波形中提取协议事务、测量握手延迟、检测超时或挂死等异常。
- **高性能波形处理**：VCD / FSDB 解析器经 Cython 优化，波形数据以 Numpy 数组为后端，加载快、内存省，轻松应对大容量仿真文件。

## 安装

```bash
pip install wavekit
```

**FSDB 支持说明**：读取 FSDB 文件需要 Verdi 运行时库（`libNPI.so`）在运行时可访问。可通过以下任一方式配置：

- `WAVEKIT_NPI_LIB` — 直接指定 `libNPI.so` 的路径
- `VERDI_HOME` — Verdi 安装目录（库文件会在 `$VERDI_HOME/share/NPI/lib/...` 下自动查找）
- `LD_LIBRARY_PATH` — 系统库搜索路径

## 快速上手

> 以下示例中的文件名（如 `sim.vcd`）和信号路径均为占位符，请替换为你实际的 VCD / FSDB 文件及设计层级路径。

### 1. 批量提取信号

通过大括号展开或正则表达式，一次加载多个相关信号。

```python
from wavekit import VcdReader

with VcdReader("jtag.vcd") as f:
    # 大括号展开：同时加载 J_state 和 J_next
    # 返回：{ ('state',): Waveform, ('next',): Waveform }
    waves = f.load_matched_waveforms(
        "tb.u0.J_{state,next}[3:0]",
        clock_pattern="tb.tck",
    )

    # 正则模式（@ 前缀）：捕获组作为字典键
    waves = f.load_matched_waveforms(
        r"tb.u0.@J_([a-z]+)",
        clock_pattern="tb.tck",
    )
```

### 2. 信号分析

Waveform 支持 Numpy 风格的算术运算、掩码过滤和边沿检测。

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

    # zip 模式：大括号模式按 key 逐组展开，每组求值一次
    # 返回：{ (0,): Waveform, (1,): Waveform, (2,): Waveform, (3,): Waveform }
    occupancies = f.eval(
        "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
        clock="tb.clk",
        mode="zip",
    )
```

### 4. 时序模式匹配

用声明式的方式描述一段时序事件，引擎会在单次扫描中找出所有匹配的实例。

**AXI-Lite 读延迟测量**

```python
from wavekit import VcdReader, Pattern

with VcdReader("axi_tb.vcd") as f:
    clk     = "tb.clk"
    arvalid = f.load_waveform("tb.dut.arvalid",     clock=clk)
    arready = f.load_waveform("tb.dut.arready",     clock=clk)
    rvalid  = f.load_waveform("tb.dut.rvalid",      clock=clk)
    rready  = f.load_waveform("tb.dut.rready",      clock=clk)
    rdata   = f.load_waveform("tb.dut.rdata[31:0]", clock=clk)

    result = (
        Pattern()
        .wait(arvalid & arready)   # AR 握手 → 事务开始
        .wait(rvalid  & rready)    # R  握手 → 事务结束
        .capture("rdata", rdata)
        .timeout(256)
        .match()
    )

    valid = result.filter_valid()
    print(f"读延迟（周期）: {valid.duration.value}")
    print(f"读数据: {valid.captures['rdata'].value}")
```

**AXI 写突发（多拍数据）**

```python
beat = Pattern().wait(wvalid & wready).capture("beats", wdata, mode="list")

result = (
    Pattern()
    .wait(awvalid & awready)   # AW 握手 → 突发开始
    .loop(beat, until=wlast)   # 收集每拍数据，直到 wlast
    .timeout(512)
    .match()
)

for i, inst in enumerate(result.filter_valid()):
    print(f"突发 {i}: {len(inst.captures['beats'])} 拍")
```

**Stall 检测**

```python
stall = valid & (ready == 0)

result = (
    Pattern()
    .wait(stall.rising_edge())             # stall 开始
    .loop(Pattern().delay(1), when=stall)  # 持续等待，直到 stall 结束
    .match()
)

stalls = result.filter_valid()
print(f"Stall 持续时间: {stalls.duration.value} 周期")
```

## API 参考

### Reader

| 方法 | 说明 |
|------|------|
| `VcdReader(file)` / `FsdbReader(file)` | 打开波形文件。建议作为上下文管理器使用。`FsdbReader` 需要 Verdi 运行时环境（通过 `WAVEKIT_NPI_LIB`、`VERDI_HOME` 或 `LD_LIBRARY_PATH` 配置）。 |
| `reader.load_waveform(signal, clock, ...)` | 加载单个信号，按时钟边沿采样，返回 `Waveform`。 |
| `reader.load_matched_waveforms(pattern, clock_pattern, ...)` | 按模式批量加载信号，返回 `dict[tuple, Waveform]`。 |
| `reader.eval(expr, clock, mode='single'\|'zip', ...)` | 对包含信号路径的算术表达式直接求值。 |
| `reader.get_matched_signals(pattern)` | 将模式解析为信号路径列表，不加载数据。 |
| `reader.top_scope_list()` | 返回信号层级的根 `Scope` 节点。 |

**信号路径支持的模式语法**：

| 语法 | 示例 | 作用 |
|------|------|------|
| `{a,b,c}` | `sig_{read,write}` | 枚举命名变体 |
| `{N..M}` | `fifo_{0..3}.ptr` | 整数范围 |
| `{N..M..step}` | `lane_{0..6..2}` | 带步长的范围 |
| `@<regex>` | `@([a-z]+)_valid` | 正则匹配，捕获组作为字典键 |
| `$ModName` | `tb.$fifo_unit.ptr` | 按模块名匹配直接子层级（仅 FSDB） |
| `$$ModName` | `tb.$$fifo_unit.ptr` | 按模块名匹配任意深度后代（仅 FSDB） |

### Waveform

`Waveform` 内部封装了三个平行的 Numpy 数组（`.value`、`.clock`、`.time`）。所有操作均返回新的 `Waveform` 实例。

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

| 方法 | 说明 |
|------|------|
| `.wait(cond, *, require=None, channel=None, tick=True)` | 阻塞等待 `cond` 为真。`require` 在等待期间每周期检查（违反 → `REQUIRE_VIOLATED`）。`channel` 将该 wait 绑定到一个共享 FIFO 消费组（详见下方 [Channel 通道](#channel-通道)）。`tick=False` 表示在当前周期命中后不消耗周期，下一步在同一 cycle 上继续判定（零周期 wait）。 |
| `.delay(n, *, require=None)` | 前进 n 个周期。`delay(0)` 为空操作。`require` 在每个延迟周期都必须为真。 |
| `.capture(name, signal, *, mode='last')` | 在当前周期记录信号值。`mode='last'`（默认）覆盖写入；`'first'` 仅记录第一次写入；`'list'` 追加到列表。 |
| `.require(cond)` | 断言条件；为假时实例标记为 `REQUIRE_VIOLATED`。 |
| `.loop(body, *, until=None, when=None)` | `until`：do-while（先执行 body，条件为真时退出）；`when`：while（条件为假时直接退出）。 |
| `.repeat(body, n)` | 重复执行 body 恰好 n 次。n 可为可调用对象。 |
| `.branch(cond, true_body, false_body)` | 条件分支。 |
| `.timeout(max_cycles)` | 对未完成的实例标记 `TIMEOUT`。 |
| `.match(start_cycle=None, end_cycle=None)` | 运行匹配引擎，返回 `MatchResult`。 |

**Channel 通道**

`Channel` 是表示共享 FIFO 消费组的身份对象：每周期至多一个在飞实例可以消费它。每个 `wait()` 步骤都自带一个隐式 channel，因此同一模板派生的多个实例会自动按一周期一个的节奏在该步骤上排队。当你需要打破这个默认的串行化时，就显式传一个 `Channel`（或 `callable(index, captures) -> Channel`）—— 典型场景是事件来自物理上并行的多条总线（多 Bank Cache、多 lane Retire 等），希望多个实例能**同周期**各自消费自己的 channel。

```python
from collections import defaultdict
from wavekit import Channel, Pattern

# 多 Bank Cache：每个 bank 有独立的响应端口，多个 bank 可以在同一周期返回数据。
# 不分区的话，默认的串行化规则会让其中一个实例多等一拍；按 bank 分 Channel
# 可以让每个在飞读请求各自消费对应 bank 的响应。
banks = defaultdict(Channel)

result = (
    Pattern()
    .wait(req_valid)
    .capture('bank', req_addr & 1)
    .wait(
        lambda i, cap: bank_valid[cap['bank']].value[i],
        channel=lambda i, cap: banks[cap['bank']],
    )
    .capture('rdata',
        lambda i, cap: bank_data[cap['bank']].value[i])
    .match()
)
```

**`MatchResult`**

| 字段 | 说明 |
|------|------|
| `.start` / `.end` | 每个匹配实例的起始和结束周期（均包含）。 |
| `.duration` | 持续时间，即 `end - start + 1` 周期。 |
| `.status` | `MatchStatus.OK`、`TIMEOUT` 或 `REQUIRE_VIOLATED`。 |
| `.captures` | 捕获的字典，`dict[str, Waveform]`。 |
| `.filter_valid()` | 仅返回状态为 `OK` 的匹配实例。 |

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

欢迎提交 Issue 和 PR！在提交代码前，请确保测试通过且代码检查无报错：

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
```

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](./LICENSE) 文件。
