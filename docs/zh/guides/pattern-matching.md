# 模式匹配

模式匹配用于描述多个信号跨多个时钟周期的时序关系，让复杂的多周期行为更容易分析。它适用于协议等高层次时序分析，包括事务提取、延迟测量和超时检测。

有两种方式可以描述一个事务：

- **声明式模式：**使用 `Pattern` 链式 API 描述相对固定的流程。如果事务结构事先已知，并且适合用一组清晰的步骤表达，可以使用这种方式。
- **编程式模式：**使用普通的同步 Python 函数描述动态的事务流程，适合依赖信号值的分支、动态长度或需要构造 Python 结果的场景。

两种方式使用相同的事件模型和结果概念。同一模式中使用的所有波形，必须具有同源时钟、相同的采样边沿，以及相同的时间窗口或周期窗口。

## 声明式模式

声明式模式由 `Pattern` 连接一组步骤组成。步骤可以等待或消费事件、推进时间、记录值、检查条件，以及表达循环和分支：

- `wait()` 等待条件满足，但不占用事件；
- `consume()` 等待条件满足并在逻辑 channel 上占用事件；
- `delay()` 前进固定的周期数；
- `capture()` 记录值，但不推进时间；
- `require()` 检查条件，条件不满足时报告失败的匹配；
- `loop()`、`repeat()` 和 `branch()` 表达循环或条件流程。

```python
from wavekit.pattern import Pattern, match

request = req_valid & req_ready
response = rsp_valid & rsp_ready

pattern = (
    Pattern()
    .wait(request)
    .consume(response)
    .capture('rsp_data', rsp_data)
)
result = match(pattern)
```

第一个阻塞步骤决定模式从哪些周期开始匹配。后续阻塞步骤描述匹配内部的时序。某个阻塞步骤满足后，后续步骤会在同一个周期执行；如果需要推进到下一个周期，应显式使用 `delay(1)`。

对于重复或有条件的流程，可以组合使用 `loop()`、`repeat()` 和 `branch()`：

```python
beat = Pattern().consume(w_valid & w_ready).capture(
    'data', w_data, mode='list'
)
pattern = Pattern().wait(aw_valid & aw_ready).loop(beat, until=w_last)
result = match(pattern)
```

## 编程式模式

编程式模式使用和声明式模式相同的模式操作，但这些操作是在函数体内通过 `ctx` 调用的。上下文对象 `ctx` 提供 `ctx.value()`、`ctx.wait()`、`ctx.consume()`、`ctx.try_consume()`、`ctx.delay()`、`ctx.capture()` 和 `ctx.require()`。这些操作的作用与声明式对应项相同；其中 `ctx.value()` 会读取当前采样点的标量值。

编程式模式适合后续流程取决于波形值的场景。运行时会从扫描范围内的每个可能起始周期调用一次函数体，因此可以在函数体内使用普通的 Python 条件和循环，根据匹配过程中读取到的值决定后续操作。

因此，固定的 `Waveform` 表达式应放在函数体外预先计算，避免重复处理完整波形；函数体内需要根据当前周期做判断时，使用 `ctx.value()`。

```python
from wavekit.pattern import collect

cmd_fire = cmd_valid & cmd_ready
rsp_fire = rsp_valid & rsp_ready


def read_command(ctx):
    if not ctx.value(cmd_fire):
        return None

    opcode = int(ctx.value(cmd_op))
    length = int(ctx.value(cmd_len))
    data = []

    if opcode == 0:
        ctx.consume(rsp_fire, channel='response')
        for _ in range(length):
            ctx.consume(r_valid & r_ready, channel='read-data')
            data.append(int(ctx.value(r_data)))
        return {'opcode': 'read', 'data': data}

    if opcode == 1:
        for _ in range(length):
            ctx.consume(w_valid & w_ready, channel='write-data')
            data.append(int(ctx.value(w_data)))
        ctx.consume(rsp_fire, channel='response')
        return {'opcode': 'write', 'data': data}

    ctx.require(False, message=f'unknown opcode {opcode}')
    return None

commands = collect(read_command)
```

需要非阻塞仲裁或轮询时，使用 `ctx.try_consume()`。对于线性的 burst，`ctx.consume()` 通常更直观。

## 事件消费与 channel

`wait()` 只等待条件满足，不会占用匹配到的事件。`consume()` 在条件满足时占用当前事件，并通过逻辑 channel 处理多个匹配之间的竞争。

多个事务等待同一个响应流时，可以共享一个 channel：

```python
pattern = (
    Pattern()
    .wait(request_fire)
    .consume(response_fire, channel='response')
)
```

这样可以确保同一个周期的响应最多只被一个匹配占用。不同 channel 上的事件可以独立消费。

声明式 `consume()` 省略 `channel` 时，会为该步骤分配一个独立的 channel。如果多个 consume 步骤需要共享同一所有权范围，应显式指定 channel。编程式 `ctx.consume()` 和 `ctx.try_consume()` 始终要求显式传入 channel。

## 结果与失败处理

模式执行有两种结果：`match()` 返回匹配记录，`collect()` 返回 Python 值。

### 匹配记录

`match()` 接受声明式 `Pattern` 或编程式函数体，并返回 `MatchRecords`。

对于编程式函数体：

- 返回 `ctx.OK`，记录一次成功的匹配；
- 返回 `None`，跳过当前起始周期，不产生记录。

`MatchRecords` 包含 `start`、`end`、`duration`、`status` 和命名 capture 值。

`end` 对应的周期也包含在匹配范围内。需要提取某次匹配对应的波形窗口时，使用：

```python
cycle_slice(start.clock, end.clock + 1)
```

可使用以下方法筛选记录：

- `filter_ok()` 保留成功的记录；
- `filter_failed()` 保留失败的记录；
- `filter_status(MatchStatus.Timeout)` 选择指定的状态。

### 收集值

`collect()` 接收编程式函数体，返回其产生的所有非 `None` 值组成的列表。

### 失败和超时

向 `match()` 或 `collect()` 传入 `timeout=<cycles>`，限制每个候选匹配的持续时间。

- `match()` 会在结果状态中记录超时和 `require()` 检查失败；
- `collect()` 在发生超时或 `require()` 检查失败时抛出 `PatternError`。
