# wavekit 文档

**使用 Python 进行高层次数字波形分析。**

数字波形文件记录的是时间戳和值变化，而硬件工程师更关注时钟周期、信号关系和多周期行为。wavekit 提供灵活的信号查询，以及从周期级到事务级的分析能力，让用户能够方便地从波形数据中提取更高层次的信息。

## wavekit 主要功能

wavekit 通过同一套 API 加载 VCD、FST 和 FSDB 文件，将信号按时钟采样并表示为 `Waveform` 对象，用于后续的波形操作和分析。

主要功能包括：

1. **批量信号查询：**支持多种路径匹配方式，从层次化波形数据中查找并批量加载相关信号。
2. **周期级分析：**使用多种 Waveform 操作处理时钟采样数据，分析接口反压、FIFO 水位等周期级行为。
3. **事务级分析：**使用时序模式匹配描述跨多个时钟周期的信号关系，分析协议行为、提取事务并测量延迟。

## 从这里开始

- [安装 wavekit](getting-started/installation.md)。
- 从[第一个波形教程](getting-started/first-waveform.md)开始。
- 根据波形格式选择对应的 [Reader](guides/reader.md)。
- 学习[信号查询](guides/signal-query.md)、[波形分析](guides/waveform-analysis.md)和[模式匹配](guides/pattern-matching.md)。
- 查看 [API 参考](reference/api.md) 或[完整示例](examples.md)。

[wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) 是一个可选项目，让 AI 工具能够通过 MCP 调用 wavekit 进行波形分析。
