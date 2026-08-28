# API 参考

本页列出 wavekit 的主要公共 API。API 的类名、函数名和签名保持代码中的英文形式，便于与实际调用对应；具体 API 说明由源代码自动生成。

## Waveform

`Waveform` 是波形分析的核心对象，包含采样值以及对应的周期和时间轴。

::: wavekit.Waveform

## Reader

不同格式的 Reader 使用同一套加载、查询和表达式求值 API。

::: wavekit.VcdReader

::: wavekit.FstReader

::: wavekit.FsdbReader

::: wavekit.has_fsdb_support

## 模式匹配

模式匹配 API 用于描述跨多个时钟周期的信号关系，并执行事务级分析。

::: wavekit.pattern.Pattern

::: wavekit.pattern.match

::: wavekit.pattern.collect

::: wavekit.pattern.MatchRecords

::: wavekit.pattern.MatchRecord

::: wavekit.pattern.MatchPoint

::: wavekit.pattern.MatchStatus

::: wavekit.pattern.Channel

::: wavekit.pattern.PatternError

## 信号层次结构和查询

这些对象表示波形文件中的层次结构、信号、范围以及查询结果中的 capture。

::: wavekit.Node

::: wavekit.Scope

::: wavekit.Signal

::: wavekit.Range

::: wavekit.SignalCompositeType

::: wavekit.Capture


::: wavekit.ExactCapture

::: wavekit.BraceCapture

::: wavekit.RegexCapture

::: wavekit.WildcardCapture
