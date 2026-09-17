# API reference

## Waveform

`Waveform` is the core object for waveform analysis, holding sampled values
together with their cycle and time axes.

::: wavekit.Waveform

## Readers

Format-specific readers share the common loading, query, and expression APIs.

::: wavekit.VcdReader

::: wavekit.FstReader

::: wavekit.FsdbReader

::: wavekit.has_fsdb_support

## Pattern matching

The pattern-matching API describes signal relationships across multiple
clock cycles and performs transaction-level analysis.

::: wavekit.pattern.Pattern

::: wavekit.pattern.match

::: wavekit.pattern.collect

::: wavekit.pattern.MatchRecords

::: wavekit.pattern.MatchRecord

::: wavekit.pattern.MatchPoint

::: wavekit.pattern.MatchStatus

::: wavekit.pattern.Channel

::: wavekit.pattern.PatternError

## Signal hierarchy and queries

These objects represent the hierarchy, signals, ranges, and captures in
query results within a waveform file.

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

## Clock domains

`ClockDomain` bundles a clock signal with a sampling recipe (edge, time or
cycle window) that `Signal.w`/`Signal.m` and related methods read from the
ambient context.

::: wavekit.ClockDomain
