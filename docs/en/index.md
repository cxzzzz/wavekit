# wavekit documentation

**High-level digital waveform analysis in Python.**

Waveform files record timestamps and value changes, while hardware engineers
focus on clock cycles, signal relationships, and multi-cycle behavior.
Wavekit provides flexible signal queries and processing at the cycle and
transaction levels, making complex hardware behavior easier to analyze.

## What wavekit provides

Wavekit loads VCD, FST, and FSDB files through the same API, representing
signals as clock-sampled `Waveform` objects for further waveform operations
and analysis.

Its main features include:

1. **Flexible signal queries:** find and batch-load related signals from
   hierarchical waveform data using multiple path-matching options.
2. **Cycle-level analysis:** use a range of waveform operations on clock-sampled
   data to analyze cycle-based behavior such as interface backpressure and FIFO
   occupancy.
3. **Transaction-level analysis:** use temporal pattern matching to describe
   signal relationships across multiple clock cycles for protocol analysis,
   transaction extraction, and latency measurement.

## Start here

- [Install wavekit](getting-started/installation.md).
- Follow [the first waveform tutorial](getting-started/first-waveform.md).
- Choose a [reader for your waveform format](guides/reader.md).
- Learn [signal queries](guides/signal-query.md), [waveform analysis](guides/waveform-analysis.md),
  and [pattern matching](guides/pattern-matching.md).
- Browse the [API reference](reference/api.md) or [complete examples](examples.md).

The optional [wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) project exposes
wavekit analysis through MCP tools for AI-assisted workflows.
