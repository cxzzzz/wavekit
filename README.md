# wavekit

[![CI](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml/badge.svg)](https://github.com/cxzzzz/wavekit/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Downloads](https://pepy.tech/badge/wavekit)](https://pepy.tech/project/wavekit)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

English | [中文](README_ZH.md)

**High-level digital waveform analysis in Python.**

Waveform files expose timestamps and value changes, while hardware engineers
reason about clock cycles, signal relationships, and multi-cycle behavior.
wavekit bridges that abstraction gap with flexible signal queries and analysis
at both the cycle and transaction levels.

> **AI integration:** [wavekit-mcp](https://github.com/cxzzzz/wavekit-mcp) exposes
> wavekit analysis through MCP tools for AI-assisted workflows.

## Features

- **Flexible signal queries:** find and batch-load related signals from
  hierarchical waveform data using multiple path-matching options.
- **Cycle-level analysis:** use a range of waveform operations on clock-sampled
  data to analyze cycle-based behavior such as interface backpressure and FIFO
  occupancy.
- **Transaction-level analysis:** use temporal pattern matching to describe
  signal relationships across multiple clock cycles for protocol analysis,
  transaction extraction, and latency measurement.
- **Multi-format waveform support:** load and analyze VCD, FST, and FSDB files
  using the same API.

## Installation

```console
python -m pip install wavekit
```

wavekit supports Python 3.9 and newer. FSDB support requires the Verdi NPI
runtime (`libNPI.so`); see the [reader setup guide](docs/guides/reader.md) for
setup details.

## Quick start

Calculate the occupancy of a FIFO from a VCD waveform:

```python
import numpy as np

from wavekit import VcdReader

with VcdReader('simulation.vcd') as r:
    clock = 'tb.clk'
    depth = 16

    w_ptr = r.load_waveform(
        'tb.u_fifo.w_ptr',
        clock=clock,
    )
    r_ptr = r.load_waveform(
        'tb.u_fifo.r_ptr',
        clock=clock,
    )

    occupancy = (w_ptr + depth - r_ptr) % depth

    print('Average occupancy:', np.mean(occupancy.value))
```

See the [first waveform tutorial](docs/getting-started/first-waveform.md) for a
complete runnable FIFO example.

## Documentation

See the [documentation](docs/index.md) for the full reference and guides:

- [Installation](docs/getting-started/installation.md)
- [First waveform](docs/getting-started/first-waveform.md)
- [Reader and FSDB setup](docs/guides/reader.md)
- [Signal queries](docs/guides/signal-query.md)
- [Waveform analysis](docs/guides/waveform-analysis.md)
- [Pattern matching](docs/guides/pattern-matching.md)
- [API reference](docs/reference/api.md)
- [Complete examples](docs/examples.md)

## Development

Install the development dependencies and run the checks with Poetry:

```console
poetry install
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
```

Build the documentation locally with Python 3.10 or newer:

```console
poetry install --with docs
poetry run zensical build --clean --strict
```

See [Contributing](docs/contributing.md) for contribution guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
