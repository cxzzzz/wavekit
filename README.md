# wavekit

[![PyPI version](https://img.shields.io/pypi/v/wavekit.svg)](https://pypi.org/project/wavekit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/wavekit.svg)](https://pypi.org/project/wavekit/)
[![License](https://img.shields.io/github/license/cxzzzz/wavekit.svg)](LICENSE)

**Wavekit** is a fundamental Python library for digital waveform analysis. By seamlessly converting VCD and FSDB data into Numpy arrays, it empowers engineers to perform high-performance signal processing, protocol analysis, and automated verification with ease.

## ✨ Features

- **High Performance**: Cython-optimized parsers and Numpy-backed storage deliver exceptional speed and memory efficiency for large waveform files.
- **Flexible Extraction**: Effortlessly batch-extract signals using glob patterns or regular expressions.
- **Rich Analysis Tools**: Comprehensive toolkit for bit-precise manipulation, temporal analysis, and signal transformation, streamlining complex analysis tasks with succinct, Numpy-like APIs.

## 📦 Installation

### Using pip (For Users)

You can install the library using pip:

```bash
pip install wavekit
```

**Note**: To read FSDB files, ensure the `VERDI_HOME` environment variable is set before installation.

### From Source (For Developers)

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Clone the repository:

   ```bash
   git clone https://github.com/cxzzzz/wavekit.git
   cd wavekit
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

## 🚀 Quick Start

### Batch Signal Extraction

Wavekit simplifies extracting multiple signals using brace expansion or regular expressions.

**1. Using Brace Expansion**
Use brace patterns (e.g., `{state,next}` or `{0..7}`) to load related signals in one go.

```python
from wavekit import VcdReader

with VcdReader("jtag.vcd") as f:
    # Load both J_state and J_next signals
    # Returns: { ('state',): Waveform(...), ('next',): Waveform(...) }
    waves = f.load_matched_waveforms(
        "tb.u0.J_{state,next}[3:0]",
        clock="tb.tck"
    )

    for (suffix,), wave in waves.items():
        print(f"Signal J_{suffix}: width={wave.width}")
```

**2. Using Regular Expressions**
Prefix the pattern with `@` to enable regex matching. Capture groups `(...)` become the dictionary keys.

```python
with VcdReader("jtag.vcd") as f:
    # Use '@' prefix for regex mode
    # Match signals like J_state[3:0] and J_next[3:0]
    # Capture the suffix (state/next)
    regex_pattern = r"tb.u0.@J_([a-z]+)"

    # Returns: { ('state',): Waveform(...), ('next',): Waveform(...) }
    waves = f.load_matched_waveforms(regex_pattern, clock="tb.tck")

    for (name_part,), wave in waves.items():
        print(f"Matched: {name_part}")
```

### Basic: FIFO Occupancy Analysis

Here is a simple example demonstrating how to read a VCD file and calculate the average occupancy level of a FIFO:

```python
import numpy as np
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    clock = "fifo_tb.s_fifo.clk"
    depth = 8

    # Load signals with clock synchronization
    w_ptr = f.load_waveform("fifo_tb.s_fifo.w_ptr[2:0]", clock=clock)
    r_ptr = f.load_waveform("fifo_tb.s_fifo.r_ptr[2:0]", clock=clock)

    # Perform vectorized arithmetic operations directly on waveforms
    fifo_water_level = (w_ptr + depth - r_ptr) % depth

    # Calculate average occupancy using Numpy
    average_level = np.mean(fifo_water_level.value)
    print(f"Average FIFO occupancy: {average_level:.2f}")
```

### Advanced: Data Validity Filtering

Use the powerful masking capabilities to filter data based on valid signals:

```python
from wavekit import VcdReader

with VcdReader("bus_tb.vcd") as f:
    clock = "top.clk"

    # Load data and valid signals
    data = f.load_waveform("top.bus_data[31:0]", clock=clock)
    valid = f.load_waveform("top.bus_valid", clock=clock)

    # Filter data where valid is high (1)
    # The mask() method accepts a boolean waveform or numpy array
    valid_data = data.mask(valid == 1)

    # Analyze the valid transactions
    print(f"Total valid transactions: {len(valid_data.value)}")
    print(f"First valid data: {hex(valid_data.value[0])}")
```

### Expression Evaluation

Use `eval` to compute waveform expressions directly from signal path strings, without manually loading each signal.

**Single mode** — all paths match exactly one signal, returns a single `Waveform`:

```python
from wavekit import VcdReader

with VcdReader("fifo_tb.vcd") as f:
    clock = "fifo_tb.clk"

    # Equivalent to loading w_ptr and r_ptr separately and computing the difference
    occupancy = f.eval(
        "fifo_tb.s_fifo.w_ptr[2:0] - fifo_tb.s_fifo.r_ptr[2:0]",
        clock=clock,
    )
    print(f"Occupancy width: {occupancy.width}")
```

Bit-slicing on the expression result is also supported:

```python
    # Load a 4-bit signal and slice out the lower 2 bits
    low_bits = f.eval("tb.dut.data[3:0][1:0]", clock=clock)
    assert low_bits.width == 2
```

**Zip mode** — paths with brace/regex patterns expand to multiple signals; the expression is evaluated once per matched key and a `dict` is returned:

```python
with VcdReader("multi_fifo_tb.vcd") as f:
    clock = "tb.clk"

    # Evaluate occupancy for fifo_0, fifo_1, fifo_2, fifo_3 in one call
    # Returns: { (0,): Waveform, (1,): Waveform, (2,): Waveform, (3,): Waveform }
    occupancies = f.eval(
        "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
        clock=clock,
        mode="zip",
    )

    for (idx,), wave in occupancies.items():
        print(f"fifo_{idx} occupancy width: {wave.width}")
```

Single-match paths in zip mode are **broadcast** — the same waveform is reused across all keys:

```python
    # base_offset matches one signal; it is reused for every fifo index
    adjusted = f.eval(
        "tb.fifo_{0..3}.w_ptr[2:0] - tb.dut.base_offset[2:0]",
        clock=clock,
        mode="zip",
    )
```

## 🛠️ Development

This project adheres to strict code quality standards using modern Python tooling:

- **Linting & Formatting**: [Ruff](https://github.com/astral-sh/ruff)
- **Type Checking**: [Mypy](https://mypy-lang.org/)

Ensure all checks pass before submitting a Pull Request:

```bash
# Run linting and formatting checks
poetry run ruff check .
poetry run ruff format --check .

# Run type checks
poetry run mypy .
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
