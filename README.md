# wavekit

Wavekit is a fundamental Python package for digital circuit waveform analysis that provides convenient batch processing of signals and a wide range of mathematical operations for waveforms.

## Features

- Read various waveform files (VCD, FSDB, etc.), especially optimized for large FSDB files.
- Extract waveforms in batches using absolute paths, brace expansions, and regular expressions.
- Perform arithmetic, bitwise, functional, and other operations on waveforms.

## Installation

Install Wavekit with Python 3.9 or later.

### Using PIP

You can install the library using pip:

```bash
pip3 install wavekit
```

### From Source

To install the library from the source, follow these steps:

Clone the repository:

```bash
git clone https://github.com/cxzzzz/wavekit.git
```

Navigate to the project directory:

```bash
cd wavekit
```

Install the library using pip:

```bash
pip install .
```

## Quick Start

Here is a simple [example](./example/fifo_average_occupancy_level/) demonstrating how to use the library to read a VCD file and calculate the average occupancy level of the FIFO :

```python
import numpy as np
from wavekit import VcdReader


with VcdReader("fifo_tb.vcd") as f: # open the VCD file
    clock = "fifo_tb.s_fifo.clk"
    depth = 8

    #calculate the average FIFO occupancy level.
    w_ptr = f.load_wave("fifo_tb.s_fifo.w_ptr[2:0]", clock=clock)   # load fifo write pointer signal
    r_ptr = f.load_wave("fifo_tb.s_fifo.r_ptr[2:0]", clock=clock)   # load fifo read pointer signal
    fifo_water_level = (w_ptr + depth - r_ptr) % depth              # calculate the occupancy level
    average_fifo_water_level = np.mean(fifo_water_level.value)      # calculate the average occupancy level using numpy
    print( f"average fifo occupancy level: {average_fifo_water_level}" )

```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
