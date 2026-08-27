# Installation

Install wavekit from PyPI with Python 3.9 or newer:

```console
python -m pip install wavekit
```

For development, clone the repository and install its Poetry environment:

```console
git clone https://github.com/cxzzzz/wavekit.git
cd wavekit
poetry install
```

## FSDB support

FSDB support requires the Verdi NPI runtime (`libNPI.so`).

In a typical Verdi setup, `VERDI_HOME` is already set, so no additional
configuration is usually required. If it is not, make `libNPI.so` discoverable
using one of these options:

- `WAVEKIT_NPI_LIB`: direct path to `libNPI.so`;
- `VERDI_HOME`: Verdi installation directory;
- `LD_LIBRARY_PATH`: a library search path containing `libNPI.so`.

Check the runtime before opening an FSDB file:

```python
from wavekit import has_fsdb_support

print(f"FSDB support: {has_fsdb_support()}")
```

## Verify the installation

```console
python -c "import wavekit; print(wavekit.__version__)"
```

Then continue with the [FIFO occupancy tutorial](first-waveform.md).
