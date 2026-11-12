"""Clock-synchronised analysis for VCD, FST, and FSDB waveforms."""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import metadata
from typing import Any

from .readers.base import Reader

try:
    __version__ = metadata.version('wavekit')
except metadata.PackageNotFoundError:
    __version__ = 'unknown'

from .readers.clock_domain import ClockDomain as ClockDomain
from .readers.fst.reader import FstReader as FstReader
from .readers.hierarchy import Node as Node
from .readers.hierarchy import Scope as Scope
from .readers.hierarchy import Signal as Signal
from .readers.hierarchy import SignalCompositeType as SignalCompositeType
from .readers.matcher import BraceCapture as BraceCapture
from .readers.matcher import Capture as Capture
from .readers.matcher import ExactCapture as ExactCapture
from .readers.matcher import RegexCapture as RegexCapture
from .readers.matcher import WildcardCapture as WildcardCapture
from .readers.range import Range as Range
from .readers.vcd.reader import VcdReader as VcdReader
from .waveform import Waveform as Waveform

__all__ = [
    'Waveform',
    'VcdReader',
    'FsdbReader',
    'FstReader',
    'ClockDomain',
    'Node',
    'Scope',
    'Range',
    'Signal',
    'SignalCompositeType',
    'Capture',
    'ExactCapture',
    'BraceCapture',
    'RegexCapture',
    'WildcardCapture',
    'has_fsdb_support',
]

try:
    from .readers.fsdb.npi_fsdb_reader import fsdb_runtime_available as _fsdb_runtime_available
    from .readers.fsdb.reader import FsdbReader as FsdbReader
except Exception as _fsdb_import_error:
    _fsdb_available = False

    def has_fsdb_support() -> bool:
        """Check whether the Verdi FSDB runtime is available."""
        return False

    class _FsdbReaderStub:
        """Placeholder that raises an error when the Verdi FSDB runtime is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                'FsdbReader requires the Verdi FSDB runtime (libNPI.so).\n\n'
                'Set WAVEKIT_NPI_LIB to the library path, set VERDI_HOME to the Verdi '
                'installation directory, or ensure libNPI.so is in LD_LIBRARY_PATH.\n\n'
                f'Import error: {_fsdb_import_error}'  # noqa: F821
            )

    FsdbReader = _FsdbReaderStub  # type: ignore[assignment]
else:
    _fsdb_available = True

    def has_fsdb_support() -> bool:
        """Check whether the Verdi FSDB runtime library is installed."""
        return _fsdb_runtime_available()


def open(file: str, **kwargs: Any) -> Reader:
    """Open a waveform file with the reader for its file extension.

    Parameters
    ----------
    file:
        Path to a ``.vcd``, ``.fst``, or ``.fsdb`` file.
    **kwargs:
        Passed to the reader, e.g. ``quiet`` for ``FsdbReader``.

    Returns
    -------
    Reader
        A ``VcdReader``, ``FstReader``, or ``FsdbReader``.

    Raises
    ------
    ValueError:
        If the extension is not supported.
    FileNotFoundError:
        If *file* does not exist.
    """
    readers: dict[str, Callable[..., Reader]] = {
        '.vcd': VcdReader,
        '.fst': FstReader,
        '.fsdb': FsdbReader,
    }
    reader = readers.get(os.path.splitext(file)[1].lower())
    if reader is None:
        raise ValueError(
            f"unsupported waveform file '{file}': expected one of {', '.join(readers)}"
        )
    return reader(file, **kwargs)
