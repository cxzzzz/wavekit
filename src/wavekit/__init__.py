#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""Python Package Template"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version('wavekit')
except metadata.PackageNotFoundError:
    __version__ = 'unknown'

from .pattern import Channel as Channel
from .pattern import MatchResult as MatchResult
from .pattern import MatchStatus as MatchStatus
from .pattern import Pattern as Pattern
from .pattern import PatternError as PatternError
from .readers.fst.reader import FstReader as FstReader
from .readers.hierarchy import Node as Node
from .readers.hierarchy import Range as Range
from .readers.hierarchy import Scope as Scope
from .readers.hierarchy import Signal as Signal
from .readers.hierarchy import SignalCompositeType as SignalCompositeType
from .readers.matcher import BraceCapture as BraceCapture
from .readers.matcher import Capture as Capture
from .readers.matcher import CaptureKey as CaptureKey
from .readers.matcher import ExactCapture as ExactCapture
from .readers.matcher import RegexCapture as RegexCapture
from .readers.matcher import WildcardCapture as WildcardCapture
from .readers.vcd.reader import VcdReader as VcdReader
from .waveform import Waveform as Waveform

__all__ = [
    'Waveform',
    'VcdReader',
    'FsdbReader',
    'FstReader',
    'Node',
    'Scope',
    'Range',
    'Signal',
    'SignalCompositeType',
    'Capture',
    'CaptureKey',
    'ExactCapture',
    'BraceCapture',
    'RegexCapture',
    'WildcardCapture',
    'Pattern',
    'MatchResult',
    'MatchStatus',
    'PatternError',
    'Channel',
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
        """Check whether the Verdi FSDB runtime is available right now."""
        return _fsdb_runtime_available()
