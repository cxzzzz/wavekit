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

from .pattern import MatchResult as MatchResult
from .pattern import MatchStatus as MatchStatus
from .pattern import Pattern as Pattern
from .pattern import PatternError as PatternError
from .readers.vcd.reader import VcdReader as VcdReader
from .scope import Scope as Scope
from .signal import Signal as Signal
from .signal import SignalCompositeType as SignalCompositeType
from .waveform import Waveform as Waveform

__all__ = [
    'Waveform',
    'VcdReader',
    'FsdbReader',
    'Scope',
    'Signal',
    'SignalCompositeType',
    'Pattern',
    'MatchResult',
    'MatchStatus',
    'PatternError',
    'has_fsdb_support',
]

# Detect if FSDB support is available (compiled with VERDI_HOME)
import importlib.util

_fsdb_available = (
    importlib.util.find_spec('.readers.fsdb.npi_fsdb_reader', package=__name__) is not None
)


def has_fsdb_support() -> bool:
    """Check if FsdbReader is available in the current installation."""
    return _fsdb_available


class _FsdbReaderStub:
    """Placeholder that raises an error when FsdbReader is used without Verdi support."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'FsdbReader requires Verdi to be installed and configured.\n\n'
            '1. Set VERDI_HOME environment variable to your Verdi installation path.\n'
            '2. Reinstall wavekit:\n'
            '    pip install --force-reinstall --no-cache-dir --no-deps wavekit'
        )


if _fsdb_available:
    from .readers.fsdb.reader import FsdbReader as FsdbReader
else:
    FsdbReader = _FsdbReaderStub  # type: ignore[assignment]
