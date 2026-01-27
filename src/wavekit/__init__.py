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

from .readers.vcd.reader import VcdReader as VcdReader
from .scope import Scope as Scope
from .signal import Signal as Signal
from .waveform import Waveform as Waveform

__all__ = ['Waveform', 'VcdReader', 'FsdbReader', 'Scope', 'Signal']

try:
    from .readers.fsdb.reader import FsdbReader as FsdbReader

    __all__.append('FsdbReader')
except ImportError:
    pass
