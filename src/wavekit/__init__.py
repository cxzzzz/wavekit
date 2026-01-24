#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""Python Package Template"""

from __future__ import annotations

__version__ = '0.1.2'

from .readers.vcd.reader import VcdReader as VcdReader
from .scope import Scope as Scope
from .signal import Signal as Signal
from .waveform import Waveform as Waveform

try:
    from .readers.fsdb.reader import FsdbReader as FsdbReader

    __all__ = ['Waveform', 'VcdReader', 'FsdbReader', 'Scope', 'Signal']
except ImportError:
    __all__ = ['Waveform', 'VcdReader', 'Scope', 'Signal']
