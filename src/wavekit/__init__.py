#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""Python Package Template"""

from __future__ import annotations

__version__ = '0.1.2'

from .vcd_reader import VcdReader as VcdReader
from .waveform import Waveform as Waveform

try:
    from .fsdb_reader import FsdbReader as FsdbReader

    __all__ = ['Waveform', 'VcdReader', 'FsdbReader']
except ImportError:
    __all__ = ['Waveform', 'VcdReader']
