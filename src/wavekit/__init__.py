#   -------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""Python Package Template"""
from __future__ import annotations

__version__ = "0.0.1"

from .waveform import Waveform
from .vcd_reader import VcdReader
try:
    from .fsdb_reader import FsdbReader
except ImportError:
    pass