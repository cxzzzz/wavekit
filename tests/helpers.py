"""Shared helpers for tests."""

import numpy as np

from wavekit import Waveform


def wf(values, width=1, signed=False):
    """Build a test Waveform from a list of values."""
    value = np.asarray(values, dtype=np.int64)
    cycle = np.arange(len(value), dtype=np.int64)
    time = cycle * 10
    return Waveform(value, cycle=cycle, time=time, width=width, signed=signed)


def bool_wf(values):
    """Build a 1-bit unsigned Waveform (boolean-like)."""
    return wf(values, width=1, signed=False)
