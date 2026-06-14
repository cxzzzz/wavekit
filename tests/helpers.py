"""Shared helpers for tests."""

import numpy as np

from wavekit import Signal, Waveform


def wf(values, width=1, signed=False):
    """Build a test Waveform from a list of values."""
    value = np.asarray(values, dtype=np.int64)
    clock = np.arange(len(value), dtype=np.int64)
    time = clock * 10
    return Waveform(value, clock, time, signal=Signal('', '', width, None, signed))


def bool_wf(values):
    """Build a 1-bit unsigned Waveform (boolean-like)."""
    return wf(values, width=1, signed=False)
