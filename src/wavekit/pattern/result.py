from __future__ import annotations

from enum import IntEnum

import numpy as np

from ..waveform import Waveform


class MatchStatus(IntEnum):
    """Status codes for pattern match instances."""

    OK = 0
    TIMEOUT = 1
    REQUIRE_VIOLATED = 2


class MatchResult:
    """Struct-of-arrays result of a pattern match.

    All fields are :class:`~wavekit.Waveform` objects whose ``clock`` axis
    is the ``start_cycle`` of each match (simulation cycle), so they live
    in the same time coordinate system as ordinary signal waveforms.

    Attributes
    ----------
    start : Waveform
        Start cycle of each match (inclusive).
    end : Waveform
        End cycle of each match (inclusive, i.e. the last cycle where the
        pattern was active).
    duration : Waveform
        ``end - start + 1`` for each match (number of cycles occupied).
        To use with ``cycle_slice``, pass ``cycle_slice(start, end + 1)``.
    status : Waveform
        :class:`MatchStatus` value (uint8) for each match.
    captures : dict[str, Waveform]
        Named capture values.  Scalar captures are plain Waveforms;
        list captures (``name[]``) are Waveforms with ``object`` dtype
        where each element is a Python list.
    """

    def __init__(
        self,
        start: Waveform,
        end: Waveform,
        duration: Waveform,
        status: Waveform,
        captures: dict[str, Waveform],
    ):
        self.start = start
        self.end = end
        self.duration = duration
        self.status = status
        self.captures = captures

    @property
    def valid(self) -> Waveform:
        """Boolean mask: ``status == OK``."""
        return self.status == MatchStatus.OK

    def filter_valid(self) -> MatchResult:
        """Return a new MatchResult keeping only ``status == OK`` matches."""
        mask = self.valid
        return MatchResult(
            start=self.start.mask(mask),
            end=self.end.mask(mask),
            duration=self.duration.mask(mask),
            status=self.status.mask(mask),
            captures={name: val.mask(mask) for name, val in self.captures.items()},
        )

    def __len__(self) -> int:
        return len(self.start.value)

    def __repr__(self) -> str:
        n = len(self)
        ok = int(np.sum(self.status.value == MatchStatus.OK))
        return f'MatchResult({n} matches, {ok} OK)'
