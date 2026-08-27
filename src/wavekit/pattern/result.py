from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from ..waveform import Waveform


class MatchStatusValue:
    """Base class for internal match status value typing."""


class MatchStatus:
    """Terminal status namespace for pattern match records."""

    @dataclass(frozen=True)
    class OK(MatchStatusValue):
        """Successful completion of a pattern candidate."""

    @dataclass(frozen=True)
    class Timeout(MatchStatusValue):
        """A pattern candidate did not complete within its allowed duration."""

        message: str | None = None

    @dataclass(frozen=True)
    class RequireViolated(MatchStatusValue):
        """A non-blocking ``require`` check failed."""

        message: str | None = None


@dataclass(frozen=True)
class MatchPoint:
    """One match boundary point.

    ``index`` is the waveform-array sample index; ``cycle`` and ``time`` are the
    corresponding absolute cycle number and simulation timestamp.
    """

    index: int
    cycle: int
    time: int


@dataclass(frozen=True)
class MatchRecord:
    """One pattern match record.

    Attributes
    ----------
    start, end:
        Inclusive match boundary points. ``MatchPoint.index`` is the waveform-array
        sample index, ``MatchPoint.cycle`` is the absolute clock cycle, and
        ``MatchPoint.time`` is the simulation timestamp.
    status:
        Terminal status object: ``MatchStatus.OK()``, ``MatchStatus.Timeout(...)``,
        or ``MatchStatus.RequireViolated(...)``.
    captures:
        Per-record captured Python values.
    """

    start: MatchPoint
    end: MatchPoint
    status: MatchStatusValue
    captures: dict[str, Any]

    @property
    def duration(self) -> int:
        """Return the inclusive duration in sampled cycles.

        Returns
        -------
        int
            ``end.index - start.index + 1``.
        """
        return self.end.index - self.start.index + 1


class MatchRecords(Sequence[MatchRecord]):
    """Columnar batch of pattern match records.

    ``start`` and ``end`` are point waveforms: ``.value`` stores waveform-array
    sample indices, ``.clock`` stores absolute cycle numbers, and ``.time`` stores
    simulation timestamps. ``duration.value`` is ``end.value - start.value + 1``.
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
    def ok(self) -> Waveform:
        """Return a boolean result-row mask for successful matches.

        Returns
        -------
        Waveform
            One-bit waveform aligned to result rows. ``value`` is true where
            ``status`` is ``MatchStatus.OK``.
        """
        value = np.array(
            [isinstance(status, MatchStatus.OK) for status in self.status.value], dtype=bool
        )
        return Waveform(value, self.start.clock.copy(), self.start.time.copy(), width=1)

    @property
    def failed(self) -> Waveform:
        """Return a boolean result-row mask for failed matches.

        Returns
        -------
        Waveform
            One-bit waveform aligned to result rows. ``value`` is true where
            ``status`` is not ``MatchStatus.OK``.
        """
        value = np.array(
            [not isinstance(status, MatchStatus.OK) for status in self.status.value], dtype=bool
        )
        return Waveform(value, self.start.clock.copy(), self.start.time.copy(), width=1)

    def filter_ok(self) -> MatchRecords:
        """Return records whose status is ``MatchStatus.OK``.

        Returns
        -------
        MatchRecords
            A row-masked batch with all fields and captures filtered together.
        """
        return self.filter_status(MatchStatus.OK)

    def filter_status(self, status: type[MatchStatusValue]) -> MatchRecords:
        """Return records matching a status class.

        Parameters
        ----------
        status:
            Status class such as ``MatchStatus.Timeout``.

        Returns
        -------
        MatchRecords
            A row-masked batch with all fields and captures filtered together.
        """
        if not isinstance(status, type):
            raise TypeError('filter_status() requires a status class')
        if not issubclass(status, MatchStatusValue):
            raise TypeError('filter_status() requires a MatchStatus class')
        mask = np.array([isinstance(value, status) for value in self.status.value], dtype=bool)
        return self._mask(mask)

    def filter_failed(self) -> MatchRecords:
        """Return records whose status is not ``MatchStatus.OK``.

        Returns
        -------
        MatchRecords
            A row-masked batch with all fields and captures filtered together.
        """
        return self._mask(self.failed.value.astype(np.bool_))

    def _mask(self, mask: np.ndarray) -> MatchRecords:
        return MatchRecords(
            start=self.start.mask(mask),
            end=self.end.mask(mask),
            duration=self.duration.mask(mask),
            status=self.status.mask(mask),
            captures={name: val.mask(mask) for name, val in self.captures.items()},
        )

    @overload
    def __getitem__(self, index: int) -> MatchRecord: ...

    @overload
    def __getitem__(self, index: slice) -> MatchRecords: ...

    def __getitem__(self, index: int | slice) -> MatchRecord | MatchRecords:
        """Return one row or a sliced batch.

        Parameters
        ----------
        index:
            Integer row index or Python slice.

        Returns
        -------
        MatchRecord or MatchRecords
            Integer indexing returns a row object; slicing returns another
            ``MatchRecords`` with all columns sliced together.
        """
        if isinstance(index, slice):
            indices = np.arange(len(self))[index]
            return MatchRecords(
                start=self.start.take(indices),
                end=self.end.take(indices),
                duration=self.duration.take(indices),
                status=self.status.take(indices),
                captures={name: val.take(indices) for name, val in self.captures.items()},
            )
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('MatchRecords index out of range')
        captures = {name: waveform.value[index] for name, waveform in self.captures.items()}
        return MatchRecord(
            start=MatchPoint(
                index=int(self.start.value[index]),
                cycle=int(self.start.clock[index]),
                time=int(self.start.time[index]),
            ),
            end=MatchPoint(
                index=int(self.end.value[index]),
                cycle=int(self.end.clock[index]),
                time=int(self.end.time[index]),
            ),
            status=self.status.value[index],
            captures=captures,
        )

    def __iter__(self) -> Iterator[MatchRecord]:
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.start.value)

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return 'MatchRecords(0 records)'
        counts = Counter(type(status).__name__ for status in self.status.value)
        summary = ', '.join(f'{count} {name}' for name, count in counts.items())
        return f'MatchRecords({n} records: {summary})'
