from __future__ import annotations

import contextvars
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import ClassVar

from .hierarchy import Signal


@dataclass(eq=False)
class ClockDomain(ContextDecorator):
    """A clock signal paired with its sampling edge and time/cycle window."""

    clock: Signal
    sample_on_posedge: bool = False
    start_time: int | None = None
    end_time: int | None = None
    start_cycle: int | None = None
    end_cycle: int | None = None
    _tokens: list[contextvars.Token] = field(default_factory=list, repr=False, compare=False)

    _current: ClassVar[contextvars.ContextVar[ClockDomain | None]] = contextvars.ContextVar(
        'wavekit_clock_domain', default=None
    )

    @classmethod
    def current(cls) -> ClockDomain | None:
        """Return the ambient ``ClockDomain``, or ``None`` outside a ``with`` block."""
        return cls._current.get()

    def __enter__(self) -> ClockDomain:
        self._tokens.append(self._current.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        self._current.reset(self._tokens.pop())
