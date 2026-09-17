from __future__ import annotations

import contextvars
import dataclasses
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, ClassVar

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
    _tokens: list[contextvars.Token] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    _current: ClassVar[contextvars.ContextVar[ClockDomain | None]] = contextvars.ContextVar(
        'wavekit_clock_domain', default=None
    )

    @classmethod
    def current(cls) -> ClockDomain:
        """Return the ambient ``ClockDomain``.

        Raises ``RuntimeError`` if no domain is active.
        """
        domain = cls._current.get()
        if domain is None:
            raise RuntimeError(
                'requires an active clock domain.\n'
                '  Enter one with:  with reader.clock_domain(clock=...):'
            )
        return domain

    def sampling_kwargs(self) -> dict[str, Any]:
        """Return this domain's sampling fields as ``load_waveform``-style kwargs."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in ('clock', '_tokens')
        }

    def __enter__(self) -> ClockDomain:
        self._tokens.append(self._current.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        self._current.reset(self._tokens.pop())
