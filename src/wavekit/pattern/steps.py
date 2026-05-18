from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union

from ..waveform import Waveform


class Channel:
    """Identity object for shared FIFO consumption across wait steps.

    A ``Channel`` represents a logical event stream from which at most one
    pattern instance may consume per cycle.  Multiple wait steps that bind
    to the same ``Channel`` instance form a single FIFO consumer group
    (oldest in-flight instance wins).

    By default each ``WaitStep`` owns a private ``_auto_channel`` that is
    shared across all clones of that step (so multiple instances of the
    same pattern still serialize one-per-cycle on that step), but is not
    shared across distinct steps.

    Internal state:
        ``(_consumed_epoch, _consumed_at)`` records *which engine run* and
        *which cycle* this channel was last consumed at.  Each ``run()``
        gets a fresh epoch, so consumption state from a previous run is
        automatically invalidated — both static and user-managed dynamic
        channels (e.g. ``defaultdict(Channel)`` lookups) are safe to reuse
        across successive ``match()`` calls without explicit reset.
    """

    __slots__ = ('_consumed_epoch', '_consumed_at')

    def __init__(self) -> None:
        self._consumed_epoch: int = -1
        self._consumed_at: int = -1


# Type aliases for step parameters
Condition = Union[Waveform, Callable[[int, dict], bool]]
IntValue = Union[int, Callable[[int, dict], int]]
SignalValue = Union[Waveform, Callable[[int, dict], Any]]
ChannelValue = Union[Channel, Callable[[int, dict], Channel]]

CaptureMode = Literal['last', 'first', 'list']
_VALID_CAPTURE_MODES = ('last', 'first', 'list')

Step = Union[
    'WaitStep',
    'DelayStep',
    'CaptureStep',
    'RequireStep',
    'LoopStep',
    'RepeatStep',
    'BranchStep',
]


@dataclass
class WaitStep:
    """Blocking: advance when *cond* is True.

    Attributes
    ----------
    cond:
        Waveform or ``callable(index, captures) -> bool``.
    require:
        Optional condition that must hold every cycle while waiting;
        violation terminates the instance with ``REQUIRE_VIOLATED``.
    channel:
        Optional explicit ``Channel`` (or ``callable`` returning one) that
        binds this wait step to a shared FIFO consumer group.  When
        ``None``, the step's private ``_auto_channel`` is used.
    tick:
        When ``True`` (default), a successful match consumes one cycle
        before the next step is evaluated.  When ``False`` the next step
        is evaluated on the same cycle (zero-cycle wait).
    """

    cond: Condition
    require: Condition | None = None
    channel: ChannelValue | None = None
    tick: bool = True
    # mutable per-template state; clone() must preserve identity so that
    # all in-flight instances share the same auto-channel for FIFO consumption.
    _auto_channel: Channel = field(default_factory=Channel, repr=False)

    def clone(self) -> WaitStep:
        return WaitStep(
            cond=self.cond,
            require=self.require,
            channel=self.channel,
            tick=self.tick,
            _auto_channel=self._auto_channel,  # SHARE
        )


@dataclass
class DelayStep:
    """Blocking: unconditionally wait *n* cycles."""

    n: IntValue
    require: Condition | None = None
    # mutable per-instance state
    remaining: int | None = field(default=None, repr=False)

    def clone(self) -> DelayStep:
        return DelayStep(n=self.n, require=self.require, remaining=None)

    def init_remaining(self, index: int, captures: dict) -> None:
        if callable(self.n):
            self.remaining = self.n(index, captures)
        else:
            self.remaining = self.n


@dataclass
class CaptureStep:
    """Epsilon: record a signal value into captures.

    ``mode``:
        * ``'last'``  – overwrite each time (default; ``cap[name]`` is scalar)
        * ``'first'`` – keep only the first write (``cap[name]`` is scalar)
        * ``'list'``  – append every write (``cap[name]`` is a Python list)
    """

    name: str
    signal: SignalValue
    mode: CaptureMode = 'last'

    def __post_init__(self) -> None:
        if self.mode not in _VALID_CAPTURE_MODES:
            raise ValueError(
                f'CaptureStep mode must be one of {_VALID_CAPTURE_MODES}, got {self.mode!r}'
            )

    def clone(self) -> CaptureStep:
        return CaptureStep(name=self.name, signal=self.signal, mode=self.mode)


@dataclass
class RequireStep:
    """Epsilon: assert cond is True, else REQUIRE_VIOLATED."""

    cond: Condition

    def clone(self) -> RequireStep:
        return RequireStep(cond=self.cond)


@dataclass
class LoopStep:
    """Epsilon entry: conditional loop over *body_template*.

    Exactly one of *until* or *when* must be set.

    * ``until``: do-while — execute body first, then check; exit when True.
    * ``when``:  while   — check before body; skip/exit when False.
    """

    body_template: list[Step]
    until: Condition | None = None
    when: Condition | None = None
    # mutable per-instance state
    iteration_count: int = field(default=0, repr=False)

    def clone(self) -> LoopStep:
        return LoopStep(
            body_template=self.body_template,  # template is shared, never mutated
            until=self.until,
            when=self.when,
            iteration_count=0,
        )


@dataclass
class RepeatStep:
    """Epsilon entry: execute body exactly *n* times."""

    body_template: list[Step]
    n: IntValue
    # mutable per-instance state
    times_remaining: int | None = field(default=None, repr=False)

    def clone(self) -> RepeatStep:
        return RepeatStep(
            body_template=self.body_template,
            n=self.n,
            times_remaining=None,
        )

    def init_remaining(self, index: int, captures: dict) -> None:
        if callable(self.n):
            self.times_remaining = self.n(index, captures)
        else:
            self.times_remaining = self.n


@dataclass
class BranchStep:
    """Epsilon: conditional branch."""

    cond: Condition
    true_body: list[Step] | None = None
    false_body: list[Step] | None = None

    def clone(self) -> BranchStep:
        return BranchStep(
            cond=self.cond,
            true_body=self.true_body,
            false_body=self.false_body,
        )


def clone_steps(steps: list[Step]) -> list[Step]:
    """Create a fresh copy of a step list with reset mutable state."""
    return [s.clone() for s in steps]
