from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Union, runtime_checkable

from typing_extensions import TypeAlias

from ..waveform import Waveform


@runtime_checkable
class HashableKey(Protocol):
    def __hash__(self) -> int: ...


class Channel:
    """Identity object for shared FIFO consumption across wait steps.

    A ``Channel`` represents a logical event stream from which at most one
    pattern instance may consume per cycle.  Multiple wait steps that bind
    to the same ``Channel`` instance form a single FIFO consumer group
    (oldest in-flight instance wins).

    By default each ``WaitStep`` owns a private ``_auto_channel`` so multiple
    in-flight instances of the same pattern still serialize one-per-cycle on
    that step, while distinct wait steps do not share consumption state.

    Internal state:
        ``(_consumed_epoch, _consumed_at)`` records *which runtime run* and
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
ChannelValue: TypeAlias = Union[
    HashableKey,
    Channel,
    Callable[[int, dict], Union[HashableKey, Channel]],
]

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
    """

    cond: Condition
    require: Condition | None = None
    channel: ChannelValue | None = None
    # Per-step channel shared by all in-flight instances for FIFO consumption.
    _auto_channel: Channel = field(default_factory=Channel, repr=False)


@dataclass
class DelayStep:
    """Blocking: unconditionally wait *n* cycles."""

    n: IntValue
    require: Condition | None = None


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


@dataclass
class RequireStep:
    """Epsilon: assert cond is True, else REQUIRE_VIOLATED."""

    cond: Condition


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


@dataclass
class RepeatStep:
    """Epsilon entry: execute body exactly *n* times."""

    body_template: list[Step]
    n: IntValue


@dataclass
class BranchStep:
    """Epsilon: conditional branch."""

    cond: Condition
    true_body: list[Step] | None = None
    false_body: list[Step] | None = None
