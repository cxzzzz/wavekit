from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, get_args

from typing_extensions import TypeAlias

from ..waveform import Waveform


class Channel:
    """Identity object for explicit consume ownership.

    A ``Channel`` represents a logical event stream from which at most one
    pattern instance may consume per cycle. Plain ``wait`` steps are
    observational and do not consume channels.

    """

    __slots__ = ()


# Type aliases for step parameters. Callable ``index`` values are waveform-array
# sample indices, not cycle numbers and not rebased by match(start_cycle=...).
Condition = Union[Waveform, Callable[[int, dict], bool], bool]
IntValue = Union[int, Callable[[int, dict], int]]
SignalValue = Union[Waveform, Callable[[int, dict], Any]]
MessageValue = Union[str, Callable[[int, dict], str]]
ChannelValue: TypeAlias = Union[
    Hashable,
    Channel,
    Callable[[int, dict], Union[Hashable, Channel]],
]

CaptureMode = Literal['last', 'first', 'list']


class Step:
    """Base class for all declarative pattern steps."""


@dataclass
class WaitStep(Step):
    """Blocking: observe cycles until *cond* is True.

    Attributes
    ----------
    cond:
        Waveform or ``callable(index, captures) -> bool``. ``index`` is the
        waveform-array sample index, not a cycle number.
    require:
        Optional condition that must hold every cycle while waiting;
        violation terminates the instance with ``MatchStatus.RequireViolated``.
    require_message:
        Optional human-readable violation message.
    """

    cond: Condition
    require: Condition | None = None
    require_message: MessageValue | None = None


@dataclass
class ConsumeStep(Step):
    """Blocking: wait for *cond* and consume an explicit channel.

    Attributes
    ----------
    cond:
        Waveform or ``callable(index, captures) -> bool``. ``index`` is the
        waveform-array sample index, not a cycle number.
    channel:
        Explicit ``Channel`` / hashable key (or ``callable`` returning one) for
        consume ownership. Callable ``index`` is the absolute waveform sample
        index.
    require:
        Optional condition that must hold every cycle while waiting or blocked
        by channel arbitration; violation terminates the instance with
        ``MatchStatus.RequireViolated``.
    require_message:
        Optional human-readable violation message.
    """

    cond: Condition
    channel: ChannelValue
    require: Condition | None = None
    require_message: MessageValue | None = None


@dataclass
class DelayStep(Step):
    """Blocking: unconditionally wait *n* cycles.

    Callable ``n`` receives the current waveform-array sample index, not a cycle
    number. ``require_message`` is used when the optional ``require`` guard fails.
    """

    n: IntValue
    require: Condition | None = None
    require_message: MessageValue | None = None


@dataclass
class CaptureStep(Step):
    """Epsilon: record a signal value into captures.

    ``mode``:
        * ``'last'``  – overwrite each time (default; ``cap[name]`` is scalar)
        * ``'first'`` – keep only the first write (``cap[name]`` is scalar)
        * ``'list'``  – append every write (``cap[name]`` is a Python list)

    Callable ``signal`` receives the current waveform-array sample index, not a cycle number.
    """

    name: str
    signal: SignalValue
    mode: CaptureMode = 'last'

    def __post_init__(self) -> None:
        allowed = get_args(CaptureMode)
        if self.mode not in allowed:
            raise ValueError(f'CaptureStep mode must be one of {allowed}, got {self.mode!r}')


@dataclass
class RequireStep(Step):
    """Epsilon: assert cond is True, else MatchStatus.RequireViolated."""

    cond: Condition
    message: MessageValue | None = None


@dataclass
class LoopStep(Step):
    """Epsilon entry: conditional loop over *body_template*.

    Exactly one of *until* or *when* must be set.

    * ``until``: do-while — execute body first, then check; exit when True.
    * ``when``:  while   — check before body; skip/exit when False.
    """

    body_template: list[Step]
    until: Condition | None = None
    when: Condition | None = None


@dataclass
class RepeatStep(Step):
    """Epsilon entry: execute body exactly *n* times."""

    body_template: list[Step]
    n: IntValue


@dataclass
class BranchStep(Step):
    """Epsilon: conditional branch."""

    cond: Condition
    true_body: list[Step] | None = None
    false_body: list[Step] | None = None
