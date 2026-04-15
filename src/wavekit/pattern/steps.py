from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union

from ..waveform import Waveform

# Type aliases for step parameters
Condition = Union[Waveform, Callable[[int, dict], bool]]
IntValue = Union[int, Callable[[int, dict], int]]
SignalValue = Union[Waveform, Callable[[int, dict], Any]]
QueueValue = Union[str, Callable[[int, dict], str]]

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
    """Blocking: advance when *cond* is True."""

    cond: Condition
    guard: Condition | None = None
    queue: QueueValue | None = None

    def clone(self) -> WaitStep:
        return WaitStep(cond=self.cond, guard=self.guard, queue=self.queue)


@dataclass
class DelayStep:
    """Blocking: unconditionally wait *n* cycles."""

    n: IntValue
    guard: Condition | None = None
    # mutable per-instance state
    remaining: int | None = field(default=None, repr=False)

    def clone(self) -> DelayStep:
        return DelayStep(n=self.n, guard=self.guard, remaining=None)

    def init_remaining(self, index: int, captures: dict) -> None:
        if callable(self.n):
            self.remaining = self.n(index, captures)
        else:
            self.remaining = self.n


@dataclass
class CaptureStep:
    """Epsilon: record a signal value into captures."""

    name: str
    signal: SignalValue

    def clone(self) -> CaptureStep:
        return CaptureStep(name=self.name, signal=self.signal)


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
