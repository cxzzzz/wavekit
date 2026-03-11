from __future__ import annotations

from dataclasses import dataclass, field

from .steps import Step

# Internal-only status value for instances still being processed.
IN_PROGRESS = -1


@dataclass
class Frame:
    """One level of the instance call stack."""

    steps: list[Step]  # cloned, mutable
    step_idx: int = 0


@dataclass
class Instance:
    """A single in-flight pattern match attempt."""

    frame_stack: list[Frame]
    captures: dict = field(default_factory=dict)
    start_cycle: int = 0
    elapsed: int = 1  # fork cycle counts as 1
    end_cycle: int = 0
    status: int = IN_PROGRESS
