from __future__ import annotations

from typing import TYPE_CHECKING

from .steps import (
    BranchStep,
    CaptureStep,
    DelayStep,
    LoopStep,
    RepeatStep,
    RequireStep,
    Step,
    WaitStep,
)

if TYPE_CHECKING:
    from .result import MatchResult
    from .steps import Condition, IntValue, SignalValue


class Pattern:
    """Builder for a temporal pattern over waveform signals.

    Methods are chained to construct a step sequence::

        p = (
            Pattern()
            .wait(valid & ready)
            .capture('data', data_signal)
            .timeout(256)
        )
        result = p.match()
    """

    def __init__(self) -> None:
        self._steps: list[Step] = []
        self._timeout_cycles: int | None = None

    # ------------------------------------------------------------------
    # Blocking steps
    # ------------------------------------------------------------------

    def wait(
        self,
        cond: Condition,
        guard: Condition | None = None,
        channel: str | None = None,
    ) -> Pattern:
        """Block until *cond* becomes True.

        Parameters
        ----------
        cond:
            Waveform (static) or ``callable(index, captures) -> bool`` (dynamic).
        guard:
            Must remain True every cycle while waiting; violation terminates
            the instance with ``REQUIRE_VIOLATED``.
        channel:
            Named channel for FIFO event-consumption (see channel docs).
        """
        self._steps.append(WaitStep(cond=cond, guard=guard, channel=channel))
        return self

    def delay(self, n: IntValue, guard: Condition | None = None) -> Pattern:
        """Block for exactly *n* cycles.

        Parameters
        ----------
        n:
            Static ``int`` (>= 0) or ``callable(index, captures) -> int``.
            ``delay(0)`` is an epsilon step (no cycle consumed).
        guard:
            Must remain True during the delay; violation terminates
            the instance with ``REQUIRE_VIOLATED``.
        """
        if isinstance(n, int) and n < 0:
            raise ValueError(f'delay() requires n >= 0, got {n}')
        self._steps.append(DelayStep(n=n, guard=guard))
        return self

    # ------------------------------------------------------------------
    # Epsilon steps
    # ------------------------------------------------------------------

    def capture(self, name: str, signal: SignalValue) -> Pattern:
        """Record a signal value at the current cycle.

        Use ``name[]`` to append to a list (for use inside loop/repeat).
        """
        self._steps.append(CaptureStep(name=name, signal=signal))
        return self

    def require(self, cond: Condition) -> Pattern:
        """Assert *cond* at the current cycle; else ``REQUIRE_VIOLATED``."""
        self._steps.append(RequireStep(cond=cond))
        return self

    def loop(
        self,
        body: Pattern,
        *,
        until: Condition | None = None,
        when: Condition | None = None,
    ) -> Pattern:
        """Conditional loop over *body*.

        Exactly one of *until* / *when* must be given.

        * ``until``: do-while — execute body, then check; exit when True.
        * ``when``:  while   — check before each iteration; skip when False.
        """
        if until is not None and when is not None:
            raise ValueError("Cannot specify both 'until' and 'when' in loop()")
        if until is None and when is None:
            raise ValueError("Must specify either 'until' or 'when' in loop()")
        self._steps.append(
            LoopStep(
                body_template=body._steps,
                until=until,
                when=when,
            )
        )
        return self

    def repeat(self, body: Pattern, n: IntValue) -> Pattern:
        """Execute *body* exactly *n* times.

        *n* can be a static ``int`` or ``callable(index, captures) -> int``.
        """
        self._steps.append(RepeatStep(body_template=body._steps, n=n))
        return self

    def branch(
        self,
        cond: Condition,
        true_body: Pattern | None = None,
        false_body: Pattern | None = None,
    ) -> Pattern:
        """Conditional branch at the current cycle."""
        self._steps.append(
            BranchStep(
                cond=cond,
                true_body=true_body._steps if true_body is not None else None,
                false_body=false_body._steps if false_body is not None else None,
            )
        )
        return self

    # ------------------------------------------------------------------
    # Pattern-level attributes
    # ------------------------------------------------------------------

    def timeout(self, max_cycles: int) -> Pattern:
        """Set the maximum number of cycles for the entire pattern."""
        self._timeout_cycles = max_cycles
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def match(
        self,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> MatchResult:
        """Run the pattern engine and return a :class:`MatchResult`.

        If the first step is a ``wait``, its condition is used as the trigger
        (only fork instances when the condition is True).  Otherwise, an
        instance is forked on every cycle.

        Parameters
        ----------
        start_cycle:
            Limit scanning to cycles >= start_cycle.
        end_cycle:
            Limit scanning to cycles < end_cycle.
        """
        from .engine import PatternEngine

        engine = PatternEngine(self)
        return engine.run(start_cycle=start_cycle, end_cycle=end_cycle)
