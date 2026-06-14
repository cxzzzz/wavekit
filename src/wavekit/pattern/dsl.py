from __future__ import annotations

import inspect
import warnings
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .compiler import compile_declarative_pattern, infer_declarative_axis
from .errors import PatternError
from .runtime import PatternRuntime
from .steps import (
    _VALID_CAPTURE_MODES,
    BranchStep,
    CaptureStep,
    ChannelValue,
    ConsumeStep,
    DelayStep,
    LoopStep,
    RepeatStep,
    RequireStep,
    Step,
    WaitStep,
)

if TYPE_CHECKING:
    from ..waveform import Waveform
    from .result import MatchResult
    from .runtime import PatternContext
    from .steps import CaptureMode, Condition, IntValue, SignalValue


PatternBody = Callable[['PatternContext'], Awaitable[object]]


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

    def __init__(
        self,
        program: PatternBody | None = None,
        *,
        axis: Waveform | None = None,
        timeout: int | None = None,
        max_active: int = 32768,
    ) -> None:
        # ``program is None`` is declarative builder mode: methods append Step
        # AST nodes and match() compiles them later.  Supplying an async program
        # enters programmable mode immediately and the body is executed directly
        # by PatternRuntime; the fluent step list is intentionally unused there.
        if program is not None and not inspect.iscoroutinefunction(program):
            raise PatternError('Pattern(program) requires an async function body')
        self._steps: list[Step] = []
        self._timeout_cycles: int | None = timeout
        self._program: PatternBody | None = program
        self._axis = axis
        self._max_active = max_active

    # ------------------------------------------------------------------
    # Blocking steps
    # ------------------------------------------------------------------

    def wait(
        self,
        cond: Condition,
        *,
        require: Condition | None = None,
    ) -> Pattern:
        """Observe cycles until *cond* becomes True.

        ``wait`` is non-consuming: every in-flight instance that observes a true
        condition in the same cycle may advance.  Use :meth:`consume` when an
        event must be owned by only one instance through FIFO arbitration.

        Parameters
        ----------
        cond:
            Waveform (static) or ``callable(index, captures) -> bool`` (dynamic).
            ``index`` is the absolute waveform sample index, not relative to
            ``match(start_cycle=...)``.
        require:
            Optional condition that must hold every cycle while waiting.
            Violation terminates the instance with ``REQUIRE_VIOLATED``.
        Examples
        --------
        Zero-cycle wait::

            (Pattern()
                .wait(valid)
                .wait(valid & ready))   # measures latency in same-cycle case
        """
        self._steps.append(WaitStep(cond=cond, require=require))
        return self

    def consume(
        self,
        cond: Condition,
        channel: ChannelValue,
        *,
        require: Condition | None = None,
    ) -> Pattern:
        """Block until *cond* is true and atomically consume *channel*.

        This is the explicit FIFO/exclusive form: when multiple in-flight
        instances reach the same channel in the same cycle, only the oldest
        instance advances.  ``channel`` may be a :class:`Channel`, any hashable
        key, or ``callable(index, captures) -> Channel | Hashable`` for dynamic
        routing such as per-ID response matching.  ``index`` is the absolute
        waveform sample index, not relative to ``match(start_cycle=...)``.
        """
        self._steps.append(ConsumeStep(cond=cond, channel=channel, require=require))
        return self

    def delay(
        self,
        n: IntValue,
        *,
        require: Condition | None = None,
    ) -> Pattern:
        """Block for exactly *n* cycles.

        Parameters
        ----------
        n:
            Static ``int`` or ``callable(index, captures) -> int``.
            ``index`` is the absolute waveform sample index, not relative to
            ``match(start_cycle=...)``.
            ``delay(0)`` is an epsilon step (no cycle consumed).
        require:
            Optional condition that must hold every cycle during the delay.
            Violation terminates the instance with ``REQUIRE_VIOLATED``.
        """
        self._steps.append(DelayStep(n=n, require=require))
        return self

    # ------------------------------------------------------------------
    # Epsilon steps
    # ------------------------------------------------------------------

    def capture(
        self,
        name: str,
        signal: SignalValue,
        *,
        mode: CaptureMode = 'last',
    ) -> Pattern:
        """Record a signal value at the current cycle.

        Parameters
        ----------
        name:
            Capture key.
        signal:
            Waveform (static) or ``callable(index, captures) -> Any``.
            ``index`` is the absolute waveform sample index, not relative to
            ``match(start_cycle=...)``.
        mode:
            * ``'last'`` (default) – ``cap[name]`` holds the most recent value
            * ``'first'`` – ``cap[name]`` holds the first value; subsequent
              writes are ignored
            * ``'list'`` – ``cap[name]`` is a list, each write appends
        """
        if mode not in _VALID_CAPTURE_MODES:
            raise ValueError(f'capture mode must be one of {_VALID_CAPTURE_MODES}, got {mode!r}')
        self._steps.append(CaptureStep(name=name, signal=signal, mode=mode))
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
        ``index`` is the absolute waveform sample index, not relative to
        ``match(start_cycle=...)``.
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
        warnings.warn(
            'Pattern.timeout() is deprecated; pass timeout=... to Pattern(...) instead',
            DeprecationWarning,
            stacklevel=2,
        )
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

        Parameters
        ----------
        start_cycle:
            Limit scanning to cycles >= start_cycle.
        end_cycle:
            Limit scanning to cycles < end_cycle.
        """
        if self._program is not None:
            return PatternRuntime(
                self._program,
                axis=self._axis,
                timeout_cycles=self._timeout_cycles,
                max_active=self._max_active,
            ).match(start_cycle=start_cycle, end_cycle=end_cycle)

        program = compile_declarative_pattern(self._steps)
        runtime_axis = self._axis
        if runtime_axis is None and (start_cycle is not None or end_cycle is not None):
            # Only pass the compiler's first-static-waveform hint when the
            # runtime needs an axis before scanning.  Otherwise runtime
            # observation remains the authority for lazy axis discovery and
            # alignment validation.
            runtime_axis = infer_declarative_axis(self._steps)
        return PatternRuntime(
            program,
            axis=runtime_axis,
            timeout_cycles=self._timeout_cycles,
            max_active=self._max_active,
        ).match(start_cycle=start_cycle, end_cycle=end_cycle)

    def collect(
        self,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> list[Any]:
        """Run a programmable pattern and collect non-``None`` return values."""
        if self._program is None:
            raise PatternError('Pattern.collect() is only available for programmable Pattern')

        return PatternRuntime(
            self._program,
            axis=self._axis,
            timeout_cycles=self._timeout_cycles,
            max_active=self._max_active,
        ).collect(start_cycle=start_cycle, end_cycle=end_cycle)
