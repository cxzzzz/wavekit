from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from .steps import (
    _VALID_CAPTURE_MODES,
    BranchStep,
    CaptureStep,
    ChannelValue,
    DelayStep,
    LoopStep,
    RepeatStep,
    RequireStep,
    Step,
    WaitStep,
)

if TYPE_CHECKING:
    from ..waveform import Waveform
    from .program import ProgramContext
    from .result import MatchResult
    from .steps import CaptureMode, Condition, IntValue, SignalValue


ProgramBody = Callable[['ProgramContext'], Coroutine[Any, Any, object]]


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
        program: ProgramBody | None = None,
        *,
        axis: Waveform | None = None,
        timeout: int | None = None,
        max_active: int = 32768,
    ) -> None:
        if program is not None and not inspect.iscoroutinefunction(program):
            from .engine import PatternError

            raise PatternError('Pattern(program) requires an async function body')
        self._steps: list[Step] = []
        self._timeout_cycles: int | None = timeout
        self._program = program
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
        channel: ChannelValue | None = None,
        tick: bool = True,
    ) -> Pattern:
        """Block until *cond* becomes True.

        By default each ``wait()`` step has its own private FIFO consumer
        group: when multiple in-flight instances reach this step, only one
        (the oldest) advances per cycle.  Pass an explicit :class:`Channel`
        via ``channel=`` to share the consumer group across multiple wait
        steps (e.g., for AXI request/response pairing).

        Parameters
        ----------
        cond:
            Waveform (static) or ``callable(index, captures) -> bool`` (dynamic).
        require:
            Optional condition that must hold every cycle while waiting.
            Violation terminates the instance with ``REQUIRE_VIOLATED``.
        channel:
            Explicit :class:`Channel` (or ``callable(index, captures) -> Channel``
            for dynamic key-based partitioning) that binds this wait to a
            shared FIFO consumer group.  ``None`` (default) uses an implicit
            per-step channel.
        tick:
            When ``True`` (default), a successful match advances time by one
            cycle before the next step is evaluated.  When ``False`` the next
            step evaluates on the **same** cycle (zero-cycle wait), useful
            for measuring intervals like ``valid → valid & ready``.

        Examples
        --------
        Static channel shared between request and response steps::

            rsp_chan = Channel()
            (Pattern()
                .wait(req)
                .wait(rsp, channel=rsp_chan))

        Dynamic per-ID partitioning for AXI out-of-order reads::

            from collections import defaultdict
            chans = defaultdict(Channel)
            (Pattern()
                .wait(arvalid & arready)
                .capture('arid', arid)
                .wait(
                    lambda i, cap: bool((rvalid & rready)[i]) and rid[i] == cap['arid'],
                    channel=lambda i, cap: chans[cap['arid']],
                ))

        Zero-cycle wait::

            (Pattern()
                .wait(valid)
                .wait(valid & ready, tick=False))   # measures latency in same-cycle case
        """
        self._steps.append(WaitStep(cond=cond, require=require, channel=channel, tick=tick))
        return self

    def consume(
        self,
        cond: Condition,
        channel: ChannelValue,
        *,
        require: Condition | None = None,
        tick: bool = True,
    ) -> Pattern:
        """Block until *cond* is true and atomically consume *channel*.

        This is the explicit spelling for a channel-consuming wait and is
        equivalent to ``wait(cond, channel=channel, require=require, tick=tick)``.
        """
        return self.wait(cond, require=require, channel=channel, tick=tick)

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
            Static ``int`` (>= 0) or ``callable(index, captures) -> int``.
            ``delay(0)`` is an epsilon step (no cycle consumed).
        require:
            Optional condition that must hold every cycle during the delay.
            Violation terminates the instance with ``REQUIRE_VIOLATED``.
        """
        if isinstance(n, int) and n < 0:
            raise ValueError(f'delay() requires n >= 0, got {n}')
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
            from .program import ProgramRuntime

            return ProgramRuntime(self).match(start_cycle=start_cycle, end_cycle=end_cycle)

        from .engine import PatternEngine

        engine = PatternEngine(self)
        return engine.run(start_cycle=start_cycle, end_cycle=end_cycle)

    def collect(
        self,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> list[Any]:
        """Run a programmable pattern and collect non-``None`` return values."""
        if self._program is None:
            from .engine import PatternError

            raise PatternError('Pattern.collect() is only available for programmable Pattern')

        from .program import ProgramRuntime

        return ProgramRuntime(self).collect(start_cycle=start_cycle, end_cycle=end_cycle)
