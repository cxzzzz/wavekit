from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from typing import Any, get_args

from ..waveform import Waveform
from .compiler import compile_declarative_pattern, infer_declarative_axis
from .errors import PatternError
from .result import MatchRecords
from .runtime import PatternContext, PatternRuntime
from .steps import (
    BranchStep,
    CaptureMode,
    CaptureStep,
    Channel,
    ConsumeStep,
    DelayStep,
    LoopStep,
    RepeatStep,
    RequireStep,
    Step,
    WaitStep,
)


class Pattern:
    """Declarative temporal pattern builder over waveform signals.

    Build steps with ``Pattern().wait(...).capture(...)`` and execute with
    module-level :func:`match`. ``Pattern`` stores only the declarative step AST;
    programmable checking/extraction uses :func:`match(body)` or :func:`collect(body)`.

    The first blocking step selects candidate start cycles. Later blocking steps
    wait within a matched transaction.

    Declarative callbacks receive ``(index, captures)``. ``index`` is the
    current sample index into waveform arrays, not a cycle number and not rebased
    by ``start_cycle``. ``captures`` is the current match's capture dict.
    """

    def __init__(self) -> None:
        self._steps: list[Step] = []

    def wait(
        self,
        cond: Waveform | Callable[[int, dict[str, Any]], bool] | bool,
        *,
        require: Waveform | Callable[[int, dict[str, Any]], bool] | bool | None = None,
        require_message: str | Callable[[int, dict[str, Any]], str] | None = None,
    ) -> Pattern:
        """Observe cycles until *cond* becomes true, without consuming the event.

        Parameters
        ----------
        cond:
            ``Waveform``, ``bool``, or ``callable(index, captures) -> bool``.
            When true at the current cycle, the step completes at that cycle.
            Callback ``index`` is the current waveform-array sample index, not a
            cycle number. ``captures`` is the current match's capture dict.
        require:
            Optional condition checked only while ``cond`` is false. It is not
            checked on the cycle where ``cond`` becomes true. Failure records
            ``MatchStatus.RequireViolated(require_message)``.
        require_message:
            Optional message for ``MatchStatus.RequireViolated``. A callable uses
            the same ``(index, captures)`` arguments and is evaluated only on failure.

        Returns
        -------
        Pattern
            This pattern, for chaining.
        """
        self._steps.append(WaitStep(cond=cond, require=require, require_message=require_message))
        return self

    def consume(
        self,
        cond: Waveform | Callable[[int, dict[str, Any]], bool] | bool,
        channel: Channel | Hashable | Callable[[int, dict[str, Any]], Channel | Hashable],
        *,
        require: Waveform | Callable[[int, dict[str, Any]], bool] | bool | None = None,
        require_message: str | Callable[[int, dict[str, Any]], str] | None = None,
    ) -> Pattern:
        """Wait for *cond* and exclusively claim ``(channel, cycle)``.

        Parameters
        ----------
        cond:
            ``Waveform``, ``bool``, or ``callable(index, captures) -> bool``.
            When true, this step tries to consume the current event cycle.
            Callback ``index`` is the current waveform-array sample index, not a
            cycle number. ``captures`` is the current match's capture dict.
        channel:
            ``Channel``, hashable key, or ``callable(index, captures) -> Channel | Hashable``.
            Channel callbacks use the same ``(index, captures)`` arguments.
            The resolved ``(channel, cycle)`` can be claimed by at most one match;
            earlier start cycles win. This does not reserve the channel while
            ``cond`` is false.
        require:
            Optional condition checked while ``cond`` is false, or while ``cond`` is
            true but the current ``(channel, cycle)`` was already claimed. It is not
            checked on the successful consume cycle.
        require_message:
            Optional message for ``MatchStatus.RequireViolated``. A callable uses
            the same ``(index, captures)`` arguments and is evaluated only on failure.

        Returns
        -------
        Pattern
            This pattern, for chaining.
        """
        self._steps.append(
            ConsumeStep(
                cond=cond, channel=channel, require=require, require_message=require_message
            )
        )
        return self

    def delay(
        self,
        n: int | Callable[[int, dict[str, Any]], int],
        *,
        require: Waveform | Callable[[int, dict[str, Any]], bool] | bool | None = None,
        require_message: str | Callable[[int, dict[str, Any]], str] | None = None,
    ) -> Pattern:
        """Wait exactly *n* cycles; ``delay(0)`` is an epsilon no-op.

        Parameters
        ----------
        n:
            Non-negative ``int`` or ``callable(index, captures) -> int``.
            Callback ``index`` is the current waveform-array sample index, not a
            cycle number. ``captures`` is the current match's capture dict.
        require:
            Optional condition checked before each cycle advance during the delay.
            Failure records ``MatchStatus.RequireViolated(require_message)``.
        require_message:
            Optional message for ``MatchStatus.RequireViolated``. A callable uses
            the same ``(index, captures)`` arguments and is evaluated only on failure.

        Returns
        -------
        Pattern
            This pattern, for chaining.
        """
        self._steps.append(DelayStep(n=n, require=require, require_message=require_message))
        return self

    def capture(
        self,
        name: str,
        signal: Waveform | Callable[[int, dict[str, Any]], Any],
        *,
        mode: CaptureMode = 'last',
    ) -> Pattern:
        """Record a value into ``captures[name]`` at the current cycle.

        Parameters
        ----------
        name:
            Capture key.
        signal:
            ``Waveform`` or ``callable(index, captures) -> Any``. Waveforms are read
            as ``waveform.value[index]`` at the current cycle. Callback ``index`` is
            the current waveform-array sample index, not a cycle number. ``captures``
            is the current match's capture dict.
        mode:
            ``'last'`` overwrites existing values, ``'first'`` keeps the first value,
            and ``'list'`` appends each captured value to a list.

        Returns
        -------
        Pattern
            This pattern, for chaining.
        """
        allowed = get_args(CaptureMode)
        if mode not in allowed:
            raise ValueError(f'capture mode must be one of {allowed}, got {mode!r}')
        self._steps.append(CaptureStep(name=name, signal=signal, mode=mode))
        return self

    def require(
        self,
        cond: Waveform | Callable[[int, dict[str, Any]], bool] | bool,
        *,
        message: str | Callable[[int, dict[str, Any]], str] | None = None,
    ) -> Pattern:
        """Assert *cond* at the current cycle, else record RequireViolated.

        Parameters
        ----------
        cond:
            ``Waveform``, ``bool``, or ``callable(index, captures) -> bool``.
            Callback ``index`` is the current waveform-array sample index, not a
            cycle number. ``captures`` is the current match's capture dict.
        message:
            Optional failure message. A callable uses the same ``(index, captures)``
            arguments and is evaluated only on failure.

        Returns
        -------
        Pattern
            This pattern, for chaining.
        """
        self._steps.append(RequireStep(cond=cond, message=message))
        return self

    def loop(
        self,
        body: Pattern,
        *,
        until: Waveform | Callable[[int, dict[str, Any]], bool] | bool | None = None,
        when: Waveform | Callable[[int, dict[str, Any]], bool] | bool | None = None,
    ) -> Pattern:
        """Run *body* as a do-while (``until``) or while (``when``) loop.

        Parameters
        ----------
        body:
            Nested declarative pattern body.
        until:
            Optional do-while exit condition. The body runs first; the loop exits
            when ``until`` becomes true. Callable conditions use ``(index, captures)``.
        when:
            Optional while condition. The condition is checked before each iteration;
            the loop exits when ``when`` becomes false. Callable conditions use
            ``(index, captures)``.

        Returns
        -------
        Pattern
            This pattern, for chaining.

        Notes
        -----
        Exactly one of ``until`` and ``when`` is required. Conditions use the same
        ``Waveform`` / ``bool`` / callable forms as :meth:`wait`.
        """
        if until is not None and when is not None:
            raise ValueError("Cannot specify both 'until' and 'when' in loop()")
        if until is None and when is None:
            raise ValueError("Must specify either 'until' or 'when' in loop()")
        self._steps.append(LoopStep(body_template=body._steps, until=until, when=when))
        return self

    def repeat(self, body: Pattern, n: int | Callable[[int, dict[str, Any]], int]) -> Pattern:
        """Run *body* exactly *n* times.

        Parameters
        ----------
        body:
            Nested declarative pattern body.
        n:
            Non-negative ``int`` or ``callable(index, captures) -> int``.
            Callback ``index`` is the current waveform-array sample index, not a
            cycle number. ``captures`` is the current match's capture dict.

        Returns
        -------
        Pattern
            This pattern, for chaining.
        """
        self._steps.append(RepeatStep(body_template=body._steps, n=n))
        return self

    def branch(
        self,
        cond: Waveform | Callable[[int, dict[str, Any]], bool] | bool,
        true_body: Pattern | None = None,
        false_body: Pattern | None = None,
    ) -> Pattern:
        """Run one of two epsilon bodies based on *cond* at the current cycle.

        Parameters
        ----------
        cond:
            ``Waveform``, ``bool``, or ``callable(index, captures) -> bool``.
            Callback ``index`` is the current waveform-array sample index, not a
            cycle number. ``captures`` is the current match's capture dict.
        true_body:
            Optional body run when ``cond`` is true.
        false_body:
            Optional body run when ``cond`` is false.

        Returns
        -------
        Pattern
            This pattern, for chaining.
        """
        self._steps.append(
            BranchStep(
                cond=cond,
                true_body=true_body._steps if true_body is not None else None,
                false_body=false_body._steps if false_body is not None else None,
            )
        )
        return self


def match(
    body: Pattern | Callable[[PatternContext], Any],
    *,
    axis: Waveform | None = None,
    timeout: int | None = None,
    timeout_message: str | None = None,
    start_cycle: int | None = None,
    end_cycle: int | None = None,
) -> MatchRecords:
    """Run a declarative pattern or programmable check body.

    Parameters
    ----------
    body:
        Declarative ``Pattern`` or normal callable ``body(ctx)``. Check bodies run
        once per scanned start cycle and must return ``ctx.OK`` to emit an OK row,
        or ``None`` to skip that start. Other non-``None`` values are errors.
    axis:
        Optional waveform that defines the scan axis, cycle numbers, and result
        timestamps. Declarative patterns usually infer it from observed waveforms.
        Pass ``axis`` when the body may not observe a waveform before blocking, or
        when using ``start_cycle`` / ``end_cycle`` without an inferable waveform.
    timeout:
        Optional positive integer per-start maximum duration in cycles. Exceeding it records
        ``MatchStatus.Timeout(timeout_message)``.
    timeout_message:
        Optional human-readable message stored in timeout statuses.
    start_cycle, end_cycle:
        Optional absolute cycle scan window. ``start_cycle`` is inclusive and
        ``end_cycle`` is exclusive.

    Returns
    -------
    MatchRecords
        Ordered batch of match records and captured columns.

    Raises
    ------
    PatternError
        If the body is invalid, waveform axes are incompatible, or execution cannot
        infer a scan axis.
    """
    if isinstance(body, Pattern):
        runtime_axis = axis
        if runtime_axis is None and (start_cycle is not None or end_cycle is not None):
            runtime_axis = infer_declarative_axis(body._steps)
        return PatternRuntime(
            compile_declarative_pattern(body._steps),
            axis=runtime_axis,
            timeout=timeout,
            timeout_message=timeout_message,
        ).match(start_cycle=start_cycle, end_cycle=end_cycle)

    if not callable(body) or inspect.iscoroutinefunction(body):
        raise PatternError('match() requires a Pattern or normal callable body')

    return PatternRuntime(
        body,
        axis=axis,
        timeout=timeout,
        timeout_message=timeout_message,
    ).match(start_cycle=start_cycle, end_cycle=end_cycle)


def collect(
    body: Callable[[PatternContext], Any],
    *,
    axis: Waveform | None = None,
    timeout: int | None = None,
    timeout_message: str | None = None,
    start_cycle: int | None = None,
    end_cycle: int | None = None,
) -> list[Any]:
    """Run a programmable extraction body and collect returned items.

    Parameters
    ----------
    body:
        Normal callable ``body(ctx)``. It runs once per scanned start cycle; each
        non-``None`` return value is appended to the output list. Declarative
        ``Pattern`` objects are intentionally unsupported.
    axis:
        Optional waveform that defines the scan axis. Pass it when the body may not
        observe a waveform before blocking, or when using a scan window.
    timeout:
        Optional positive integer per-start maximum duration in cycles. Timeout raises
        ``PatternError`` instead of returning a status row.
    timeout_message:
        Optional human-readable timeout message.
    start_cycle, end_cycle:
        Optional absolute cycle scan window. ``start_cycle`` is inclusive and
        ``end_cycle`` is exclusive.

    Returns
    -------
    list[Any]
        Non-``None`` values returned by the extraction body.

    Raises
    ------
    PatternError
        If ``body`` is not a normal callable, if a timeout/require failure occurs,
        or if execution cannot infer a scan axis.
    """
    if isinstance(body, Pattern):
        raise PatternError('collect() requires a callable extraction body, not Pattern')
    if not callable(body) or inspect.iscoroutinefunction(body):
        raise PatternError('collect() requires a normal callable extraction body')
    return PatternRuntime(
        body,
        axis=axis,
        timeout=timeout,
        timeout_message=timeout_message,
    ).collect(start_cycle=start_cycle, end_cycle=end_cycle)
