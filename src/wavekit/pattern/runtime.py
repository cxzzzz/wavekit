from __future__ import annotations

from collections.abc import Awaitable, Callable, Hashable, Iterator
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
from typing_extensions import TypeAlias

from ..waveform import Waveform
from .errors import PatternError
from .result import MatchResult, MatchStatus
from .steps import CaptureMode, Channel

_MAX_SAME_CYCLE_STEPS = 100_000
_UNSET = object()

# Runtime conditions deliberately use zero-argument callables.  Declarative
# callables that need (index, captures) are adapted by compiler.py so this
# scheduler can drive both public APIs through one small operation protocol.
RuntimeCondition: TypeAlias = Union[Waveform, Callable[[], bool], bool]
RuntimeChannel: TypeAlias = Union[
    Channel,
    Hashable,
    Callable[[int, dict[str, Any]], Union[Channel, Hashable]],
]
PatternBody: TypeAlias = Callable[['PatternContext'], Awaitable[object]]


class _OkSentinel:
    def __repr__(self) -> str:
        return 'ctx.OK'


OK = _OkSentinel()


class _RequireViolation(PatternError):
    pass


class RuntimeOp:
    """Cooperative operation yielded by ``await ctx.*`` calls.

    The programmable API looks async, but it is not scheduled by ``asyncio``.
    ``__await__`` yields the operation object back to ``PatternRuntime`` so the
    runtime can decide, once per waveform cycle, whether this instance can keep
    running in the same cycle or must remain suspended until a later cycle.
    """

    def __await__(self) -> Iterator[RuntimeOp]:
        yield self
        return None

    def ready(self, ctx: PatternContext, runtime: PatternRuntime) -> bool:
        raise NotImplementedError

    def commit(self, ctx: PatternContext, runtime: PatternRuntime) -> None:
        pass


@dataclass
class WaitOp(RuntimeOp):
    cond: RuntimeCondition
    require: RuntimeCondition | None = None

    def ready(self, ctx: PatternContext, runtime: PatternRuntime) -> bool:
        # Condition-first is part of the blocking-guard contract: a successful
        # wait never evaluates require on the match cycle.  The guard only protects
        # cycles where the instance is still blocked.
        if not runtime.eval_condition(self.cond, ctx):
            self._check_require(ctx, runtime)
            return False
        return True

    def _check_require(self, ctx: PatternContext, runtime: PatternRuntime) -> None:
        if self.require is not None and not runtime.eval_condition(self.require, ctx):
            raise _RequireViolation


@dataclass
class ConsumeOp(RuntimeOp):
    cond: RuntimeCondition
    channel: RuntimeChannel
    require: RuntimeCondition | None = None
    _ready_channel: Channel | None = field(default=None, init=False, repr=False)

    def ready(self, ctx: PatternContext, runtime: PatternRuntime) -> bool:
        # Condition-first matches wait(require=...) semantics: the guard protects
        # blocked cycles, including cycles blocked by FIFO arbitration, but not
        # the success cycle.
        if not runtime.eval_condition(self.cond, ctx):
            self._check_require(ctx, runtime)
            return False
        channel = runtime.resolve_channel(self.channel, ctx)
        if runtime.channel_free(channel, ctx.index):
            self._ready_channel = channel
            return True
        self._ready_channel = None
        self._check_require(ctx, runtime)
        return False

    def commit(self, ctx: PatternContext, runtime: PatternRuntime) -> None:
        # Channel consumption is committed after all readiness checks, so two
        # instances that become ready in the same cycle arbitrate through the
        # runtime epoch/index marker instead of through waveform state.
        if self._ready_channel is None:
            raise PatternError('consume operation committed before a channel was reserved')
        runtime.consume_channel(self._ready_channel, ctx.index)
        self._ready_channel = None

    def _check_require(self, ctx: PatternContext, runtime: PatternRuntime) -> None:
        if self.require is not None and not runtime.eval_condition(self.require, ctx):
            raise _RequireViolation


@dataclass
class DelayOp(RuntimeOp):
    n: int
    start_index: int
    require: RuntimeCondition | None = None

    def ready(self, ctx: PatternContext, runtime: PatternRuntime) -> bool:
        # delay(0) is epsilon.  For delay(n>0) started at t, this op is not
        # ready on t..t+n-1, checks the guard on those blocked cycles, and then
        # resumes at t+n without evaluating the guard for this delay again.
        if self.n == 0:
            return True
        if ctx.index >= self.start_index + self.n:
            return True
        if self.require is not None and not runtime.eval_condition(self.require, ctx):
            raise _RequireViolation
        return False


@dataclass
class PatternContext:
    _runtime: PatternRuntime
    _instance: PatternInstance
    _index: int

    OK = OK

    @property
    def index(self) -> int:
        """Absolute waveform sample index for the current cycle.

        This is not rebased to zero when ``match(start_cycle=...)`` is used.
        """
        return self._index

    @property
    def captures(self) -> dict[str, Any]:
        """Captures accumulated by the current pattern instance."""
        return self._instance.captures

    def value(self, waveform: Waveform, offset: int = 0) -> Any:
        index = self._index + offset
        # Waveform observation is the validation boundary.  This keeps
        # declarative and programmable patterns lazy: unexecuted branches and
        # untouched captures do not need to be pre-collected or pre-validated.
        self._runtime.note_waveform(waveform)
        return waveform.value[index]

    def cycle(self, waveform: Waveform, offset: int = 0) -> Any:
        index = self._index + offset
        self._runtime.note_waveform(waveform)
        return waveform.clock[index]

    def time(self, waveform: Waveform, offset: int = 0) -> Any:
        index = self._index + offset
        self._runtime.note_waveform(waveform)
        return waveform.time[index]

    def wait(
        self,
        cond: RuntimeCondition,
        *,
        require: RuntimeCondition | None = None,
    ) -> WaitOp:
        """Observe cycles until *cond* is true.

        ``require`` is a blocking guard: it is checked only on cycles where the
        wait remains blocked.  If *cond* is true, the wait succeeds without
        checking the guard on that success cycle.  ``wait`` does not consume or
        arbitrate events; use :meth:`consume` for exclusive FIFO ownership.
        """
        return WaitOp(cond=cond, require=require)

    def consume(
        self,
        cond: RuntimeCondition,
        channel: RuntimeChannel,
        *,
        require: RuntimeCondition | None = None,
    ) -> ConsumeOp:
        """Block until *cond* is true and atomically consume *channel*.

        ``channel`` may be a :class:`Channel` object, any hashable key, or a
        ``callable(index, captures)`` that returns either.  ``require`` follows
        ``wait(..., require=...)`` blocking-guard semantics, including cycles
        blocked by channel arbitration.
        """
        return ConsumeOp(cond=cond, channel=channel, require=require)

    def try_consume(self, cond: RuntimeCondition, channel: RuntimeChannel) -> bool:
        op = ConsumeOp(cond=cond, channel=channel)
        if not op.ready(self, self._runtime):
            return False
        op.commit(self, self._runtime)
        return True

    def delay(self, n: int, *, require: RuntimeCondition | None = None) -> DelayOp:
        """Block for *n* cycles.

        ``delay(0)`` is a no-op and checks no guard.  For ``n > 0``, ``require``
        is checked on blocked cycles from the starting cycle through the cycle
        before resume; the resume cycle itself is not checked by this delay.
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise PatternError('ctx.delay(n) requires an integer cycle count')
        if n < 0:
            raise PatternError(f'ctx.delay(n) requires n >= 0, got {n}')
        return DelayOp(n=n, start_index=self._index, require=require)

    def capture(self, name: str, value: Any, mode: CaptureMode = 'last') -> None:
        if isinstance(value, Waveform):
            value = self.value(value)
        if mode == 'last':
            self.captures[name] = value
        elif mode == 'first':
            self.captures.setdefault(name, value)
        elif mode == 'list':
            self.captures.setdefault(name, []).append(value)
        else:
            raise PatternError("ctx.capture() mode must be 'last', 'first', or 'list'")

    def require(self, cond: RuntimeCondition) -> None:
        if not self._runtime.eval_condition(cond, self):
            raise _RequireViolation


@dataclass
class PatternInstance:
    coroutine: Any
    context: PatternContext | None
    start_index: int
    order: int
    captures: dict[str, Any] = field(default_factory=dict)
    current_op: RuntimeOp | None = None
    status: MatchStatus | None = None
    end_index: int = 0
    return_value: Any = None
    discarded: bool = False


class PatternRuntime:
    """Unified cycle-major runtime for programmable and declarative patterns.

    A new candidate instance starts at each scanned cycle.  Each active
    coroutine then runs as far as possible in that same cycle until it completes,
    discards itself by returning ``None``, or yields a blocking ``RuntimeOp``.
    Declarative patterns are compiled to the same coroutine shape, so there is
    only one production execution backend.
    """

    _next_epoch = 0

    def __init__(
        self,
        program: PatternBody,
        *,
        axis: Waveform | None = None,
        timeout_cycles: int | None = None,
        max_active: int = 32768,
    ) -> None:
        self._program = program
        self._timeout_cycles = timeout_cycles
        self._max_active = max_active
        PatternRuntime._next_epoch += 1
        self._epoch = PatternRuntime._next_epoch
        self._key_channels: dict[Hashable, Channel] = {}
        self._axis: Waveform | None = axis
        self._validated_waveform_ids: set[int] = set()
        if axis is not None:
            self._validated_waveform_ids.add(id(axis))
        self._order = 0

    def match(self, start_cycle: int | None = None, end_cycle: int | None = None) -> MatchResult:
        completed = self._run(start_cycle, end_cycle)
        for inst in completed:
            if inst.status == MatchStatus.OK and inst.return_value is not OK:
                raise PatternError('programmable Pattern .match() body must return ctx.OK or None')
        completed = [inst for inst in completed if not inst.discarded]
        axis = self._axis
        if axis is None:
            raise PatternError(
                'Pattern runtime could not determine scan axis; pass axis=<waveform>'
            )
        completed.sort(key=lambda i: (int(axis.clock[i.start_index]), i.order))
        start_arr = np.array([int(axis.clock[i.start_index]) for i in completed], dtype=np.int64)
        end_arr = np.array([int(axis.clock[i.end_index]) for i in completed], dtype=np.int64)
        status_arr = np.array([i.status for i in completed], dtype=np.uint8)
        duration_arr = end_arr - start_arr + 1
        clock = start_arr.copy()
        time = start_arr.copy()

        def wf(value: np.ndarray) -> Waveform:
            return Waveform(value, clock.copy(), time.copy())

        all_keys: set[str] = set()
        for inst in completed:
            all_keys.update(inst.captures.keys())

        captures: dict[str, Waveform] = {}
        for name in sorted(all_keys):
            vals = [inst.captures.get(name) for inst in completed]
            try:
                arr = np.array(vals, dtype=np.int64)
            except (ValueError, TypeError, OverflowError):
                arr = np.array(vals, dtype=object)
            captures[name] = wf(arr)

        return MatchResult(
            start=wf(start_arr),
            end=wf(end_arr),
            duration=wf(duration_arr),
            status=wf(status_arr),
            captures=captures,
        )

    def collect(self, start_cycle: int | None = None, end_cycle: int | None = None) -> list[Any]:
        completed = self._run(start_cycle, end_cycle)
        axis = self._axis
        if axis is None:
            raise PatternError(
                'Pattern runtime could not determine scan axis; pass axis=<waveform>'
            )
        for inst in completed:
            if inst.status is None or inst.status == MatchStatus.OK:
                continue
            kind = {
                MatchStatus.TIMEOUT: 'timed out',
                MatchStatus.REQUIRE_VIOLATED: 'require failed',
            }[inst.status]
            raise PatternError(
                f'Pattern {kind}; '
                f'start_cycle={axis.clock[inst.start_index]}, '
                f'failure_cycle={axis.clock[inst.end_index]}'
            )
        return [inst.return_value for inst in completed if inst.return_value is not None]

    def eval_condition(self, cond: Any, ctx: PatternContext) -> bool:
        if isinstance(cond, Waveform):
            self.note_waveform(cond)
            return bool(cond.value[ctx.index])
        if isinstance(cond, bool):
            return cond
        if callable(cond):
            return bool(cond())
        raise PatternError(
            'condition must be a Waveform, zero-argument callable, or bool; '
            f'got {type(cond).__name__}'
        )

    def note_waveform(self, waveform: Waveform) -> None:
        # This is the single clock-axis validation point for both APIs.  The
        # first observed waveform establishes the scan axis; every later
        # waveform must have the same clock array, not merely the same length.
        waveform_id = id(waveform)
        if waveform_id in self._validated_waveform_ids:
            return
        if self._axis is None:
            self._axis = waveform
            self._validated_waveform_ids.add(waveform_id)
            return
        if len(waveform.clock) != len(self._axis.clock):
            raise PatternError('Waveform clock arrays have different lengths')
        if not np.array_equal(waveform.clock, self._axis.clock):
            raise PatternError('Waveform clock arrays are not aligned')
        self._validated_waveform_ids.add(waveform_id)

    def resolve_channel(self, key: Any, ctx: PatternContext) -> Channel:
        if callable(key) and not isinstance(key, Channel):
            key = key(ctx.index, ctx.captures)
        if isinstance(key, Channel):
            return key
        if not isinstance(key, Hashable):
            raise PatternError(
                f'channel must be a Channel or hashable key, got {type(key).__name__}'
            )
        return self._key_channels.setdefault(key, Channel())

    def channel_free(self, channel: Channel, index: int) -> bool:
        return channel._consumed_epoch != self._epoch or channel._consumed_at != index

    def consume_channel(self, channel: Channel, index: int) -> None:
        channel._consumed_epoch = self._epoch
        channel._consumed_at = index

    def _run(
        self,
        start_cycle: int | None,
        end_cycle: int | None,
    ) -> list[PatternInstance]:
        ref_wf = self._axis
        start_index = 0
        end_index: int | None = None
        if ref_wf is not None:
            n = len(ref_wf.value)
            start_index = (
                0 if start_cycle is None else int(np.searchsorted(ref_wf.clock, start_cycle))
            )
            end_index = n if end_cycle is None else int(np.searchsorted(ref_wf.clock, end_cycle))
            start_index = max(0, min(start_index, n))
            end_index = max(start_index, min(end_index, n))
        elif start_cycle is not None or end_cycle is not None:
            raise PatternError(
                'Pattern runtime requires axis=<waveform> when start/end cycle is used'
            )

        active: list[PatternInstance] = []
        completed: list[PatternInstance] = []

        t = start_index
        while end_index is None or t < end_index:
            assert self._program is not None
            inst = PatternInstance(
                coroutine=None,
                context=None,
                start_index=t,
                order=self._order,
            )
            self._order += 1
            ctx = PatternContext(self, inst, t)
            inst.context = ctx
            inst.coroutine = self._program(ctx)
            active.append(inst)

            still_active: list[PatternInstance] = []
            for inst in active:
                self._advance_instance(inst, t)
                if inst.status is None and not inst.discarded:
                    still_active.append(inst)
                else:
                    completed.append(inst)
            active = still_active

            if len(active) > self._max_active:
                raise PatternError(
                    f'programmable Pattern exceeded max_active={self._max_active}; '
                    'too many candidates are suspended. Use a current-cycle start guard '
                    'such as "if ctx.value(fire): ..." instead of starting many '
                    'candidates with "await ctx.wait(...)", or increase max_active '
                    'if this accumulation is intentional.'
                )

            if t == start_index and ref_wf is None:
                if self._axis is not None:
                    ref_wf = self._axis
                    end_index = len(ref_wf.value)
                else:
                    raise PatternError(
                        'Pattern runtime could not determine scan axis; pass axis=<waveform>'
                    )
            t += 1

        assert ref_wf is not None
        stop_index = end_index if end_index is not None else len(ref_wf.value)
        last_index = max(start_index, stop_index) - 1
        for inst in active:
            # Instances still suspended when the scan window closes are reported
            # as TIMEOUT with an inclusive end index at the final scanned cycle.
            inst.status = MatchStatus.TIMEOUT
            inst.end_index = last_index
            completed.append(inst)

        return completed

    def _advance_instance(self, inst: PatternInstance, index: int) -> None:
        if self._timeout_cycles is not None and index - inst.start_index + 1 > self._timeout_cycles:
            inst.end_index = index
            inst.status = MatchStatus.TIMEOUT
            return
        ctx = inst.context
        assert ctx is not None
        ctx._index = index
        steps = 0
        while inst.status is None and not inst.discarded:
            steps += 1
            if steps > _MAX_SAME_CYCLE_STEPS:
                # Protect against a program that loops forever without yielding a
                # blocking op (for example, repeated delay(0) transitions).
                raise PatternError('programmable Pattern exceeded same-cycle step limit')

            if inst.current_op is not None:
                try:
                    if not inst.current_op.ready(ctx, self):
                        return
                    inst.current_op.commit(ctx, self)
                except _RequireViolation:
                    inst.end_index = index
                    inst.status = MatchStatus.REQUIRE_VIOLATED
                    return
                inst.current_op = None

            try:
                yielded = inst.coroutine.send(None)
            except StopIteration as exc:
                inst.return_value = exc.value
                inst.end_index = index
                if exc.value is None:
                    inst.discarded = True
                else:
                    inst.status = MatchStatus.OK
                return
            except _RequireViolation:
                inst.end_index = index
                inst.status = MatchStatus.REQUIRE_VIOLATED
                return

            if not isinstance(yielded, RuntimeOp):
                raise PatternError(
                    f'programmable Pattern yielded unsupported awaitable {type(yielded)}'
                )
            inst.current_op = yielded
