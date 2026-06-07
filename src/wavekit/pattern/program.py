from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
from typing_extensions import TypeAlias

from ..waveform import Waveform
from .engine import PatternError
from .result import MatchResult, MatchStatus
from .steps import CaptureMode, Channel

_MAX_SAME_CYCLE_STEPS = 100_000
_UNSET = object()

ProgramCondition: TypeAlias = Union[Waveform, Callable[[], bool], bool]
ProgramChannel: TypeAlias = Union[Channel, Hashable]


class _OkSentinel:
    def __repr__(self) -> str:
        return 'ctx.OK'


OK = _OkSentinel()


class _RequireViolation(PatternError):
    pass


class ProgramOp:
    def __await__(self) -> Iterator[ProgramOp]:
        yield self
        return None

    def ready(self, ctx: ProgramContext, runtime: ProgramRuntime) -> bool:
        raise NotImplementedError

    def commit(self, ctx: ProgramContext, runtime: ProgramRuntime) -> None:
        pass


@dataclass
class WaitOp(ProgramOp):
    cond: ProgramCondition
    consume: bool = True
    channel: ProgramChannel | None = None

    def ready(self, ctx: ProgramContext, runtime: ProgramRuntime) -> bool:
        if not runtime.eval_condition(self.cond, ctx):
            return False
        if not self.consume:
            return True
        channel = runtime.resolve_channel(self.channel)
        return runtime.channel_free(channel, ctx.index)

    def commit(self, ctx: ProgramContext, runtime: ProgramRuntime) -> None:
        if self.consume:
            runtime.consume_channel(runtime.resolve_channel(self.channel), ctx.index)


@dataclass
class ConsumeOp(WaitOp):
    consume: bool = True


@dataclass
class DelayOp(ProgramOp):
    n: int
    start_index: int

    def ready(self, ctx: ProgramContext, runtime: ProgramRuntime) -> bool:
        return ctx.index >= self.start_index + self.n


@dataclass
class ProgramContext:
    _runtime: ProgramRuntime
    _instance: ProgramInstance
    _index: int

    OK = OK

    @property
    def index(self) -> int:
        return self._index

    def value(self, waveform: Waveform, offset: int = 0) -> Any:
        index = self._index + offset
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
        cond: ProgramCondition,
        *,
        consume: bool = True,
        channel: ProgramChannel | None = None,
    ) -> WaitOp:
        self._runtime.validate_condition(cond)
        if not consume and channel is not None:
            raise PatternError('ctx.wait(..., consume=False) cannot specify channel')
        if consume and channel is None:
            raise PatternError('ctx.wait(..., consume=True) requires an explicit channel')
        return WaitOp(cond=cond, consume=consume, channel=channel)

    def consume(self, cond: ProgramCondition, channel: ProgramChannel) -> ConsumeOp:
        self._runtime.validate_condition(cond)
        return ConsumeOp(cond=cond, channel=channel)

    def try_consume(self, cond: ProgramCondition, channel: ProgramChannel) -> bool:
        self._runtime.validate_condition(cond)
        op = ConsumeOp(cond=cond, channel=channel)
        if not op.ready(self, self._runtime):
            return False
        op.commit(self, self._runtime)
        return True

    def delay(self, n: int) -> DelayOp:
        if isinstance(n, bool) or not isinstance(n, int):
            raise PatternError('ctx.delay(n) requires an integer cycle count')
        if n < 0:
            raise PatternError(f'ctx.delay(n) requires n >= 0, got {n}')
        return DelayOp(n=n, start_index=self._index)

    def capture(self, name: str, value: Any, mode: CaptureMode = 'last') -> None:
        if isinstance(value, Waveform):
            value = self.value(value)
        if mode == 'last':
            self._instance.captures[name] = value
        elif mode == 'first':
            self._instance.captures.setdefault(name, value)
        elif mode == 'list':
            self._instance.captures.setdefault(name, []).append(value)
        else:
            raise PatternError("ctx.capture() mode must be 'last', 'first', or 'list'")

    def require(self, cond: ProgramCondition) -> None:
        self._runtime.validate_condition(cond)
        if not self._runtime.eval_condition(cond, self):
            raise _RequireViolation


@dataclass
class ProgramInstance:
    coroutine: Any
    context: ProgramContext | None
    start_index: int
    order: int
    captures: dict[str, Any] = field(default_factory=dict)
    current_op: ProgramOp | None = None
    status: MatchStatus | None = None
    end_index: int = 0
    return_value: Any = None
    discarded: bool = False


class ProgramRuntime:
    """Cycle-major runtime for programmable ``Pattern(async_body, ...)``."""

    _next_epoch = 0

    def __init__(self, pattern) -> None:
        self._program = pattern._program
        self._timeout_cycles = pattern._timeout_cycles
        self._max_active = pattern._max_active
        ProgramRuntime._next_epoch += 1
        self._epoch = ProgramRuntime._next_epoch
        self._key_channels: dict[Hashable, Channel] = {}
        self._axis: Waveform | None = pattern._axis
        self._order = 0

    def match(self, start_cycle: int | None = None, end_cycle: int | None = None) -> MatchResult:
        completed = self._run(start_cycle, end_cycle)
        for inst in completed:
            if inst.status == MatchStatus.OK and inst.return_value is not OK:
                raise PatternError('programmable Pattern .match() body must return ctx.OK or None')
        completed = [inst for inst in completed if not inst.discarded]
        if self._axis is None:
            raise PatternError(
                'Pattern(program) could not determine scan axis; pass axis=<waveform>'
            )
        axis = self._axis
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
                'Pattern(program) could not determine scan axis; pass axis=<waveform>'
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

    def validate_condition(self, cond: Any) -> None:
        if isinstance(cond, Waveform):
            self.note_waveform(cond)
            return
        if isinstance(cond, bool):
            return
        if callable(cond):
            return
        raise PatternError(
            'condition must be a Waveform, zero-argument callable, or bool; '
            f'got {type(cond).__name__}'
        )

    def eval_condition(self, cond: Any, ctx: ProgramContext) -> bool:
        self.validate_condition(cond)
        if isinstance(cond, Waveform):
            return bool(cond.value[ctx.index])
        if isinstance(cond, bool):
            return cond
        return bool(cond())

    def note_waveform(self, waveform: Waveform) -> None:
        if self._axis is None:
            self._axis = waveform
            return
        if len(waveform.clock) != len(self._axis.clock):
            raise PatternError('Waveform clock arrays have different lengths')

    def resolve_channel(self, key: Any) -> Channel:
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
    ) -> list[ProgramInstance]:
        ref_wf = self._axis
        start_index = 0
        end_index: int | None = None
        if ref_wf is not None:
            n = len(ref_wf.value)
            start_index = (
                0 if start_cycle is None else int(np.searchsorted(ref_wf.clock, start_cycle))
            )
            end_index = (
                n if end_cycle is None else int(np.searchsorted(ref_wf.clock, end_cycle))
            )
            start_index = max(0, min(start_index, n))
            end_index = max(start_index, min(end_index, n))
        elif start_cycle is not None or end_cycle is not None:
            raise PatternError(
                'Pattern(program) requires axis=<waveform> when start/end cycle is used'
            )

        active: list[ProgramInstance] = []
        completed: list[ProgramInstance] = []

        t = start_index
        while end_index is None or t < end_index:
            # Start one candidate at every scanned cycle.
            assert self._program is not None
            inst = ProgramInstance(
                coroutine=None,
                context=None,
                start_index=t,
                order=self._order,
            )
            self._order += 1
            ctx = ProgramContext(self, inst, t)
            inst.context = ctx
            inst.coroutine = self._program(ctx)
            active.append(inst)

            still_active: list[ProgramInstance] = []
            for inst in active:
                self._tick(inst, t)
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

            if ref_wf is None and self._axis is not None:
                ref_wf = self._axis
                end_index = len(ref_wf.value)
            elif ref_wf is None:
                raise PatternError(
                    'Pattern(program) could not determine scan axis; pass axis=<waveform>'
                )
            t += 1

        assert ref_wf is not None
        stop_index = end_index if end_index is not None else len(ref_wf.value)
        last_index = max(start_index, stop_index) - 1
        for inst in active:
            inst.status = MatchStatus.TIMEOUT
            inst.end_index = last_index
            completed.append(inst)

        return completed

    def _tick(self, inst: ProgramInstance, index: int) -> None:
        if self._timeout_cycles is not None and index - inst.start_index + 1 > self._timeout_cycles:
            inst.end_index = index
            inst.status = MatchStatus.TIMEOUT
            return
        assert inst.context is not None
        inst.context._index = index
        steps = 0
        while inst.status is None and not inst.discarded:
            steps += 1
            if steps > _MAX_SAME_CYCLE_STEPS:
                raise PatternError('programmable Pattern exceeded same-cycle step limit')

            if inst.current_op is not None:
                if not inst.current_op.ready(inst.context, self):
                    return
                inst.current_op.commit(inst.context, self)
                inst.current_op = None

            try:
                yielded = inst.coroutine.send(None)
            except StopIteration as exc:
                # Record normal completion; match/collect interpret the return value later.
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

            if not isinstance(yielded, ProgramOp):
                raise PatternError(
                    f'programmable Pattern yielded unsupported awaitable {type(yielded)}'
                )
            inst.current_op = yielded
