from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Union

import numpy as np
from typing_extensions import TypeAlias

from ..waveform import Waveform
from .errors import PatternError
from .result import MatchRecords, MatchStatus, MatchStatusValue
from .steps import CaptureMode, Channel

_MAX_SAME_CYCLE_STEPS = 100_000

PatternBody: TypeAlias = Callable[['PatternContext'], object]
Message: TypeAlias = Union[str, Callable[[Any], str]]


class _StopPattern(PatternError):
    def __init__(self, status: MatchStatusValue):
        self.status = status


@dataclass
class PatternInstance:
    start_index: int
    order: int
    captures: dict[str, Any] = field(default_factory=dict)
    status: MatchStatusValue | None = None
    end_index: int = 0
    return_value: Any = None
    discarded: bool = False


@dataclass
class PatternContext:
    _runtime: PatternRuntime
    _instance: PatternInstance
    _index: int
    _same_cycle_steps: int = 0

    OK = MatchStatus.OK()

    @property
    def index(self) -> int:
        """Current waveform-array sample index, not a cycle number."""
        return self._index

    @property
    def captures(self) -> dict[str, Any]:
        """Captures accumulated by the current pattern instance."""
        return self._instance.captures

    def _touch(self) -> None:
        self._instance.end_index = self._index

    def _message_text(self, message: Message | None) -> str | None:
        if message is None:
            return None
        if isinstance(message, str):
            return message
        if callable(message):
            return str(message(self))
        raise PatternError(f'message must be a str or callable, got {type(message).__name__}')

    def _timeout_status(self, message: Message | None) -> MatchStatusValue:
        return MatchStatus.Timeout(self._message_text(message) or self._runtime._timeout_message)

    def _require_status(self, message: Message | None) -> MatchStatusValue:
        return MatchStatus.RequireViolated(self._message_text(message))

    def _guard_operation(self) -> None:
        self._same_cycle_steps += 1
        if self._same_cycle_steps > _MAX_SAME_CYCLE_STEPS:
            raise PatternError('programmable Pattern exceeded same-cycle step limit')
        timeout = self._runtime._timeout_cycles
        if timeout is not None and self._index - self._instance.start_index + 1 > timeout:
            raise _StopPattern(self._timeout_status(None))
        self._touch()

    def _advance_cycle(self) -> None:
        if self._runtime._axis is None:
            raise PatternError(
                'Pattern runtime could not determine scan axis; pass axis=<waveform> '
                'or observe a Waveform before blocking operations'
            )
        if self._index >= self._runtime.scan_end_index - 1:
            raise _StopPattern(self._timeout_status(None))
        self._index += 1
        self._same_cycle_steps = 0
        self._touch()

    def _wait_next(
        self,
        require: Waveform | Callable[[], bool] | bool | None,
        require_message: Message | None,
    ) -> None:
        if self._runtime.eval_condition(require, self):
            self._advance_cycle()
            return
        raise _StopPattern(self._require_status(require_message))

    def value(self, waveform: Waveform, offset: int = 0) -> Any:
        index = self._index + offset
        self._runtime.note_waveform(waveform)
        self._touch()
        return waveform.value[index]

    def cycle(self, waveform: Waveform, offset: int = 0) -> Any:
        index = self._index + offset
        self._runtime.note_waveform(waveform)
        self._touch()
        return waveform.clock[index]

    def time(self, waveform: Waveform, offset: int = 0) -> Any:
        index = self._index + offset
        self._runtime.note_waveform(waveform)
        self._touch()
        return waveform.time[index]

    def wait(
        self,
        cond: Waveform | Callable[[], bool] | bool,
        *,
        require: Waveform | Callable[[], bool] | bool | None = None,
        require_message: Message | None = None,
    ) -> None:
        while True:
            self._guard_operation()
            if self._runtime.eval_condition(cond, self):
                return
            self._wait_next(require, require_message)

    def consume(
        self,
        cond: Waveform | Callable[[], bool] | bool,
        channel: Channel | Hashable | Callable[[int, dict[str, Any]], Channel | Hashable],
        *,
        require: Waveform | Callable[[], bool] | bool | None = None,
        require_message: Message | None = None,
    ) -> None:
        while True:
            self._guard_operation()
            if self._runtime.eval_condition(cond, self):
                resolved = self._runtime.resolve_channel(channel, self)
                if self._runtime.channel_free(resolved, self._index):
                    self._runtime.consume_channel(resolved, self._index)
                    return
            self._wait_next(require, require_message)

    def try_consume(
        self,
        cond: Waveform | Callable[[], bool] | bool,
        channel: Channel | Hashable | Callable[[int, dict[str, Any]], Channel | Hashable],
    ) -> bool:
        self._guard_operation()
        if not self._runtime.eval_condition(cond, self):
            return False
        resolved = self._runtime.resolve_channel(channel, self)
        if not self._runtime.channel_free(resolved, self._index):
            return False
        self._runtime.consume_channel(resolved, self._index)
        return True

    def delay(
        self,
        n: int,
        *,
        require: Waveform | Callable[[], bool] | bool | None = None,
        require_message: Message | None = None,
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise PatternError('ctx.delay(n) requires an integer cycle count')
        if n < 0:
            raise PatternError(f'ctx.delay(n) requires n >= 0, got {n}')
        if n == 0:
            return
        target_index = self._index + n
        while self._index < target_index:
            self._guard_operation()
            self._wait_next(require, require_message)

    def capture(self, name: str, value: Any, mode: CaptureMode = 'last') -> None:
        self._guard_operation()
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

    def require(
        self, cond: Waveform | Callable[[], bool] | bool, *, message: Message | None = None
    ) -> None:
        self._guard_operation()
        if not self._runtime.eval_condition(cond, self):
            raise _StopPattern(self._require_status(message))


class PatternRuntime:
    """Synchronous start-major runtime for declarative and programmable patterns."""

    def __init__(
        self,
        program: PatternBody,
        *,
        axis: Waveform | None = None,
        timeout_cycles: int | None = None,
        timeout_message: str | None = None,
    ) -> None:
        self._program = program
        self._timeout_cycles = timeout_cycles
        self._timeout_message = timeout_message
        self._axis: Waveform | None = axis
        self._start_cycle: int | None = None
        self._end_cycle: int | None = None
        self._key_channels: dict[Hashable, np.ndarray] = {}
        self._validated_waveform_ids: set[int] = set()
        if axis is not None:
            self._validated_waveform_ids.add(id(axis))
        self._order = 0

    @cached_property
    def scan_start_index(self) -> int:
        if self._start_cycle is None:
            return 0
        axis = self._require_axis()
        return int(np.searchsorted(axis.clock, self._start_cycle))

    @cached_property
    def scan_end_index(self) -> int:
        axis = self._require_axis()
        if self._end_cycle is None:
            return len(axis.value)
        return int(np.searchsorted(axis.clock, self._end_cycle))

    def match(self, start_cycle: int | None = None, end_cycle: int | None = None) -> MatchRecords:
        completed = self._run(start_cycle, end_cycle)
        for inst in completed:
            if isinstance(inst.status, MatchStatus.OK) and not isinstance(
                inst.return_value, MatchStatus.OK
            ):
                raise PatternError('pattern match body must return ctx.OK or None')
        return self._records([inst for inst in completed if not inst.discarded])

    def collect(self, start_cycle: int | None = None, end_cycle: int | None = None) -> list[Any]:
        completed = self._run(start_cycle, end_cycle)
        axis = self._require_axis()
        for inst in completed:
            if inst.status is None or isinstance(inst.status, MatchStatus.OK):
                continue
            raise PatternError(
                f'Pattern failed with {inst.status}; '
                f'start_cycle={axis.clock[inst.start_index]}, '
                f'failure_cycle={axis.clock[inst.end_index]}'
            )
        return [inst.return_value for inst in completed if inst.return_value is not None]

    def _records(self, completed: list[PatternInstance]) -> MatchRecords:
        axis = self._require_axis()
        completed.sort(key=lambda i: (int(axis.clock[i.start_index]), i.order))
        start_index_arr = np.array([i.start_index for i in completed], dtype=np.int64)
        end_index_arr = np.array([i.end_index for i in completed], dtype=np.int64)
        start_cycle_arr = np.array(
            [int(axis.clock[i.start_index]) for i in completed], dtype=np.int64
        )
        end_cycle_arr = np.array([int(axis.clock[i.end_index]) for i in completed], dtype=np.int64)
        start_time_arr = np.array(
            [int(axis.time[i.start_index]) for i in completed], dtype=np.int64
        )
        end_time_arr = np.array([int(axis.time[i.end_index]) for i in completed], dtype=np.int64)
        duration_arr = end_index_arr - start_index_arr + 1
        status_arr = np.array([i.status for i in completed], dtype=object)

        def row_wf(value: np.ndarray) -> Waveform:
            return Waveform(value, start_cycle_arr.copy(), start_time_arr.copy())

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
            captures[name] = row_wf(arr)

        return MatchRecords(
            start=Waveform(start_index_arr, start_cycle_arr, start_time_arr),
            end=Waveform(end_index_arr, end_cycle_arr, end_time_arr),
            duration=row_wf(duration_arr),
            status=row_wf(status_arr),
            captures=captures,
        )

    def eval_condition(
        self, cond: Waveform | Callable[[], bool] | bool | None, ctx: PatternContext
    ) -> bool:
        if cond is None:
            return True
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

    def resolve_channel(
        self,
        key: Channel | Hashable | Callable[[int, dict[str, Any]], Channel | Hashable],
        ctx: PatternContext,
    ) -> Hashable:
        if callable(key) and not isinstance(key, Channel):
            key = key(ctx.index, ctx.captures)
        if isinstance(key, Channel):
            return key
        if not isinstance(key, Hashable):
            raise PatternError(
                f'channel must be a Channel or hashable key, got {type(key).__name__}'
            )
        return key

    def channel_free(self, channel: Hashable, index: int) -> bool:
        mask = self._key_channels.get(channel)
        if mask is None or index >= len(mask):
            return True
        return not bool(mask[index])

    def consume_channel(self, channel: Hashable, index: int) -> None:
        mask = self._key_channels.get(channel)
        if mask is None:
            length = self.scan_end_index if self._axis is not None else index + 1
            mask = np.zeros(length, dtype=bool)
            self._key_channels[channel] = mask
        elif self._axis is not None and len(mask) < self.scan_end_index:
            new_mask = np.zeros(self.scan_end_index, dtype=bool)
            new_mask[: len(mask)] = mask
            self._key_channels[channel] = mask = new_mask
        elif index >= len(mask):
            new_mask = np.zeros(index + 1, dtype=bool)
            new_mask[: len(mask)] = mask
            self._key_channels[channel] = mask = new_mask
        mask[index] = True

    def _run_candidate(self, start_index: int, order: int) -> PatternInstance:
        inst = PatternInstance(start_index=start_index, order=order, end_index=start_index)
        ctx = PatternContext(self, inst, start_index)
        try:
            inst.return_value = self._program(ctx)
        except _StopPattern as stop:
            inst.status = stop.status
        else:
            if inst.return_value is None:
                inst.discarded = True
            else:
                inst.status = MatchStatus.OK()
        return inst

    def _run(self, start_cycle: int | None, end_cycle: int | None) -> list[PatternInstance]:
        self._start_cycle = start_cycle
        self._end_cycle = end_cycle

        if self._axis is None and (start_cycle is not None or end_cycle is not None):
            raise PatternError(
                'Pattern runtime requires axis=<waveform> when start/end cycle is used'
            )

        start_index = self.scan_start_index
        end_index: int | None = self.scan_end_index if self._axis is not None else None

        completed: list[PatternInstance] = []
        t = start_index
        while end_index is None or t < end_index:
            inst = self._run_candidate(t, self._order)
            self._order += 1
            completed.append(inst)

            if self._axis is None:
                raise PatternError(
                    'Pattern runtime could not determine scan axis; pass axis=<waveform>'
                )
            if end_index is None:
                end_index = self.scan_end_index

            t += 1

        return completed

    def _require_axis(self) -> Waveform:
        if self._axis is None:
            raise PatternError(
                'Pattern runtime could not determine scan axis; pass axis=<waveform>'
            )
        return self._axis
