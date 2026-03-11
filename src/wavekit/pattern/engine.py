from __future__ import annotations

from typing import Any

import numpy as np

from ..waveform import Waveform
from .instance import IN_PROGRESS, Frame, Instance
from .result import MatchResult, MatchStatus
from .steps import (
    BranchStep,
    CaptureStep,
    Condition,
    DelayStep,
    LoopStep,
    RepeatStep,
    RequireStep,
    Step,
    WaitStep,
    clone_steps,
)


class PatternError(Exception):
    """Raised for pattern definition or runtime errors."""


# Safety limit: max epsilon transitions per tick to detect infinite loops.
_MAX_EPSILON = 100_000


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _eval_bool(cond: Condition, index: int, captures: dict) -> bool:
    if isinstance(cond, Waveform):
        return bool(cond.value[index])
    if callable(cond):
        return bool(cond(index, captures))
    raise TypeError(f'Invalid condition type: {type(cond)}')


def _eval_int(val, index: int, captures: dict) -> int:
    if isinstance(val, int):
        return val
    if callable(val):
        return int(val(index, captures))
    raise TypeError(f'Invalid int value type: {type(val)}')


def _eval_signal(signal, index: int, captures: dict) -> Any:
    if isinstance(signal, Waveform):
        return signal.value[index]
    if callable(signal):
        return signal(index, captures)
    raise TypeError(f'Invalid signal type: {type(signal)}')


# ---------------------------------------------------------------------------
# Waveform collection & validation
# ---------------------------------------------------------------------------


def _collect_waveforms(steps: list[Step]) -> list[Waveform]:
    """Walk the step tree and collect all referenced Waveform objects."""
    result: list[Waveform] = []
    for step in steps:
        if isinstance(step, WaitStep):
            if isinstance(step.cond, Waveform):
                result.append(step.cond)
            if isinstance(step.guard, Waveform):
                result.append(step.guard)
        elif isinstance(step, DelayStep):
            if isinstance(step.guard, Waveform):
                result.append(step.guard)
        elif isinstance(step, CaptureStep):
            if isinstance(step.signal, Waveform):
                result.append(step.signal)
        elif isinstance(step, RequireStep):
            if isinstance(step.cond, Waveform):
                result.append(step.cond)
        elif isinstance(step, LoopStep):
            if isinstance(step.until, Waveform):
                result.append(step.until)
            if isinstance(step.when, Waveform):
                result.append(step.when)
            result.extend(_collect_waveforms(step.body_template))
        elif isinstance(step, RepeatStep):
            result.extend(_collect_waveforms(step.body_template))
        elif isinstance(step, BranchStep):
            if isinstance(step.cond, Waveform):
                result.append(step.cond)
            if step.true_body:
                result.extend(_collect_waveforms(step.true_body))
            if step.false_body:
                result.extend(_collect_waveforms(step.false_body))
    return result


def _validate_waveforms(waveforms: list[Waveform]) -> None:
    """Ensure all waveforms share the same clock axis."""
    if not waveforms:
        raise PatternError('Pattern contains no Waveform references; cannot determine time axis')
    ref = waveforms[0]
    for wf in waveforms[1:]:
        if len(wf.clock) != len(ref.clock):
            raise PatternError(
                f'Waveform clock arrays have different lengths '
                f'({len(wf.clock)} vs {len(ref.clock)})'
            )
        if not np.array_equal(wf.clock, ref.clock):
            raise PatternError('Waveform clock arrays are not aligned')


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PatternEngine:
    """NFA-style engine that scans a time axis and runs pattern instances."""

    def __init__(self, pattern) -> None:
        self._steps: list[Step] = pattern._steps
        self._timeout_cycles: int | None = pattern._timeout_cycles
        self._consumed_channels: set[str] = set()

    def run(
        self,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> MatchResult:
        # ---- collect & validate waveforms ----
        waveforms = _collect_waveforms(self._steps)
        _validate_waveforms(waveforms)
        ref_wf = waveforms[0]
        n = len(ref_wf.value)

        # ---- determine trigger mode ----
        # If first step is a WaitStep, use its cond as trigger and skip it on fork.
        # Otherwise, fork an instance every cycle.
        skip_first_wait = False
        trigger: Condition | None = None
        if self._steps and isinstance(self._steps[0], WaitStep):
            trigger = self._steps[0].cond
            skip_first_wait = True

        # ---- determine scan range (index space) ----
        start_idx = 0
        end_idx = n
        if start_cycle is not None:
            start_idx = int(np.searchsorted(ref_wf.clock, start_cycle))
        if end_cycle is not None:
            end_idx = int(np.searchsorted(ref_wf.clock, end_cycle))
        start_idx = max(0, min(start_idx, n))
        end_idx = max(start_idx, min(end_idx, n))

        # ---- pre-compute static trigger positions ----
        trigger_positions: set[int] | None = None
        if trigger is not None and isinstance(trigger, Waveform):
            mask = trigger.value.astype(bool)
            trigger_positions = set(
                int(i) for i in np.where(mask[start_idx:end_idx])[0] + start_idx
            )

        # ---- main loop ----
        active: list[Instance] = []
        completed: list[Instance] = []

        for t in range(start_idx, end_idx):
            # Reset per-cycle channel consumption tracking (oldest instance wins)
            self._consumed_channels = set()

            # Tick all active instances in creation order (oldest first)
            still_active: list[Instance] = []
            for inst in active:
                self._tick(inst, t, ref_wf)
                if inst.status == IN_PROGRESS:
                    still_active.append(inst)
                else:
                    completed.append(inst)
            active = still_active

            # Fork new instance if trigger fires
            should_fork = False
            if trigger is None:
                # No trigger — fork every cycle
                should_fork = True
            elif trigger_positions is not None:
                should_fork = t in trigger_positions
            else:
                should_fork = _eval_bool(trigger, t, {})

            if should_fork:
                inst = Instance(
                    frame_stack=[Frame(steps=clone_steps(self._steps), step_idx=0)],
                    captures={},
                    start_cycle=int(ref_wf.clock[t]),
                    elapsed=1,
                    end_cycle=0,
                    status=IN_PROGRESS,
                )

                if skip_first_wait:
                    # First wait consumed by trigger — advance past it
                    inst.frame_stack[-1].step_idx = 1

                # Execute epsilon steps on the fork cycle
                self._advance(inst, t, ref_wf, blocking_budget=0)

                if inst.status == IN_PROGRESS:
                    active.append(inst)
                else:
                    completed.append(inst)

        # ---- remaining active instances → TIMEOUT ----
        last_clock = int(ref_wf.clock[end_idx - 1]) if end_idx > start_idx else 0
        for inst in active:
            inst.status = MatchStatus.TIMEOUT
            inst.end_cycle = last_clock
            completed.append(inst)

        return self._build_result(completed)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def _tick(self, inst: Instance, t: int, ref_wf: Waveform) -> None:
        """Process one clock cycle for an instance."""
        inst.elapsed += 1
        if self._timeout_cycles is not None and inst.elapsed > self._timeout_cycles:
            inst.status = MatchStatus.TIMEOUT
            inst.end_cycle = int(ref_wf.clock[t])
            return
        self._advance(inst, t, ref_wf, blocking_budget=1)

    # ------------------------------------------------------------------
    # Advance (core step processing)
    # ------------------------------------------------------------------

    def _advance(self, inst: Instance, t: int, ref_wf: Waveform, blocking_budget: int) -> None:
        """Advance the instance, consuming at most *blocking_budget* blocking steps.

        Instances are always processed oldest-first within a cycle, so the first
        instance to reach a channel WaitStep wins and marks the channel consumed
        in ``self._consumed_channels``; later instances see it as blocked.
        """
        epsilon_count = 0

        while inst.status == IN_PROGRESS:
            epsilon_count += 1
            if epsilon_count > _MAX_EPSILON:
                raise PatternError(
                    'Infinite loop detected: too many epsilon transitions in a single tick'
                )

            # ---- frame complete? ----
            if not inst.frame_stack:
                inst.status = MatchStatus.OK
                inst.end_cycle = int(ref_wf.clock[t])
                return

            frame = inst.frame_stack[-1]

            if frame.step_idx >= len(frame.steps):
                # Current frame is done — pop and let parent step re-process itself
                inst.frame_stack.pop()
                if not inst.frame_stack:
                    inst.status = MatchStatus.OK
                    inst.end_cycle = int(ref_wf.clock[t])
                    return
                continue

            # ---- process current step ----
            step = frame.steps[frame.step_idx]

            # -- blocking steps --

            if isinstance(step, WaitStep):
                # Channel check: only the first instance to reach this channel
                # this cycle may advance; later instances are blocked.
                channel_free = step.channel is None or step.channel not in self._consumed_channels
                if blocking_budget > 0 and channel_free and _eval_bool(step.cond, t, inst.captures):
                    if step.channel is not None:
                        self._consumed_channels.add(step.channel)
                    frame.step_idx += 1
                    blocking_budget -= 1
                    continue
                # Still waiting — check guard
                if step.guard is not None and not _eval_bool(step.guard, t, inst.captures):
                    inst.status = MatchStatus.REQUIRE_VIOLATED
                    inst.end_cycle = int(ref_wf.clock[t])
                    return
                return  # wait for next tick (budget exhausted or channel blocked)

            if isinstance(step, DelayStep):
                if step.remaining is None:
                    step.init_remaining(t, inst.captures)
                assert step.remaining is not None
                if step.remaining <= 0:
                    # delay(0): epsilon — advance without consuming blocking budget
                    frame.step_idx += 1
                    continue
                if blocking_budget <= 0:
                    return
                if step.guard is not None and not _eval_bool(step.guard, t, inst.captures):
                    inst.status = MatchStatus.REQUIRE_VIOLATED
                    inst.end_cycle = int(ref_wf.clock[t])
                    return
                step.remaining -= 1
                if step.remaining <= 0:
                    frame.step_idx += 1
                    blocking_budget -= 1
                    continue
                else:
                    return  # delay still counting down

            # -- epsilon steps --

            if isinstance(step, CaptureStep):
                name = step.name
                val = _eval_signal(step.signal, t, inst.captures)
                if name.endswith('[]'):
                    key = name[:-2]
                    inst.captures.setdefault(key, []).append(val)
                else:
                    inst.captures[name] = val
                frame.step_idx += 1
                continue

            if isinstance(step, RequireStep):
                if not _eval_bool(step.cond, t, inst.captures):
                    inst.status = MatchStatus.REQUIRE_VIOLATED
                    inst.end_cycle = int(ref_wf.clock[t])
                    return
                frame.step_idx += 1
                continue

            if isinstance(step, LoopStep):
                # 'until': check after body (iteration_count > 0); skip on first entry
                if step.until is not None and step.iteration_count > 0:
                    if _eval_bool(step.until, t, inst.captures):
                        frame.step_idx += 1  # exit loop
                        continue
                # 'when': check before every iteration (including first)
                elif step.when is not None:
                    if not _eval_bool(step.when, t, inst.captures):
                        frame.step_idx += 1  # skip/exit loop
                        continue
                step.iteration_count += 1
                inst.frame_stack.append(
                    Frame(
                        steps=clone_steps(step.body_template),
                        step_idx=0,
                    )
                )
                continue

            if isinstance(step, RepeatStep):
                if step.times_remaining is None:
                    step.init_remaining(t, inst.captures)
                assert step.times_remaining is not None
                if step.times_remaining <= 0:
                    frame.step_idx += 1  # all iterations done
                    continue
                step.times_remaining -= 1
                inst.frame_stack.append(
                    Frame(
                        steps=clone_steps(step.body_template),
                        step_idx=0,
                    )
                )
                continue

            if isinstance(step, BranchStep):
                if _eval_bool(step.cond, t, inst.captures):
                    body = step.true_body
                else:
                    body = step.false_body
                # Advance past BranchStep before pushing body so frame-pop
                # naturally lands on the next step without special handling.
                frame.step_idx += 1
                if body:
                    inst.frame_stack.append(
                        Frame(
                            steps=clone_steps(body),
                            step_idx=0,
                        )
                    )
                continue

            raise PatternError(f'Unknown step type: {type(step)}')

    # ------------------------------------------------------------------
    # Build MatchResult
    # ------------------------------------------------------------------

    def _build_result(self, completed: list[Instance]) -> MatchResult:
        """Pack completed instances into a MatchResult."""
        completed.sort(key=lambda i: i.start_cycle)

        start_arr = np.array([i.start_cycle for i in completed], dtype=np.int64)
        end_arr = np.array([i.end_cycle for i in completed], dtype=np.int64)
        status_arr = np.array([i.status for i in completed], dtype=np.uint8)
        duration_arr = end_arr - start_arr + 1

        clock = start_arr.copy()
        time = start_arr.copy()

        def _wf(value: np.ndarray) -> Waveform:
            return Waveform(value, clock.copy(), time.copy())

        # Build captures dict
        all_keys: set[str] = set()
        for inst in completed:
            all_keys.update(inst.captures.keys())

        captures: dict[str, Waveform] = {}
        for name in sorted(all_keys):
            vals = []
            for inst in completed:
                vals.append(inst.captures.get(name))
            try:
                arr = np.array(vals, dtype=np.int64)
            except (ValueError, TypeError, OverflowError):
                arr = np.array(vals, dtype=object)
            captures[name] = _wf(arr)

        return MatchResult(
            start=_wf(start_arr),
            end=_wf(end_arr),
            duration=_wf(duration_arr),
            status=_wf(status_arr),
            captures=captures,
        )
