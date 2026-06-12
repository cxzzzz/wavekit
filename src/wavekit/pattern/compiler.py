from __future__ import annotations

from collections.abc import Callable, Coroutine, Hashable
from typing import Any

from ..waveform import Waveform
from .errors import PatternError
from .program import _MAX_SAME_CYCLE_STEPS, ProgramContext
from .steps import (
    BranchStep,
    CaptureStep,
    Channel,
    DelayStep,
    LoopStep,
    RepeatStep,
    RequireStep,
    Step,
    WaitStep,
)

ProgramBody = Callable[[ProgramContext], Coroutine[Any, Any, object]]


def compile_declarative_pattern(pattern) -> ProgramBody:
    """Compile declarative pattern steps into an internal async program body."""
    steps = pattern._steps
    first_step = steps[0] if steps else None

    def captures(ctx: ProgramContext) -> dict[str, Any]:
        return ctx._instance.captures

    def eval_condition(cond: Any, ctx: ProgramContext) -> bool:
        if isinstance(cond, Waveform):
            return bool(ctx.value(cond))
        if callable(cond):
            try:
                return bool(cond(ctx.index, captures(ctx)))
            except PatternError:
                raise
            except Exception as exc:
                raise PatternError(f'condition callable failed: {exc}') from exc
        raise PatternError(f'condition must be a Waveform or callable, got {type(cond).__name__}')

    def eval_signal(signal: Any, ctx: ProgramContext) -> Any:
        if isinstance(signal, Waveform):
            return ctx.value(signal)
        if callable(signal):
            try:
                return signal(ctx.index, captures(ctx))
            except PatternError:
                raise
            except Exception as exc:
                raise PatternError(f'signal callable failed: {exc}') from exc
        raise PatternError(f'signal must be a Waveform or callable, got {type(signal).__name__}')

    def eval_int(value: Any, ctx: ProgramContext) -> int:
        if isinstance(value, bool):
            raise PatternError('integer value must not be bool')
        if isinstance(value, int):
            return value
        if callable(value):
            try:
                result = value(ctx.index, captures(ctx))
            except PatternError:
                raise
            except Exception as exc:
                raise PatternError(f'integer callable failed: {exc}') from exc
            if isinstance(result, bool):
                raise PatternError('integer callable must not return bool')
            try:
                return int(result)
            except (TypeError, ValueError) as exc:
                raise PatternError(
                    'integer callable must return an int-compatible value, '
                    f'got {type(result).__name__}'
                ) from exc
        raise PatternError(f'integer value must be an int or callable, got {type(value).__name__}')

    def eval_channel(channel: Any, ctx: ProgramContext) -> Channel | Hashable | None:
        if channel is None:
            return None
        if callable(channel) and not isinstance(channel, Channel):
            try:
                channel = channel(ctx.index, captures(ctx))
            except PatternError:
                raise
            except Exception as exc:
                raise PatternError(f'channel callable failed: {exc}') from exc
        if channel is None or isinstance(channel, Channel) or isinstance(channel, Hashable):
            return channel
        raise PatternError(
            f'channel must be a Channel or hashable key, got {type(channel).__name__}'
        )

    def raise_infinite_loop() -> None:
        raise PatternError('Infinite loop detected: too many epsilon transitions in a single tick')

    async def run_steps(
        ctx: ProgramContext,
        step_list: list[Step],
        first_wait_ready: bool | None = None,
    ) -> None:
        for step in step_list:
            if isinstance(step, WaitStep):
                wait_step = step
                channel = eval_channel(step.channel, ctx)
                if channel is None:
                    channel = step._auto_channel
                cached_index = ctx.index
                cached_ready = first_wait_ready

                def condition(
                    step: WaitStep = wait_step,
                    cached_index: int = cached_index,
                ) -> bool:
                    nonlocal cached_ready
                    if cached_ready is not None and ctx.index == cached_index:
                        ready = cached_ready
                        cached_ready = None
                        return ready
                    return eval_condition(step.cond, ctx)

                await ctx._wait_internal(
                    condition,
                    channel=channel,
                    tick=step.tick,
                    require=None
                    if step.require is None
                    else lambda step=step: eval_condition(step.require, ctx),
                )
                first_wait_ready = None
            elif isinstance(step, DelayStep):
                await ctx._delay_internal(
                    eval_int(step.n, ctx),
                    require=None
                    if step.require is None
                    else lambda step=step: eval_condition(step.require, ctx),
                )
            elif isinstance(step, CaptureStep):
                ctx.capture(step.name, eval_signal(step.signal, ctx), mode=step.mode)
            elif isinstance(step, RequireStep):
                ctx.require(lambda step=step: eval_condition(step.cond, ctx))
            elif isinstance(step, BranchStep):
                if eval_condition(step.cond, ctx):
                    if step.true_body:
                        await run_steps(ctx, step.true_body)
                elif step.false_body:
                    await run_steps(ctx, step.false_body)
            elif isinstance(step, RepeatStep):
                count = eval_int(step.n, ctx)
                if count < 0:
                    raise PatternError(f'repeat count must be non-negative, got {count}')
                for _ in range(count):
                    await run_steps(ctx, step.body_template)
            elif isinstance(step, LoopStep):
                iterations = 0
                if step.when is not None:
                    while eval_condition(step.when, ctx):
                        iterations += 1
                        if iterations > _MAX_SAME_CYCLE_STEPS:
                            raise_infinite_loop()
                        await run_steps(ctx, step.body_template)
                elif step.until is not None:
                    while True:
                        iterations += 1
                        if iterations > _MAX_SAME_CYCLE_STEPS:
                            raise_infinite_loop()
                        await run_steps(ctx, step.body_template)
                        if eval_condition(step.until, ctx):
                            break
                else:
                    raise PatternError("loop step must specify either 'until' or 'when'")
            else:
                raise PatternError(f'Unknown step type: {type(step).__name__}')

    async def body(ctx: ProgramContext) -> object:
        first_wait_ready = None
        if isinstance(first_step, WaitStep) and first_step.require is None:
            first_wait_ready = eval_condition(first_step.cond, ctx)
            if not first_wait_ready:
                return None
        await run_steps(ctx, steps, first_wait_ready=first_wait_ready)
        return ctx.OK

    return body
