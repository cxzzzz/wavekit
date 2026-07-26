from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

from ..waveform import Waveform
from .errors import PatternError
from .runtime import PatternContext
from .steps import (
    BranchStep,
    CaptureStep,
    Channel,
    ChannelValue,
    Condition,
    ConsumeStep,
    DelayStep,
    IntValue,
    LoopStep,
    MessageValue,
    RepeatStep,
    RequireStep,
    SignalValue,
    Step,
    WaitStep,
)


def infer_declarative_axis(steps: list[Step]) -> Waveform | None:
    """Infer a minimal first-static-waveform scan axis hint from steps."""
    # Axis inference is intentionally minimal.  It only finds a static waveform
    # so start/end_cycle can be translated to indices; full clock alignment
    # validation remains lazy in PatternRuntime.note_waveform().
    for step in steps:
        if isinstance(step, (WaitStep, ConsumeStep)):
            if isinstance(step.cond, Waveform):
                return step.cond
            if isinstance(step.require, Waveform):
                return step.require
        elif isinstance(step, DelayStep):
            if isinstance(step.require, Waveform):
                return step.require
        elif isinstance(step, CaptureStep):
            if isinstance(step.signal, Waveform):
                return step.signal
        elif isinstance(step, RequireStep):
            if isinstance(step.cond, Waveform):
                return step.cond
        elif isinstance(step, LoopStep):
            if isinstance(step.until, Waveform):
                return step.until
            if isinstance(step.when, Waveform):
                return step.when
            waveform = infer_declarative_axis(step.body_template)
            if waveform is not None:
                return waveform
        elif isinstance(step, RepeatStep):
            waveform = infer_declarative_axis(step.body_template)
            if waveform is not None:
                return waveform
        elif isinstance(step, BranchStep):
            if isinstance(step.cond, Waveform):
                return step.cond
            if step.true_body:
                waveform = infer_declarative_axis(step.true_body)
                if waveform is not None:
                    return waveform
            if step.false_body:
                waveform = infer_declarative_axis(step.false_body)
                if waveform is not None:
                    return waveform
    return None


def compile_declarative_pattern(steps: list[Step]) -> Callable[[PatternContext], Any]:
    """Compile declarative pattern steps into a program body."""
    first_step = steps[0] if steps else None

    def eval_condition(cond: Condition, ctx: PatternContext) -> bool:
        if isinstance(cond, Waveform):
            return bool(ctx.value(cond))
        if isinstance(cond, bool):
            return cond
        if callable(cond):
            return bool(cond(ctx.index, ctx.captures))
        raise PatternError(
            f'condition must be a Waveform, bool, or callable, got {type(cond).__name__}'
        )

    def eval_signal(signal: SignalValue, ctx: PatternContext) -> Any:
        if isinstance(signal, Waveform):
            return ctx.value(signal)
        if callable(signal):
            return signal(ctx.index, ctx.captures)
        raise PatternError(f'signal must be a Waveform or callable, got {type(signal).__name__}')

    def eval_int(value: IntValue, ctx: PatternContext) -> int:
        if isinstance(value, int):
            return value
        if callable(value):
            result = value(ctx.index, ctx.captures)
            return int(result)
        raise PatternError(f'integer value must be an int or callable, got {type(value).__name__}')

    def eval_message(message: MessageValue | None, ctx: PatternContext) -> str | None:
        if message is None or isinstance(message, str):
            return message
        if callable(message):
            return str(message(ctx.index, ctx.captures))
        raise PatternError(f'message must be a str or callable, got {type(message).__name__}')

    def adapt_message(message: MessageValue | None, ctx: PatternContext):
        if message is None:
            return None
        return lambda message=message: eval_message(message, ctx)

    def adapt_channel(
        channel: ChannelValue, ctx: PatternContext
    ) -> Channel | Hashable | Callable[[], Channel | Hashable]:
        if callable(channel):
            return lambda channel=channel: channel(ctx.index, ctx.captures)
        return channel

    def run_steps(ctx: PatternContext, step_list: list[Step]) -> None:
        for step in step_list:
            if isinstance(step, WaitStep):
                ctx.wait(
                    lambda step=step: eval_condition(step.cond, ctx),
                    require=None
                    if step.require is None
                    else lambda step=step: eval_condition(step.require, ctx),
                    require_message=adapt_message(step.require_message, ctx),
                )
            elif isinstance(step, ConsumeStep):
                ctx.consume(
                    lambda step=step: eval_condition(step.cond, ctx),
                    channel=adapt_channel(step.channel, ctx),
                    require=None
                    if step.require is None
                    else lambda step=step: eval_condition(step.require, ctx),
                    require_message=adapt_message(step.require_message, ctx),
                )
            elif isinstance(step, DelayStep):
                ctx.delay(
                    eval_int(step.n, ctx),
                    require=None
                    if step.require is None
                    else lambda step=step: eval_condition(step.require, ctx),
                    require_message=adapt_message(step.require_message, ctx),
                )
            elif isinstance(step, CaptureStep):
                ctx.capture(step.name, eval_signal(step.signal, ctx), mode=step.mode)
            elif isinstance(step, RequireStep):
                ctx.require(
                    lambda step=step: eval_condition(step.cond, ctx),
                    message=adapt_message(step.message, ctx),
                )
            elif isinstance(step, BranchStep):
                if eval_condition(step.cond, ctx):
                    if step.true_body:
                        run_steps(ctx, step.true_body)
                elif step.false_body:
                    run_steps(ctx, step.false_body)
            elif isinstance(step, RepeatStep):
                count = eval_int(step.n, ctx)
                if count < 0:
                    raise PatternError(f'repeat count must be non-negative, got {count}')
                for _ in range(count):
                    run_steps(ctx, step.body_template)
            elif isinstance(step, LoopStep):
                if step.when is not None:
                    while eval_condition(step.when, ctx):
                        run_steps(ctx, step.body_template)
                elif step.until is not None:
                    while True:
                        run_steps(ctx, step.body_template)
                        if eval_condition(step.until, ctx):
                            break
                else:
                    raise PatternError("loop step must specify either 'until' or 'when'")
            else:
                raise PatternError(f'Unknown step type: {type(step).__name__}')

    def body(ctx: PatternContext) -> object:
        if isinstance(first_step, WaitStep) and first_step.require is None:
            # Treat the first unguarded wait as the trigger: candidates whose
            # first condition is false are discarded immediately instead of
            # accumulating as blocked instances. Guarded first waits cannot use
            # this shortcut because require must be evaluated while blocked.
            # Consume steps are not trigger-shortcut candidates because they
            # need FIFO/ownership arbitration through the runtime.
            if not eval_condition(first_step.cond, ctx):
                return None
        run_steps(ctx, steps)
        return ctx.OK

    return body
