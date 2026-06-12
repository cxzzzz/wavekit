from __future__ import annotations

import numpy as np

from ..waveform import Waveform
from .errors import PatternError
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


def _collect_waveforms(steps: list[Step]) -> list[Waveform]:
    """Walk the step tree and collect all referenced Waveform objects."""
    result: list[Waveform] = []
    for step in steps:
        if isinstance(step, WaitStep):
            if isinstance(step.cond, Waveform):
                result.append(step.cond)
            if isinstance(step.require, Waveform):
                result.append(step.require)
        elif isinstance(step, DelayStep):
            if isinstance(step.require, Waveform):
                result.append(step.require)
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
