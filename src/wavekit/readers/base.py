from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..scope import Scope, traverse_scope
from ..signal import Signal
from ..waveform import Waveform
from .pattern_parser import (
    PatternMap,
    expand_brace_pattern,
    split_by_hierarchy,
)
from .value_change import value_change_to_value_array


class Reader:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # exception wont be suppressed
        return False

    @abstractmethod
    def load_waveform(
        self,
        signal: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> Waveform:
        pass

    @abstractmethod
    def get_signal_width(self, signal: str) -> int:
        pass
    
    @abstractmethod
    def get_signal_range(self, signal: str) -> tuple[int]:
        pass

    @staticmethod
    def value_change_to_waveform(
        value_change: np.ndarray,
        clock_changes: np.ndarray,
        width: int | None,
        signed: bool,
        sample_on_posedge: bool = False,
        signal: str = '',
    ) -> Waveform:
        value, clock, time = value_change_to_value_array(
            value_change, clock_changes, sample_on_posedge=sample_on_posedge
        )

        return Waveform(
            value=value,
            clock=clock,
            time=time,
            signal=Signal(signal, width, signed),
        )

    @abstractmethod
    def top_scope_list(self) -> Sequence[Scope]:
        pass

    def get_matched_signals(
        self,
        pattern: str,
    ) -> dict[tuple[Any, ...], str]:
        def combine_dict(
            dict1: dict[tuple[Any, ...], str],
            dict2: dict[tuple[Any, ...], str],
        ) -> dict[tuple[Any, ...], str]:
            common_keys = set(dict1.keys()).intersection(dict2.keys())

            if common_keys:
                signal1s = [dict1[k] for k in common_keys]
                signal2s = [dict2[k] for k in common_keys]
                raise Exception(
                    'found more than one signal with the same keys: '
                    f'keys:{list(common_keys)} , signals:{signal1s + signal2s}'
                )

            return {**dict1, **dict2}

        pattern = pattern.strip()
        expanded_pattern_list: list[PatternMap] = [
            expand_brace_pattern(p) for p in split_by_hierarchy(pattern)
        ]

        matched_signals: dict[tuple[Any, ...], str] = {}
        for scope in self.top_scope_list():
            matched_signals = combine_dict(
                matched_signals,
                traverse_scope(scope, expanded_pattern_list),
            )
        return matched_signals

    def load_matched_waveforms(
        self,
        pattern: str,
        clock_pattern: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> dict[tuple[Any, ...], Waveform]:
        clock_patterns = self.get_matched_signals(clock_pattern)
        if not clock_patterns:
            raise Exception(f'clock pattern {clock_pattern} can not match any signal')
        if len(clock_patterns) > 1:
            raise Exception(
                f'clock pattern {clock_pattern} match more than one signal: {clock_patterns}'
            )
        clock_full_name = next(iter(clock_patterns.values()))

        matched_signals = self.get_matched_signals(pattern)

        return {
            k: self.load_waveform(
                s,
                clock_full_name,
                xz_value=xz_value,
                signed=signed,
                sample_on_posedge=sample_on_posedge,
                begin_time=begin_time,
                end_time=end_time,
            )
            for k, s in matched_signals.items()
        }

    @abstractmethod
    def close(self):
        pass
