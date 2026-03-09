from __future__ import annotations

import ast
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..scope import Scope, traverse_scope
from ..signal import Signal
from ..waveform import Waveform
from .expr_parser import extract_wave_paths
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

    # ------------------------------------------------------------------
    # High-level expression APIs
    # ------------------------------------------------------------------

    def eval(
        self,
        expr: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        mode: Literal['single', 'zip'] = 'single',
    ) -> Waveform | dict[tuple[Any, ...], Waveform]:
        """Evaluate a waveform expression string.

        Wave signal paths embedded in *expr* are automatically extracted,
        loaded, and substituted before evaluating the resulting Python
        arithmetic expression.

        Parameters
        ----------
        expr:
            Expression string.
            Single mode example: ``"tb.dut.w_ptr[3:0] - tb.dut.r_ptr[3:0]"``.
            Zip mode example: ``"tb.dut.fifo_{0..3}.w_ptr[3:0] - tb.dut.fifo_{0..3}.r_ptr[3:0]"``.
        clock:
            Clock signal used for all waveform loads.
        mode:
            ``'single'`` (default) — every path must match exactly one
            signal; returns a single :class:`Waveform`.
            ``'zip'`` — paths matching multiple signals must all share the
            same set of pattern keys; single-match paths are broadcast;
            returns ``dict[tuple, Waveform]``.
        """
        substituted, path_entries = extract_wave_paths(expr)

        load_kwargs: dict[str, Any] = dict(
            clock=clock,
            xz_value=xz_value,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
        )

        # Resolve each path to its matched signal(s)
        matched_per_path: list[tuple[str, str, dict[tuple[Any, ...], str]]] = []
        for placeholder, path in path_entries:
            matched = self.get_matched_signals(path)
            if not matched:
                raise ValueError(f"path '{path}' matched no signals")
            matched_per_path.append((placeholder, path, matched))

        if mode == 'single':
            for placeholder, path, matched in matched_per_path:
                if len(matched) > 1:
                    raise ValueError(
                        f"path '{path}' matched {len(matched)} signals in mode='single',"
                        f" use mode='zip'. Matched: {list(matched.values())}"
                    )
            ns: dict[str, Any] = {
                placeholder: self.load_waveform(next(iter(matched.values())), **load_kwargs)
                for placeholder, _, matched in matched_per_path
            }
            try:
                code = compile(ast.parse(substituted, mode='eval'), '<eval_expr>', 'eval')
                return eval(code, {'__builtins__': {}}, ns)  # noqa: S307
            except Exception as exc:
                raise ValueError(
                    f"failed to evaluate expression '{expr}' (substituted: '{substituted}')"
                ) from exc

        elif mode == 'zip':
            multi_paths = [
                (ph, p, m) for ph, p, m in matched_per_path if len(m) > 1
            ]
            single_paths = [
                (ph, p, m) for ph, p, m in matched_per_path if len(m) == 1
            ]

            if multi_paths:
                # All multi-match paths must share identical key sets
                ref_ph, ref_p, ref_matched = multi_paths[0]
                ref_keys = set(ref_matched.keys())
                for ph, p, matched in multi_paths[1:]:
                    if set(matched.keys()) != ref_keys:
                        raise ValueError(
                            f"inconsistent match keys between paths: "
                            f"'{ref_p}' has keys {sorted(ref_keys)}, "
                            f"'{p}' has keys {sorted(matched.keys())}"
                        )
                zip_keys: list[tuple[Any, ...]] = list(ref_keys)
            else:
                # All paths are single-match; degenerate to single behaviour
                zip_keys = [()]

            # Pre-load broadcast waveforms (single-match paths)
            broadcast_ns: dict[str, Waveform] = {
                placeholder: self.load_waveform(next(iter(matched.values())), **load_kwargs)
                for placeholder, _, matched in single_paths
            }

            result: dict[tuple[Any, ...], Waveform] = {}
            try:
                code = compile(ast.parse(substituted, mode='eval'), '<eval_expr>', 'eval')
            except SyntaxError as exc:
                raise ValueError(
                    f"invalid expression '{expr}' (substituted: '{substituted}')"
                ) from exc

            for key in zip_keys:
                ns = dict(broadcast_ns)
                for placeholder, _, matched in multi_paths:
                    ns[placeholder] = self.load_waveform(matched[key], **load_kwargs)
                try:
                    result[key] = eval(code, {'__builtins__': {}}, ns)  # noqa: S307
                except Exception as exc:
                    raise ValueError(
                        f"failed to evaluate expression '{expr}' for key {key}"
                    ) from exc

            return result

        else:
            raise ValueError(f"unknown mode '{mode}', expected 'single' or 'zip'")
