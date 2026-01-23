from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np

from .pattern_parser import (
    PatternMap,
    expand_brace_pattern,
    split_by_hierarchy,
    split_by_range_expr,
)
from .value_change import value_change_to_value_array
from .waveform import Waveform


def traverse_signal(
    scope: Scope, descendant_scope_pattern_list: list[PatternMap]
) -> dict[tuple[Any, ...], str]:
    res: dict[tuple[Any, ...], str] = {}
    if len(descendant_scope_pattern_list) == 1:
        for signal in scope.signal_list:
            signal_name, signal_range = split_by_range_expr(signal)
            for k, p in descendant_scope_pattern_list[0].items():
                if p[0] == '@':
                    if match := re.fullmatch(p[1:], signal):
                        assert len(k) == 0
                        key = (match.groups(),)
                        if key in res:
                            raise Exception(f'pattern {p[1:]} match more than one signal')
                        res[key] = signal
                else:
                    p_no_range, range_expr = split_by_range_expr(p)
                    if p_no_range != signal_name:
                        continue
                    key = k
                    assert key not in res
                    final_range = range_expr or signal_range
                    res[key] = f'{signal_name}{final_range}'
                    break
    return res


def traverse_scope(
    scope: Scope, descendant_scope_pattern_list: list[PatternMap]
) -> dict[tuple[Any, ...], str]:
    res: dict[tuple[Any, ...], str] = {}
    if len(descendant_scope_pattern_list) == 0:
        return res

    for k, p in descendant_scope_pattern_list[0].items():
        if len(p) >= 2 and p[0] == '$':
            if p[1] == '$':
                module_name = p[2:]
                depth = 0
            else:
                module_name = p[1:]
                depth = 1

            module_scopes = (
                scope._module_cache[p]
                if p in scope._module_cache
                else scope.find_scope_by_module(module_name=module_name, depth=depth)
            )
            scope._module_cache[p] = module_scopes

            if len(module_scopes) == 1 and module_scopes[0] == scope:
                for sk, ss in traverse_signal(
                    scope,
                    descendant_scope_pattern_list[1:],
                ).items():
                    key = (scope.name,) + sk
                    assert key not in res
                    res[key] = f'{scope.name}.{ss}'

                for child_scope in scope.child_scope_list:
                    for ck, cs in traverse_scope(
                        child_scope,
                        descendant_scope_pattern_list[1:],
                    ).items():
                        res[(scope.name,) + ck] = f'{scope.name}.{cs}'
            else:
                for child_scope in module_scopes:
                    for ck, cs in traverse_scope(
                        child_scope,
                        descendant_scope_pattern_list[0:],
                    ).items():
                        parent_scope = child_scope.parent_scope
                        if parent_scope is None:
                            raise ValueError('parent scope is None')
                        parent_name = parent_scope.full_name(scope)
                        key = (f'{parent_name}.{ck[0]}',) + ck[1:]
                        res[key] = f'{parent_name}.{cs}'
        else:
            matched = False
            if p[0] == '@':
                if match := re.fullmatch(p[1:], scope.name):
                    matched = True
                    assert len(k) == 0
                    new_k = (match.groups(),)
            else:
                if p == scope.name:
                    matched = True
                    new_k = k

            if matched:
                for sk, ss in traverse_signal(
                    scope,
                    descendant_scope_pattern_list[1:],
                ).items():
                    key = new_k + sk
                    if key in res:
                        raise Exception(f'pattern {p} match more than one signal')
                    res[key] = f'{scope.name}.{ss}'

                for child_scope in scope.child_scope_list:
                    for ck, cs in traverse_scope(
                        child_scope,
                        descendant_scope_pattern_list[1:],
                    ).items():
                        key = new_k + ck
                        if key in res:
                            raise Exception(f'pattern {p} match more than one signal')
                        res[key] = f'{scope.name}.{cs}'
    return res


class Scope:
    def __init__(self, name: str):
        self.name = name
        self._module_cache: dict[str, list[Scope]] = {}
        self.parent_scope: Scope | None = None

    @cached_property
    @abstractmethod
    def signal_list(self) -> Sequence[str]:
        pass

    @cached_property
    @abstractmethod
    def child_scope_list(self) -> Sequence[Scope]:
        pass

    def full_name(self, root: Scope | None = None) -> str:
        ancestors: list[Scope] = []
        parent: Scope | None = self
        while parent is not None:
            ancestors.append(parent)
            if parent == root:
                break
            parent = parent.parent_scope
        return '.'.join([x.name for x in reversed(ancestors)])

    def find_scope_by_module(self, module_name: str, depth: int = 0) -> list[Scope]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def begin_time(self) -> int:
        pass

    @property
    @abstractmethod
    def end_time(self) -> int:
        pass


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
            width=width,
            signed=signed,
            signal=signal,
        )

    @abstractmethod
    def top_scope_list(self) -> Sequence[Scope]:
        pass

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
        pattern = pattern.strip()
        expanded_pattern_list: list[PatternMap] = [
            expand_brace_pattern(p) for p in split_by_hierarchy(pattern)
        ]

        expanded_clock_pattern_list: list[PatternMap] = [
            expand_brace_pattern(p) for p in split_by_hierarchy(clock_pattern)
        ]

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

        clock_patterns: dict[tuple[Any, ...], str] = {}
        for scope in self.top_scope_list():
            clock_patterns = combine_dict(
                clock_patterns,
                traverse_scope(scope, expanded_clock_pattern_list),
            )
        if len(clock_patterns) == 0:
            raise Exception(f'clock pattern {clock_pattern} can not match any signal')
        clock_full_name = list(clock_patterns.values())[0]

        signal_patterns: dict[tuple[Any, ...], str] = {}
        for scope in self.top_scope_list():
            signal_patterns = combine_dict(
                signal_patterns,
                traverse_scope(scope, expanded_pattern_list),
            )

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
            for k, s in signal_patterns.items()
        }

    @abstractmethod
    def close(self):
        pass
