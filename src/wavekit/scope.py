from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Any

from .readers.pattern_parser import (
    PatternMap,
    split_by_range_expr,
)
from .signal import Signal


class Scope:
    """A node in the hierarchical scope tree of a waveform file.

    Waveform formats (VCD, FSDB) organise signals in a tree of named scopes that
    mirrors the RTL module hierarchy.  Each ``Scope`` node exposes the signals
    declared at that level and the child scopes one level below.

    The concrete implementations (``VcdScope``, ``FsdbScope``) are created
    automatically by the corresponding :class:`~wavekit.readers.base.Reader`
    and are returned via :meth:`~wavekit.readers.base.Reader.top_scope_list`.
    You typically traverse scopes to resolve pattern-matched signal paths; for
    direct signal loading use the Reader methods instead.

    Attributes
    ----------
    name:
        The local (non-qualified) scope name, e.g. ``"dut"``.
    parent_scope:
        Parent ``Scope`` node, or ``None`` for top-level scopes.

    Abstract properties (implemented by subclasses)
    -------------------------------------------------
    signal_list:
        All :class:`~wavekit.signal.Signal` objects declared in this scope
        (not recursively).
    child_scope_list:
        Direct child :class:`Scope` nodes.
    begin_time / end_time:
        Simulation time boundaries for this scope (inherited from the file).
    """

    def __init__(self, name: str):
        self.name = name
        self._module_cache: dict[str, list[Scope]] = {}
        self.parent_scope: Scope | None = None

    @cached_property
    @abstractmethod
    def signal_list(self) -> Sequence[Signal]:
        pass

    @cached_property
    @abstractmethod
    def child_scope_list(self) -> Sequence[Scope]:
        pass

    def full_name(self, root: Scope | None = None) -> str:
        """Return the fully-qualified dotted name of this scope.

        Walks up the parent chain and joins names with ``"."``.

        Parameters
        ----------
        root:
            If provided, stop ascending at this ancestor scope so the returned
            name is relative to *root* rather than the absolute top.
        """
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


def traverse_signal(
    scope: Scope, descendant_scope_pattern_list: list[PatternMap]
) -> dict[tuple[Any, ...], str]:
    res: dict[tuple[Any, ...], str] = {}
    if len(descendant_scope_pattern_list) == 1:
        for signal in scope.signal_list:
            signal_name, signal_range = split_by_range_expr(signal.name)
            for k, p in descendant_scope_pattern_list[0].items():
                if p[0] == '@':
                    if match := re.fullmatch(p[1:], signal.name):
                        assert len(k) == 0
                        key = (match.groups(),)
                        if key in res:
                            raise Exception(f'pattern {p[1:]} match more than one signal')
                        res[key] = signal.name
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
