from __future__ import annotations

import dataclasses
import re
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, TypeVar

from .readers.pattern_parser import (
    PatternMap,
    split_by_range_expr,
)
from .signal import Signal, SignalCompositeType

T = TypeVar('T')


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


def _prepend_scope_name(value: Any, scope_name: str) -> Any:
    if isinstance(value, Signal):
        return dataclasses.replace(value, full_name=f'{scope_name}.{value.full_name}')
    return value


def _traverse_scope_tree(
    scope: Scope,
    descendant_scope_pattern_list: list[PatternMap],
    leaf_fn: Callable[[Scope, list[PatternMap]], dict[tuple[Any, ...], T]],
) -> dict[tuple[Any, ...], T]:
    """Traverse the scope tree, calling *leaf_fn* at each matched scope node.

    Parameters
    ----------
    scope:
        The scope to match the first element of *descendant_scope_pattern_list*
        against.
    descendant_scope_pattern_list:
        Remaining scope-level patterns to match.  When this list is empty the
        function returns an empty dict (nothing left to match).
    leaf_fn:
        Called with ``(matched_scope, remaining_patterns)`` whenever a scope
        matches.  ``remaining_patterns`` is ``descendant_scope_pattern_list[1:]``.
        Callers supply this as a locally-defined function: ``match_signals``
        passes its nested ``match_signals_in_scope``; ``match_scopes`` passes
        its nested ``leaf``.

    Returns
    -------
    dict mapping expansion key tuples to values produced by *leaf_fn*.
    """

    res: dict[tuple[Any, ...], T] = {}
    if len(descendant_scope_pattern_list) == 0:
        return res

    for k, p in descendant_scope_pattern_list[0].items():
        if len(p) >= 2 and p[0] == '$':
            # Module-name pattern: $$ModName (any depth) or $ModName (direct child)
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

            remaining = descendant_scope_pattern_list[1:]

            if len(module_scopes) == 1 and module_scopes[0] == scope:
                # Current scope IS the target module — apply leaf_fn here
                for lk, lv in leaf_fn(scope, remaining).items():
                    key = (scope.name,) + lk
                    assert key not in res
                    res[key] = _prepend_scope_name(lv, scope.name)

                for child_scope in scope.child_scope_list:
                    for ck, cv in _traverse_scope_tree(child_scope, remaining, leaf_fn).items():
                        res[(scope.name,) + ck] = _prepend_scope_name(cv, scope.name)
            else:
                for child_scope in module_scopes:
                    for ck, cv in _traverse_scope_tree(
                        child_scope, descendant_scope_pattern_list, leaf_fn
                    ).items():
                        parent_scope = child_scope.parent_scope
                        if parent_scope is None:
                            raise ValueError('parent scope is None')
                        parent_name = parent_scope.full_name(scope)
                        key = (f'{parent_name}.{ck[0]}',) + ck[1:]
                        res[key] = _prepend_scope_name(cv, parent_name)
        else:
            # Exact or regex match against current scope name
            matched = False
            new_k: tuple[Any, ...]
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
                remaining = descendant_scope_pattern_list[1:]

                for lk, lv in leaf_fn(scope, remaining).items():
                    key = new_k + lk
                    if key in res:
                        raise Exception(f'pattern {p} match more than one result')
                    res[key] = _prepend_scope_name(lv, scope.name)

                for child_scope in scope.child_scope_list:
                    for ck, cv in _traverse_scope_tree(child_scope, remaining, leaf_fn).items():
                        key = new_k + ck
                        if key in res:
                            raise Exception(f'pattern {p} match more than one result')
                        res[key] = _prepend_scope_name(cv, scope.name)
    return res


def match_signals(
    scope: Scope, descendant_scope_pattern_list: list[PatternMap]
) -> dict[tuple[Any, ...], Signal]:
    """Search for signals matching a hierarchical pattern starting at *scope*.

    The last element of *descendant_scope_pattern_list* is matched against
    signal names; all preceding elements are matched against scope names.

    Returns a ``dict`` mapping expansion key tuples to :class:`~wavekit.signal.Signal`
    objects whose ``full_name`` is the complete hierarchical path.
    """

    def _match_signals_in_list(
        signals: Sequence[Signal], pattern_list: list[PatternMap]
    ) -> dict[tuple[Any, ...], Signal]:
        """Match *pattern_list* against *signals*, recursing into composite members.

        The first element of *pattern_list* is matched against signal names in
        *signals*.  When more patterns remain and the matched signal is composite
        (``signal.member_list is not None``), the function recurses into those
        members with the remaining patterns, allowing patterns to address
        struct/union members across multiple levels.

        **Array signals** require special treatment because NPI reports each
        element's name as an extension of the parent name with no ``"."``
        separator.  For example, signal ``a`` (ARRAY) has members ``a[0]``,
        ``a[1]``, which in turn have members ``a[0][0]``, ``a[0][1]``, etc.
        A user pattern like ``a[10][0]`` therefore cannot be resolved by a
        single exact-name match at the top level.  Instead, for ARRAY signals
        we check whether the signal name is a string prefix of the pattern:
        if yes, we recurse into members with the **same** pattern (not
        advancing to the next element), letting each member self-select by the
        same prefix rule until an exact match is found.
        """
        if not pattern_list:
            return {}

        def resolve_leaf(sig: Signal, sig_bare: str, range_suffix: str) -> Signal:
            if range_suffix:
                # Extract (high, low) from the innermost bracket of range_suffix,
                # e.g. "[7:0]" → (7, 0), "[3]" → (3, 3).
                m = re.search(r'\[(\d+)(?::(\d+))?\]$', range_suffix)
                h = int(m.group(1))
                low = int(m.group(2)) if m.group(2) is not None else h
                new_range: tuple[int, int] | None = (h, low)
            else:
                new_range = sig.range
            return dataclasses.replace(
                sig,
                full_name=f'{sig_bare}{range_suffix}' if range_suffix else sig.name,
                range=new_range,
            )

        res: dict[tuple[Any, ...], Signal] = {}
        for sig in signals:
            sig_bare, _ = split_by_range_expr(sig.name)
            for k, p in pattern_list[0].items():
                if p[0] == '@':
                    name_regex, range_suffix = split_by_range_expr(p[1:])
                    if match := re.fullmatch(name_regex, sig.name):
                        assert len(k) == 0
                        key = (match.groups(),)
                        if len(pattern_list) == 1:
                            assert (
                                key not in res
                            ), f'pattern {name_regex} matches more than one signal'
                            res[key] = resolve_leaf(sig, sig_bare, range_suffix)
                        elif sig.member_list is not None:
                            for ck, cv in _match_signals_in_list(
                                sig.member_list, pattern_list[1:]
                            ).items():
                                res[key + ck] = _prepend_scope_name(cv, sig.name)
                else:
                    p_bare, range_suffix = split_by_range_expr(p)
                    if p_bare != sig_bare:
                        continue
                    key = k
                    if sig.composite_type == SignalCompositeType.ARRAY and range_suffix:
                        if sig.member_list is not None:
                            for ck, cv in _match_signals_in_list(
                                sig.member_list, pattern_list
                            ).items():
                                res[key + ck] = cv
                    elif len(pattern_list) == 1:
                        assert key not in res
                        res[key] = resolve_leaf(sig, sig_bare, range_suffix)
                    elif sig.member_list is not None:
                        for ck, cv in _match_signals_in_list(
                            sig.member_list, pattern_list[1:]
                        ).items():
                            res[key + ck] = _prepend_scope_name(cv, sig.name)
                    break
        return res

    def match_signals_in_scope(
        scope: Scope, signal_pattern_list: list[PatternMap]
    ) -> dict[tuple[Any, ...], Signal]:
        return _match_signals_in_list(scope.signal_list, signal_pattern_list)

    return _traverse_scope_tree(scope, descendant_scope_pattern_list, match_signals_in_scope)


def match_scopes(
    scope: Scope, descendant_scope_pattern_list: list[PatternMap]
) -> dict[tuple[Any, ...], Scope]:
    """Search for scopes matching a hierarchical pattern starting at *scope*.

    All elements of *descendant_scope_pattern_list* are matched against scope
    names; the final matched scope is returned as the value.

    Returns a ``dict`` mapping expansion key tuples to matched :class:`Scope`
    objects.
    """

    def leaf(sc: Scope, remaining: list[PatternMap]) -> dict[tuple[Any, ...], Scope]:
        # Only return a result when all patterns are consumed (this scope is the leaf)
        if not remaining:
            return {(): sc}
        return {}

    return _traverse_scope_tree(scope, descendant_scope_pattern_list, leaf)
