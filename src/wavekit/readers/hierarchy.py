from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import TypeVar, cast

from .matcher import (
    Capture,
    CaptureKey,
    ExactCapture,
    Matcher,
    PathStep,
    WildcardCapture,
    WildcardMatcher,
    parse_query_path,
)
from .range import Range

NodeT = TypeVar('NodeT', bound='Node')


class SignalCompositeType(Enum):
    """Composite signal type as reported by a waveform backend."""

    ARRAY = 'array'
    STRUCT = 'struct'
    UNION = 'union'
    TAGGED_UNION = 'tagged_union'
    RECORD = 'record'


@dataclass(frozen=True, eq=False)
class Node(ABC):
    """An immutable node in a waveform-file hierarchy."""

    base_name: str
    parent: Node | None
    _recursive_match_cache: dict[Matcher, tuple[Node, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def name(self) -> str:
        """Return this node's local name, including a signal range when present."""
        if isinstance(self, Signal) and self.range is not None:
            return f'{self.base_name}{self.range}'
        return self.base_name

    @property
    def is_range_selectable(self) -> bool:
        """Return whether this node supports a trailing bit-range selection."""
        return False

    @cached_property
    def full_name(self) -> str:
        """Return this node's fully-qualified real hierarchy name."""
        parent = self.parent
        while isinstance(parent, Signal) and parent.composite_type == SignalCompositeType.ARRAY:
            parent = parent.parent
        return f'{parent.full_name}.{self.name}' if parent is not None else self.name

    @property
    @abstractmethod
    def children(self) -> tuple[Node, ...]:
        """Return this node's direct children."""

    def _match_path(
        self,
        path: list[PathStep],
        node_filter: Callable[[Node, list[PathStep]], bool],
        *,
        step_anchor_node: Node | None = None,
    ) -> dict[tuple[Capture, ...], Node]:
        steps = path
        if not steps:
            raise ValueError('query path must not be empty')

        step = steps[0]
        step_anchor_node = self if step_anchor_node is None else step_anchor_node
        results: dict[tuple[Capture, ...], Node] = {}
        recursive_cache_key = (
            step.matcher
            if step.recursive and not isinstance(step.matcher, WildcardMatcher)
            else None
        )

        def add_match(key: tuple[Capture, ...], node: Node) -> None:
            if key in results:
                raise ValueError(
                    f'Query path step matched more than one node for key {key!r}: '
                    f'{results[key].name!r} vs {node.name!r}'
                )
            results[key] = node

        if (
            recursive_cache_key is not None
            and (cached_parents := self._recursive_match_cache.get(recursive_cache_key)) is not None
        ):
            replay_step = dataclasses.replace(step, recursive=False)
            replay_steps = [replay_step, *steps[1:]]
            for parent in cached_parents:
                for key, node in parent._match_path(
                    replay_steps,
                    node_filter,
                    step_anchor_node=step_anchor_node,
                ).items():
                    add_match(key, node)
            return results

        candidate_parents: dict[Node, None] = {}

        for child in self.children:
            matched = step.matcher.match(child, anchor_node=step_anchor_node)
            if recursive_cache_key is not None and matched is not None:
                candidate_parents.setdefault(self, None)

            # Consume both direct and range-view matches uniformly. The matcher
            # has already resolved suffix/range semantics; a different matched
            # node means it produced a terminal range view.
            if matched is not None:
                assert matched.node is not None
                matched_node = matched.node
                if matched_node is not child and len(steps) != 1:
                    raise ValueError(
                        f'Range-selected signal {child.full_name!r} cannot be followed '
                        'by another hierarchy path component'
                    )
                if node_filter(matched_node, steps):
                    if len(steps) == 1:
                        add_match((matched,), matched_node)
                    else:
                        for key, node in child._match_path(steps[1:], node_filter).items():
                            add_match((matched, *key), node)

            # Both recursive steps and transparent ARRAY containers search a
            # child with the current step. ARRAY descent preserves ``arr[0]`` as
            # a concrete member match; recursive descent extends that same search
            # to all ordinary hierarchy children. Run the shared descent once.
            if step.recursive or (
                isinstance(child, Signal)
                and child.composite_type == SignalCompositeType.ARRAY
                and not (matched is not None and matched.node is child)
            ):
                for key, node in child._match_path(
                    steps,
                    node_filter,
                    step_anchor_node=step_anchor_node,
                ).items():
                    add_match(key, node)

                # Cache entries are replay parents, so a recursive parent must
                # inherit the candidate parents discovered in this child subtree.
                if recursive_cache_key is not None:
                    for parent in child._recursive_match_cache[recursive_cache_key]:
                        candidate_parents.setdefault(parent, None)

        if recursive_cache_key is not None:
            self._recursive_match_cache[recursive_cache_key] = tuple(candidate_parents)

        return results

    def _match_query_path(
        self,
        path: str,
        node_filter: Callable[[Node, list[PathStep]], bool],
    ) -> dict[CaptureKey, Node]:
        """Execute a public query path and return its public capture keys."""
        raw_steps = parse_query_path(path)

        # Lower public ``**.matcher`` syntax into the recursive execution form.
        match_steps: list[PathStep] = []
        steps = iter(raw_steps)
        for step in steps:
            if not (step.recursive and isinstance(step.matcher, WildcardMatcher)):
                match_steps.append(step)
                continue

            next_step = next(steps, None)
            if next_step is None:
                match_steps.append(step)
                break

            match_steps.append(
                dataclasses.replace(next_step, recursive=True, native_recursive=False)
            )

        internal_matches = self._match_path(match_steps, node_filter)

        # Restore public ``**`` captures and remove non-binding exact captures.
        results: dict[CaptureKey, Node] = {}
        for captures, node in internal_matches.items():
            key: list[Capture] = []
            for step, capture in zip(match_steps, captures):
                restore_wildcard = (
                    step.recursive
                    and not step.native_recursive
                    and not isinstance(step.matcher, WildcardMatcher)
                )
                if not restore_wildcard:
                    key.append(capture)
                    continue

                # Split the internal recursive capture at the matched node's logical
                # parent: ``**`` consumes anchor -> parent, and the matcher consumes
                # parent -> node. Physical ARRAY containers are transparent here.
                assert capture.node is not None
                parent = capture.node.parent
                while (
                    isinstance(parent, Signal)
                    and parent.composite_type == SignalCompositeType.ARRAY
                ):
                    parent = parent.parent

                if parent is None:
                    raise ValueError(f'Recursive match has no parent: {capture.node.full_name!r}')
                key.extend(
                    (
                        WildcardCapture(anchor_node=capture.anchor_node, node=parent),
                        capture.with_anchor_node(parent),
                    )
                )

            public_key = tuple(
                capture
                for capture in key
                if not (isinstance(capture, ExactCapture) and capture.definition is None)
            )
            if public_key in results:
                raise ValueError(
                    f'Query path matched more than one node for key {public_key!r}: '
                    f'{results[public_key].full_name!r} vs {node.full_name!r}'
                )
            results[public_key] = node
        return results

    def get_matched_nodes(self, path: str) -> dict[CaptureKey, Node]:
        """Return matching descendant nodes keyed by binding captures."""
        return self._match_query_path(path, lambda _node, _steps: True)

    def get_matched_signals(self, path: str) -> dict[CaptureKey, Signal]:
        """Return matching descendant signals keyed by binding captures."""
        return cast(
            dict[CaptureKey, Signal],
            self._match_query_path(
                path,
                lambda node, remaining_steps: len(remaining_steps) > 1 or isinstance(node, Signal),
            ),
        )

    def get_matched_scopes(self, path: str) -> dict[CaptureKey, Scope]:
        """Return matching descendant scopes keyed by binding captures."""
        return cast(
            dict[CaptureKey, Scope],
            self._match_query_path(path, lambda node, _steps: isinstance(node, Scope)),
        )


@dataclass(frozen=True, eq=False)
class Scope(Node):
    """A real hierarchy scope."""


@dataclass(frozen=True, eq=False)
class Signal(Node):
    """An immutable signal view, optionally narrowed by a selection range."""

    range: Range | None
    composite_type: SignalCompositeType | None = None
    native_range: Range | None = None

    def __post_init__(self) -> None:
        if self.composite_type not in (SignalCompositeType.ARRAY, None) and self.range is not None:
            raise ValueError(
                f"Signal '{self.full_name}' has composite type {self.composite_type} "
                f'but also has a range {self.range}. Only array signals can have a range.'
            )

    @property
    def is_range_selectable(self) -> bool:
        """Return whether this signal supports a trailing bit-range selection."""
        return self.composite_type is None

    @property
    def is_leaf(self) -> bool:
        """Return whether this signal has no composite children."""
        return self.composite_type is None

    def _calc_width(self, selected_range: Range | None) -> int:
        if self.composite_type is None:
            return abs(selected_range.start - selected_range.end) + 1 if selected_range else 1
        if self.composite_type == SignalCompositeType.ARRAY:
            if selected_range is None:
                raise ValueError(f"Array signal '{self.full_name}' has no selected range")

            child_widths = [child.width for child in self.children if isinstance(child, Signal)]
            if not child_widths:
                raise ValueError(f"Array signal '{self.full_name}' has no children")
            if any(child_widths[0] != width for width in child_widths[1:]):
                raise ValueError(
                    f"Array signal '{self.full_name}' has children with different widths: "
                    f'{child_widths}'
                )
            return (abs(selected_range.start - selected_range.end) + 1) * child_widths[0]

        if selected_range is not None:
            raise ValueError(
                f"Signal '{self.full_name}' has composite type {self.composite_type} "
                f'but also has a range {selected_range}. Only array signals can have a range.'
            )

        child_widths = [child.width for child in self.children if isinstance(child, Signal)]
        if not child_widths:
            raise ValueError(f"Composite signal '{self.full_name}' has no children")
        if self.composite_type == SignalCompositeType.UNION:
            return max(child_widths)
        return sum(child_widths)

    @cached_property
    def native_width(self) -> int:
        """Return the width of the complete signal before range selection."""
        return self._calc_width(self.native_range)

    @cached_property
    def width(self) -> int:
        """Return the width of the current selected signal view."""
        return self._calc_width(self.range)

    def with_range(self, selected_range: Range | None) -> Signal:
        """Return a view with *selected_range*, or restore the native range for ``None``."""
        if selected_range is None:
            return dataclasses.replace(self, range=self.native_range)

        if not self.is_range_selectable:
            raise TypeError(f'Signal {self.full_name!r} does not support range selection')

        if self.native_range is None:
            if self.native_width != 1 or selected_range != Range(0, 0):
                raise ValueError(
                    f'Signal {self.name} has no native range; only scalar bit [0] is valid'
                )
            return dataclasses.replace(self, range=selected_range)

        if (selected_range.start - selected_range.end) * (
            self.native_range.start - self.native_range.end
        ) < 0:
            raise ValueError(
                f'Selected range {selected_range} has opposite direction from native range '
                f'{self.native_range} for signal {self.full_name}'
            )

        native_max = max(self.native_range.end, self.native_range.start)
        native_min = min(self.native_range.end, self.native_range.start)
        selected_max = max(selected_range.end, selected_range.start)
        selected_min = min(selected_range.end, selected_range.start)
        if selected_max > native_max or selected_min < native_min:
            raise ValueError(
                f'Range [{selected_range.start}:{selected_range.end}] out of native range '
                f"[{self.native_range.start}:{self.native_range.end}] for signal '{self.full_name}'"
            )

        return dataclasses.replace(self, range=selected_range)
