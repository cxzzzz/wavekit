from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TypeVar, cast

from .matcher import Capture, CaptureKey, ExactCapture, PathStep, parse_query_path
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

    @property
    def name(self) -> str:
        """Return this node's local name, including a signal range when present."""
        if isinstance(self, Signal) and self.range is not None:
            return f'{self.base_name}{self.range}'
        return self.base_name

    @property
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
        path: list[PathStep] | str,
        node_filter: Callable[[Node, list[PathStep]], bool],
    ) -> dict[tuple[Capture, ...], Node]:
        steps = parse_query_path(path) if isinstance(path, str) else path
        if not steps:
            raise ValueError('query path must not be empty')

        results: dict[tuple[Capture, ...], Node] = {}
        step = steps[0]

        def add_match(key: tuple[Capture, ...], node: Node) -> None:
            if key in results:
                raise ValueError(
                    f'Query path step matched more than one node for key {key!r}: '
                    f'{results[key].name!r} vs {node.name!r}'
                )
            results[key] = node

        for child in self.children:
            matched = step.matcher.match(child)

            # A. A direct base-name match always wins.  In particular, an ARRAY
            # element such as ``arr[0]`` must win over interpreting ``[0]`` as
            # a range on its ARRAY parent.
            if matched is not None and matched[1] is None and node_filter(child, steps):
                capture, _ = matched
                if len(steps) == 1:
                    results[(capture,)] = (
                        child.with_range(None) if isinstance(child, Signal) else child
                    )
                else:
                    for key, node in child._match_path(steps[1:], node_filter).items():
                        add_match((capture,) + key, node)

            # B. If the current node is an ARRAY, try the same PathStep on its
            # children.  FSDB array members carry the cumulative local name
            # (``arr[0]``, ``arr[0][1]``, ...), so this makes ARRAY containers
            # transparent for hierarchy matching.
            array_matches: dict[tuple[Capture, ...], Node] = {}
            if (
                isinstance(child, Signal)
                and child.composite_type == SignalCompositeType.ARRAY
                and not (matched is not None and matched[1] is None)
            ):
                array_matches = child._match_path(steps, node_filter)
                for key, node in array_matches.items():
                    add_match(key, node)

            if array_matches:
                continue

            # C. If the current node matched through a range suffix, materialize
            # a range view only when this is the terminal path step.  A range
            # view is not a hierarchy node and therefore cannot be followed by
            # another path component.
            if matched is not None and matched[1] is not None:
                capture, selected_range = matched
                if not node_filter(child, steps):
                    continue
                if len(steps) != 1:
                    raise ValueError(
                        f'Range-selected signal {child.full_name!r} cannot be followed '
                        'by another hierarchy path component'
                    )
                if not isinstance(child, Signal):
                    raise TypeError(f'Node type {child.__class__.__name__} does not support ranges')
                results[(capture,)] = child.with_range(selected_range)

            if step.recursive:
                for key, node in child._match_path(steps, node_filter).items():
                    recursive_capture = key[0].with_prefix(child.name) if child.name else key[0]
                    add_match((recursive_capture, *key[1:]), node)

        return results

    @staticmethod
    def _normalize_match_keys(matches: dict[CaptureKey, NodeT]) -> dict[CaptureKey, NodeT]:
        """Remove non-binding exact-name captures from public result keys."""
        results: dict[CaptureKey, NodeT] = {}
        for captures, node in matches.items():
            key = tuple(
                capture
                for capture in captures
                if not (isinstance(capture, ExactCapture) and capture.definition is None)
            )
            if key in results:
                raise ValueError(
                    f'Query path matched more than one node for key {key!r}: '
                    f'{results[key].full_name!r} vs {node.full_name!r}'
                )
            results[key] = node
        return results

    def get_matched_nodes(self, path: list[PathStep] | str) -> dict[CaptureKey, Node]:
        """Return matching descendant nodes keyed by binding captures."""
        return self._normalize_match_keys(self._match_path(path, lambda _node, _steps: True))

    def get_matched_signals(self, path: list[PathStep] | str) -> dict[CaptureKey, Signal]:
        """Return matching descendant signals keyed by binding captures."""
        return cast(
            dict[CaptureKey, Signal],
            self._normalize_match_keys(
                self._match_path(
                    path, lambda node, steps: len(steps) > 1 or isinstance(node, Signal)
                )
            ),
        )

    def get_matched_scopes(self, path: list[PathStep] | str) -> dict[CaptureKey, Scope]:
        """Return matching descendant scopes keyed by binding captures."""
        return cast(
            dict[CaptureKey, Scope],
            self._normalize_match_keys(
                self._match_path(path, lambda node, _steps: isinstance(node, Scope))
            ),
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
        """Return whether this signal supports a trailing range selection."""
        return self.composite_type in (None, SignalCompositeType.ARRAY)

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
        return sum(child.width for child in self.children if isinstance(child, Signal))

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
