"""Query path matcher types and parser.

This module implements wavekit's signal/scope query-path syntax:

* ``name`` — exact local-name match
* ``{a,b}``, ``{0..3}``, ``{0..7..2}`` — brace expansion
* ``*`` / ``**`` — single-level / recursive wildcard
* ``/regex/`` — regular-expression match; legacy ``@regex`` is also accepted
* ``$ModName`` / ``$$ModName`` — direct / recursive module-definition match

Matchers deliberately depend only on the small node surface they consume
(``base_name``, ``definition``, and ``is_range_selectable``). The hierarchy
module imports this parser, so importing concrete hierarchy types here would
create a cycle.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from .hierarchy import Node, Signal

from .range import Range

MatchTarget = Literal['name', 'definition']
_RANGE_BRACE = re.compile(r'\{(\d+)\.\.(\d+)(?:\.\.(\d+))?\}')
_LIST_BRACE = re.compile(r'\{([^{}]+)\}')
_TRAILING_RANGE = re.compile(r'(\[(\d+)(?::(\d+))?\])$')


def split_trailing_range(segment: str) -> tuple[str, str, Range | None]:
    """Split a trailing ``[N]`` or ``[H:L]`` from a path segment."""
    matched = _TRAILING_RANGE.search(segment)
    if matched is None:
        return segment, '', None
    suffix = matched.group(1)
    start = int(matched.group(2))
    end = int(matched.group(3)) if matched.group(3) is not None else start
    return segment[: matched.start()], suffix, Range(start, end)


@dataclass(frozen=True)
class Capture:
    """Base class for typed bindings returned in a capture tuple.

    Matcher-specific code may create a partial Capture with no node context.
    ``Matcher.match()`` completes it before it can escape the matcher layer.
    """

    anchor_node: Node | None = field(default=None, repr=False, compare=False)
    node: Node | None = field(default=None, repr=False, compare=False)
    definition: str | None = None
    path: str | None = field(init=False)

    def __post_init__(self) -> None:
        if self.anchor_node is None or self.node is None:
            object.__setattr__(self, 'path', None)
            return
        start_name = self.anchor_node.full_name
        end_name = self.node.full_name
        suffix = end_name[len(start_name) :]
        object.__setattr__(self, 'path', suffix.removeprefix('.'))

    def finalize(
        self,
        *,
        anchor_node: Node,
        node: Node,
        definition: str | None = None,
    ) -> Capture:
        """Complete a partial Capture with its matched node context."""
        return replace(
            self,
            anchor_node=anchor_node,
            node=node,
            definition=definition,
        )

    def with_anchor_node(self, anchor_node: Node) -> Capture:
        """Return a complete Capture rebased to *anchor_node*."""
        if self.node is None:
            raise ValueError('Cannot rebase a Capture without a matched node')
        return self.finalize(
            anchor_node=anchor_node,
            node=self.node,
            definition=self.definition,
        )


@dataclass(frozen=True)
class Matcher(ABC):
    """Match a single hierarchy node and create a typed Capture."""

    target: MatchTarget
    pattern: str = ''
    suffix: str = ''
    range: Range | None = None

    def match(self, node: Node, *, anchor_node: Node) -> Capture | None:
        """Match *node* and return a complete Capture, if successful."""
        if self.target == 'definition':
            if not hasattr(node, 'definition'):
                raise ValueError(
                    'Cannot use module matcher ($/$$) on a backend without definition '
                    'matching support; use FSDB. '
                    f'(node: {node.name!r})'
                )
            candidate_name = cast(Any, node).definition
        else:
            candidate_name = node.base_name

        if candidate_name is not None and candidate_name.endswith(self.suffix):
            partial = self._match_name(candidate_name.removesuffix(self.suffix))
            if partial is not None:
                return partial.finalize(
                    anchor_node=anchor_node,
                    node=node,
                    definition=candidate_name if self.target == 'definition' else None,
                )

        if self.target != 'definition' and self.range is not None and node.is_range_selectable:
            partial = self._match_name(node.base_name)
            if partial is not None:
                selected_node = cast('Signal', node).with_range(self.range)
                return partial.finalize(
                    anchor_node=anchor_node,
                    node=selected_node,
                )
        return None

    @abstractmethod
    def _match_name(self, name: str) -> Capture | None:
        """Return a matcher-specific partial Capture for *name*."""


@dataclass(frozen=True)
class ExactCapture(Capture):
    """Exact-name binding; only module-definition matches are public."""


@dataclass(frozen=True)
class ExactMatcher(Matcher):
    """Exact name or module-definition matcher."""

    def _match_name(self, name: str) -> Capture | None:
        return ExactCapture() if self.pattern == name else None


@dataclass(frozen=True)
class BraceCapture(Capture):
    """Binding produced by brace expansion; ``groups`` stores brace values."""

    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class BraceMatcher(Matcher):
    """Brace expansion matcher delegated to exact-name matching."""

    matchers: dict[tuple[str, ...], ExactMatcher] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'matchers',
            {
                key: ExactMatcher(self.target, expanded)
                for key, expanded in self.expand(self.pattern).items()
            },
        )

    def _match_name(self, name: str) -> Capture | None:
        for key, matcher in self.matchers.items():
            if matcher._match_name(name) is not None:
                return BraceCapture(groups=key)
        return None

    @staticmethod
    def expand(segment: str) -> dict[tuple[str, ...], str]:
        """Expand ``{a,b}``, ``{0..3}``, and ``{0..7..2}`` syntax."""
        parts: list[str] = []
        index = 0
        while index < len(segment):
            if segment[index] == '{':
                end = segment.find('}', index)
                if end == -1:
                    raise ValueError(f'Unmatched brace in: {segment!r}')
                parts.append(segment[index : end + 1])
                index = end + 1
            else:
                end = index
                while end < len(segment) and segment[end] != '{':
                    end += 1
                parts.append(segment[index:end])
                index = end

        def expand_one(part: str) -> dict[tuple[str, ...], str]:
            if matched := _RANGE_BRACE.fullmatch(part):
                start, end = int(matched.group(1)), int(matched.group(2))
                step = int(matched.group(3)) if matched.group(3) else 1
                return {(str(value),): str(value) for value in range(start, end + 1, step)}
            if matched := _LIST_BRACE.fullmatch(part):
                return {(value,): value for value in matched.group(1).split(',')}
            return {(): part}

        expanded: dict[tuple[str, ...], str] = {(): ''}
        for part in parts:
            expanded = {
                prefix_key + key: prefix_value + value
                for prefix_key, prefix_value in expanded.items()
                for key, value in expand_one(part).items()
            }
        return expanded


@dataclass(frozen=True)
class WildcardCapture(Capture):
    """Binding produced by ``*`` or ``**`` wildcard matching."""


@dataclass(frozen=True)
class WildcardMatcher(Matcher):
    """Single-level (``*``) or recursive (``**``) wildcard."""

    def _match_name(self, name: str) -> Capture | None:
        assert self.target != 'definition'
        return WildcardCapture()


@dataclass(frozen=True)
class RegexCapture(Capture):
    """Binding produced by regex matching; ``groups`` stores regex groups."""

    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegexMatcher(Matcher):
    """Regular-expression matcher for local names or module definitions."""

    regex: re.Pattern[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'regex', re.compile(self.pattern))

    def _match_name(self, name: str) -> Capture | None:
        matched = self.regex.fullmatch(name)
        return RegexCapture(groups=matched.groups()) if matched is not None else None


@dataclass(frozen=True)
class PathStep:
    """One parsed query-path segment.

    Attributes
    ----------
    matcher:
        Matcher for this segment.
    recursive:
        Whether this segment should traverse recursively through descendants.
    native_recursive:
        Whether a recursive step came directly from ``$$`` rather than from
        lowering a preceding ``**`` wildcard.
    """

    matcher: Matcher
    recursive: bool = False
    native_recursive: bool = False


def parse_query_path(path: str) -> list[PathStep]:
    """Parse a query path string into ``PathStep`` objects."""
    path = path.strip()

    def find_regex_close(text: str) -> int:
        """Find the closing slash of a regex literal beginning at offset zero."""
        index = 1
        while index < len(text):
            if text[index] == '\\' and index + 1 < len(text):
                index += 2
            elif text[index] == '/':
                return index
            else:
                index += 1
        return -1

    def split_hierarchy(text: str) -> list[str]:
        """Split on dots while preserving brace and regex contents."""
        parts: list[str] = []
        current: list[str] = []
        brace_depth = 0
        index = 0
        while index < len(text):
            char = text[index]
            if char == '\\' and index + 1 < len(text):
                current.extend((char, text[index + 1]))
                index += 2
                continue
            if char == '/' and brace_depth == 0:
                close = find_regex_close(text[index:])
                if close == -1:
                    current.append(text[index:])
                    break
                current.append(text[index : index + close + 1])
                index += close + 1
                continue
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth = max(0, brace_depth - 1)
            elif char == '.' and brace_depth == 0:
                parts.append(''.join(current))
                current = []
                index += 1
                continue
            current.append(char)
            index += 1
        if current:
            parts.append(''.join(current))
        return parts

    def parse_segment(segment: str) -> PathStep:
        if segment == '*':
            return PathStep(matcher=WildcardMatcher(target='name'))
        if segment == '**':
            return PathStep(matcher=WildcardMatcher(target='name'), recursive=True)

        target: MatchTarget = 'name'
        recursive = False
        native_recursive = False
        rest = segment
        if segment.startswith('$$'):
            target = 'definition'
            recursive = True
            native_recursive = True
            rest = segment[2:]
        elif segment.startswith('$'):
            target = 'definition'
            rest = segment[1:]

        if target == 'definition':
            if rest.startswith('@'):
                prefix = '$$' if native_recursive else '$'
                raise ValueError(
                    f'{prefix}@regex not supported; use {prefix}/regex/ instead: {segment!r}'
                )
            if rest in ('*', '**'):
                prefix = '$$' if native_recursive else '$'
                raise ValueError(
                    f'{prefix}* not supported; use ** for recursive name matching: {segment!r}'
                )

        matcher: Matcher
        if rest.startswith('@'):
            pattern, suffix, selected_range = split_trailing_range(rest[1:])
            matcher = RegexMatcher(
                target=target,
                pattern=pattern,
                suffix=suffix,
                range=selected_range,
            )
        elif rest.startswith('/'):
            close = find_regex_close(rest)
            if close == -1:
                raise ValueError(f'Unclosed regex literal: {rest!r}')
            trailing = rest[close + 1 :]
            base, suffix, selected_range = split_trailing_range(trailing)
            if base:
                raise ValueError(f'Regex literal has trailing content {trailing!r}')
            pattern = rest[1:close]
            matcher = RegexMatcher(
                target=target,
                pattern=pattern,
                suffix=suffix,
                range=selected_range,
            )
        else:
            pattern, suffix, selected_range = split_trailing_range(rest)
            matcher_type = BraceMatcher if '{' in pattern else ExactMatcher
            matcher = matcher_type(
                target=target,
                pattern=pattern,
                suffix=suffix,
                range=selected_range,
            )

        return PathStep(
            matcher=matcher,
            recursive=recursive,
            native_recursive=native_recursive,
        )

    raw_steps = [parse_segment(segment) for segment in split_hierarchy(path) if segment]
    for step, next_step in zip(raw_steps, raw_steps[1:]):
        if (
            step.recursive
            and isinstance(step.matcher, WildcardMatcher)
            and (next_step.recursive or isinstance(next_step.matcher, WildcardMatcher))
        ):
            raise ValueError(
                'Recursive wildcard must be followed by a non-wildcard, non-recursive matcher'
            )
    return raw_steps
