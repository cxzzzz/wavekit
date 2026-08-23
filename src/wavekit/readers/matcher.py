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
from abc import abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .hierarchy import Node

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
    """Base class for typed bindings returned in a :data:`CaptureKey`."""

    anchor_node: Node = field(repr=False, compare=False)
    node: Node = field(repr=False, compare=False)
    definition: str | None = None
    path: str = field(init=False)

    def __post_init__(self) -> None:
        start_name = self.anchor_node.full_name
        end_name = self.node.full_name
        suffix = end_name[len(start_name) :]
        object.__setattr__(self, 'path', suffix.removeprefix('.'))

    def with_anchor_node(self, anchor_node: Node) -> Capture:
        """Return a copy rebased to *anchor_node*."""
        return replace(self, anchor_node=anchor_node)

    def with_node(self, node: Node) -> Capture:
        """Return a copy ending at *node*."""
        return replace(self, node=node)


CaptureKey = tuple[Capture, ...]


class Matcher:
    """Match a single hierarchy node.

    A successful match returns the capture and an optional :class:`Range`.
    The hierarchy layer owns validation against the matched signal's native range.
    """

    target: MatchTarget

    @abstractmethod
    def match(self, node: Node) -> tuple[Capture, Range | None] | None:
        """Return the capture and optional range selection for *node*."""


@dataclass(frozen=True)
class ExactCapture(Capture):
    """Exact-name binding; only module-definition matches are public."""


@dataclass(frozen=True)
class ExactMatcher(Matcher):
    """Exact name or module-definition matcher."""

    target: MatchTarget
    pattern: str
    name: str = field(init=False, compare=False)
    suffix: str = field(init=False, compare=False)
    range: Range | None = field(init=False, compare=False)

    def __post_init__(self) -> None:
        name, suffix, selected_range = split_trailing_range(self.pattern)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'suffix', suffix)
        object.__setattr__(self, 'range', selected_range)

    def match(self, node: Node) -> tuple[Capture, Range | None] | None:
        """Match a single node by exact name or module definition."""
        # Definition regexes ($/regex/ and $$/regex/) match the FSDB module
        # definition name rather than the hierarchy node name.
        if self.target == 'definition':
            if not hasattr(node, 'definition'):
                raise ValueError(
                    'Cannot use module matcher ($/$$) on a backend without definition '
                    'matching support; use FSDB. '
                    f'(node: {node.name!r})'
                )
            definition = node.definition
            if self.pattern == definition:
                return ExactCapture(anchor_node=node, node=node, definition=definition), None
            return None

        if node.base_name == self.pattern:
            return ExactCapture(anchor_node=node, node=node), None
        if (
            self.range is not None
            and getattr(node, 'is_range_selectable', False)
            and node.base_name == self.name
        ):
            return ExactCapture(anchor_node=node, node=node), self.range
        return None


@dataclass(frozen=True)
class BraceCapture(Capture):
    """Binding produced by brace expansion; ``groups`` stores brace values."""

    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class BraceMatcher(Matcher):
    """Brace expansion matcher delegated to exact matchers."""

    target: MatchTarget
    pattern: str
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

    def match(self, node: Node) -> tuple[Capture, Range | None] | None:
        """Match a node against every brace-expanded alternative."""
        for key, matcher in self.matchers.items():
            matched = matcher.match(node)
            if matched is not None:
                capture, selected_range = matched
                return (
                    BraceCapture(
                        anchor_node=capture.anchor_node,
                        node=capture.node,
                        definition=capture.definition,
                        groups=key,
                    ),
                    selected_range,
                )
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

    target: MatchTarget

    def match(self, node: Node) -> tuple[Capture, Range | None] | None:
        """Match any node allowed by the wildcard syntax."""
        assert self.target != 'definition'
        return WildcardCapture(anchor_node=node, node=node), None


@dataclass(frozen=True)
class RegexCapture(Capture):
    """Binding produced by regex matching; ``groups`` stores regex groups."""

    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegexMatcher(Matcher):
    """Regular-expression matcher for local names or module definitions."""

    target: MatchTarget
    pattern: str
    suffix: str = field(init=False, compare=False)
    range: Range | None = field(init=False, compare=False)
    regex: re.Pattern[str] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        regex, suffix, selected_range = split_trailing_range(self.pattern)
        object.__setattr__(self, 'suffix', suffix)
        object.__setattr__(self, 'range', selected_range)
        object.__setattr__(self, 'regex', re.compile(regex))

    def match(self, node: Node) -> tuple[Capture, Range | None] | None:
        """Match a node against a compiled regex and optional range suffix."""
        # Definition regexes ($/regex/ and $$/regex/) match the FSDB module
        # definition name rather than the hierarchy node name.
        if self.target == 'definition':
            if not hasattr(node, 'definition'):
                raise ValueError(
                    'Cannot use module matcher ($/$$) on a backend without definition '
                    'matching support; use FSDB. '
                    f'(node: {node.name!r})'
                )
            if self.range is not None:
                raise ValueError(
                    f'Range selector {self.suffix!r} is not allowed on definition matchers: '
                    f'{self.pattern!r}'
                )
            definition = node.definition
            if definition is not None and (matched := self.regex.fullmatch(definition)):
                return (
                    RegexCapture(
                        anchor_node=node,
                        node=node,
                        definition=definition,
                        groups=matched.groups(),
                    ),
                    None,
                )
            return None

        # No trailing selection was parsed outside the regex literal. First
        # try a direct match against the real local base name.
        if self.range is None:
            if matched := self.regex.fullmatch(node.base_name):
                return RegexCapture(anchor_node=node, node=node, groups=matched.groups()), None
            # Compatibility case: the regex itself may contain a native range,
            # for example /(counter\[3:0\]|overflow)/. Match the displayed
            # name and return the node's existing range as the selection.
            if (
                getattr(node, 'is_range_selectable', False)
                and (selected_range := getattr(node, 'range', None)) is not None
                and (matched := self.regex.fullmatch(node.name))
            ):
                return (
                    RegexCapture(anchor_node=node, node=node, groups=matched.groups()),
                    selected_range,
                )
            return None

        # A parsed trailing suffix may already be part of a real ARRAY member's
        # base name, e.g. /arr/[0] matching the concrete child arr[0]. Treat it
        # as a direct node match rather than a range view of the ARRAY parent.
        if node.base_name.endswith(self.suffix) and (
            matched := self.regex.fullmatch(node.base_name[: -len(self.suffix)])
        ):
            return RegexCapture(anchor_node=node, node=node, groups=matched.groups()), None
        # Otherwise the parsed suffix is a range selection on the current
        # signal, e.g. /data/[7:0] matching base_name == 'data'.
        if getattr(node, 'is_range_selectable', False) and (
            matched := self.regex.fullmatch(node.base_name)
        ):
            return RegexCapture(
                anchor_node=node,
                node=node,
                groups=matched.groups(),
            ), self.range
        return None


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
    """Parse a query path string into :class:`PathStep` objects."""
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

    def parse_regex(literal: str, target: MatchTarget) -> RegexMatcher:
        close = find_regex_close(literal)
        if close == -1:
            raise ValueError(f'Unclosed regex literal: {literal!r}')
        trailing = literal[close + 1 :]
        if trailing:
            base, _, _ = split_trailing_range(trailing)
            if base:
                raise ValueError(f'Regex literal has trailing content {trailing!r}')
        pattern = literal[1:close] + trailing
        try:
            re.compile(literal[1:close])
        except re.error as error:
            raise ValueError(f'Invalid regex /{literal[1:close]}/: {error}') from error
        return RegexMatcher(target=target, pattern=pattern)

    def parse_name(segment: str, target: MatchTarget) -> Matcher:
        if segment.startswith('/'):
            return parse_regex(segment, target)
        if '{' in segment:
            return BraceMatcher(target=target, pattern=segment)
        return ExactMatcher(target=target, pattern=segment)

    def parse_segment(segment: str) -> PathStep:
        if segment.startswith('$$'):
            rest = segment[2:]
            if rest.startswith('@'):
                raise ValueError(f'$$@regex not supported; use $$/regex/ instead: {segment!r}')
            if rest in ('*', '**'):
                raise ValueError(
                    f'$$* not supported; use ** for recursive name matching: {segment!r}'
                )
            matcher = (
                parse_regex(rest, 'definition')
                if rest.startswith('/')
                else parse_name(rest, 'definition')
            )
            return PathStep(matcher=matcher, recursive=True, native_recursive=True)

        if segment.startswith('$'):
            rest = segment[1:]
            if rest.startswith('@'):
                raise ValueError(f'$@regex not supported; use $/regex/ instead: {segment!r}')
            if rest in ('*', '**'):
                raise ValueError(
                    f'$* not supported; use * for single-level name matching: {segment!r}'
                )
            matcher = (
                parse_regex(rest, 'definition')
                if rest.startswith('/')
                else parse_name(rest, 'definition')
            )
            return PathStep(matcher=matcher, recursive=False)

        if segment == '*':
            return PathStep(matcher=WildcardMatcher(target='name'))
        if segment == '**':
            return PathStep(matcher=WildcardMatcher(target='name'), recursive=True)
        if segment.startswith('@'):
            try:
                re.compile(segment[1:])
            except re.error as error:
                raise ValueError(f'Invalid @regex pattern {segment[1:]!r}: {error}') from error
            return PathStep(matcher=RegexMatcher(target='name', pattern=segment[1:]))
        return PathStep(matcher=parse_name(segment, 'name'))

    raw_steps = [parse_segment(segment) for segment in split_hierarchy(path) if segment]
    for step, next_step in zip(raw_steps, raw_steps[1:]):
        if step.recursive and isinstance(step.matcher, WildcardMatcher) and (
            next_step.recursive or isinstance(next_step.matcher, WildcardMatcher)
        ):
            raise ValueError(
                'Recursive wildcard must be followed by a non-wildcard, non-recursive matcher'
            )
    return raw_steps
