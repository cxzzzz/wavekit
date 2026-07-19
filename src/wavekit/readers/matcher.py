"""Query path matcher types and parser.

This module implements wavekit's signal/scope query-path syntax:

* ``name`` — exact local-name match
* ``{a,b}``, ``{0..3}``, ``{0..7..2}`` — brace expansion
* ``*`` / ``**`` — single-level / recursive wildcard
* ``/regex/`` or legacy ``@regex`` — regular-expression match
* ``$ModName`` / ``$$ModName`` — direct / recursive module-definition match

Matchers deliberately depend only on the small node surface they consume
(``name``, ``base_name``, ``def_name``, ``composite_type``, and
``with_range``).  The hierarchy module imports this parser, so importing the
concrete hierarchy types here would create a cycle.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .hierarchy import Node

MatchTarget = Literal['name', 'definition']
RangeSelection = tuple[int, int] | None

_RANGE_BRACE = re.compile(r'\{(\d+)\.\.(\d+)(?:\.\.(\d+))?\}')
_LIST_BRACE = re.compile(r'\{([^{}]+)\}')
_TRAILING_RANGE = re.compile(r'(\[(\d+)(?::(\d+))?\])$')


def split_trailing_range(segment: str) -> tuple[str, str, RangeSelection]:
    """Split a trailing ``[N]`` or ``[H:L]`` from a path segment."""
    matched = _TRAILING_RANGE.search(segment)
    if matched is None:
        return segment, '', None
    suffix = matched.group(1)
    start = int(matched.group(2))
    end = int(matched.group(3)) if matched.group(3) is not None else start
    return segment[: matched.start()], suffix, (start, end)


@dataclass(frozen=True)
class Capture:
    path: str
    definition: str | None = None

    def with_prefix(self, prefix: str) -> Capture:
        return replace(self, path=f'{prefix}.{self.path}')


CaptureKey = tuple[Capture, ...]


class Matcher:
    """Match a single hierarchy node.

    A successful match returns the capture and an optional raw ``(start, end)``
    range selection.  The hierarchy layer owns conversion to :class:`Range`
    and validation against the matched signal's native range.
    """

    target: MatchTarget

    @abstractmethod
    def match(self, node: Node) -> tuple[Capture, RangeSelection] | None:
        """Return the capture and optional range selection for *node*."""


def _is_range_selectable(node: Node) -> bool:
    """Return whether a node may accept a trailing range selector.

    The concrete hierarchy keeps the authoritative validation in
    ``Signal.with_range``.  This parser-level check only rejects scopes and
    non-array composite signals before attempting a selection.
    """
    if not callable(getattr(node, 'with_range', None)):
        return False
    composite_type = getattr(node, 'composite_type', None)
    return composite_type is None or getattr(composite_type, 'value', composite_type) == 'array'


@dataclass(frozen=True)
class ExactCapture(Capture):
    pass


class ExactMatcher(Matcher):
    """Exact name or module-definition matcher."""

    def __init__(self, target: MatchTarget, pattern: str):
        self.target = target
        self.pattern = pattern
        self.name, self.suffix, self.range = split_trailing_range(pattern)

    def match(self, node: Node) -> tuple[Capture, RangeSelection] | None:
        if self.target == 'definition':
            if not hasattr(node, 'def_name'):
                raise ValueError(
                    'Cannot use module matcher ($/$$) on a scope without def_name. '
                    'VCD/FST backends do not support module def_name matching; use FSDB. '
                    f'(scope: {node.name!r})'
                )
            definition = node.def_name
            if self.pattern == definition:
                return ExactCapture(path=node.name, definition=definition), None
            return None

        if node.name == self.pattern:
            return ExactCapture(path=self.pattern), None
        if _is_range_selectable(node) and node.base_name == self.name:
            return ExactCapture(path=self.pattern), self.range
        return None


@dataclass(frozen=True)
class BraceCapture(Capture):
    groups: tuple[str, ...] = ()


class BraceMatcher(Matcher):
    """Brace expansion matcher delegated to exact matchers."""

    def __init__(self, target: MatchTarget, pattern: str):
        self.target = target
        self.pattern = pattern
        self.matchers = {
            key: ExactMatcher(target, expanded) for key, expanded in self.expand(pattern).items()
        }

    def match(self, node: Node) -> tuple[Capture, RangeSelection] | None:
        for key, matcher in self.matchers.items():
            matched = matcher.match(node)
            if matched is not None:
                capture, selected_range = matched
                return (
                    BraceCapture(
                        path=capture.path,
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
    pass


class WildcardMatcher(Matcher):
    """Single-level (``*``) or recursive (``**``) wildcard."""

    def __init__(self, target: MatchTarget):
        self.target = target

    def match(self, node: Node) -> tuple[Capture, RangeSelection] | None:
        assert self.target != 'definition'
        return WildcardCapture(path=node.name), None


@dataclass(frozen=True)
class RegexCapture(Capture):
    groups: tuple[str, ...] = ()


class RegexMatcher(Matcher):
    """Regular-expression matcher for local names or module definitions."""

    def __init__(self, target: MatchTarget, pattern: str):
        self.target = target
        self.pattern = pattern
        regex, self.suffix, self.range = split_trailing_range(pattern)
        self.regex = re.compile(regex)

    def match(self, node: Node) -> tuple[Capture, RangeSelection] | None:
        if self.target == 'definition':
            if not hasattr(node, 'def_name'):
                raise ValueError(
                    'Cannot use module matcher ($/$$) on a scope without def_name. '
                    'VCD/FST backends do not support module def_name matching; use FSDB. '
                    f'(scope: {node.name!r})'
                )
            if self.range is not None:
                raise ValueError(
                    f'Range selector {self.suffix!r} is not allowed on definition matchers: '
                    f'{self.pattern!r}'
                )
            definition = node.def_name
            if definition is not None and (matched := self.regex.fullmatch(definition)):
                return (
                    RegexCapture(
                        path=node.name,
                        definition=definition,
                        groups=matched.groups(),
                    ),
                    None,
                )
            return None

        if self.range is None:
            if matched := self.regex.fullmatch(node.name):
                return RegexCapture(path=node.name, groups=matched.groups()), None
            return None

        if node.name.endswith(self.suffix) and (
            matched := self.regex.fullmatch(node.name[: -len(self.suffix)])
        ):
            return RegexCapture(path=node.name, groups=matched.groups()), None
        if _is_range_selectable(node) and (matched := self.regex.fullmatch(node.base_name)):
            return RegexCapture(
                path=f'{node.base_name}{self.suffix}', groups=matched.groups()
            ), self.range
        return None


@dataclass(frozen=True)
class PathStep:
    matcher: Matcher
    recursive: bool = False


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
            return PathStep(matcher=matcher, recursive=True)

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

    return [parse_segment(segment) for segment in split_hierarchy(path) if segment]
