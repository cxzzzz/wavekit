"""Unit tests for the query-path matcher parser."""

import pytest

from wavekit.readers.matcher import (
    BraceMatcher,
    ExactMatcher,
    RegexMatcher,
    WildcardMatcher,
    parse_query_path,
)
from wavekit.readers.range import Range


def test_matcher_parser_splits_hierarchy_and_range():
    steps = parse_query_path('tb.u0.J_state[3:0]')

    assert len(steps) == 3
    assert all(isinstance(step.matcher, ExactMatcher) for step in steps)
    assert steps[-1].matcher.name == 'J_state'
    assert steps[-1].matcher.range == Range(3, 0)


def test_matcher_parser_preserves_regex_hierarchy_content():
    steps = parse_query_path(r'tb./u[0-3]\.core/.data[7:0]')

    assert len(steps) == 3
    assert isinstance(steps[1].matcher, RegexMatcher)
    assert steps[1].matcher.regex.pattern == r'u[0-3]\.core'


def test_matcher_parser_distinguishes_regex_range_from_escaped_brackets():
    selected = parse_query_path(r'/data/[1:0]')[0].matcher
    escaped = parse_query_path(r'/data\[7:0\]/')[0].matcher

    assert isinstance(selected, RegexMatcher)
    assert selected.regex.pattern == 'data'
    assert selected.range == Range(1, 0)
    assert isinstance(escaped, RegexMatcher)
    assert escaped.regex.pattern == r'data\[7:0\]'
    assert escaped.range is None


def test_matcher_parser_expands_braces():
    matcher = parse_query_path('unit_{a,b}.sig_{0..1}')[0].matcher

    assert isinstance(matcher, BraceMatcher)
    assert BraceMatcher.expand(matcher.pattern) == {
        ('a',): 'unit_a',
        ('b',): 'unit_b',
    }


def test_matcher_parser_rejects_unmatched_brace():
    with pytest.raises(ValueError, match='Unmatched brace'):
        parse_query_path('unit_{a,b')


def test_matcher_parser_keeps_recursive_wildcard_raw():
    steps = parse_query_path('a.**.b')

    assert len(steps) == 3
    assert steps[0].recursive is False
    assert isinstance(steps[1].matcher, WildcardMatcher)
    assert steps[1].recursive is True
    assert steps[1].native_recursive is False
    assert isinstance(steps[2].matcher, ExactMatcher)
    assert steps[2].recursive is False


def test_matcher_parser_keeps_native_recursive_steps():
    steps = parse_query_path('$$A.$$B')

    assert len(steps) == 2
    assert all(step.recursive and step.native_recursive for step in steps)


def test_matcher_parser_allows_native_recursive_suffixes():
    assert parse_query_path('$$A.**')[-1].recursive is True
    assert isinstance(parse_query_path('$$A.**')[-1].matcher, WildcardMatcher)


def test_matchers_have_value_semantics():
    first = ExactMatcher(target='name', pattern='data[7:0]')
    second = ExactMatcher(target='name', pattern='data[7:0]')

    assert first == second
    assert hash(first) == hash(second)


def test_matcher_parser_rejects_ambiguous_recursive_wildcard():
    for path in ('a.**.*', 'a.**.**', 'a.**.$$B'):
        with pytest.raises(ValueError, match='Recursive wildcard'):
            parse_query_path(path)
