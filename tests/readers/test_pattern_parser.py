"""Unit tests for the query-path matcher parser."""

import pytest

from wavekit.readers.matcher import BraceMatcher, ExactMatcher, RegexMatcher, parse_query_path
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
