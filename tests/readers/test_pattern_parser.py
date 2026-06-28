"""Unit tests for the brace/regex pattern parser (extracted from test_vcd.py)."""

import pytest

from wavekit.readers.pattern_parser import (
    expand_brace_pattern,
    split_by_hierarchy,
    split_by_range_expr,
)


def test_pattern_parsing_split_by_range_expr():
    pattern, range_expr = split_by_range_expr('tb.u0.J_state[3:0]')
    assert pattern == 'tb.u0.J_state'
    assert range_expr == '[3:0]'

    pattern, range_expr = split_by_range_expr('tb.u0.signal')
    assert pattern == 'tb.u0.signal'
    assert range_expr == ''


def test_pattern_parsing_split_by_hierarchy():
    assert split_by_hierarchy('tb.u0.J_state[3:0]') == ['tb', 'u0', 'J_state[3:0]']


def test_pattern_parsing_expand_brace_pattern():
    expanded = expand_brace_pattern('u{0,1}.sig{2..3}')
    assert expanded[('0', 2)] == 'u0.sig2'
    assert expanded[('1', 3)] == 'u1.sig3'


def test_pattern_parsing_expand_brace_pattern_invalid():
    with pytest.raises(ValueError):
        _ = expand_brace_pattern('u{0,1')
