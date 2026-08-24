"""Tests for WaveKit expression parsing and evaluation."""

import numpy as np
import pytest

from wavekit import Waveform
from wavekit.expression import evaluate_expression, parse_expression


def build_waveform(values, width=8, signed=False):
    value = np.array(values)
    clock = np.arange(len(value))
    time = clock * 10
    return Waveform(value, clock, time, width=width, signed=signed)


def evaluate_source_expression(expr, waveforms):
    substituted, path_entries = parse_expression(expr)
    namespace = {placeholder: waveforms[path] for placeholder, path in path_entries}
    return evaluate_expression(substituted, namespace)


def assert_same_waveform(actual, expected):
    assert np.array_equal(actual.value, expected.value)
    assert np.array_equal(actual.clock, expected.clock)
    assert np.array_equal(actual.time, expected.time)
    assert actual.width == expected.width
    assert actual.signed == expected.signed


# ==========================================
# Parser
# ==========================================


def test_parse_expression_operators_without_whitespace():
    expected = {
        'a*b': '__wave_0__*__wave_1__',
        'a * b': '__wave_0__*__wave_1__',
        'a**b': '__wave_0__**__wave_1__',
        'a/b': '__wave_0__/__wave_1__',
        'a//b': '__wave_0__//__wave_1__',
        'a<<b': '__wave_0__<<__wave_1__',
        'a!=b': '__wave_0__!=__wave_1__',
        'a&~b': '__wave_0__&~__wave_1__',
    }
    for expr, expected_expr in expected.items():
        substituted, paths = parse_expression(expr)
        assert substituted == expected_expr
        assert paths == [('__wave_0__', 'a'), ('__wave_1__', 'b')]


def test_parse_expression_distinguishes_matchers_from_operators():
    substituted, paths = parse_expression('tb.*.valid&tb.**.ready')

    assert substituted == '__wave_0__&__wave_1__'
    assert paths == [
        ('__wave_0__', 'tb.*.valid'),
        ('__wave_1__', 'tb.**.ready'),
    ]


def test_parse_expression_supports_numeric_literals():
    for literal in ('0x12_34', '0b1010_0011', '0o7_55', '1_234', '1_234.5e-2'):
        substituted, paths = parse_expression(f'data+{literal}')
        assert substituted == f'__wave_0__+{literal}'
        assert paths == [('__wave_0__', 'data')]


def test_parse_expression_supports_calls_and_nested_calls():
    substituted, paths = parse_expression('rising_edge(falling_edge(valid))')

    assert substituted == 'rising_edge(falling_edge(__wave_0__))'
    assert paths == [('__wave_0__', 'valid')]


def test_parse_expression_distinguishes_signal_name_from_registered_call():
    signal_call_name, call_paths = parse_expression('rising_edge(valid)')
    signal_name, signal_paths = parse_expression('rising_edge')

    assert signal_call_name == 'rising_edge(__wave_0__)'
    assert call_paths == [('__wave_0__', 'valid')]
    assert signal_name == '__wave_0__'
    assert signal_paths == [('__wave_0__', 'rising_edge')]


def test_parse_expression_treats_function_name_as_signal_outside_call():
    substituted, paths = parse_expression('rising_edge&valid')

    assert substituted == '__wave_0__&__wave_1__'
    assert paths == [
        ('__wave_0__', 'rising_edge'),
        ('__wave_1__', 'valid'),
    ]


def test_parse_expression_preserves_special_paths_and_subscripts():
    substituted, paths = parse_expression('tb.fifo_{0..3}.data[7:0][1:0]+tb.u0./valid/[0]')

    assert substituted == '__wave_0__+__wave_1__'
    assert paths == [
        ('__wave_0__', 'tb.fifo_{0..3}.data[7:0][1:0]'),
        ('__wave_1__', 'tb.u0./valid/[0]'),
    ]


def test_parse_expression_supports_module_matchers():
    substituted, paths = parse_expression('tb.$fifo_{0..3}.ptr+tb.$$fifo.ptr')

    assert substituted == '__wave_0__+__wave_1__'
    assert paths == [
        ('__wave_0__', 'tb.$fifo_{0..3}.ptr'),
        ('__wave_1__', 'tb.$$fifo.ptr'),
    ]


def test_parse_expression_rejects_legacy_regex_and_attribute_calls():
    with pytest.raises(Exception):
        parse_expression('tb.u0.@valid')
    with pytest.raises(Exception):
        parse_expression('valid.rising_edge()')
    with pytest.raises(Exception):
        parse_expression('Waveform.rising_edge(valid)')
    with pytest.raises(Exception):
        parse_expression('unknown(valid)')


def test_parse_expression_rejects_embedded_wildcard_segments():
    with pytest.raises(Exception):
        parse_expression('tb.foo*')


# ==========================================
# Evaluator
# ==========================================


def test_evaluate_expression_operators_and_comparisons():
    data = build_waveform([1, 2, 3])
    valid = build_waveform([0, 1, 1], width=1)

    result = evaluate_source_expression('data+1', {'data': data})
    assert np.array_equal(result.value, np.array([2, 3, 4]))

    result = evaluate_source_expression(
        '(data>=2)&valid',
        {'data': data, 'valid': valid},
    )
    assert np.array_equal(result.value, np.array([0, 1, 1]))
    assert result.width == 1
    assert result.signed is False


def test_evaluate_expression_respects_operator_precedence_and_parentheses():
    a = build_waveform([2, 4])
    b = build_waveform([3, 5])
    c = build_waveform([4, 2])

    result = evaluate_source_expression('a+b*c', {'a': a, 'b': b, 'c': c})
    expected = a + b * c
    assert_same_waveform(result, expected)

    result = evaluate_source_expression('(a+b)*c', {'a': a, 'b': b, 'c': c})
    expected = (a + b) * c
    assert_same_waveform(result, expected)


def test_evaluate_expression_registered_functions():
    data = build_waveform([1, 15, 3], width=4)
    valid = build_waveform([0, 1, 1, 0], width=1)

    for expr, expected in [
        ('bit_count(data)', data.bit_count()),
        ('ahead(data, 2)', data.ahead(2)),
        ('back(data, 2)', data.back(2)),
        ('as_signed(data)', data.as_signed()),
        ('rising_edge(valid)', valid.rising_edge()),
        ('falling_edge(valid)', valid.falling_edge()),
    ]:
        result = evaluate_source_expression(expr, {'data': data, 'valid': valid})
        assert_same_waveform(result, expected)

    signed_data = build_waveform([-1, 1], width=4, signed=True)
    result = evaluate_source_expression('as_unsigned(signed_data)', {'signed_data': signed_data})
    assert_same_waveform(result, signed_data.as_unsigned())


def test_evaluate_expression_propagates_waveform_errors():
    unsigned = build_waveform([1, 2], width=4)
    signed = build_waveform([1, 2], width=4, signed=True)

    with pytest.raises(ValueError, match='signedness mismatch'):
        evaluate_source_expression('unsigned+signed', {'unsigned': unsigned, 'signed': signed})


def test_evaluate_expression_registered_nested_functions():
    valid = build_waveform([0, 1, 1, 0], width=1)

    result = evaluate_source_expression(
        'rising_edge(falling_edge(valid))',
        {'valid': valid},
    )

    assert np.array_equal(result.value, np.array([0, 0, 0, 1]))
    assert np.array_equal(result.clock, valid.clock)
    assert np.array_equal(result.time, valid.time)


def test_evaluate_expression_executes_substituted_expression():
    data = build_waveform([1, 2, 3])

    result = evaluate_expression('__wave_0__+1', {'__wave_0__': data})

    assert np.array_equal(result.value, np.array([2, 3, 4]))
