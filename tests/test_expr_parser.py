"""Unit tests for the expression parser path extraction (extracted from test_vcd.py)."""

from wavekit.readers.expr_parser import extract_wave_paths


def test_extract_wave_paths_simple():
    subst, paths = extract_wave_paths('tb.u0.sig[3:0] + tb.u0.other')
    assert paths == [('__wave_0__', 'tb.u0.sig[3:0]'), ('__wave_1__', 'tb.u0.other')]
    assert subst == '__wave_0__ + __wave_1__'


def test_extract_wave_paths_dollar_prefix():
    subst, paths = extract_wave_paths('$mod.sig + 1')
    assert paths[0] == ('__wave_0__', '$mod.sig')
    assert '+ 1' in subst


def test_extract_wave_paths_no_path():
    subst, paths = extract_wave_paths('1 + 2')
    assert paths == []
    assert subst == '1 + 2'


def test_extract_wave_paths_bit_slice_preserved():
    # The signal range [3:0] is consumed by the regex; the extra [1:0] remains
    # as a Python subscript on the placeholder.
    subst, paths = extract_wave_paths('tb.u0.sig[3:0][1:0]')
    assert paths == [('__wave_0__', 'tb.u0.sig[3:0]')]
    assert subst == '__wave_0__[1:0]'
