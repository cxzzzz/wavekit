from pathlib import Path

import numpy as np
import pytest

from wavekit import VcdReader, Waveform
from wavekit.readers.base import Reader
from wavekit.readers.expr_parser import extract_wave_paths
from wavekit.readers.pattern_parser import (
    expand_brace_pattern,
    split_by_hierarchy,
    split_by_range_expr,
)


@pytest.fixture()
def vcd_path():
    return Path(__file__).resolve().parent / 'testdata' / 'jtag.vcd'


def test_pattern_parsing():
    pattern, range_expr = split_by_range_expr('tb.u0.J_state[3:0]')
    assert pattern == 'tb.u0.J_state'
    assert range_expr == '[3:0]'

    assert split_by_hierarchy('tb.u0.J_state[3:0]') == ['tb', 'u0', 'J_state[3:0]']

    expanded = expand_brace_pattern('u{0,1}.sig{2..3}')
    assert expanded[('0', 2)] == 'u0.sig2'
    assert expanded[('1', 3)] == 'u1.sig3'

    pattern, range_expr = split_by_range_expr('tb.u0.signal')
    assert pattern == 'tb.u0.signal'
    assert range_expr == ''

    with pytest.raises(ValueError):
        _ = expand_brace_pattern('u{0,1')


def test_vcd_reader_load_waveform(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    j_state = vcd_reader.load_waveform(
        'tb.u0.J_state[3:0]',
        clock='tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert j_state.name == 'tb.u0.J_state[3:0]'
    assert j_state.width == 4
    assert j_state.signed is True
    assert len(j_state.value) > 0


def test_vcd_reader_load_waveform_without_range(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    j_next = vcd_reader.load_waveform(
        'tb.u0.J_next',
        clock='tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert j_next.name == 'tb.u0.J_next[3:0]'
    assert j_next.width == 4


def test_vcd_reader_load_matched_waveforms_regex(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    j_regex = vcd_reader.load_matched_waveforms(
        r'tb.u0.@J_([a-z]+\[3:0\])',
        'tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert len(j_regex) == 2
    assert {k[0][0] for k in j_regex} == {'next[3:0]', 'state[3:0]'}
    assert all(wave.width == 4 for wave in j_regex.values())


def test_vcd_reader_load_matched_waveforms_brace_expansion(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    j_states = vcd_reader.load_matched_waveforms(
        'tb.u0.J_{state,next}[3:0]',
        'tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert set(j_states.keys()) == {('next',), ('state',)}
    assert {wave.name for wave in j_states.values()} == {
        'tb.u0.J_next[3:0]',
        'tb.u0.J_state[3:0]',
    }
    assert all(wave.width == 4 for wave in j_states.values())


def test_vcd_reader_load_matched_waveforms_regex_key_conflict(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    with pytest.raises(Exception):
        vcd_reader.load_matched_waveforms(
            r'tb.u0.@J_[A-Za-z0-9_]+\[3:0\]',
            'tb.tck',
        )


def test_vcd_reader_load_matched_waveforms_uses_signal_range(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    waves = vcd_reader.load_matched_waveforms(
        'tb.u0.J_state',
        'tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert list(waves.keys()) == [()]
    wave = waves[()]
    assert wave.name == 'tb.u0.J_state[3:0]'
    assert wave.width == 4


def test_vcd_reader_clock_pattern_error(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))
    with pytest.raises(Exception):
        vcd_reader.load_matched_waveforms('tb.u0.J_state[3:0]', 'tb.no_clock')


def test_vcd_reader_clock_pattern_key_mismatch_error(vcd_path):
    # clock brace expansion yields different keys than the signal pattern
    vcd_reader = VcdReader(str(vcd_path))
    with pytest.raises(Exception, match='do not match signal pattern keys'):
        vcd_reader.load_matched_waveforms(
            'tb.u0.J_{state,next}[3:0]',   # keys: {('state',), ('next',)}
            'tb.{tck,tms}',                 # keys: {('tck',), ('tms',)} — mismatch
        )


def test_value_change_to_waveform_sample_on_posedge():
    value_change = np.array([[0, 0], [5, 1], [10, 0]], dtype=np.uint64)
    clock_changes = np.array([[0, 0], [5, 1], [10, 0], [15, 1]], dtype=np.uint64)

    wave = Reader.value_change_to_waveform(
        value_change,
        clock_changes,
        width=1,
        signed=False,
        sample_on_posedge=True,
        signal='tb.sig',
    )

    assert np.all(wave.value == np.array([1, 0]))
    assert np.all(wave.clock == np.array([0, 1]))
    assert np.all(wave.time == np.array([5, 15]))
    assert wave.name == 'tb.sig'


# ------------------------------------------------------------------
# expr_parser unit tests
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# eval integration tests
# ------------------------------------------------------------------


def test_eval_single_mode_arithmetic(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        result = reader.eval(
            'tb.u0.J_state[3:0] + tb.u0.J_next[3:0]',
            clock='tb.tck',
        )
    assert isinstance(result, Waveform)
    assert result.width == 5  # addition increases width by 1


def test_eval_single_mode_scalar(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        result = reader.eval(
            'tb.u0.J_state[3:0] + 1',
            clock='tb.tck',
        )
    assert isinstance(result, Waveform)


def test_eval_bit_slice(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        result = reader.eval(
            'tb.u0.J_state[3:0][1:0]',
            clock='tb.tck',
        )
    assert isinstance(result, Waveform)
    assert result.width == 2


def test_eval_single_mode_error_on_multi_match(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        with pytest.raises(ValueError, match="mode='single'"):
            reader.eval(
                'tb.u0.J_{state,next}[3:0]',
                clock='tb.tck',
                mode='single',
            )


def test_eval_zip_mode_brace_expansion(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        result = reader.eval(
            'tb.u0.J_{state,next}[3:0] + 1',
            clock='tb.tck',
            mode='zip',
        )
    assert isinstance(result, dict)
    assert set(result.keys()) == {('state',), ('next',)}
    assert all(isinstance(w, Waveform) for w in result.values())


def test_eval_zip_mode_broadcast(vcd_path):
    # J_state matches 1 signal (broadcast), J_{state,next} matches 2
    with VcdReader(str(vcd_path)) as reader:
        result = reader.eval(
            'tb.u0.J_{state,next}[3:0] + tb.u0.J_state[3:0]',
            clock='tb.tck',
            mode='zip',
        )
    assert isinstance(result, dict)
    assert len(result) == 2


def test_eval_no_match_raises(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        with pytest.raises(ValueError, match='matched no signals'):
            reader.eval('tb.u0.nonexistent_signal', clock='tb.tck')
