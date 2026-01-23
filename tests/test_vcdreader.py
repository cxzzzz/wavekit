from pathlib import Path

import numpy as np
import pytest

from wavekit import VcdReader
from wavekit.pattern_parser import (
    expand_brace_pattern,
    split_by_hierarchy,
    split_by_range_expr,
)
from wavekit.reader import Reader


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

    assert j_state.signal == 'tb.u0.J_state[3:0]'
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

    assert j_next.signal == 'tb.u0.J_next[3:0]'
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
    assert {wave.signal for wave in j_states.values()} == {
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
    assert wave.signal == 'tb.u0.J_state[3:0]'
    assert wave.width == 4


def test_vcd_reader_clock_pattern_error(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))
    with pytest.raises(Exception):
        vcd_reader.load_matched_waveforms('tb.u0.J_state[3:0]', 'tb.no_clock')


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
    assert wave.signal == 'tb.sig'
