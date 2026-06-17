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

    assert j_next.name == 'tb.u0.J_next'
    assert j_next.width == 4


def test_vcd_reader_subrange_load(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        low_bits = reader.load_waveform('tb.u0.J_state[1:0]', clock='tb.tck')
        matched_low_bits = reader.load_matched_waveforms('tb.u0.J_state[1:0]', 'tb.tck')[()]

    assert low_bits.width == 2
    assert matched_low_bits.width == 2
    assert np.array_equal(matched_low_bits.value, low_bits.value)


@pytest.fixture()
def unknown_vcd_path():
    return Path(__file__).resolve().parent / 'testdata' / 'unknown_states.vcd'


def test_vcd_reader_load_unknown_mask_include_flags(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        both = reader.load_unknown_mask('tb.bus[3:0]', clock='tb.clk', begin_cycle=1, end_cycle=6)
        x_only = reader.load_unknown_mask(
            'tb.bus[3:0]', clock='tb.clk', include_z=False, begin_cycle=1, end_cycle=6
        )
        z_only = reader.load_unknown_mask(
            'tb.bus[3:0]', clock='tb.clk', include_x=False, begin_cycle=1, end_cycle=6
        )
        values_x0 = reader.load_waveform(
            'tb.bus[3:0]', clock='tb.clk', xz_value=0, begin_cycle=1, end_cycle=6
        )

    assert both.name == 'unknown_mask(tb.bus[3:0])'
    assert both.width == 4
    assert both.signed is False
    assert np.array_equal(
        both.value,
        np.array([0b1111, 0b1111, 0b0010, 0b0101, 0], dtype=np.uint64),
    )
    assert np.array_equal(x_only.value, np.array([0b1111, 0, 0b0010, 0b0001, 0], dtype=np.uint64))
    assert np.array_equal(z_only.value, np.array([0, 0b1111, 0, 0b0100, 0], dtype=np.uint64))
    assert np.array_equal(both.clock, values_x0.clock)
    assert np.array_equal(both.time, values_x0.time)


def test_vcd_reader_load_unknown_mask_range_selection(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        low = reader.load_unknown_mask('tb.bus[1:0]', clock='tb.clk', begin_cycle=1, end_cycle=6)

    assert low.width == 2
    assert low.name == 'unknown_mask(tb.bus[1:0])'
    assert np.array_equal(low.value, np.array([0b11, 0b11, 0b10, 0b01, 0], dtype=np.uint64))


def test_load_waveform_name_signed(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        w = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck', signed=True)
    assert w.name == 'tb.u0.J_state[3:0]'
    assert w.signed is True


def test_load_waveform_name_signed_default(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        w = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')
    assert w.name == 'tb.u0.J_state[3:0]'
    assert w.signed is False


def test_load_unknown_mask_name_signed(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        w = reader.load_unknown_mask('tb.bus[3:0]', clock='tb.clk', begin_cycle=1, end_cycle=6)
    assert w.name == 'unknown_mask(tb.bus[3:0])'
    assert w.signed is False


def test_load_waveform_signal_object_name(vcd_path):
    from wavekit.signal import Signal
    sig = Signal(name='J_state[3:0]', full_name='tb.u0.J_state[3:0]', width=4, range=(3, 0))
    with VcdReader(str(vcd_path)) as reader:
        w = reader.load_waveform(sig, clock='tb.tck', signed=True)
    assert w.name == 'tb.u0.J_state[3:0]'
    assert w.signed is True


def test_load_unknown_mask_signal_object_name(unknown_vcd_path):
    from wavekit.signal import Signal
    sig = Signal(name='bus[3:0]', full_name='tb.bus[3:0]', width=4, range=(3, 0))
    with VcdReader(str(unknown_vcd_path)) as reader:
        w = reader.load_unknown_mask(sig, clock='tb.clk', begin_cycle=1, end_cycle=6)
    assert w.name == 'unknown_mask(tb.bus[3:0])'
    assert w.signed is False


def test_load_waveform_subrange_name(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        w = reader.load_waveform('tb.u0.J_state[1:0]', clock='tb.tck')
    assert w.name == 'tb.u0.J_state[1:0]'
    assert w.width == 2


def test_load_matched_waveforms_name_with_brace(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        waves = reader.load_matched_waveforms(
            'tb.u0.J_{state,next}[3:0]', 'tb.tck', signed=True
        )
    assert waves[('state',)].name == 'tb.u0.J_state[3:0]'
    assert waves[('next',)].name == 'tb.u0.J_next[3:0]'
    assert waves[('state',)].signed is True
    assert waves[('next',)].signed is True


def test_load_matched_unknown_masks_name(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        masks = reader.load_matched_unknown_masks(
            'tb.data_{0,1}[3:0]', 'tb.clk', begin_cycle=1, end_cycle=6
        )
    assert masks[('0',)].name == 'unknown_mask(tb.data_0[3:0])'
    assert masks[('1',)].name == 'unknown_mask(tb.data_1[3:0])'
    assert masks[('0',)].signed is False
    assert masks[('1',)].signed is False


def test_vcd_reader_load_unknown_mask_fully_known_is_zero(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        mask = reader.load_unknown_mask(
            'tb.data_0[3:0]', clock='tb.clk', begin_cycle=4, end_cycle=5
        )

    assert np.array_equal(mask.value, np.array([0], dtype=np.uint64))


def test_vcd_reader_load_unknown_mask_both_false_is_all_zero(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        mask = reader.load_unknown_mask(
            'tb.bus[3:0]', clock='tb.clk', include_x=False, include_z=False,
            begin_cycle=1, end_cycle=6,
        )

    assert mask.name == 'unknown_mask(tb.bus[3:0])'
    assert mask.width == 4
    assert np.array_equal(mask.value, np.zeros(5, dtype=np.uint64))


def test_vcd_reader_load_matched_unknown_masks(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        masks = reader.load_matched_unknown_masks(
            'tb.data_{0,1}[3:0]', 'tb.clk', begin_cycle=1, end_cycle=6
        )
        values = reader.load_matched_waveforms(
            'tb.data_{0,1}[3:0]', 'tb.clk', begin_cycle=1, end_cycle=6
        )

    assert set(masks) == set(values) == {('0',), ('1',)}
    assert masks[('0',)].name == 'unknown_mask(tb.data_0[3:0])'
    assert masks[('1',)].name == 'unknown_mask(tb.data_1[3:0])'
    assert np.array_equal(masks[('0',)].value, np.array([0, 0b1000, 0b0001, 0, 0], dtype=np.uint64))
    assert np.array_equal(masks[('1',)].value, np.array([0b1111, 0, 0b0101, 0, 0], dtype=np.uint64))


def test_vcd_reader_matched_unknown_mask_both_false_is_all_zero(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        masks = reader.load_matched_unknown_masks(
            'tb.data_{0,1}[3:0]', 'tb.clk', include_x=False, include_z=False,
            begin_cycle=1, end_cycle=6,
        )

    assert set(masks) == {('0',), ('1',)}
    assert np.array_equal(masks[('0',)].value, np.zeros(5, dtype=np.uint64))
    assert np.array_equal(masks[('1',)].value, np.zeros(5, dtype=np.uint64))


def test_vcd_reader_rejects_invalid_xz_value(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.load_waveform('tb.bus[3:0]', clock='tb.clk', xz_value=2)
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.load_matched_waveforms('tb.bus[3:0]', 'tb.clk', xz_value=2)
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.eval('tb.bus[3:0] + 1', clock='tb.clk', xz_value=2)


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


def test_vcd_reader_module_name_matching_is_unsupported(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        with pytest.raises(NotImplementedError):
            reader.get_matched_signals('tb.$u0.J_state[3:0]')


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
            'tb.u0.J_{state,next}[3:0]',  # keys: {('state',), ('next',)}
            'tb.{tck,tms}',  # keys: {('tck',), ('tms',)} — mismatch
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


# ------------------------------------------------------------------
# begin_time / end_time / begin_cycle / end_cycle tests
# ------------------------------------------------------------------


def test_load_waveform_begin_end_time(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')
        windowed = reader.load_waveform(
            'tb.u0.J_state[3:0]', clock='tb.tck', begin_time=105, end_time=205
        )

    # Windowed result should be a strict subset of the full waveform
    assert len(windowed.value) == 10
    assert windowed.time[0] == 105
    assert windowed.time[-1] == 195
    # Clock values are absolute: cycle 10 is at time 105 (negedge 0 at t=5, period=10)
    assert windowed.clock[0] == 10
    assert windowed.clock[-1] == 19
    # Values should match the corresponding slice of the full waveform
    assert np.array_equal(windowed.value, full.value[10:20])


def test_load_waveform_begin_end_cycle(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')
        windowed = reader.load_waveform(
            'tb.u0.J_state[3:0]', clock='tb.tck', begin_cycle=10, end_cycle=20
        )

    assert len(windowed.value) == 10
    assert windowed.clock[0] == 10
    assert windowed.clock[-1] == 19
    assert np.array_equal(windowed.value, full.value[10:20])


def test_load_waveform_cycle_equals_time_window(vcd_path):
    # begin_cycle=20 / end_cycle=30 should produce identical results to the
    # corresponding begin_time / end_time window (cycle 20 is at time 205)
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')
        by_time = reader.load_waveform(
            'tb.u0.J_state[3:0]', clock='tb.tck', begin_time=205, end_time=305
        )
        by_cycle = reader.load_waveform(
            'tb.u0.J_state[3:0]', clock='tb.tck', begin_cycle=20, end_cycle=30
        )

    assert np.array_equal(by_time.value, by_cycle.value)
    assert np.array_equal(by_time.clock, by_cycle.clock)
    assert np.array_equal(by_time.time, by_cycle.time)
    assert np.array_equal(by_cycle.value, full.value[20:30])


def test_load_waveform_mutually_exclusive_begin(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform(
                'tb.u0.J_state[3:0]',
                clock='tb.tck',
                begin_time=100,
                begin_cycle=10,
            )


def test_load_waveform_mutually_exclusive_end(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform(
                'tb.u0.J_state[3:0]',
                clock='tb.tck',
                end_time=200,
                end_cycle=20,
            )


def test_value_change_to_waveform_clock_offset():
    # Verify that clock_offset shifts the .clock array to start from a given value
    value_change = np.array([[0, 0], [5, 1], [10, 0]], dtype=np.uint64)
    clock_changes = np.array([[0, 0], [5, 1], [10, 0], [15, 1]], dtype=np.uint64)

    wave_no_offset = Reader.value_change_to_waveform(
        value_change,
        clock_changes,
        width=1,
        signed=False,
        sample_on_posedge=True,
        signal='tb.sig',
    )
    wave_with_offset = Reader.value_change_to_waveform(
        value_change,
        clock_changes,
        width=1,
        signed=False,
        sample_on_posedge=True,
        signal='tb.sig',
        clock_offset=50,
    )

    assert np.all(wave_no_offset.clock == np.array([0, 1]))
    assert np.all(wave_with_offset.clock == np.array([50, 51]))
    # Values are the same regardless of offset
    assert np.array_equal(wave_no_offset.value, wave_with_offset.value)


def test_cycle_slice(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')

    sliced = full.cycle_slice(10, 20)
    assert len(sliced.value) == 10
    assert sliced.clock[0] == 10
    assert sliced.clock[-1] == 19
    assert np.array_equal(sliced.value, full.value[10:20])


def test_cycle_slice_include_end(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')

    sliced = full.cycle_slice(10, 20, include_end=True)
    assert len(sliced.value) == 11
    assert sliced.clock[-1] == 20
