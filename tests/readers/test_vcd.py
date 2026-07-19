from pathlib import Path

import numpy as np
import pytest

from wavekit import Scope, Signal, VcdReader, Waveform
from wavekit.readers.base import Reader  # noqa: F401  # used in test_vcd_value_change_to_waveform_*
from wavekit.readers.hierarchy import Range


def _scopes(node):
    return tuple(child for child in node.children if isinstance(child, Scope))


def _signals(node):
    return tuple(child for child in node.children if isinstance(child, Signal))


def _capture_groups(results):
    return {key[0].groups for key in results}


def _by_group(results, group):
    return next(value for key, value in results.items() if key[0].groups == (group,))


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def vcd_path():
    return Path(__file__).resolve().parent / 'fixtures' / 'vcd' / 'jtag.vcd'


@pytest.fixture()
def compare_vcd_path():
    return Path(__file__).resolve().parent / 'fixtures' / 'vcd' / 'compare.vcd'


@pytest.fixture()
def compare_xz_vcd_path():
    return Path(__file__).resolve().parent / 'fixtures' / 'vcd' / 'compare_xz.vcd'


@pytest.fixture()
def unknown_vcd_path():
    return Path(__file__).resolve().parent / 'fixtures' / 'vcd' / 'unknown_states.vcd'


@pytest.fixture()
def nonzero_vcd_path():
    return Path(__file__).resolve().parent / 'fixtures' / 'vcd' / 'nonzero_ranges.vcd'


# ------------------------------------------------------------------
# Exports / hierarchy
# ------------------------------------------------------------------


def test_vcd_reader_exported():
    assert VcdReader.__name__ == 'VcdReader'


def test_vcd_reader_top_scope_list(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        top = reader.top_scope_list()
        tb = top[0]
        dut = next(scope for scope in _scopes(tb) if scope.name == 'dut')

        assert tb.name == 'compare_tb'
        assert dut.name == 'dut'
        assert {sig.base_name for sig in _signals(tb)} == {'clk', 'rst_n'}
        assert {sig.base_name for sig in _signals(dut)} >= {'clk', 'rst_n', 'counter', 'status'}

        child_names = {scope.name for scope in _scopes(dut)}
        assert {'unit_a', 'unit_b'} <= child_names

        unit_a = next(scope for scope in _scopes(dut) if scope.name == 'unit_a')
        unit_a_signals = {sig.base_name for sig in _signals(unit_a)}
        assert {'data', 'nonzero_data', 'zero_range', 'bus', 'data_0', 'data_1'} <= unit_a_signals
        unit_a_children = {scope.name for scope in _scopes(unit_a)}
        assert {'pkt', 'u', 'gen_blk[0]', 'gen_blk[1]', 'gen_blk[2]'} <= unit_a_children


# ------------------------------------------------------------------
# Basic load / range metadata (jtag.vcd)
# ------------------------------------------------------------------


def test_vcd_reader_load_waveform(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    j_state = vcd_reader.load_waveform(
        'tb.u0.J_state[3:0]',
        clock='tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert j_state.signal.full_name == 'tb.u0.J_state[3:0]'
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

    assert j_next.signal.full_name == 'tb.u0.J_next[3:0]'
    assert j_next.width == 4


def test_vcd_reader_subrange_load(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        low_bits = reader.load_waveform('tb.u0.J_state[1:0]', clock='tb.tck')
        matched_low_bits = reader.load_matched_waveforms('tb.u0.J_state[1:0]', 'tb.tck')[()]

    assert low_bits.width == 2
    assert matched_low_bits.width == 2
    assert np.array_equal(matched_low_bits.value, low_bits.value)


def test_vcd_reader_midrange_load(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')
        high_bits = reader.load_waveform('tb.u0.J_state[3:2]', clock='tb.tck')

    assert high_bits.width == 2
    assert np.array_equal(high_bits.value, (full.value >> 2) & 0x3)


def test_vcd_reader_native_range_metadata(nonzero_vcd_path):
    with VcdReader(str(nonzero_vcd_path)) as reader:
        tb = _scopes(reader.top_scope_list()[0])[0]
        signals = {sig.base_name: sig for sig in _signals(tb)}

    assert signals['packed_vec'].full_name == 'TOP.tb.packed_vec[3:0]'
    assert signals['packed_vec'].range == Range(3, 0)
    assert signals['packed_vec'].native_range == Range(3, 0)
    assert signals['packed_nonzero'].native_range == Range(7, 4)
    assert signals['packed_nonzero'].range == Range(7, 4)
    assert signals['packed_arr[10]'].native_range == Range(2, 0)
    assert signals['packed_arr[10]'].range == Range(2, 0)
    assert signals['arr_elem[10][0]'].width == 1
    assert signals['arr_elem[10][0]'].range is None
    assert signals['arr_elem[10][0]'].native_range is None
    assert signals['zero_range'].full_name == 'TOP.tb.zero_range[0]'
    assert signals['zero_range'].range == Range(0, 0)
    assert signals['zero_range'].native_range == Range(0, 0)
    assert signals['zero_range'].width == 1


def test_vcd_reader_scalar_bit_select(nonzero_vcd_path):
    with VcdReader(str(nonzero_vcd_path)) as reader:
        clk = reader.load_waveform('TOP.tb.clk', clock='TOP.tb.clk', begin_cycle=0, end_cycle=3)
        clk_bit0 = reader.load_waveform(
            'TOP.tb.clk[0]', clock='TOP.tb.clk', begin_cycle=0, end_cycle=3
        )

    assert clk.width == 1
    assert clk_bit0.width == 1
    assert np.array_equal(clk_bit0.value, clk.value)


def test_vcd_reader_single_bracket_array_element_load(nonzero_vcd_path):
    with VcdReader(str(nonzero_vcd_path)) as reader:
        elem0 = reader.load_waveform(
            'TOP.tb.arr_elem[10][0]', clock='TOP.tb.clk', begin_cycle=0, end_cycle=3
        )
        elem1 = reader.load_waveform(
            'TOP.tb.arr_elem[10][1]', clock='TOP.tb.clk', begin_cycle=0, end_cycle=3
        )

    assert elem0.width == 1
    assert elem1.width == 1
    assert np.array_equal(elem0.value, np.array([1, 0, 0], dtype=np.uint64))
    assert np.array_equal(elem1.value, np.array([0, 1, 0], dtype=np.uint64))


def test_vcd_reader_nonzero_native_range_loads(nonzero_vcd_path):
    with VcdReader(str(nonzero_vcd_path)) as reader:
        full = reader.load_waveform(
            'TOP.tb.packed_nonzero[7:4]', clock='TOP.tb.clk', begin_cycle=0, end_cycle=3
        )
        base = reader.load_waveform(
            'TOP.tb.packed_nonzero', clock='TOP.tb.clk', begin_cycle=0, end_cycle=3
        )
        view = reader.load_waveform(
            'TOP.tb.packed_nonzero[6:5]', clock='TOP.tb.clk', begin_cycle=0, end_cycle=3
        )
        matched = reader.get_matched_signals('TOP.tb.packed_nonzero[6:5]')[()]
        masks = reader.load_matched_unknown_masks(
            'TOP.tb.packed_nonzero[6:5]', 'TOP.tb.clk', begin_cycle=0, end_cycle=3
        )

    assert np.array_equal(full.value, np.array([0b1100, 0b1010, 0b0101], dtype=np.uint64))
    assert np.array_equal(base.value, full.value)
    assert np.array_equal(view.value, np.array([0b10, 0b01, 0b10], dtype=np.uint64))
    assert view.width == 2
    assert view.signal.range == Range(6, 5)
    assert matched.base_name == 'packed_nonzero'
    assert matched.full_name == 'TOP.tb.packed_nonzero[6:5]'
    assert matched.width == 2
    assert matched.range == Range(6, 5)
    assert masks[()].width == 2
    assert np.array_equal(masks[()].value, np.zeros(3, dtype=np.uint64))


# ------------------------------------------------------------------
# Signed / signal-object / name metadata (compare.vcd)
# ------------------------------------------------------------------


def test_vcd_reader_packed_range_directions(compare_vcd_path):
    """Exercise ascending and descending nonzero packed ranges from real dumps."""
    with VcdReader(str(compare_vcd_path)) as reader:
        asc_zero_signal = reader.get_matched_signals('compare_tb.dut.unit_a.asc_zero')[()]
        asc_nonzero_signal = reader.get_matched_signals('compare_tb.dut.unit_a.asc_nonzero')[()]
        desc_nonzero_signal = reader.get_matched_signals('compare_tb.dut.unit_a.desc_nonzero')[()]

        asc_zero = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_zero', clock='compare_tb.clk', begin_cycle=1, end_cycle=4
        )
        asc_nonzero = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_nonzero',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        desc_nonzero = reader.load_waveform(
            'compare_tb.dut.unit_a.desc_nonzero',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )

        asc_zero_left = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_zero[0:1]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        asc_zero_right = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_zero[2:3]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        asc_nonzero_left = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_nonzero[1:2]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        asc_nonzero_right = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_nonzero[2:3]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        desc_nonzero_left = reader.load_waveform(
            'compare_tb.dut.unit_a.desc_nonzero[3:2]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        desc_nonzero_right = reader.load_waveform(
            'compare_tb.dut.unit_a.desc_nonzero[2:1]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )

        asc_zero_msb = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_zero[0]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        asc_zero_lsb = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_zero[3]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        asc_nonzero_msb = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_nonzero[1]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        asc_nonzero_lsb = reader.load_waveform(
            'compare_tb.dut.unit_a.asc_nonzero[3]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        desc_nonzero_msb = reader.load_waveform(
            'compare_tb.dut.unit_a.desc_nonzero[3]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        desc_nonzero_lsb = reader.load_waveform(
            'compare_tb.dut.unit_a.desc_nonzero[1]',
            clock='compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )
        matched = reader.load_matched_waveforms(
            'compare_tb.dut.unit_{a,b}.asc_nonzero[1:2]',
            'compare_tb.clk',
            begin_cycle=1,
            end_cycle=4,
        )

    assert (asc_zero_signal.native_range.start, asc_zero_signal.native_range.end) == (0, 3)
    assert (asc_nonzero_signal.native_range.start, asc_nonzero_signal.native_range.end) == (1, 3)
    assert (desc_nonzero_signal.native_range.start, desc_nonzero_signal.native_range.end) == (3, 1)
    assert asc_zero_signal.full_name == 'compare_tb.dut.unit_a.asc_zero[0:3]'
    assert asc_nonzero_signal.full_name == 'compare_tb.dut.unit_a.asc_nonzero[1:3]'
    assert desc_nonzero_signal.full_name == 'compare_tb.dut.unit_a.desc_nonzero[3:1]'

    assert np.array_equal(asc_zero.value, np.array([0b1100, 0b0011, 0b1100], dtype=np.uint64))
    assert np.array_equal(asc_nonzero.value, np.array([0b110, 0b001, 0b110], dtype=np.uint64))
    assert np.array_equal(desc_nonzero.value, asc_nonzero.value)

    assert np.array_equal(asc_zero_left.value, np.array([0b11, 0b00, 0b11], dtype=np.uint64))
    assert np.array_equal(asc_zero_right.value, np.array([0b00, 0b11, 0b00], dtype=np.uint64))
    assert np.array_equal(asc_nonzero_left.value, asc_zero_left.value)
    assert np.array_equal(desc_nonzero_left.value, asc_zero_left.value)
    assert np.array_equal(asc_nonzero_right.value, np.array([0b10, 0b01, 0b10], dtype=np.uint64))
    assert np.array_equal(desc_nonzero_right.value, asc_nonzero_right.value)

    assert np.array_equal(asc_zero_msb.value, np.array([1, 0, 1], dtype=np.uint64))
    assert np.array_equal(asc_zero_lsb.value, np.array([0, 1, 0], dtype=np.uint64))
    assert np.array_equal(asc_nonzero_msb.value, asc_zero_msb.value)
    assert np.array_equal(desc_nonzero_msb.value, asc_zero_msb.value)
    assert np.array_equal(asc_nonzero_lsb.value, asc_zero_lsb.value)
    assert np.array_equal(desc_nonzero_lsb.value, asc_zero_lsb.value)

    assert _capture_groups(matched) == {('a',), ('b',)}
    assert np.array_equal(_by_group(matched, 'a').value, asc_nonzero_left.value)
    assert np.array_equal(
        _by_group(matched, 'b').value, np.array([0b00, 0b11, 0b00], dtype=np.uint64)
    )


def test_vcd_load_waveform_signed(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        w = reader.load_waveform(
            'compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk', signed=True
        )
    assert w.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert w.signed is True
    assert w.width == 8


def test_vcd_load_waveform_signed_default(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        w = reader.load_waveform('compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk')
    assert w.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert w.signed is False


def test_vcd_load_waveform_signal_object(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        sig = reader.get_matched_signals('compare_tb.dut.unit_a.data[7:0]')[()]
        w = reader.load_waveform(sig, clock='compare_tb.clk', signed=True)
    assert w.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert w.signed is True


def test_vcd_load_waveform_subrange_name(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        w = reader.load_waveform('tb.u0.J_state[1:0]', clock='tb.tck')
    assert w.signal.full_name == 'tb.u0.J_state[1:0]'
    assert w.width == 2


# ------------------------------------------------------------------
# Pattern matching (brace / regex / range)
# ------------------------------------------------------------------


def test_vcd_load_matched_waveforms_brace_expansion(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        waves = reader.load_matched_waveforms(
            'compare_tb.dut.unit_{a,b}.data[7:0]', 'compare_tb.clk'
        )

    assert _capture_groups(waves) == {('a',), ('b',)}
    assert {wave.signal.full_name for wave in waves.values()} == {
        'compare_tb.dut.unit_a.data[7:0]',
        'compare_tb.dut.unit_b.data[7:0]',
    }
    assert all(wave.width == 8 for wave in waves.values())


def test_vcd_load_matched_waveforms_regex(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        waves = reader.load_matched_waveforms(
            r'compare_tb.dut.@(unit_a|unit_b).data[7:0]', 'compare_tb.clk'
        )

    assert len(waves) == 2
    assert _capture_groups(waves) == {('unit_a',), ('unit_b',)}
    assert all(wave.width == 8 for wave in waves.values())


def test_vcd_load_matched_waveforms_brace_expansion_jtag(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    j_states = vcd_reader.load_matched_waveforms(
        'tb.u0.J_{state,next}[3:0]',
        'tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert _capture_groups(j_states) == {('next',), ('state',)}
    assert {wave.signal.full_name for wave in j_states.values()} == {
        'tb.u0.J_next[3:0]',
        'tb.u0.J_state[3:0]',
    }
    assert all(wave.width == 4 for wave in j_states.values())


def test_vcd_reader_load_matched_waveforms_regex(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    j_regex = vcd_reader.load_matched_waveforms(
        r'tb.u0.@J_([a-z]+)[3:0]',
        'tb.tck',
        signed=True,
        sample_on_posedge=False,
    )

    assert len(j_regex) == 2
    assert _capture_groups(j_regex) == {('next',), ('state',)}
    assert all(wave.width == 4 for wave in j_regex.values())


def test_vcd_reader_load_matched_waveforms_regex_without_groups(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))

    waves = vcd_reader.load_matched_waveforms(r'tb.u0./(?:J_state\[3:0\]|J_next\[3:0\])/', 'tb.tck')
    assert len(waves) > 1


def test_vcd_load_matched_waveforms_uses_signal_range(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        waves = reader.load_matched_waveforms('compare_tb.dut.unit_a.data', 'compare_tb.clk')

    assert list(waves.keys()) == [()]
    wave = waves[()]
    assert wave.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert wave.width == 8


def test_vcd_load_matched_waveforms_name_with_brace(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        waves = reader.load_matched_waveforms('tb.u0.J_{state,next}[3:0]', 'tb.tck', signed=True)
    assert _by_group(waves, 'state').signal.full_name == 'tb.u0.J_state[3:0]'
    assert _by_group(waves, 'next').signal.full_name == 'tb.u0.J_next[3:0]'
    assert _by_group(waves, 'state').signed is True
    assert _by_group(waves, 'next').signed is True


def test_vcd_reader_module_name_matching_is_unsupported(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        with pytest.raises(ValueError):
            reader.get_matched_signals('tb.$u0.J_state[3:0]')


def test_vcd_reader_clock_pattern_error(vcd_path):
    vcd_reader = VcdReader(str(vcd_path))
    with pytest.raises(Exception):
        vcd_reader.load_matched_waveforms('tb.u0.J_state[3:0]', 'tb.no_clock')


def test_vcd_reader_clock_pattern_key_mismatch_error(vcd_path):
    # clock brace expansion yields different keys than the signal pattern
    vcd_reader = VcdReader(str(vcd_path))
    with pytest.raises(Exception, match='no clock key is a prefix'):
        vcd_reader.load_matched_waveforms(
            'tb.u0.J_{state,next}[3:0]',  # keys: {('state',), ('next',)}
            'tb.{tck,tms}',  # keys: {('tck',), ('tms',)} — mismatch
        )


def test_vcd_reader_load_waveform_no_match_raises(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        with pytest.raises(ValueError, match="signal 'compare_tb.dut.unit_a.nope' not found"):
            reader.load_waveform('compare_tb.dut.unit_a.nope', clock='compare_tb.clk')


# ------------------------------------------------------------------
# Unknown-mask tests (compare_xz.vcd — real X/Z states)
# ------------------------------------------------------------------


def test_vcd_reader_load_unknown_mask_include_flags(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        both = reader.load_unknown_mask('compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk')
        x_only = reader.load_unknown_mask(
            'compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk', include_z=False
        )
        z_only = reader.load_unknown_mask(
            'compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk', include_x=False
        )
        values = reader.load_waveform(
            'compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk', xz_value=0
        )

    assert both.signal.full_name == 'compare_xz_tb.bus[3:0]'
    assert both.width == 4
    assert both.signed is False
    assert np.array_equal(both.value, np.array([0, 15, 15, 2, 5, 0], dtype=np.uint64))
    assert np.array_equal(x_only.value, np.array([0, 15, 0, 2, 1, 0], dtype=np.uint64))
    assert np.array_equal(z_only.value, np.array([0, 0, 15, 0, 4, 0], dtype=np.uint64))
    assert np.array_equal(both.clock, values.clock)
    assert np.array_equal(both.time, values.time)


def test_vcd_reader_load_unknown_mask_range_selection(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        full = reader.load_unknown_mask('compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk')
        mid = reader.load_unknown_mask('compare_xz_tb.bus[3:2]', clock='compare_xz_tb.clk')
        low = reader.load_unknown_mask('compare_xz_tb.bus[1:0]', clock='compare_xz_tb.clk')

    assert mid.width == 2
    assert mid.signal.full_name == 'compare_xz_tb.bus[3:2]'
    assert np.array_equal(mid.value, (full.value >> 2) & 0x3)
    assert low.width == 2
    assert low.signal.full_name == 'compare_xz_tb.bus[1:0]'
    assert np.array_equal(low.value, full.value & 0x3)


def test_vcd_load_unknown_mask_name_signed(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        w = reader.load_unknown_mask('compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk')
    assert w.signal.full_name == 'compare_xz_tb.bus[3:0]'
    assert w.signed is False


def test_vcd_load_unknown_mask_signal_object_name(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        sig = reader.get_matched_signals('compare_xz_tb.bus[3:0]')[()]
        w = reader.load_unknown_mask(sig, clock='compare_xz_tb.clk')
    assert w.signal.full_name == 'compare_xz_tb.bus[3:0]'
    assert w.signed is False


def test_vcd_reader_load_unknown_mask_fully_known_is_zero(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        # data_0 is fully known at cycle 0 (mask bit pattern all zero there)
        mask = reader.load_unknown_mask(
            'compare_xz_tb.data_0[3:0]', clock='compare_xz_tb.clk', begin_cycle=0, end_cycle=1
        )

    assert np.array_equal(mask.value, np.array([0], dtype=np.uint64))


def test_vcd_reader_load_unknown_mask_both_false_is_all_zero(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        mask = reader.load_unknown_mask(
            'compare_xz_tb.bus[3:0]',
            clock='compare_xz_tb.clk',
            include_x=False,
            include_z=False,
        )

    assert mask.signal.full_name == 'compare_xz_tb.bus[3:0]'
    assert mask.width == 4
    assert np.array_equal(mask.value, np.zeros(len(mask.value), dtype=np.uint64))


def test_vcd_load_matched_unknown_masks_name(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        masks = reader.load_matched_unknown_masks(
            'compare_xz_tb.data_{0,1}[3:0]', 'compare_xz_tb.clk'
        )
    assert _by_group(masks, '0').signal.full_name == 'compare_xz_tb.data_0[3:0]'
    assert _by_group(masks, '1').signal.full_name == 'compare_xz_tb.data_1[3:0]'
    assert _by_group(masks, '0').signed is False
    assert _by_group(masks, '1').signed is False


def test_vcd_reader_load_matched_unknown_masks(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        masks = reader.load_matched_unknown_masks(
            'compare_xz_tb.data_{0,1}[3:0]', 'compare_xz_tb.clk'
        )
        values = reader.load_matched_waveforms(
            'compare_xz_tb.data_{0,1}[3:0]', 'compare_xz_tb.clk', xz_value=0
        )

    assert _capture_groups(masks) == _capture_groups(values) == {('0',), ('1',)}
    assert _by_group(masks, '0').signal.full_name == 'compare_xz_tb.data_0[3:0]'
    assert _by_group(masks, '1').signal.full_name == 'compare_xz_tb.data_1[3:0]'
    # data_0 has X bits at cycles 2 (bit 3) and 3 (bit 0); data_1 has X/Z bits at cycles 1,3
    assert np.any(_by_group(masks, '0').value != 0)
    assert np.any(_by_group(masks, '1').value != 0)


def test_vcd_reader_matched_unknown_mask_both_false_is_all_zero(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        masks = reader.load_matched_unknown_masks(
            'compare_xz_tb.data_{0,1}[3:0]',
            'compare_xz_tb.clk',
            include_x=False,
            include_z=False,
        )

    assert _capture_groups(masks) == {('0',), ('1',)}
    assert np.array_equal(
        _by_group(masks, '0').value, np.zeros(len(_by_group(masks, '0').value), dtype=np.uint64)
    )
    assert np.array_equal(
        _by_group(masks, '1').value, np.zeros(len(_by_group(masks, '1').value), dtype=np.uint64)
    )


# ------------------------------------------------------------------
# Verilator composite scope reads (unknown_states.vcd — backend-specific)
# ------------------------------------------------------------------


def test_vcd_reader_verilator_composites_expose_structs_as_scopes(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        top = reader.top_scope_list()[0]
        tb = _scopes(top)[0]
        signals = {sig.base_name: sig for sig in _signals(tb)}
        children = {scope.name: scope for scope in _scopes(tb)}

    assert top.name == 'TOP'
    assert tb.name == 'tb'
    assert 'pkt[3:0]' not in signals
    assert 'packed_arr[32:0]' not in signals
    assert 'pkt_packed_arr[7:0]' not in signals
    assert signals['packed_arr[0]'].width == 3
    assert signals['packed_arr[0]'].native_range == Range(2, 0)
    assert signals['packed_arr[0]'].composite_type is None
    assert signals['packed_arr[10]'].width == 3
    assert signals['unpacked_arr[0]'].width == 11
    assert signals['unpacked_arr[1]'].composite_type is None
    assert set(children) == {
        'pkt',
        'pkt_arr[0]',
        'pkt_arr[1]',
        'pkt_packed_arr[0]',
        'pkt_packed_arr[1]',
    }
    assert {sig.base_name for sig in _signals(children['pkt'])} == {'valid', 'data'}
    assert all(sig.composite_type is None for sig in _signals(children['pkt']))


def test_vcd_reader_verilator_packed_struct_member_reads(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        valid = reader.load_waveform(
            'TOP.tb.pkt.valid', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        data = reader.load_waveform(
            'TOP.tb.pkt.data[2:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )

    assert np.array_equal(valid.value, np.array([1, 1, 0, 1, 0], dtype=np.uint64))
    assert np.array_equal(data.value, np.array([1, 2, 7, 4, 5], dtype=np.uint64))
    assert np.array_equal((valid.value << 3) | data.value, np.array([9, 10, 7, 12, 5]))


def test_vcd_reader_verilator_logic_array_element_reads(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        packed_0 = reader.load_waveform(
            'TOP.tb.packed_arr[0][2:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        packed_10 = reader.load_waveform(
            'TOP.tb.packed_arr[10][2:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        unpacked_0 = reader.load_waveform(
            'TOP.tb.unpacked_arr[0][10:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        unpacked_1 = reader.load_waveform(
            'TOP.tb.unpacked_arr[1][10:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        unpacked_2 = reader.load_waveform(
            'TOP.tb.unpacked_arr[2][10:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )

    assert np.array_equal(packed_0.value, np.array([1, 2, 3, 4, 5], dtype=np.uint64))
    assert np.array_equal(packed_10.value, packed_0.value)
    assert np.array_equal(unpacked_0.value, np.array([1, 2, 3, 4, 5], dtype=np.uint64))
    assert np.array_equal(unpacked_1.value, np.array([0x101, 0x102, 0x103, 0x104, 0x105]))
    assert np.array_equal(unpacked_2.value, np.array([0x201, 0x202, 0x203, 0x204, 0x205]))


def test_vcd_reader_verilator_struct_array_member_reads(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        pkt_arr_0_valid = reader.load_waveform(
            'TOP.tb.pkt_arr[0].valid', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        pkt_arr_0_data = reader.load_waveform(
            'TOP.tb.pkt_arr[0].data[2:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        pkt_arr_1_valid = reader.load_waveform(
            'TOP.tb.pkt_arr[1].valid', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        pkt_arr_1_data = reader.load_waveform(
            'TOP.tb.pkt_arr[1].data[2:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        packed_0_valid = reader.load_waveform(
            'TOP.tb.pkt_packed_arr[0].valid', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        packed_0_data = reader.load_waveform(
            'TOP.tb.pkt_packed_arr[0].data[2:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        packed_1_valid = reader.load_waveform(
            'TOP.tb.pkt_packed_arr[1].valid', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )
        packed_1_data = reader.load_waveform(
            'TOP.tb.pkt_packed_arr[1].data[2:0]', clock='TOP.tb.clk', begin_cycle=1, end_cycle=6
        )

    assert np.array_equal(pkt_arr_0_valid.value, np.array([1, 1, 0, 1, 0], dtype=np.uint64))
    assert np.array_equal(pkt_arr_0_data.value, np.array([1, 2, 3, 4, 5], dtype=np.uint64))
    assert np.array_equal(pkt_arr_1_valid.value, np.array([0, 0, 1, 0, 1], dtype=np.uint64))
    assert np.array_equal(pkt_arr_1_data.value, np.array([6, 5, 4, 3, 2], dtype=np.uint64))
    assert np.array_equal(packed_0_valid.value, pkt_arr_0_valid.value)
    assert np.array_equal(packed_0_data.value, pkt_arr_0_data.value)
    assert np.array_equal(packed_1_valid.value, pkt_arr_1_valid.value)
    assert np.array_equal(packed_1_data.value, pkt_arr_1_data.value)


def test_vcd_reader_verilator_whole_aggregate_reads_fail(unknown_vcd_path):
    with VcdReader(str(unknown_vcd_path)) as reader:
        for signal in [
            'TOP.tb.pkt[3:0]',
            'TOP.tb.packed_arr[32:0]',
            'TOP.tb.pkt_arr[0]',
            'TOP.tb.pkt_packed_arr[7:0]',
        ]:
            with pytest.raises(ValueError, match='not found'):
                reader.load_waveform(signal, clock='TOP.tb.clk')


def test_vcd_reader_rejects_invalid_xz_value(compare_xz_vcd_path):
    with VcdReader(str(compare_xz_vcd_path)) as reader:
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.load_waveform('compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk', xz_value=2)
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.load_matched_waveforms('compare_xz_tb.bus[3:0]', 'compare_xz_tb.clk', xz_value=2)
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.eval('compare_xz_tb.bus[3:0] + 1', clock='compare_xz_tb.clk', xz_value=2)


# ------------------------------------------------------------------
# eval integration tests
# ------------------------------------------------------------------


def test_vcd_eval_smoke(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        result = reader.eval('compare_tb.dut.unit_a.data[7:0] + 1', clock='compare_tb.clk')
        bit_slice = reader.eval('compare_tb.dut.unit_a.data[7:0][1:0]', clock='compare_tb.clk')

    assert isinstance(result, Waveform)
    assert result.width == 9  # addition increases width by 1
    assert isinstance(bit_slice, Waveform)
    assert bit_slice.width == 2


def test_vcd_eval_no_match_raises(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        with pytest.raises(ValueError, match='matched no signals'):
            reader.eval('compare_tb.dut.unit_a.nonexistent', clock='compare_tb.clk')


def test_vcd_eval_error_on_multi_match(compare_vcd_path):
    with VcdReader(str(compare_vcd_path)) as reader:
        with pytest.raises(ValueError, match="mode='single'"):
            reader.eval(
                'compare_tb.dut.unit_{a,b}.data[7:0]',
                clock='compare_tb.clk',
                mode='single',
            )


def test_vcd_eval_zip_mode_brace_expansion(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        result = reader.eval(
            'tb.u0.J_{state,next}[3:0] + 1',
            clock='tb.tck',
            mode='zip',
        )
    assert isinstance(result, dict)
    assert _capture_groups(result) == {('state',), ('next',)}
    assert all(isinstance(w, Waveform) for w in result.values())


def test_vcd_eval_zip_mode_broadcast(vcd_path):
    # J_state matches 1 signal (broadcast), J_{state,next} matches 2
    with VcdReader(str(vcd_path)) as reader:
        result = reader.eval(
            'tb.u0.J_{state,next}[3:0] + tb.u0.J_state[3:0]',
            clock='tb.tck',
            mode='zip',
        )
    assert isinstance(result, dict)
    assert len(result) == 2


# ------------------------------------------------------------------
# begin_time / end_time / begin_cycle / end_cycle tests (jtag.vcd)
# ------------------------------------------------------------------


def test_vcd_begin_end_time(vcd_path):
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


def test_vcd_begin_end_cycle(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')
        windowed = reader.load_waveform(
            'tb.u0.J_state[3:0]', clock='tb.tck', begin_cycle=10, end_cycle=20
        )

    assert len(windowed.value) == 10
    assert windowed.clock[0] == 10
    assert windowed.clock[-1] == 19
    assert np.array_equal(windowed.value, full.value[10:20])


def test_vcd_cycle_equals_time_window(vcd_path):
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


def test_vcd_mutually_exclusive_errors(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform(
                'tb.u0.J_state[3:0]',
                clock='tb.tck',
                begin_time=100,
                begin_cycle=10,
            )
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform(
                'tb.u0.J_state[3:0]',
                clock='tb.tck',
                end_time=200,
                end_cycle=20,
            )


# ------------------------------------------------------------------
# value_change -> waveform internals
# ------------------------------------------------------------------


def test_vcd_value_change_to_waveform_sample_on_posedge():
    value_change = np.array([[0, 0], [5, 1], [10, 0]], dtype=np.uint64)
    clock_changes = np.array([[0, 0], [5, 1], [10, 0], [15, 1]], dtype=np.uint64)

    wave = Reader._value_change_to_waveform(
        value_change,
        clock_changes,
        width=1,
        signed=False,
        sample_on_posedge=True,
    )

    assert np.all(wave.value == np.array([1, 0]))
    assert np.all(wave.clock == np.array([0, 1]))
    assert np.all(wave.time == np.array([5, 15]))
    assert wave.signal is None


def test_vcd_value_change_to_waveform_clock_offset():
    # Verify that clock_offset shifts the .clock array to start from a given value
    value_change = np.array([[0, 0], [5, 1], [10, 0]], dtype=np.uint64)
    clock_changes = np.array([[0, 0], [5, 1], [10, 0], [15, 1]], dtype=np.uint64)

    wave_no_offset = Reader._value_change_to_waveform(
        value_change,
        clock_changes,
        width=1,
        signed=False,
        sample_on_posedge=True,
    )
    wave_with_offset = Reader._value_change_to_waveform(
        value_change,
        clock_changes,
        width=1,
        signed=False,
        sample_on_posedge=True,
        clock_offset=50,
    )

    assert np.all(wave_no_offset.clock == np.array([0, 1]))
    assert np.all(wave_with_offset.clock == np.array([50, 51]))
    # Values are the same regardless of offset
    assert np.array_equal(wave_no_offset.value, wave_with_offset.value)


# ------------------------------------------------------------------
# cycle_slice helpers
# ------------------------------------------------------------------


def test_vcd_cycle_slice(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')

    sliced = full.cycle_slice(10, 20)
    assert len(sliced.value) == 10
    assert sliced.clock[0] == 10
    assert sliced.clock[-1] == 19
    assert np.array_equal(sliced.value, full.value[10:20])


def test_vcd_cycle_slice_include_end(vcd_path):
    with VcdReader(str(vcd_path)) as reader:
        full = reader.load_waveform('tb.u0.J_state[3:0]', clock='tb.tck')

    sliced = full.cycle_slice(10, 20, include_end=True)
    assert len(sliced.value) == 11
    assert sliced.clock[-1] == 20
