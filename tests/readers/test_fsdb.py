from pathlib import Path

import numpy as np
import pytest

from wavekit import (
    BraceCapture,
    ExactCapture,
    FsdbReader,
    RegexCapture,
    Scope,
    Signal,
    SignalCompositeType,
    Waveform,
    has_fsdb_support,
)
from wavekit.readers.range import Range


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


@pytest.fixture(scope='module')
def simple_fsdb_path():
    path = Path(__file__).resolve().parent / 'fixtures' / 'fsdb' / 'simple.fsdb'
    if not path.exists():
        pytest.skip(
            'simple.fsdb fixture is unavailable; run tests/readers/fixtures/fsdb/build_fsdb.sh'
        )
    return path


@pytest.fixture(scope='module')
def compare_fsdb_path():
    path = Path(__file__).resolve().parent / 'fixtures' / 'fsdb' / 'compare.fsdb'
    if not path.exists():
        pytest.skip('compare.fsdb fixture is unavailable')
    return path


@pytest.fixture(scope='module')
def compare_xz_fsdb_path():
    path = Path(__file__).resolve().parent / 'fixtures' / 'fsdb' / 'compare_xz.fsdb'
    if not path.exists():
        pytest.skip('compare_xz.fsdb fixture is unavailable')
    return path


@pytest.fixture(scope='module')
def fsdb_runtime(simple_fsdb_path):
    if not has_fsdb_support():
        pytest.skip('FSDB tests require the Verdi NPI runtime')
    return simple_fsdb_path


@pytest.fixture(scope='module')
def compare_fsdb(compare_fsdb_path):
    if not has_fsdb_support():
        pytest.skip('FSDB tests require the Verdi NPI runtime')
    return compare_fsdb_path


@pytest.fixture(scope='module')
def compare_xz_fsdb(compare_xz_fsdb_path):
    if not has_fsdb_support():
        pytest.skip('FSDB tests require the Verdi NPI runtime')
    return compare_xz_fsdb_path


# ------------------------------------------------------------------
# Exports / hierarchy
# ------------------------------------------------------------------


def test_fsdb_reader_exported(fsdb_runtime):
    assert FsdbReader.__name__ == 'FsdbReader'


def test_fsdb_reader_top_scopes(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        top = reader.top_scopes
        tb = top[0]
        dut = next(scope for scope in _scopes(tb) if scope.name == 'dut')

        assert tb.name == 'compare_tb'
        assert dut.name == 'dut'
        # fsdb_file is a VCS artifact signal; clk/rst_n are the real top-level signals
        assert {sig.base_name for sig in _signals(tb)} >= {'clk', 'rst_n'}
        assert {sig.base_name for sig in _signals(dut)} >= {'clk', 'rst_n', 'counter', 'status'}

        child_names = {scope.name for scope in _scopes(dut)}
        assert {'unit_a', 'unit_b'} <= child_names

        unit_a = next(scope for scope in _scopes(dut) if scope.name == 'unit_a')
        unit_a_signals = {sig.base_name for sig in _signals(unit_a)}
        assert {'data', 'nonzero_data', 'zero_range', 'bus', 'data_0', 'data_1'} <= unit_a_signals
        unit_a_children = {scope.name for scope in _scopes(unit_a)}
        assert {'gen_blk[0]', 'gen_blk[1]', 'gen_blk[2]'} <= unit_a_children


# ------------------------------------------------------------------
# Basic load / range metadata (simple.fsdb)
# ------------------------------------------------------------------


def test_fsdb_reader_native_range_metadata(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        tb = reader.top_scopes[0]
        signals = {sig.base_name: sig for sig in _signals(tb)}

    assert signals['nonzero_vec'].width == 4
    assert signals['nonzero_vec'].range == Range(7, 4)
    assert signals['nonzero_vec'].full_name == 'simple_tb.nonzero_vec[7:4]'
    assert signals['data_i'].range == Range(3, 0)
    assert signals['data_i'].full_name == 'simple_tb.data_i[3:0]'
    assert signals['clk'].range is None
    assert signals['clk'].full_name == 'simple_tb.clk'


def test_fsdb_reader_scalar_bit_select(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        clk = reader.load_waveform('simple_tb.clk', clock='simple_tb.clk', end_cycle=6)
        clk_bit0 = reader.load_waveform('simple_tb.clk[0]', clock='simple_tb.clk', end_cycle=6)

    assert clk.width == 1
    assert clk_bit0.width == 1
    assert np.array_equal(clk_bit0.value, clk.value)


def test_fsdb_reader_nonzero_range_subrange(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        full = reader.load_waveform(
            'simple_tb.nonzero_vec[7:4]', clock='simple_tb.clk', end_cycle=6
        )
        view = reader.load_waveform(
            'simple_tb.nonzero_vec[6:5]', clock='simple_tb.clk', end_cycle=6
        )

    assert full.width == 4
    assert view.width == 2
    assert np.array_equal(view.value, (full.value >> 1) & 0x3)


def test_fsdb_reader_nonzero_native_range_loads(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        full = reader.load_waveform(
            'compare_tb.dut.unit_a.nonzero_data[7:4]', clock='compare_tb.clk'
        )
        base = reader.load_waveform('compare_tb.dut.unit_a.nonzero_data', clock='compare_tb.clk')
        view = reader.load_waveform(
            'compare_tb.dut.unit_a.nonzero_data[6:5]', clock='compare_tb.clk'
        )
        matched = reader.get_matched_signals('compare_tb.dut.unit_a.nonzero_data[6:5]')[()]
        masks = reader.load_matched_unknown_masks(
            'compare_tb.dut.unit_a.nonzero_data[6:5]', 'compare_tb.clk'
        )

    assert full.signal.range == Range(7, 4)
    assert np.array_equal(base.value, full.value)
    assert view.width == 2
    assert view.signal.range == Range(6, 5)
    assert matched.base_name == 'nonzero_data'
    assert matched.full_name == 'compare_tb.dut.unit_a.nonzero_data[6:5]'
    assert matched.width == 2
    assert matched.range == Range(6, 5)
    assert masks[()].width == 2
    assert np.array_equal(masks[()].value, np.zeros(len(masks[()].value), dtype=np.uint64))


def test_fsdb_reader_packed_range_directions(compare_fsdb):
    """Exercise ascending and descending nonzero packed ranges from real dumps."""
    with FsdbReader(str(compare_fsdb)) as reader:
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


# ------------------------------------------------------------------
# Signed / signal-object / name metadata (compare.fsdb)
# ------------------------------------------------------------------


def test_fsdb_reader_load_waveform(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        w = reader.load_waveform(
            'compare_tb.dut.unit_a.data[7:0]',
            clock='compare_tb.clk',
            signed=True,
            sample_on_posedge=False,
        )

    assert w.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert w.width == 8
    assert w.signed is True
    assert len(w.value) > 0


def test_fsdb_load_waveform_signed(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        w = reader.load_waveform(
            'compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk', signed=True
        )
    assert w.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert w.signed is True
    assert w.width == 8


def test_fsdb_load_waveform_signed_default(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        w = reader.load_waveform('compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk')
    assert w.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert w.signed is False


def test_fsdb_load_waveform_signal_object(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        sig = reader.get_matched_signals('compare_tb.dut.unit_a.data[7:0]')[()]
        w = reader.load_waveform(sig, clock='compare_tb.clk', signed=True)
    assert w.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert w.signed is True


def test_fsdb_load_waveform_subrange_name(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        w = reader.load_waveform('simple_tb.dut.data_o[1:0]', clock='simple_tb.clk')
    assert w.signal.full_name == 'simple_tb.dut.data_o[1:0]'
    assert w.width == 2


def test_fsdb_reader_load_waveform_without_range(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        data = reader.load_waveform('simple_tb.dut.data_o', clock='simple_tb.clk')

    assert data.signal.full_name == 'simple_tb.dut.data_o[3:0]'
    assert data.width == 4
    assert data.signed is False
    assert len(data.value) > 0
    assert np.array_equal(data.clock[:5], np.arange(5, dtype=np.uint64))


def test_fsdb_reader_subrange_load(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        full = reader.load_waveform('simple_tb.dut.data_o[3:0]', clock='simple_tb.clk')
        low_bits = reader.load_waveform('simple_tb.dut.data_o[1:0]', clock='simple_tb.clk')
        matched_low_bits = reader.load_matched_waveforms(
            'simple_tb.dut.data_o[1:0]', 'simple_tb.clk'
        )[()]

    assert low_bits.width == 2
    assert matched_low_bits.width == 2
    assert np.array_equal(low_bits.value, full.value & 0x3)
    assert np.array_equal(matched_low_bits.value, low_bits.value)


def test_fsdb_reader_midrange_load(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        full = reader.load_waveform('simple_tb.dut.data_o[3:0]', clock='simple_tb.clk')
        high_bits = reader.load_waveform('simple_tb.dut.data_o[3:2]', clock='simple_tb.clk')

    assert high_bits.width == 2
    assert np.array_equal(high_bits.value, (full.value >> 2) & 0x3)


def test_fsdb_reader_single_bracket_array_element_load(compare_fsdb):
    # FSDB exposes generate-block leaves as scope + signal (not composite array elements).
    # gen_blk[i].gen_sig is a 4-bit leaf signal under a gen_blk[i] scope.
    with FsdbReader(str(compare_fsdb)) as reader:
        g0 = reader.load_waveform(
            'compare_tb.dut.unit_a.gen_blk[0].gen_sig[3:0]', clock='compare_tb.clk'
        )
        g1 = reader.load_waveform(
            'compare_tb.dut.unit_a.gen_blk[1].gen_sig[3:0]', clock='compare_tb.clk'
        )

    assert g0.width == 4
    assert g1.width == 4
    assert g0.signal.full_name == 'compare_tb.dut.unit_a.gen_blk[0].gen_sig[3:0]'
    assert g1.signal.full_name == 'compare_tb.dut.unit_a.gen_blk[1].gen_sig[3:0]'


# ------------------------------------------------------------------
# Pattern matching (brace / regex / range)
# ------------------------------------------------------------------


def test_fsdb_load_matched_waveforms_brace_expansion(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        waves = reader.load_matched_waveforms(
            'compare_tb.dut.unit_{a,b}.data[7:0]', 'compare_tb.clk'
        )

    assert _capture_groups(waves) == {('a',), ('b',)}
    assert {wave.signal.full_name for wave in waves.values()} == {
        'compare_tb.dut.unit_a.data[7:0]',
        'compare_tb.dut.unit_b.data[7:0]',
    }
    assert all(wave.width == 8 for wave in waves.values())


def test_fsdb_reader_load_matched_waveforms_regex(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        waves = reader.load_matched_waveforms(
            r'compare_tb.dut./(unit_a|unit_b)/.data[7:0]', 'compare_tb.clk'
        )

    assert len(waves) == 2
    assert _capture_groups(waves) == {('unit_a',), ('unit_b',)}
    assert all(wave.width == 8 for wave in waves.values())


def test_fsdb_reader_load_matched_waveforms_regex_key_conflict(fsdb_runtime):
    # @regex without a capture group: all matches map to the same key -> conflict
    with FsdbReader(str(fsdb_runtime)) as reader:
        waves = reader.load_matched_waveforms(r'simple_tb.dut.@[a-z]+', 'simple_tb.clk')
        assert len(waves) > 1


def test_fsdb_load_matched_waveforms_uses_signal_range(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        waves = reader.load_matched_waveforms('compare_tb.dut.unit_a.data', 'compare_tb.clk')

    assert list(waves.keys()) == [()]
    wave = waves[()]
    assert wave.signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'
    assert wave.width == 8


def test_fsdb_reader_load_matched_waveforms(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        waves = reader.load_matched_waveforms(
            'simple_tb.dut.{data_o[3:0],overflow}', 'simple_tb.clk'
        )

    assert _capture_groups(waves) == {('data_o[3:0]',), ('overflow',)}
    assert _by_group(waves, 'data_o[3:0]').width == 4
    assert _by_group(waves, 'overflow').width == 1


def test_fsdb_reader_clock_path_error(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        with pytest.raises(Exception):
            reader.load_matched_waveforms('simple_tb.dut.data_o[3:0]', 'simple_tb.no_clock')


def test_fsdb_reader_clock_path_key_mismatch_error(fsdb_runtime):
    # clock brace expansion yields different keys than the signal pattern
    with FsdbReader(str(fsdb_runtime)) as reader:
        with pytest.raises(Exception, match='no clock key is a prefix'):
            reader.load_matched_waveforms(
                'simple_tb.dut.{data_o[3:0],overflow}',  # keys: data_o[3:0], overflow
                'simple_tb.{clk,rst_n}',  # keys: clk, rst_n — mismatch
            )


def test_fsdb_reader_load_waveform_no_match_raises(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        with pytest.raises(ValueError, match="signal 'compare_tb.dut.unit_a.nope' not found"):
            reader.load_waveform('compare_tb.dut.unit_a.nope', clock='compare_tb.clk')


# ------------------------------------------------------------------
# FSDB-specific: $ / $$ module-name matching (compare.fsdb)
# ------------------------------------------------------------------


def test_fsdb_reader_dollar_module_name_matching(compare_fsdb):
    # $ matches a direct-child scope by module/definition name.
    # compare_dut instantiates two compare_unit modules (unit_a, unit_b).
    with FsdbReader(str(compare_fsdb)) as reader:
        matched = reader.get_matched_signals('compare_tb.dut.$compare_unit.data[7:0]')

    captures = {key[0] for key in matched}
    assert all(isinstance(capture, ExactCapture) for capture in captures)
    assert {capture.path for capture in captures} == {'unit_a', 'unit_b'}
    assert {capture.definition for capture in captures} == {'compare_unit'}
    assert {s.full_name for s in matched.values()} == {
        'compare_tb.dut.unit_a.data[7:0]',
        'compare_tb.dut.unit_b.data[7:0]',
    }


def test_fsdb_reader_dollar_dollar_module_name_matching(compare_fsdb):
    # $$ matches any-depth descendant scope by module/definition name.
    with FsdbReader(str(compare_fsdb)) as reader:
        matched = reader.get_matched_signals('compare_tb.$$compare_unit.data[7:0]')

    captures = {key[0] for key in matched}
    assert all(isinstance(capture, ExactCapture) for capture in captures)
    assert {capture.path for capture in captures} == {'dut.unit_a', 'dut.unit_b'}
    assert {capture.definition for capture in captures} == {'compare_unit'}


def test_fsdb_reader_module_regex_combines_with_signal_brace(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        matched = reader.get_matched_signals(r'compare_tb.dut.$/compare_(unit)/.data_{0,1}')

    assert len(matched) == 4
    assert all(isinstance(key[0], RegexCapture) for key in matched)
    assert all(isinstance(key[1], BraceCapture) for key in matched)
    assert {key[0].path for key in matched} == {'unit_a', 'unit_b'}
    assert {key[0].definition for key in matched} == {'compare_unit'}
    assert {key[0].groups for key in matched} == {('unit',)}
    assert {key[1].path for key in matched} == {'data_0', 'data_1'}
    assert {key[1].groups for key in matched} == {('0',), ('1',)}


def test_fsdb_reader_recursive_module_partial_brace(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        matched = reader.get_matched_scopes(r'compare_tb.$$compare_{dut,unit}')

    assert len(matched) == 3
    captures = {key[0] for key in matched}
    assert all(isinstance(capture, BraceCapture) for capture in captures)
    assert {capture.path for capture in captures} == {
        'dut',
        'dut.unit_a',
        'dut.unit_b',
    }
    assert {capture.definition for capture in captures} == {'compare_dut', 'compare_unit'}
    assert {capture.groups for capture in captures} == {('dut',), ('unit',)}


def test_fsdb_reader_signal_regex_trailing_range(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        matched = reader.get_matched_signals(r'compare_tb.dut.unit_a./data/[1:0]')

    assert len(matched) == 1
    (capture,) = next(iter(matched))
    signal = next(iter(matched.values()))
    assert isinstance(capture, RegexCapture)
    assert capture.path == 'data[1:0]'
    assert capture.groups == ()
    assert signal.range == Range(1, 0)
    assert signal.full_name == 'compare_tb.dut.unit_a.data[1:0]'


def test_fsdb_reader_signal_regex_escaped_native_range(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        matched = reader.get_matched_signals(r'compare_tb.dut.unit_a./data\[7:0\]/')

    assert len(matched) == 1
    (capture,) = next(iter(matched))
    signal = next(iter(matched.values()))
    assert isinstance(capture, RegexCapture)
    assert capture.path == 'data[7:0]'
    assert capture.groups == ()
    assert signal.range == Range(7, 0)
    assert signal.full_name == 'compare_tb.dut.unit_a.data[7:0]'


def test_fsdb_reader_scope_regex_escaped_brackets(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        matched = reader.get_matched_signals(r'compare_tb.dut.unit_a./gen_blk\[0\]/.gen_sig[3:0]')

    assert len(matched) == 1
    (capture,) = next(iter(matched))
    signal = next(iter(matched.values()))
    assert isinstance(capture, RegexCapture)
    assert capture.path == 'gen_blk[0]'
    assert capture.groups == ()
    assert signal.full_name == 'compare_tb.dut.unit_a.gen_blk[0].gen_sig[3:0]'


# ------------------------------------------------------------------
# Unknown-mask tests (compare_xz.fsdb — real X/Z states)
# ------------------------------------------------------------------


def test_fsdb_reader_load_unknown_mask_include_flags(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
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


def test_fsdb_reader_load_unknown_mask_range_selection(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
        full = reader.load_unknown_mask('compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk')
        mid = reader.load_unknown_mask('compare_xz_tb.bus[3:2]', clock='compare_xz_tb.clk')
        low = reader.load_unknown_mask('compare_xz_tb.bus[1:0]', clock='compare_xz_tb.clk')

    assert mid.width == 2
    assert mid.signal.full_name == 'compare_xz_tb.bus[3:2]'
    assert np.array_equal(mid.value, (full.value >> 2) & 0x3)
    assert low.width == 2
    assert low.signal.full_name == 'compare_xz_tb.bus[1:0]'
    assert np.array_equal(low.value, full.value & 0x3)


def test_fsdb_load_unknown_mask_name_signed(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
        w = reader.load_unknown_mask('compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk')
    assert w.signal.full_name == 'compare_xz_tb.bus[3:0]'
    assert w.signed is False


def test_fsdb_load_unknown_mask_signal_object_name(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
        sig = reader.get_matched_signals('compare_xz_tb.bus[3:0]')[()]
        w = reader.load_unknown_mask(sig, clock='compare_xz_tb.clk')
    assert w.signal.full_name == 'compare_xz_tb.bus[3:0]'
    assert w.signed is False


def test_fsdb_reader_load_unknown_mask_fully_known_is_zero(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
        # data_0 is fully known at cycle 0 (mask bit pattern all zero there)
        mask = reader.load_unknown_mask(
            'compare_xz_tb.data_0[3:0]', clock='compare_xz_tb.clk', begin_cycle=0, end_cycle=1
        )

    assert np.array_equal(mask.value, np.array([0], dtype=np.uint64))


def test_fsdb_reader_load_unknown_mask_both_false_is_all_zero(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
        mask = reader.load_unknown_mask(
            'compare_xz_tb.bus[3:0]',
            clock='compare_xz_tb.clk',
            include_x=False,
            include_z=False,
        )

    assert mask.signal.full_name == 'compare_xz_tb.bus[3:0]'
    assert mask.width == 4
    assert np.array_equal(mask.value, np.zeros(len(mask.value), dtype=np.uint64))


def test_fsdb_load_matched_unknown_masks_name(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
        masks = reader.load_matched_unknown_masks(
            'compare_xz_tb.data_{0,1}[3:0]', 'compare_xz_tb.clk'
        )
    assert _by_group(masks, '0').signal.full_name == 'compare_xz_tb.data_0[3:0]'
    assert _by_group(masks, '1').signal.full_name == 'compare_xz_tb.data_1[3:0]'
    assert _by_group(masks, '0').signed is False
    assert _by_group(masks, '1').signed is False


def test_fsdb_reader_load_matched_unknown_masks(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
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


def test_fsdb_reader_matched_unknown_mask_both_false_is_all_zero(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
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


def test_fsdb_reader_rejects_invalid_xz_value(compare_xz_fsdb):
    with FsdbReader(str(compare_xz_fsdb)) as reader:
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.load_waveform('compare_xz_tb.bus[3:0]', clock='compare_xz_tb.clk', xz_value=2)
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.load_matched_waveforms('compare_xz_tb.bus[3:0]', 'compare_xz_tb.clk', xz_value=2)
        with pytest.raises(ValueError, match='xz_value must be 0 or 1'):
            reader.eval('compare_xz_tb.bus[3:0] + 1', clock='compare_xz_tb.clk', xz_value=2)


# ------------------------------------------------------------------
# FSDB-specific: composite signal whole-load + children (simple.fsdb)
# ------------------------------------------------------------------


def test_fsdb_reader_composite_metadata(fsdb_runtime):
    # Composite children are loaded lazily from NPI and require an open reader.
    with FsdbReader(str(fsdb_runtime)) as reader:
        tb = reader.top_scopes[0]
        signals = {sig.base_name: sig for sig in _signals(tb)}

        assert signals['pkt'].width == 4
        assert signals['pkt'].composite_type == SignalCompositeType.STRUCT
        pkt_members = {sig.base_name: sig for sig in _signals(signals['pkt'])}
        assert set(pkt_members) == {'valid', 'data'}
        assert pkt_members['valid'].width == 1
        assert pkt_members['data'].width == 3

        assert signals['pkt_union'].width == 4
        assert signals['pkt_union'].composite_type == SignalCompositeType.UNION
        union_members = {sig.base_name: sig for sig in _signals(signals['pkt_union'])}
        assert set(union_members) == {'raw', 'packet'}
        assert union_members['raw'].width == 4
        assert union_members['packet'].width == 4

        assert signals['packed_arr'].width == 33
        assert signals['packed_arr'].composite_type == SignalCompositeType.ARRAY
        packed_members = {sig.base_name: sig for sig in _signals(signals['packed_arr'])}
        assert len(packed_members) == 11
        assert packed_members['packed_arr[0]'].width == 3

        assert signals['unpacked_arr'].width == 33
        assert signals['unpacked_arr'].composite_type == SignalCompositeType.ARRAY
        unpacked_members = {sig.base_name: sig for sig in _signals(signals['unpacked_arr'])}
        assert set(unpacked_members) == {
            'unpacked_arr[0]',
            'unpacked_arr[1]',
            'unpacked_arr[2]',
        }
        assert unpacked_members['unpacked_arr[0]'].width == 11

        assert signals['pkt_arr'].width == 8
        assert signals['pkt_arr'].composite_type == SignalCompositeType.ARRAY
        pkt_arr_members = {sig.base_name: sig for sig in _signals(signals['pkt_arr'])}
        assert set(pkt_arr_members) == {'pkt_arr[0]', 'pkt_arr[1]'}
        assert pkt_arr_members['pkt_arr[0]'].composite_type == SignalCompositeType.STRUCT

        assert signals['pkt_packed_arr'].width == 8
        assert signals['pkt_packed_arr'].composite_type == SignalCompositeType.ARRAY
        pkt_packed_arr_members = {sig.base_name: sig for sig in _signals(signals['pkt_packed_arr'])}
        assert set(pkt_packed_arr_members) == {'pkt_packed_arr[0]', 'pkt_packed_arr[1]'}
        assert (
            pkt_packed_arr_members['pkt_packed_arr[0]'].composite_type == SignalCompositeType.STRUCT
        )


def test_fsdb_reader_array_matching_prefers_elements_then_range_views(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        packed_element = reader.get_matched_signals('simple_tb.packed_arr[0]')[()]
        packed_range = reader.get_matched_signals('simple_tb.packed_arr[2:1]')[()]
        unpacked_element = reader.get_matched_signals('simple_tb.unpacked_arr[0]')[()]
        element_range = reader.get_matched_signals('simple_tb.unpacked_arr[0][8:7]')[()]
        struct_element = reader.get_matched_signals('simple_tb.pkt_arr[0]')[()]
        struct_member = reader.get_matched_signals('simple_tb.pkt_arr[0].valid')[()]

        with pytest.raises(ValueError, match='cannot be followed'):
            reader.get_matched_signals('simple_tb.pkt_arr[1:0].valid')

    assert packed_element.base_name == 'packed_arr[0]'
    assert packed_element.range == Range(2, 0)
    assert packed_range.base_name == 'packed_arr'
    assert packed_range.range == Range(2, 1)
    assert unpacked_element.base_name == 'unpacked_arr[0]'
    assert unpacked_element.range == Range(10, 0)
    assert element_range.base_name == 'unpacked_arr[0]'
    assert element_range.range == Range(8, 7)
    assert struct_element.base_name == 'pkt_arr[0]'
    assert struct_element.composite_type == SignalCompositeType.STRUCT
    assert struct_member.full_name == 'simple_tb.pkt_arr[0].valid'


def test_fsdb_reader_rejects_partial_array_range_load(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        complete = reader.load_waveform(
            'simple_tb.packed_arr[10:0]', clock='simple_tb.clk', end_cycle=6
        )

        for path in ('simple_tb.packed_arr[2:1]', 'simple_tb.unpacked_arr[2:1]'):
            with pytest.raises(NotImplementedError, match='partial range of FSDB array'):
                reader.load_waveform(path, clock='simple_tb.clk', end_cycle=6)

    assert complete.width == 33


def test_fsdb_reader_packed_struct_whole_and_fields(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        pkt = reader.load_waveform('simple_tb.pkt', clock='simple_tb.clk', end_cycle=6)
        valid = reader.load_waveform('simple_tb.pkt.valid', clock='simple_tb.clk', end_cycle=6)
        data = reader.load_waveform('simple_tb.pkt.data', clock='simple_tb.clk', end_cycle=6)

    assert pkt.width == 4
    assert valid.width == 1
    assert data.width == 3
    assert np.array_equal(pkt.value, np.array([0, 0, 0b1001, 0b1010, 0b0111, 0b1100]))
    assert np.array_equal(valid.value, (pkt.value >> 3) & 0x1)
    assert np.array_equal(data.value, pkt.value & 0x7)
    assert np.array_equal(valid.clock, pkt.clock)
    assert np.array_equal(data.time, pkt.time)


def test_fsdb_reader_union_requires_loading_a_member(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        with pytest.raises(NotImplementedError, match='load one of its members instead'):
            reader.load_waveform('simple_tb.pkt_union', clock='simple_tb.clk', end_cycle=6)
        with pytest.raises(NotImplementedError, match='load one of its members instead'):
            reader.load_unknown_mask('simple_tb.pkt_union', clock='simple_tb.clk', end_cycle=6)


def test_fsdb_reader_union_members(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        raw = reader.load_waveform('simple_tb.pkt_union.raw', clock='simple_tb.clk', end_cycle=6)
        valid = reader.load_waveform(
            'simple_tb.pkt_union.packet.valid', clock='simple_tb.clk', end_cycle=6
        )
        data = reader.load_waveform(
            'simple_tb.pkt_union.packet.data', clock='simple_tb.clk', end_cycle=6
        )

    assert raw.width == 4
    assert valid.width == 1
    assert data.width == 3
    assert np.array_equal(raw.value, np.array([0, 0, 0x9, 0xA, 0x7, 0xC]))
    assert np.array_equal(valid.value, (raw.value >> 3) & 0x1)
    assert np.array_equal(data.value, raw.value & 0x7)
    assert np.array_equal(valid.clock, raw.clock)
    assert np.array_equal(data.time, raw.time)


def test_fsdb_reader_array_whole_and_elements(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        packed = reader.load_waveform('simple_tb.packed_arr', clock='simple_tb.clk', end_cycle=6)
        packed_0 = reader.load_waveform(
            'simple_tb.packed_arr[0]', clock='simple_tb.clk', end_cycle=6
        )
        packed_10 = reader.load_waveform(
            'simple_tb.packed_arr[10]', clock='simple_tb.clk', end_cycle=6
        )
        packed_elements = [
            reader.load_waveform(f'simple_tb.packed_arr[{idx}]', clock='simple_tb.clk', end_cycle=6)
            for idx in range(11)
        ]
        unpacked = reader.load_waveform(
            'simple_tb.unpacked_arr', clock='simple_tb.clk', end_cycle=6
        )
        unpacked_0 = reader.load_waveform(
            'simple_tb.unpacked_arr[0]', clock='simple_tb.clk', end_cycle=6
        )
        unpacked_1 = reader.load_waveform(
            'simple_tb.unpacked_arr[1]', clock='simple_tb.clk', end_cycle=6
        )
        unpacked_elements = [
            reader.load_waveform(
                f'simple_tb.unpacked_arr[{idx}]', clock='simple_tb.clk', end_cycle=6
            )
            for idx in range(3)
        ]

    assert packed.width == 33
    assert packed_0.width == packed_10.width == 3
    assert np.array_equal(
        packed.value,
        np.array([0, 0, 1227133513, 2454267026, 3681400539, 4908534052]),
    )
    assert np.array_equal(packed_0.value, np.array([0, 0, 1, 2, 3, 4]))
    assert np.array_equal(packed_10.value, packed_0.value)
    packed_reconstructed = sum(wave.value << (idx * 3) for idx, wave in enumerate(packed_elements))
    assert np.array_equal(packed.value, packed_reconstructed)

    assert unpacked.width == 33
    assert unpacked_0.width == unpacked_1.width == 11
    assert np.array_equal(
        unpacked.value,
        np.array([2290649088, 2290649088, 2152204289, 2156400642, 2160596995, 2164793348]),
    )
    assert np.array_equal(unpacked_0.value, np.array([0, 0, 1, 2, 3, 4]))
    assert np.array_equal(unpacked_1.value, np.array([273, 273, 257, 258, 259, 260]))
    unpacked_reconstructed = sum(
        wave.value << (idx * 11) for idx, wave in enumerate(unpacked_elements)
    )
    assert np.array_equal(unpacked.value, unpacked_reconstructed)


def test_fsdb_reader_struct_array_whole_and_fields(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        pkt_arr = reader.load_waveform('simple_tb.pkt_arr', clock='simple_tb.clk', end_cycle=6)
        pkt_arr_0 = reader.load_waveform('simple_tb.pkt_arr[0]', clock='simple_tb.clk', end_cycle=6)
        pkt_arr_0_valid = reader.load_waveform(
            'simple_tb.pkt_arr[0].valid', clock='simple_tb.clk', end_cycle=6
        )
        pkt_arr_0_data = reader.load_waveform(
            'simple_tb.pkt_arr[0].data', clock='simple_tb.clk', end_cycle=6
        )
        pkt_arr_1 = reader.load_waveform('simple_tb.pkt_arr[1]', clock='simple_tb.clk', end_cycle=6)

    assert pkt_arr.width == 8
    assert pkt_arr_0.width == 4
    assert np.array_equal(pkt_arr.value, np.array([240, 240, 105, 90, 195, 60]))
    assert np.array_equal(pkt_arr_0.value, np.array([0, 0, 9, 10, 3, 12]))
    assert np.array_equal(pkt_arr_0_valid.value, (pkt_arr_0.value >> 3) & 0x1)
    assert np.array_equal(pkt_arr_0_data.value, pkt_arr_0.value & 0x7)
    assert np.array_equal(pkt_arr.value, pkt_arr_0.value | (pkt_arr_1.value << 4))


def test_fsdb_reader_packed_struct_array_whole_and_fields(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        pkt_arr = reader.load_waveform(
            'simple_tb.pkt_packed_arr', clock='simple_tb.clk', end_cycle=6
        )
        pkt_arr_0 = reader.load_waveform(
            'simple_tb.pkt_packed_arr[0]', clock='simple_tb.clk', end_cycle=6
        )
        pkt_arr_1 = reader.load_waveform(
            'simple_tb.pkt_packed_arr[1]', clock='simple_tb.clk', end_cycle=6
        )
        pkt_arr_0_valid = reader.load_waveform(
            'simple_tb.pkt_packed_arr[0].valid', clock='simple_tb.clk', end_cycle=6
        )
        pkt_arr_0_data = reader.load_waveform(
            'simple_tb.pkt_packed_arr[0].data', clock='simple_tb.clk', end_cycle=6
        )

    assert pkt_arr.width == 8
    assert pkt_arr_0.width == pkt_arr_1.width == 4
    assert np.array_equal(pkt_arr.value, np.array([240, 240, 105, 90, 195, 60]))
    assert np.array_equal(pkt_arr_0.value, np.array([0, 0, 9, 10, 3, 12]))
    assert np.array_equal(pkt_arr_0_valid.value, (pkt_arr_0.value >> 3) & 0x1)
    assert np.array_equal(pkt_arr_0_data.value, pkt_arr_0.value & 0x7)
    assert np.array_equal(pkt_arr.value, pkt_arr_0.value | (pkt_arr_1.value << 4))


# ------------------------------------------------------------------
# eval integration tests (compare.fsdb)
# ------------------------------------------------------------------


def test_fsdb_eval_smoke(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        result = reader.eval('compare_tb.dut.unit_a.data[7:0] + 1', clock='compare_tb.clk')
        bit_slice = reader.eval('compare_tb.dut.unit_a.data[7:0][1:0]', clock='compare_tb.clk')

    assert isinstance(result, Waveform)
    assert result.width == 9  # addition increases width by 1
    assert isinstance(bit_slice, Waveform)
    assert bit_slice.width == 2


def test_fsdb_eval_no_match_raises(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        with pytest.raises(ValueError, match='matched no signals'):
            reader.eval('compare_tb.dut.unit_a.nonexistent', clock='compare_tb.clk')


def test_fsdb_eval_error_on_multi_match(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        with pytest.raises(ValueError, match="mode='single'"):
            reader.eval(
                'compare_tb.dut.unit_{a,b}.data[7:0]',
                clock='compare_tb.clk',
                mode='single',
            )


def test_fsdb_eval_zip_mode_brace_expansion(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        result = reader.eval(
            'compare_tb.dut.unit_{a,b}.data[7:0] + 1',
            clock='compare_tb.clk',
            mode='zip',
        )
    assert isinstance(result, dict)
    assert _capture_groups(result) == {('a',), ('b',)}
    assert all(isinstance(w, Waveform) for w in result.values())


def test_fsdb_eval_zip_mode_broadcast(compare_fsdb):
    # unit_a.data matches 1 signal (broadcast), unit_{a,b}.data matches 2
    with FsdbReader(str(compare_fsdb)) as reader:
        result = reader.eval(
            'compare_tb.dut.unit_{a,b}.data[7:0] + compare_tb.dut.unit_a.data[7:0]',
            clock='compare_tb.clk',
            mode='zip',
        )
    assert isinstance(result, dict)
    assert len(result) == 2


# ------------------------------------------------------------------
# begin_time / end_time / begin_cycle / end_cycle tests (compare.fsdb)
# ------------------------------------------------------------------


def test_fsdb_load_waveform_begin_end_time(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        full = reader.load_waveform('compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk')
        windowed = reader.load_waveform(
            'compare_tb.dut.unit_a.data[7:0]',
            clock='compare_tb.clk',
            begin_time=100,
            end_time=200,
        )

    # Windowed result should be a strict subset of the full waveform
    assert len(windowed.value) == 10
    assert windowed.time[0] == 100
    assert windowed.time[-1] == 190
    # Clock values are absolute: cycle 10 is at time 100 (period=10)
    assert windowed.clock[0] == 10
    assert windowed.clock[-1] == 19
    # Values should match the corresponding slice of the full waveform
    assert np.array_equal(windowed.value, full.value[10:20])


def test_fsdb_load_waveform_begin_end_cycle(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        full = reader.load_waveform('compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk')
        windowed = reader.load_waveform(
            'compare_tb.dut.unit_a.data[7:0]',
            clock='compare_tb.clk',
            begin_cycle=10,
            end_cycle=20,
        )

    assert len(windowed.value) == 10
    assert windowed.clock[0] == 10
    assert windowed.clock[-1] == 19
    assert np.array_equal(windowed.value, full.value[10:20])


def test_fsdb_reader_mutually_exclusive_errors(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform(
                'simple_tb.dut.data_o[3:0]', clock='simple_tb.clk', begin_time=0, begin_cycle=0
            )
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform(
                'simple_tb.dut.data_o[3:0]', clock='simple_tb.clk', end_time=10, end_cycle=1
            )


# ------------------------------------------------------------------
# cycle_slice helpers
# ------------------------------------------------------------------


def test_fsdb_cycle_slice(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        full = reader.load_waveform('compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk')

    sliced = full.cycle_slice(10, 20)
    assert len(sliced.value) == 10
    assert sliced.clock[0] == 10
    assert sliced.clock[-1] == 19
    assert np.array_equal(sliced.value, full.value[10:20])


def test_fsdb_cycle_slice_include_end(compare_fsdb):
    with FsdbReader(str(compare_fsdb)) as reader:
        full = reader.load_waveform('compare_tb.dut.unit_a.data[7:0]', clock='compare_tb.clk')

    sliced = full.cycle_slice(10, 20, include_end=True)
    assert len(sliced.value) == 11
    assert sliced.clock[-1] == 20
