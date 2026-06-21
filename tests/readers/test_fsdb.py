from pathlib import Path

import numpy as np
import pytest

from wavekit import FsdbReader, Waveform, has_fsdb_support


@pytest.fixture(scope='module')
def fsdb_path():
    path = Path(__file__).resolve().parent / 'fixtures' / 'fsdb' / 'simple.fsdb'
    if not path.exists():
        pytest.skip(
            'simple.fsdb fixture is unavailable; run ' 'tests/readers/fixtures/fsdb/build_fsdb.sh'
        )
    return path


@pytest.fixture(scope='module')
def fsdb_runtime(fsdb_path):
    if not has_fsdb_support():
        pytest.skip('FSDB tests require the Verdi NPI runtime')
    return fsdb_path


def test_fsdb_reader_exported(fsdb_runtime):
    assert FsdbReader.__name__ == 'FsdbReader'


def test_fsdb_reader_top_scope_list(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        top = reader.top_scope_list()
        tb = top[0]
        dut = next(scope for scope in tb.child_scope_list if scope.name == 'dut')

        assert tb.name == 'simple_tb'
        assert dut.name == 'dut'
        tb_signals = {sig.name: sig for sig in tb.signal_list}
        dut_signals = {sig.name: sig for sig in dut.signal_list}
        assert set(tb_signals) >= {
            'clk',
            'rst_n',
            'valid',
            'data_i',
            'bus',
            'data_0',
            'data_1',
        }
        assert set(dut_signals) >= {
            'clk',
            'rst_n',
            'valid',
            'data_i',
            'data_o',
            'overflow',
        }
        for signals, names in (
            (tb_signals, ['data_i', 'bus', 'data_0', 'data_1']),
            (dut_signals, ['data_i', 'data_o']),
        ):
            for name in names:
                assert signals[name].width == 4


def test_fsdb_reader_load_waveform_without_range(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        data = reader.load_waveform('simple_tb.dut.data_o', clock='simple_tb.clk')

    assert data.name == 'simple_tb.dut.data_o'
    assert data.width == 4
    assert data.signed is False
    assert len(data.value) > 0
    assert np.array_equal(data.clock[:5], np.arange(5, dtype=np.uint64))


def test_fsdb_reader_begin_end_time_and_cycle_match(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        full = reader.load_waveform('simple_tb.dut.data_o[3:0]', clock='simple_tb.clk')
        by_time = reader.load_waveform(
            'simple_tb.dut.data_o[3:0]',
            clock='simple_tb.clk',
            begin_time=int(full.time[2]),
            end_time=int(full.time[5]),
        )
        by_cycle = reader.load_waveform(
            'simple_tb.dut.data_o[3:0]',
            clock='simple_tb.clk',
            begin_cycle=2,
            end_cycle=5,
        )

    assert np.array_equal(by_time.value, by_cycle.value)
    assert np.array_equal(by_time.time, by_cycle.time)
    assert np.array_equal(by_time.clock, by_cycle.clock)
    assert np.array_equal(by_cycle.clock, np.arange(2, 5, dtype=np.uint64))


def test_fsdb_reader_subrange_load(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        full = reader.load_waveform('simple_tb.dut.data_o[3:0]', clock='simple_tb.clk')
        low_bits = reader.load_waveform('simple_tb.dut.data_o[1:0]', clock='simple_tb.clk')
        high_bits = reader.load_waveform('simple_tb.dut.data_o[3:2]', clock='simple_tb.clk')

    assert low_bits.width == 2
    assert high_bits.width == 2
    assert np.array_equal(low_bits.value, full.value & 0x3)
    assert np.array_equal(high_bits.value, (full.value >> 2) & 0x3)


def test_fsdb_reader_load_unknown_mask_include_flags(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        both = reader.load_unknown_mask('simple_tb.bus[3:0]', clock='simple_tb.clk', end_cycle=6)
        x_only = reader.load_unknown_mask(
            'simple_tb.bus[3:0]', clock='simple_tb.clk', include_z=False, end_cycle=6
        )
        z_only = reader.load_unknown_mask(
            'simple_tb.bus[3:0]', clock='simple_tb.clk', include_x=False, end_cycle=6
        )
        values = reader.load_waveform('simple_tb.bus[3:0]', clock='simple_tb.clk', end_cycle=6)

    assert both.name == 'unknown_mask(simple_tb.bus[3:0])'
    assert both.width == 4
    assert both.signed is False
    assert np.array_equal(
        both.value,
        np.array([0, 0, 0b1111, 0b1111, 0b0010, 0b0101], dtype=np.uint64),
    )
    assert np.array_equal(
        x_only.value,
        np.array([0, 0, 0b1111, 0, 0b0010, 0b0001], dtype=np.uint64),
    )
    assert np.array_equal(
        z_only.value,
        np.array([0, 0, 0, 0b1111, 0, 0b0100], dtype=np.uint64),
    )
    assert np.array_equal(both.clock, values.clock)
    assert np.array_equal(both.time, values.time)


def test_fsdb_reader_load_unknown_mask_range_and_matched(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        full = reader.load_unknown_mask('simple_tb.bus[3:0]', clock='simple_tb.clk', end_cycle=6)
        low = reader.load_unknown_mask('simple_tb.bus[1:0]', clock='simple_tb.clk', end_cycle=6)
        masks = reader.load_matched_unknown_masks(
            'simple_tb.data_{0,1}[3:0]', 'simple_tb.clk', end_cycle=6
        )
        values = reader.load_matched_waveforms(
            'simple_tb.data_{0,1}[3:0]', 'simple_tb.clk', end_cycle=6
        )

    assert low.width == 2
    assert np.array_equal(low.value, full.value & 0x3)
    assert set(masks) == set(values) == {('0',), ('1',)}
    assert masks[('0',)].name == 'unknown_mask(simple_tb.data_0[3:0])'
    assert masks[('1',)].name == 'unknown_mask(simple_tb.data_1[3:0])'


def test_fsdb_reader_unknown_mask_both_false_is_all_zero(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        mask = reader.load_unknown_mask(
            'simple_tb.bus[3:0]',
            clock='simple_tb.clk',
            include_x=False,
            include_z=False,
            end_cycle=6,
        )

    assert np.array_equal(mask.value, np.zeros(6, dtype=np.uint64))


def test_fsdb_reader_load_matched_waveforms(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        waves = reader.load_matched_waveforms(
            'simple_tb.dut.{data_o[3:0],overflow}', 'simple_tb.clk'
        )

    assert set(waves) == {('data_o[3:0]',), ('overflow',)}
    assert waves[('data_o[3:0]',)].width == 4
    assert waves[('overflow',)].width == 1


def test_fsdb_reader_eval_smoke(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        result = reader.eval('simple_tb.dut.data_o[3:0] + 1', clock='simple_tb.clk')
        bit_slice = reader.eval('simple_tb.dut.data_o[3:0][1:0]', clock='simple_tb.clk')

    assert isinstance(result, Waveform)
    assert result.width == 5
    assert isinstance(bit_slice, Waveform)
    assert bit_slice.width == 2


def test_fsdb_reader_load_waveform_no_match_raises(fsdb_runtime):
    with FsdbReader(str(fsdb_runtime)) as reader:
        with pytest.raises(ValueError, match="signal 'simple_tb.nope' not found"):
            reader.load_waveform('simple_tb.nope', clock='simple_tb.clk')


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
