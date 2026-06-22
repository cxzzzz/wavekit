from pathlib import Path

import numpy as np
import pytest

from wavekit import FsdbReader, Waveform, has_fsdb_support
from wavekit.signal import SignalCompositeType


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
            'pkt',
            'packed_arr',
            'unpacked_arr',
            'pkt_arr',
            'pkt_packed_arr',
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

        assert tb_signals['pkt'].width == 4
        assert tb_signals['pkt'].composite_type == SignalCompositeType.STRUCT
        pkt_members = {sig.name: sig for sig in tb_signals['pkt'].member_list or []}
        assert set(pkt_members) == {'valid', 'data'}
        assert pkt_members['valid'].width == 1
        assert pkt_members['data'].width == 3

        packed_members = {sig.name: sig for sig in tb_signals['packed_arr'].member_list or []}
        assert tb_signals['packed_arr'].width == 33
        assert tb_signals['packed_arr'].composite_type == SignalCompositeType.ARRAY
        assert len(packed_members) == 11
        assert packed_members['packed_arr[0]'].width == 3

        unpacked_members = {sig.name: sig for sig in tb_signals['unpacked_arr'].member_list or []}
        assert tb_signals['unpacked_arr'].width == 33
        assert tb_signals['unpacked_arr'].composite_type == SignalCompositeType.ARRAY
        assert set(unpacked_members) == {'unpacked_arr[0]', 'unpacked_arr[1]', 'unpacked_arr[2]'}
        assert unpacked_members['unpacked_arr[0]'].width == 11

        pkt_arr_members = {sig.name: sig for sig in tb_signals['pkt_arr'].member_list or []}
        assert tb_signals['pkt_arr'].width == 8
        assert tb_signals['pkt_arr'].composite_type == SignalCompositeType.ARRAY
        assert set(pkt_arr_members) == {'pkt_arr[0]', 'pkt_arr[1]'}
        assert pkt_arr_members['pkt_arr[0]'].composite_type == SignalCompositeType.STRUCT

        pkt_packed_arr_members = {
            sig.name: sig for sig in tb_signals['pkt_packed_arr'].member_list or []
        }
        assert tb_signals['pkt_packed_arr'].width == 8
        assert tb_signals['pkt_packed_arr'].composite_type == SignalCompositeType.ARRAY
        assert set(pkt_packed_arr_members) == {'pkt_packed_arr[0]', 'pkt_packed_arr[1]'}
        assert (
            pkt_packed_arr_members['pkt_packed_arr[0]'].composite_type == SignalCompositeType.STRUCT
        )


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
