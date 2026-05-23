from pathlib import Path

import numpy as np
import pytest

from wavekit import FstReader, Waveform


@pytest.fixture()
def fst_path():
    path = Path(__file__).resolve().parent / 'testdata' / 'counter.fst'
    if not path.exists():
        pytest.skip('counter.fst fixture is unavailable')
    return path


def test_fst_reader_exported():
    assert FstReader.__name__ == 'FstReader'


def test_fst_reader_top_scope_list_normalizes_names_and_preserves_aliases(fst_path):
    with FstReader(str(fst_path)) as reader:
        top = reader.top_scope_list()
        tb = top[0]
        dut = tb.child_scope_list[0]

        assert tb.name == 'tb'
        assert dut.name == 'dut'
        assert {sig.name for sig in tb.signal_list} >= {'clk', 'reset', 'overflow'}
        assert {sig.name for sig in dut.signal_list} >= {
            'clk',
            'reset',
            'overflow',
            'counter[3:0]',
        }

        tb_clk = next(sig for sig in tb.signal_list if sig.name == 'clk')
        dut_clk = next(sig for sig in dut.signal_list if sig.name == 'clk')
        assert tb_clk.handle == dut_clk.handle


def test_fst_reader_module_name_matching_is_unsupported(fst_path):
    with FstReader(str(fst_path)) as reader:
        with pytest.raises(NotImplementedError):
            reader.get_matched_signals('tb.$dut.counter[3:0]')


def test_fst_reader_load_waveform_without_range(fst_path):
    with FstReader(str(fst_path)) as reader:
        counter = reader.load_waveform('tb.dut.counter', clock='tb.clk', sample_on_posedge=True)

    assert counter.name == 'tb.dut.counter[3:0]'
    assert counter.width == 4
    assert counter.signed is False
    assert np.array_equal(counter.time[:5], np.array([10, 30, 50, 70, 90], dtype=np.uint64))
    assert np.array_equal(counter.clock[:5], np.arange(5, dtype=np.uint64))
    assert np.array_equal(counter.value[:5], np.array([0, 0, 0, 0, 0], dtype=np.uint64))
    assert counter.value[5] == 1


def test_fst_reader_begin_end_time_and_cycle_match(fst_path):
    with FstReader(str(fst_path)) as reader:
        by_time = reader.load_waveform(
            'tb.dut.counter[3:0]',
            clock='tb.clk',
            sample_on_posedge=True,
            begin_time=130,
            end_time=230,
        )
        by_cycle = reader.load_waveform(
            'tb.dut.counter[3:0]',
            clock='tb.clk',
            sample_on_posedge=True,
            begin_cycle=6,
            end_cycle=11,
        )

    assert np.array_equal(by_time.value, by_cycle.value)
    assert np.array_equal(by_time.time, by_cycle.time)
    assert np.array_equal(by_time.clock, by_cycle.clock)
    assert by_time.time[0] == 130
    assert by_time.time[-1] == 210
    assert np.array_equal(by_cycle.clock, np.arange(6, 11, dtype=np.uint64))


def test_fst_reader_subrange_load(fst_path):
    with FstReader(str(fst_path)) as reader:
        low_bits = reader.load_waveform('tb.dut.counter[1:0]', clock='tb.clk')
        matched_low_bits = reader.load_matched_waveforms('tb.dut.counter[1:0]', 'tb.clk')[()]

    assert low_bits.width == 2
    assert np.all(low_bits.value < 4)
    assert matched_low_bits.width == 2
    assert np.array_equal(matched_low_bits.value, low_bits.value)


def test_fst_reader_load_matched_waveforms(fst_path):
    with FstReader(str(fst_path)) as reader:
        waves = reader.load_matched_waveforms('tb.dut.{counter[3:0],overflow}', 'tb.clk')

    assert set(waves.keys()) == {('counter[3:0]',), ('overflow',)}
    assert waves[('counter[3:0]',)].width == 4
    assert waves[('overflow',)].width == 1


def test_fst_reader_load_matched_waveforms_regex(fst_path):
    with FstReader(str(fst_path)) as reader:
        waves = reader.load_matched_waveforms(r'tb.dut.@(counter\[3:0\]|overflow)', 'tb.clk')

    assert {key[0][0] for key in waves} == {'counter[3:0]', 'overflow'}
    assert {wave.width for wave in waves.values()} == {1, 4}


def test_fst_reader_load_matched_waveforms_uses_signal_range(fst_path):
    with FstReader(str(fst_path)) as reader:
        waves = reader.load_matched_waveforms('tb.dut.counter', 'tb.clk')

    assert list(waves.keys()) == [()]
    wave = waves[()]
    assert wave.name == 'tb.dut.counter[3:0]'
    assert wave.width == 4


def test_fst_reader_eval_smoke(fst_path):
    with FstReader(str(fst_path)) as reader:
        result = reader.eval('tb.dut.counter[3:0] + 1', clock='tb.clk')
        bit_slice = reader.eval('tb.dut.counter[3:0][1:0]', clock='tb.clk')

    assert isinstance(result, Waveform)
    assert result.width == 5
    assert isinstance(bit_slice, Waveform)
    assert bit_slice.width == 2


def test_fst_reader_load_waveform_no_match_raises(fst_path):
    with FstReader(str(fst_path)) as reader:
        with pytest.raises(ValueError, match="signal 'tb.dut.nope' not found"):
            reader.load_waveform('tb.dut.nope', clock='tb.clk')


def test_fst_reader_mutually_exclusive_errors(fst_path):
    with FstReader(str(fst_path)) as reader:
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform('tb.dut.counter[3:0]', clock='tb.clk', begin_time=0, begin_cycle=0)
        with pytest.raises(ValueError, match='mutually exclusive'):
            reader.load_waveform('tb.dut.counter[3:0]', clock='tb.clk', end_time=10, end_cycle=1)
