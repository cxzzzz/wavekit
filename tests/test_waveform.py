import numpy as np
import pytest

from wavekit.waveform import Waveform


def build_waveform(values, width, signed=False):
    value = np.array(values)
    clock = np.arange(len(value))
    time = clock * 10
    return Waveform(value, clock, time, width=width, signed=signed)


def test_arithmetic_with_waveform():
    wave = build_waveform([1, 2, 3, 4], width=8)
    other = build_waveform([1, 2, 3, 4], width=8)

    assert np.all((wave + other).value == np.array([2, 4, 6, 8]))
    assert (wave + other).width == 9
    assert np.all((wave - other).value == np.array([0, 0, 0, 0]))
    assert (wave - other).width == 8
    assert np.all((wave * other).value == np.array([1, 4, 9, 16]))
    assert (wave * other).width == 16
    assert np.allclose((wave / other).value, np.array([1, 1, 1, 1]))
    assert (wave / other).width is None
    assert np.all((wave // other).value == np.array([1, 1, 1, 1]))
    assert (wave // other).width == 8


def test_arithmetic_with_scalar():
    wave = build_waveform([1, 2, 3, 4], width=8)

    assert np.all((wave + 1).value == np.array([2, 3, 4, 5]))
    assert (wave + 1).width == 9
    assert np.all((wave - 1).value == np.array([0, 1, 2, 3]))
    assert (wave - 1).width == 8
    assert np.all((wave * 3).value == np.array([3, 6, 9, 12]))
    assert (wave * 3).width == 10
    assert np.allclose((wave / 5).value, np.array([0.2, 0.4, 0.6, 0.8]))
    assert (wave / 5).width is None
    assert np.all((wave // 5).value == np.array([0, 0, 0, 0]))
    assert (wave // 5).width == 8

    assert np.all((1 + wave).value == np.array([2, 3, 4, 5]))
    assert (1 + wave).width == 9
    assert np.all((10 - wave).value == np.array([9, 8, 7, 6]))
    assert (10 - wave).width == 8
    assert np.all((3 * wave).value == np.array([3, 6, 9, 12]))
    assert (3 * wave).width == 10
    assert np.allclose((20 / wave).value, np.array([20, 10, 6.6666667, 5]))
    assert (20 / wave).width is None
    assert np.all((5 // wave).value == np.array([5, 2, 1, 1]))
    assert (5 // wave).width == int.bit_length(5)


def test_arithmetic_width_limit():
    wave = build_waveform([1, 2, 3], width=64)
    other = build_waveform([1, 2, 3], width=64)

    assert (wave + other).width == 64
    assert (wave - other).width == 64
    assert (wave * other).width == 64


def test_arithmetic_width_error():
    wave = build_waveform([1, 2, 3], width=65)
    other = build_waveform([1, 2, 3], width=65)
    with pytest.raises(ValueError):
        _ = wave + other


def test_signedness_mismatch_error():
    signed_wave = build_waveform([1, 2, 3], width=8, signed=True)
    unsigned_wave = build_waveform([1, 2, 3], width=8, signed=False)
    with pytest.raises(ValueError):
        _ = signed_wave + unsigned_wave


def test_logical_ops_and_widths():
    value = np.array([1, 2, 3, 4], dtype=np.uint64)
    other_value = np.array([4, 3, 2, 1], dtype=np.uint64)
    wave = build_waveform(value, width=12)
    other = build_waveform(other_value, width=12)

    assert np.all((wave & other).value == (value & other_value))
    assert (wave & other).width == 12
    assert np.all((wave | other).value == (value | other_value))
    assert (wave | other).width == 12
    assert np.all((wave ^ other).value == (value ^ other_value))
    assert (wave ^ other).width == 12
    assert np.all((~wave).value == ((~value) & ((1 << 12) - 1)))
    assert (~wave).width == 12
    assert np.all((wave == other).value == (value == other_value))
    assert (wave == other).width == 1

    assert np.all((wave & 7).value == (value & 7))
    assert (wave & 7).width == 12
    assert np.all((wave | 7).value == (value | 7))
    assert (wave | 7).width == 12
    assert np.all((wave ^ 7).value == (value ^ 7))
    assert (wave ^ 7).width == 12
    assert np.all((wave == 7).value == (value == 7))
    assert (wave == 7).width == 1


def test_logical_width_errors():
    wave = build_waveform([1, 2, 3], width=4)
    other = build_waveform([1, 2, 3], width=5)
    with pytest.raises(ValueError):
        _ = wave & other
    with pytest.raises(ValueError):
        _ = wave & 16


def test_shift_ops():
    wave = build_waveform([1, 2, 3], width=8)
    assert np.all((wave << 2).value == np.array([4, 8, 12]))
    assert (wave << 2).width == 10
    assert np.all((wave >> 2).value == np.array([0, 0, 0]))
    assert (wave >> 2).width == 6


def test_mod_pow_and_ne():
    wave = build_waveform([2, 3, 4], width=8)
    other = build_waveform([1, 2, 3], width=8)

    assert np.all((wave % other).value == np.array([0, 1, 1]))
    assert (wave % other).width == 8
    assert np.all((5 % wave).value == np.array([1, 2, 1]))
    assert (5 % wave).width == int.bit_length(5)

    assert np.all((wave**other).value == np.array([2, 9, 64]))
    assert (wave**other).width == 64
    assert np.all((wave != other).value == np.array([1, 1, 1]))
    assert (wave != other).width == 1


def test_bitsel_and_slice_errors():
    wave = build_waveform([0b1011, 0b0101], width=4)
    assert np.all(wave[1].value == np.array([1, 0]))
    assert np.all(wave[3:2].value == np.array([2, 1]))

    with pytest.raises(Exception):
        _ = wave[3:0:2]
    with pytest.raises(Exception):
        _ = wave[1:3]


def test_sampling_and_filter():
    wave = build_waveform([1, 2, 3, 4, 5, 6], width=8)
    sampled = wave.downsample(2)
    assert np.allclose(sampled.value, np.array([1.5, 3.5, 5.5]))
    assert sampled.width is None

    masked = wave.mask(wave.value > 4)
    assert np.all(masked.value == np.array([5, 6]))
    compressed = wave.compress(lambda v: v > 4)
    assert np.all(compressed.value == np.array([5, 6]))


def test_rise_fall_and_compress():
    wave = build_waveform([0, 1, 0, 1, 1, 0], width=1)
    assert np.all(wave.rising_edge().value == np.array([0, 1, 0, 1, 0, 0]))
    assert np.all(wave.falling_edge().value == np.array([0, 0, 1, 0, 0, 1]))

    repeated = build_waveform([0, 0, 1, 1, 0, 0], width=1)
    assert np.all(repeated.unique_consecutive().value == np.array([0, 1, 0, 0]))
    empty = build_waveform([], width=1)
    assert np.all(empty.unique_consecutive().value == np.array([]))


def test_count_one_and_bits():
    value = np.array([1, 2, 3, 4, 2**32 - 1], dtype=np.uint64)
    wave = build_waveform(value, width=64)
    assert np.all(wave.bit_count().value == np.array([1, 1, 2, 1, 32]))
    assert np.all(wave[1].value == np.array([0, 1, 1, 0, 1]))
    assert np.all(wave[2:1].value == np.array([0, 1, 1, 2, 3]))

    wide_values = np.array([0, (1 << 65) + 3], dtype=np.object_)
    wide_wave = build_waveform(wide_values, width=128)
    assert np.all(wide_wave.bit_count().value == np.array([0, 3]))
    assert np.all(wide_wave[68:66].value == np.array([0, 0]))


def test_bit_count_over_64_width():
    wide_values = np.array([1 << 70, (1 << 80) + 3], dtype=np.object_)
    wide_wave = build_waveform(wide_values, width=96)
    assert np.all(wide_wave.bit_count().value == np.array([1, 3]))


def test_split_concat_and_merge():
    value = np.array([1, 2, 3], dtype=np.uint64)
    widths = [4, 6, 5]
    waves = [build_waveform(value + offset, width=w) for offset, w in zip([0, 5, 9], widths)]
    concat = Waveform.concatenate(waves)
    split = concat.split_bits(widths, padding=False)
    for original, extracted in zip(waves, split):
        assert np.all(original.value == extracted.value)

    object_value = np.array([1, 2, 3], dtype=np.object_)
    object_waves = [
        build_waveform(object_value + offset, width=w) for offset, w in zip([0, 3, 7], widths)
    ]
    object_concat = Waveform.concatenate(object_waves)
    object_split = object_concat.split_bits(widths, padding=False)
    for original, extracted in zip(object_waves, object_split):
        assert np.all(original.value == extracted.value)

    merged = Waveform.merge(
        [waves[0], waves[1]],
        lambda values: values[0] + values[1],
        width=16,
        signed=False,
    )
    assert np.all(merged.value == waves[0].value + waves[1].value)
    assert merged.width == 16


def test_split_bits_validation_error():
    wave = build_waveform([1, 2, 3], width=10)
    with pytest.raises(Exception):
        _ = wave.split_bits(3, padding=False)
    with pytest.raises(Exception):
        _ = wave.split_bits([3, 4], padding=False)


def test_signed_unsigned_conversion():
    wave = build_waveform([0, 7, 8, 15], width=4, signed=False)
    signed_wave = wave.as_signed()
    assert np.all(signed_wave.value == np.array([0, 7, -8, -1]))
    unsigned_wave = signed_wave.as_unsigned()
    assert np.all(unsigned_wave.value == np.array([0, 7, 8, 15]))


def test_waveform_metadata_and_copy():
    wave = build_waveform([1, 2, 3], width=8)
    assert str(wave) == "Waveform(signal='', width=8, signed=False)"
    assert wave.set_signal('tb.u0.sig') is wave
    assert wave.signal == 'tb.u0.sig'

    record = wave.data
    assert record.dtype.names == ('time', 'clock', 'value')
    assert np.all(record['time'] == wave.time)
    assert np.all(record['clock'] == wave.clock)
    assert np.all(record['value'] == wave.value)

    copied = wave.copy()
    copied.value[0] = 99
    assert wave.value[0] == 1


def test_map_take_and_idempotent_conversion():
    wave = build_waveform([1, 2, 3, 4], width=8)
    mapped = wave.map(lambda x: x + 1, width=9, signed=True)
    assert np.all(mapped.value == np.array([2, 3, 4, 5]))
    assert mapped.width == 9
    assert mapped.signed is True
    assert np.all(mapped.clock == wave.clock)
    assert np.all(mapped.time == wave.time)

    taken = wave.take([0, 2])
    assert np.all(taken.value == np.array([1, 3]))
    assert np.all(taken.clock == np.array([0, 2]))

    with pytest.raises(TypeError):
        wave.take(np.array([True, False, True, False]))

    unsigned_wave = build_waveform([1, 2, 3], width=8, signed=False)
    assert np.all(unsigned_wave.as_unsigned().value == unsigned_wave.value)
    signed_wave = build_waveform([1, 2, 3], width=8, signed=True)
    assert np.all(signed_wave.as_signed().value == signed_wave.value)
