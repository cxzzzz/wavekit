import numpy as np
import pytest

from wavekit import Signal, Waveform


def build_waveform(values, width, signed=False):
    value = np.array(values)
    clock = np.arange(len(value))
    time = clock * 10
    return Waveform(value, clock, time, signal=Signal('', '', width, None, signed))


# ==========================================
# Metadata and Basics
# ==========================================


def test_metadata_and_copy():
    wave = build_waveform([1, 2, 3], width=8, signed=False)
    assert str(wave) == "Waveform(Signal(name='', full_name='', width=8, signed=False, range=None))"
    wave.name = 'tb.u0.sig'
    assert wave.name == 'tb.u0.sig'

    record = wave.data
    assert len(record) == 3
    assert record.dtype.names == ('time', 'clock', 'value')
    assert np.all(record['time'] == wave.time)
    assert np.all(record['clock'] == wave.clock)
    assert np.all(record['value'] == wave.value)

    copied = wave.copy()
    copied.value[0] = 99
    assert wave.value[0] == 1


# ==========================================
# Arithmetic Operations
# ==========================================


def test_arithmetic_waveform():
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


def test_arithmetic_scalar():
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


def test_arithmetic_width():
    wave = build_waveform([1, 2, 3], width=64)
    other = build_waveform([1, 2, 3], width=64)

    assert (wave + other).width == 64
    assert (wave - other).width == 64
    assert (wave * other).width == 64

    # Error cases
    wave_large = build_waveform([1, 2, 3], width=65)
    other_large = build_waveform([1, 2, 3], width=65)
    with pytest.raises(ValueError):
        _ = wave_large + other_large


def test_arithmetic_signedness_error():
    signed_wave = build_waveform([1, 2, 3], width=8, signed=True)
    unsigned_wave = build_waveform([1, 2, 3], width=8, signed=False)
    with pytest.raises(ValueError):
        _ = signed_wave + unsigned_wave


def test_mod_pow_ne():
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


# ==========================================
# Logical Operations
# ==========================================


def test_logical_ops():
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


def test_shift_ops():
    wave = build_waveform([1, 2, 3], width=8)
    assert np.all((wave << 2).value == np.array([4, 8, 12]))
    assert (wave << 2).width == 10
    assert np.all((wave >> 2).value == np.array([0, 0, 0]))
    assert (wave >> 2).width == 6


def test_logical_width_errors():
    wave = build_waveform([1, 2, 3], width=4)
    other = build_waveform([1, 2, 3], width=5)
    with pytest.raises(ValueError):
        _ = wave & other
    with pytest.raises(ValueError):
        _ = wave & 16


# ==========================================
# Slicing and Windowing
# ==========================================


def test_getitem_bitsel():
    wave = build_waveform([0b1011, 0b0101], width=4)
    assert np.all(wave[1].value == np.array([1, 0]))
    assert np.all(wave[3:2].value == np.array([2, 1]))


def test_getitem_errors():
    wave = build_waveform([0b1011, 0b0101], width=4)
    with pytest.raises(Exception):
        _ = wave[3:0:2]
    with pytest.raises(Exception):
        _ = wave[1:3]


def test_time_slice():
    wave = build_waveform([1, 2, 3, 4], width=8)
    # time = clock * 10 -> [0, 10, 20, 30]

    sliced = wave.time_slice(begin_time=10, end_time=20)
    assert np.all(sliced.time == np.array([10]))
    assert np.all(sliced.value == np.array([2]))

    sliced_inclusive = wave.time_slice(begin_time=10, end_time=20, include_end=True)
    assert np.all(sliced_inclusive.time == np.array([10, 20]))
    assert np.all(sliced_inclusive.value == np.array([2, 3]))


def test_slice_by_index():
    wave = build_waveform([1, 2, 3, 4], width=8)

    index_slice = wave.slice(1, 2)
    assert np.all(index_slice.time == np.array([10]))
    assert np.all(index_slice.value == np.array([2]))

    index_slice_inclusive = wave.slice(1, 2, include_end=True)
    assert np.all(index_slice_inclusive.time == np.array([10, 20]))
    assert np.all(index_slice_inclusive.value == np.array([2, 3]))


def test_take_by_indices():
    wave = build_waveform([1, 2, 3, 4], width=8)
    taken = wave.take([0, 2])
    assert np.all(taken.value == np.array([1, 3]))
    assert np.all(taken.clock == np.array([0, 2]))

    with pytest.raises(TypeError):
        wave.take(np.array([True, False, True, False]))


# ==========================================
# Sampling and Filtering
# ==========================================


def test_downsample():
    wave = build_waveform([1, 2, 3, 4, 5, 6], width=8)
    sampled = wave.downsample(2)
    assert np.allclose(sampled.value, np.array([1.5, 3.5, 5.5]))
    assert sampled.width is None


def test_mask():
    wave = build_waveform([1, 2, 3, 4, 5, 6], width=8)
    masked = wave.mask(wave.value > 4)
    assert np.all(masked.value == np.array([5, 6]))


def test_mask_with_waveform():
    wave = build_waveform([1, 2, 3, 4], width=8)
    # Create a boolean mask waveform
    mask_vals = np.array([False, True, False, True], dtype=np.bool_)
    mask_wave = Waveform(
        value=mask_vals,
        clock=wave.clock,
        time=wave.time,
        signal=Signal('', '', 1, None, False),
    )
    masked = wave.mask(mask_wave)
    assert np.all(masked.value == np.array([2, 4]))


def test_filter():
    wave = build_waveform([1, 2, 3, 4, 5, 6], width=8)
    filtered = wave.filter(lambda v: v > 4)
    assert np.all(filtered.value == np.array([5, 6]))


def test_vectorized_filter():
    wave = build_waveform([1, 2, 3, 4, 5, 6], width=8)
    # vectorized_filter expects a function that takes an array and returns a boolean array
    filtered = wave.vectorized_filter(lambda x: x > 3)
    assert np.all(filtered.value == np.array([4, 5, 6]))


def test_compress():
    # compress keeps last occurrence (preserves end time)
    wave = build_waveform([1, 1, 2, 2, 3], width=8)
    compressed = wave.compress()
    assert np.all(compressed.value == np.array([1, 2, 3]))
    # compress keeps last timestamp of each group
    assert np.all(compressed.time == np.array([10, 30, 40]))


def test_unique_consecutive():
    repeated = build_waveform([0, 0, 1, 1, 0, 0], width=1)
    # unique_consecutive keeps first occurrence (matches numpy behavior)
    assert np.all(repeated.unique_consecutive().value == np.array([0, 1, 0]))
    # unique_consecutive keeps first timestamp of each group
    assert np.all(repeated.unique_consecutive().time == np.array([0, 20, 40]))

    empty = build_waveform([], width=1)
    assert np.all(empty.unique_consecutive().value == np.array([]))


# ==========================================
# Mapping and Transformation
# ==========================================


def test_map():
    wave = build_waveform([1, 2, 3, 4], width=8)
    mapped = wave.map(lambda x: x + 1, width=9, signed=True)
    assert np.all(mapped.value == np.array([2, 3, 4, 5]))
    assert mapped.width == 9
    assert mapped.signed is True
    assert np.all(mapped.clock == wave.clock)
    assert np.all(mapped.time == wave.time)


def test_vectorized_map():
    wave = build_waveform([1, 2, 3, 4], width=8)
    squared = wave.vectorized_map(np.square, width=8, signed=False)
    assert np.all(squared.value == np.array([1, 4, 9, 16]))


# ==========================================
# Edge Detection
# ==========================================


def test_edges():
    wave = build_waveform([0, 1, 0, 1, 1, 0], width=1)
    assert np.all(wave.rising_edge().value == np.array([0, 1, 0, 1, 0, 0]))
    assert np.all(wave.falling_edge().value == np.array([0, 0, 1, 0, 0, 1]))


# ==========================================
# Bit Operations
# ==========================================


def test_bit_count():
    value = np.array([1, 2, 3, 4, 2**32 - 1], dtype=np.uint64)
    wave = build_waveform(value, width=64)
    assert np.all(wave.bit_count().value == np.array([1, 1, 2, 1, 32]))
    assert np.all(wave[1].value == np.array([0, 1, 1, 0, 1]))
    assert np.all(wave[2:1].value == np.array([0, 1, 1, 2, 3]))


def test_bit_count_wide():
    wide_values = np.array([0, (1 << 65) + 3], dtype=np.object_)
    wide_wave = build_waveform(wide_values, width=128)
    assert np.all(wide_wave.bit_count().value == np.array([0, 3]))
    assert np.all(wide_wave[68:66].value == np.array([0, 0]))

    wide_values_2 = np.array([1 << 70, (1 << 80) + 3], dtype=np.object_)
    wide_wave_2 = build_waveform(wide_values_2, width=96)
    assert np.all(wide_wave_2.bit_count().value == np.array([1, 3]))


def test_split_concat_merge():
    # Split and Concat
    value = np.array([1, 2, 3], dtype=np.uint64)
    widths = [4, 6, 5]
    waves = [build_waveform(value + offset, width=w) for offset, w in zip([0, 5, 9], widths)]
    concat = Waveform.concatenate(waves)
    split = concat.split_bits(widths, padding=False)
    for original, extracted in zip(waves, split):
        assert np.all(original.value == extracted.value)

    # Wide Split and Concat
    object_value = np.array([1, 2, 3], dtype=np.object_)
    object_waves = [
        build_waveform(object_value + offset, width=w) for offset, w in zip([0, 3, 7], widths)
    ]
    object_concat = Waveform.concatenate(object_waves)
    object_split = object_concat.split_bits(widths, padding=False)
    for original, extracted in zip(object_waves, object_split):
        assert np.all(original.value == extracted.value)

    # Merge
    merged = Waveform.merge(
        [waves[0], waves[1]],
        lambda values: values[0] + values[1],
        width=16,
        signed=False,
    )
    assert np.all(merged.value == waves[0].value + waves[1].value)
    assert merged.width == 16


def test_split_bits_errors():
    wave = build_waveform([1, 2, 3], width=10)
    with pytest.raises(Exception):
        _ = wave.split_bits(3, padding=False)
    with pytest.raises(Exception):
        _ = wave.split_bits([3, 4], padding=False)


# ==========================================
# Type Conversion
# ==========================================


def test_signed_conversion():
    wave = build_waveform([0, 7, 8, 15], width=4, signed=False)
    signed_wave = wave.as_signed()
    assert np.all(signed_wave.value == np.array([0, 7, -8, -1]))
    unsigned_wave = signed_wave.as_unsigned()
    assert np.all(unsigned_wave.value == np.array([0, 7, 8, 15]))

    # Idempotent checks
    unsigned_wave_2 = build_waveform([1, 2, 3], width=8, signed=False)
    assert np.all(unsigned_wave_2.as_unsigned().value == unsigned_wave_2.value)
    signed_wave_2 = build_waveform([1, 2, 3], width=8, signed=True)
    assert np.all(signed_wave_2.as_signed().value == signed_wave_2.value)


def test_signal_synchronization():
    wave = build_waveform([1, 2, 3], width=8, signed=False)
    wave.name = 'test_sig'

    # Check initial state
    assert wave.width == 8
    assert wave.signal.width == 8
    assert wave.name == 'test_sig'
    assert wave.signal.name == 'test_sig'

    # Modify via Waveform property
    wave.width = 16
    assert wave.width == 16
    assert wave.signal.width == 16

    wave.name = 'new_name'
    assert wave.name == 'new_name'
    assert wave.signal.name == 'new_name'

    # Modify via Signal object
    wave.signal.width = 32
    assert wave.width == 32
    assert wave.signal.width == 32

    wave.signal.name = 'final_name'
    assert wave.name == 'final_name'
    assert wave.signal.name == 'final_name'

    # Check signed synchronization
    wave.signed = True
    assert wave.signed is True
    assert wave.signal.signed is True

    wave.signal.signed = False
    assert wave.signed is False
    assert wave.signal.signed is False


# ==========================================
# Relative Time Access
# ==========================================


def test_relative_forward():
    """Test relative() with positive offset (looking into future)."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)

    # Default pad='repeat' - same length
    result = wave.relative(1)
    assert np.all(result.value == np.array([2, 3, 4, 5, 5]))
    assert len(result.value) == 5

    # Forward by 2
    result = wave.relative(2)
    assert np.all(result.value == np.array([3, 4, 5, 5, 5]))


def test_relative_backward():
    """Test relative() with negative offset (looking into past)."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)

    # Default pad='repeat' - same length
    result = wave.relative(-1)
    assert np.all(result.value == np.array([1, 1, 2, 3, 4]))
    assert len(result.value) == 5

    # Backward by 2
    result = wave.relative(-2)
    assert np.all(result.value == np.array([1, 1, 1, 2, 3]))


def test_relative_zero_offset():
    """Test relative(0) returns a copy."""
    wave = build_waveform([1, 2, 3], width=8)
    result = wave.relative(0)
    assert np.all(result.value == wave.value)
    assert result is not wave


def test_relative_pad_repeat():
    """Test relative() with pad='repeat'."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)

    # Forward - result[i] = original[i+1], pad end with last value
    result = wave.relative(1, pad='repeat')
    assert np.all(result.value == np.array([2, 3, 4, 5, 5]))
    assert len(result.value) == len(wave.value)

    # Forward by 2
    result = wave.relative(2, pad='repeat')
    assert np.all(result.value == np.array([3, 4, 5, 5, 5]))

    # Backward - result[i] = original[i-1], pad start with first value
    result = wave.relative(-1, pad='repeat')
    assert np.all(result.value == np.array([1, 1, 2, 3, 4]))

    # Backward by 2 - first 2 positions padded
    result = wave.relative(-2, pad='repeat')
    assert np.all(result.value == np.array([1, 1, 1, 2, 3]))


def test_relative_pad_value():
    """Test relative() with pad='value'."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)

    # Forward - pad end with 0
    result = wave.relative(1, pad='value', pad_value=0)
    assert np.all(result.value == np.array([2, 3, 4, 5, 0]))

    # Backward - pad start with 99
    result = wave.relative(-1, pad='value', pad_value=99)
    assert np.all(result.value == np.array([99, 1, 2, 3, 4]))


def test_relative_pad_value_string():
    """Test relative() with string pad_value for string-typed signals."""
    value = np.array(['a', 'b', 'c', 'd'], dtype=object)
    clock = np.arange(len(value))
    time = clock * 10
    wave = Waveform(value, clock, time, signal=Signal('', '', None, None, False))

    result = wave.relative(1, pad='value', pad_value='X')
    assert np.all(result.value == np.array(['b', 'c', 'd', 'X'], dtype=object))

    result = wave.relative(-1, pad='value', pad_value='UNK')
    assert np.all(result.value == np.array(['UNK', 'a', 'b', 'c'], dtype=object))


def test_relative_errors():
    """Test relative() error conditions."""
    wave = build_waveform([1, 2, 3], width=8)

    with pytest.raises(ValueError, match="pad must be"):
        wave.relative(1, pad='invalid')

    with pytest.raises(ValueError, match="pad_value is required"):
        wave.relative(1, pad='value')


def test_relative_empty():
    """Test relative() on empty waveform."""
    wave = build_waveform([], width=8)
    result = wave.relative(1)
    assert len(result.value) == 0


def test_relative_offset_exceeds_length():
    """Test relative() when abs(offset) >= len(value)."""
    wave = build_waveform([1, 2, 3], width=8)

    # offset == length: all padded
    result = wave.relative(3, pad='repeat')
    assert np.all(result.value == np.array([3, 3, 3]))

    result = wave.relative(-3, pad='repeat')
    assert np.all(result.value == np.array([1, 1, 1]))

    # offset > length: all padded
    result = wave.relative(5, pad='repeat')
    assert np.all(result.value == np.array([3, 3, 3]))

    result = wave.relative(-5, pad='repeat')
    assert np.all(result.value == np.array([1, 1, 1]))

    # Same with pad='value'
    result = wave.relative(5, pad='value', pad_value=0)
    assert np.all(result.value == np.array([0, 0, 0]))

    result = wave.relative(-5, pad='value', pad_value=99)
    assert np.all(result.value == np.array([99, 99, 99]))


def test_relative_preserves_clock_time():
    """Test that relative() preserves clock and time arrays unchanged."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)

    result = wave.relative(2)
    assert np.array_equal(result.clock, wave.clock)
    assert np.array_equal(result.time, wave.time)

    result = wave.relative(-2)
    assert np.array_equal(result.clock, wave.clock)
    assert np.array_equal(result.time, wave.time)


def test_relative_preserves_dtype():
    """Test that relative() preserves value array dtype."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)
    result = wave.relative(1)
    assert result.value.dtype == wave.value.dtype


def test_ahead_back():
    """Test ahead() and back() convenience methods."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)

    # ahead() with default n=1, pad='repeat'
    result = wave.ahead()
    assert np.all(result.value == np.array([2, 3, 4, 5, 5]))

    # ahead(2) should equal relative(2)
    result = wave.ahead(2)
    expected = wave.relative(2)
    assert np.all(result.value == expected.value)

    # back() with default n=1, pad='repeat'
    result = wave.back()
    assert np.all(result.value == np.array([1, 1, 2, 3, 4]))

    # back(2) should equal relative(-2)
    result = wave.back(2)
    expected = wave.relative(-2)
    assert np.all(result.value == expected.value)

    # ahead/back with pad='value'
    result = wave.ahead(pad='value', pad_value=0)
    assert np.all(result.value == np.array([2, 3, 4, 5, 0]))

    result = wave.back(pad='value', pad_value=99)
    assert np.all(result.value == np.array([99, 1, 2, 3, 4]))


def test_rising_edge_with_ahead():
    """Test rising edge detection using ahead()."""
    wave = build_waveform([0, 0, 1, 1, 0, 1], width=1)

    # Rising edge: current is 0, next is 1
    # wave = [0, 0, 1, 1, 0, 1]
    # wave.ahead() = [0, 1, 1, 0, 1, 1]
    # wave == 0 = [1, 1, 0, 0, 1, 0]
    # rising = [0, 1, 0, 0, 1, 0]
    rising = (wave == 0) & wave.ahead()
    assert np.all(rising.value == np.array([0, 1, 0, 0, 1, 0]))


def test_relative_preserves_metadata():
    """Test that relative() preserves signal metadata."""
    wave = build_waveform([1, 2, 3, 4, 5], width=8)
    wave.name = 'test_sig'

    result = wave.ahead()
    assert result.width == 8
    assert result.name == 'test_sig'
