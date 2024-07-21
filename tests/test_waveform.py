import pytest
import numpy as np
from src.wavekit.waveform import Waveform


def test_waveform():

    value = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    clock = np.arange(len(value))
    time = clock * 10

    width = 32
    a = Waveform(value, clock, time, width=width, signed=False)
    b = Waveform(value, clock, time, width=width, signed=False)

    assert np.all((a + b).value == (value * 2))
    assert (a+b).width == width + 1
    assert np.all((a - b).value == 0)
    assert (a-b).width == width
    assert np.all((a * b).value == value * value)
    assert (a*b).width == 2*width
    assert np.all((a / b).value == value / value)
    assert (a/b).width is None
    assert np.all((a // b).value == (value // value))
    assert (a//b).width == width
    assert np.all(a.sample(2).value == np.array([1.5, 3.5, 5.5, 7.5]))

    assert np.all((a + 1).value == (value + 1))
    assert (a+1).width == width + 1
    assert np.all((a - 1).value == (value - 1))
    assert (a-1).width == width
    assert np.all((a * 3).value == (value * 3))
    assert (a*3).width == width + 2
    assert np.all((a / 5).value == value / 5)
    assert (a/5).width is None
    assert np.all((a // 5).value == value // 5)
    assert (a//5).width == width

    assert np.all((1 + a).value == (value + 1))
    assert (1+a).width == width + 1
    assert np.all((10 - a).value == (10 - value))
    assert (31 - a).width == width
    assert np.all((3*a).value == (value * 3))
    assert (3*a).width == width + 2
    assert np.all((20/a).value == (20/value))
    assert (20/a).width is None
    assert np.all((5//a).value == 5//value)
    assert (5//a).width == int.bit_length(5)

    width = 64
    a = Waveform(value, clock, time, width=width, signed=False)
    b = Waveform(value, clock, time, width=width, signed=False)
    assert np.all((a + b).value == (value * 2))
    assert (a+b).width == 64
    assert np.all((a - b).value == 0)
    assert (a-b).width == 64
    assert np.all((a * b).value == value * value)
    assert (a*b).width == 64
    assert np.all((a * 5).value == value * 5)
    assert (a*b).width == 64

    value = np.array([1, 2, 3, 4, 2**32-1, 2**32, 2 **
                     64-2, 2**64-1], dtype=np.uint64)
    width = 64
    clock = np.arange(len(value))
    time = clock * 10
    a = Waveform(value, clock, time, width=width, signed=False)
    assert np.all(a.count_one().value == np.array([1, 1, 2, 1, 32, 1, 63, 64]))
    assert np.all(a[1].value == np.array([0, 1, 1, 0, 1, 0, 1, 1]))
    assert np.all(a[2:1].value == np.array([0, 1, 1, 2, 3, 0, 3, 3]))

    width = 128
    value = np.array([1, 2, 3, 4, 2**32-1, 2**32, 2 **
                     64-2, 2**64-1, 2**66, 2**128-1])
    a = Waveform(value, clock, time, width=width, signed=False)
    assert np.all(a[68:66].value == np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 7]))

    value = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.uint64)
    width = 10
    clock = np.arange(len(value))
    time = clock * 10
    a = Waveform(value, clock, time, width=width, signed=False)
    splited_as = a.split_bits(3, padding=True)
    assert len(splited_as) == 4
    assert np.all(splited_as[0].value == np.array([1, 2, 4, 0, 0, 0, 0]))
    assert np.all(splited_as[1].value == np.array([0, 0, 0, 1, 2, 4, 0]))
    assert np.all(splited_as[2].value == np.array([0, 0, 0, 0, 0, 0, 1]))

    value = np.array([1, 2, 3, 4, 0, 666, 100, 200, 333, 444,
                     1023, 1999, 99999], dtype=np.uint64)
    value2 = value + 0x1010_0111
    width = 44
    clock = np.arange(len(value))
    time = clock * 10
    a = Waveform(value, clock, time, width=width, signed=False)
    b = Waveform(value2, clock, time, width=width, signed=False)
    assert np.all((a & b).value == (value & value2))
    assert (a & b).width == width
    assert np.all((a | b).value == (value | value2))
    assert (a | b).width == width
    assert np.all((a ^ b).value == (value ^ value2))
    assert (a ^ b).width == width
    assert np.all((~a).value == ((~value) & ((1 << width)-1)))
    assert (~a).width == width
    assert np.all((a == b).value == (value == value2))
    assert (a == b).width == 1

    assert np.all((a & 666).value == (value & 666))
    assert (a & 666).width == width
    assert np.all((a | 666).value == (value | 666))
    assert (a | 666).width == width
    assert np.all((a ^ 666).value == (value ^ 666))
    assert (a ^ 666).width == width
    assert np.all((~a).value == ((~value) & ((1 << width)-1)))
    assert (~a).width == width
    assert np.all((a == 666).value == (value == 666))
    assert (a == 666).width == 1

    value = np.array([1, 2, 3, 4, 0, 666, 100, 200, 333,
                     444, 1023, 1999], dtype=np.uint64)
    values = [value, value + 1234, value + 9999]
    width = [16, 20, 21]
    clock = np.arange(len(value))
    time = clock * 10
    waves = [Waveform(v, clock, time, width=w, signed=False)
             for v, w in zip(values, width)]
    concat_values = (values[0] + (values[1] << width[0]) +
                     (values[2] << (width[0] + width[1])))
    concat_waves = Waveform.concat(waves)
    assert np.all(concat_waves.value == concat_values)
    splited_waves = concat_waves.split_bits(width, padding=False)
    for v, w in zip(values, splited_waves):
        assert np.all(v == w.value)

    value = np.array([1, 2, 3, 4, 0, 666, 100, 200, 333,
                     444, 1023, 1999], dtype=np.object_)
    values = [value, value + 1234, value + 99999]
    width = [16, 35, 30]
    clock = np.arange(len(value))
    time = clock * 10
    waves = [Waveform(v, clock, time, width=w, signed=False)
             for v, w in zip(values, width)]
    concat_values = (values[0] + (values[1] << width[0]) +
                     (values[2] << (width[0] + width[1])))
    concat_waves = Waveform.concat(waves)
    assert np.all(concat_waves.value == concat_values)
    splited_waves = concat_waves.split_bits(width, padding=False)
    for v, w in zip(values, splited_waves):
        assert np.all(v == w.value)

    value = np.array([1, 2, 9, 4, 10, 666],)
    width = 20
    clock = np.arange(len(value))
    time = clock * 10
    a = Waveform(value, clock, time, width=width, signed=False)
    b = a.filter(lambda x: (x > 5))
    assert np.all(b.value == np.array([9, 10, 666]))

    value = np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0])
    width = 1
    clock = np.arange(len(value))
    time = clock*10
    a = Waveform(value, clock, time, width=width, signed=False)
    b = a.rise()
    assert np.all(b.value == np.array([1, 1, 1]))
    assert np.all(b.clock == np.array([1, 3, 10]))
    assert np.all(b.time == np.array([1, 3, 10]) * 10)

    b = a.fall()
    assert np.all(b.value == np.array([0, 0, 0]))
    assert np.all(b.clock == np.array([2, 7, 11]))
    assert np.all(b.time == np.array([2, 7, 11]) * 10)

    assert (np.all(a.compress().value == np.array([0, 1, 0, 1, 0, 1, 0])))

    value = np.array([])
    clock = np.arange(len(value))
    time = clock*10
    a = Waveform(value, clock, time, width=width, signed=False)
    assert (np.all(a.compress().value == np.array([])))

    value = np.array([50])
    clock = np.arange(len(value))
    time = clock*10
    a = Waveform(value, clock, time, width=width, signed=False)
    assert (np.all(a.compress().value == np.array([50])))

    value = np.array([50, 50])
    clock = np.arange(len(value))
    time = clock*10
    a = Waveform(value, clock, time, width=width, signed=False)
    assert (np.all(a.compress().value == np.array([50, 50])))

    value = np.array([50, 60])
    clock = np.arange(len(value))
    time = clock*10
    a = Waveform(value, clock, time, width=width, signed=False)
    assert (np.all(a.compress().value == np.array([50, 60])))

    value1 = np.array([50, 100, 200])
    value2 = np.array([30, 40, 90])
    clock = np.arange(len(value))
    time = clock*10
    a = Waveform(value1, clock, time, width=width, signed=False)
    b = Waveform(value2, clock, time, width=width, signed=False)
    merged = Waveform.merge([a, b], lambda x: x[0] + x[1], 32, False)

    assert np.all(merged.value == value1 + value2)
    assert merged.width == 32
    assert np.all(merged.clock == clock)
    assert np.all(merged.time == time)
