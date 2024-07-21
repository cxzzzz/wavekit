from __future__ import annotations
import numpy as np
from numba import jit
from typing import Union, Callable, List


class Waveform:

    WaveformOrScalar = Union["Waveform", int, float]

    def __init__(
        self,
        value: np.ndarray,
        clock: np.ndarray,
        time: np.ndarray,
        width: int,
        signed: bool,
        signal: str = "",
    ):

        self.clock = clock
        self.time = time

        self.width = width
        self.signed = signed
        self.signal = signal

        if width is None:
            self.value = value
        elif width > 64:
            self.value = value.astype(np.object_)
        elif signed:
            self.value = value.astype(np.int64)
        else:
            self.value = value.astype(np.uint64)

    def __str__(self):
        return f"Waveform(signal='{self.signal}', width={self.width}, signed={self.signed})"

    def set_signal(self, signal):
        self.signal = signal
        return self

    @property
    def data(self):
        return np.rec.fromarrays(
            [self.time, self.clock, self.value],
            names="time,clock,value",
        )

    def compress(self) -> Waveform:
        if len(self.value) <= 1:
            return self.copy()
        diff_mask = np.diff(self.value) != 0
        padded_diff_mask = np.concatenate(([1], diff_mask[:-1], [1]))
        diff_indices = np.where(padded_diff_mask)[0]
        return self.take(diff_indices)

    def copy(self) -> Waveform:
        return self.map(lambda x: np.copy(x))

    @staticmethod
    # @jit
    def _signed(value: np.ndarray, width: int):
        offset = 1 << width
        return np.where(
            value < (offset // 2),
            value.astype(np.uint64),
            value.astype(np.uint64) - offset,
        )

    def as_signed(self) -> Waveform:
        if self.signed:
            return self.copy()
        return self.map(lambda x: self._signed(x, self.width), signed=True)

    @staticmethod
    # @jit
    def _unsigned(value: np.ndarray, width: int):
        return value & ((1 << width) - 1)

    def as_unsigned(self) -> Waveform:
        if not self.signed:
            return self.copy()
        return self.map(lambda x: self._unsigned(x, self.width), signed=False)

    @staticmethod
    # @jit
    def _add(a, b):
        return a + b

    def _check_sign(self, other: WaveformOrScalar):
        if isinstance(other, Waveform) and self.signed != other.signed:
            raise ValueError("signedness mismatch")

    def _check_arithmetic_op_width(self, other: WaveformOrScalar):
        if self.width is not None and self.width > 64:
            raise ValueError("width too large")

        if isinstance(other, Waveform) and other.width is not None and other.width > 64:
            raise ValueError("width too large")

    def _infer_arithmetic_op_width(self, inferred_width: Callable[[], int]):
        try:
            inferred_width = min(inferred_width(), 64)
        except Exception:
            inferred_width = None

        return inferred_width

    @staticmethod
    def _get_width(other: WaveformOrScalar):
        if isinstance(other, Waveform):
            return other.width
        elif isinstance(other, int):
            return int.bit_length(other)
        elif isinstance(other, float):
            return None
        else:
            raise ValueError("unsupported type")

    @staticmethod
    def _get_value(other: WaveformOrScalar):
        if isinstance(other, Waveform):
            return other.value
        else:
            return other

    def __add__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        self._check_arithmetic_op_width(other)

        new_width = self._infer_arithmetic_op_width(
            lambda: max(self.width, self._get_width(other)) + 1
        )

        return self.map(
            lambda x: self._add(x, self._get_value(other)),
            width=new_width,
            signed=self.signed,
        )

    def __radd__(self, other: WaveformOrScalar) -> Waveform:
        return self.__add__(other)

    @staticmethod
    # @jit
    def _sub(a, b):
        return a - b

    def __sub__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)

        new_width = self._infer_arithmetic_op_width(
            lambda: max(self.width, self._get_width(other))
        )

        return self.map(
            lambda x: self._sub(x, self._get_value(other)),
            width=new_width,
            signed=self.signed,
        )

    def __rsub__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)

        new_width = self._infer_arithmetic_op_width(
            lambda: max(self.width, self._get_width(other))
        )

        return self.map(
            lambda x: self._sub(self._get_value(other), x),
            width=new_width,
            signed=self.signed,
        )

    @staticmethod
    # @jit
    def _mul(a, b):
        return a * b

    def __mul__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)

        new_width = self._infer_arithmetic_op_width(
            lambda: self.width + self._get_width(other)
        )

        return self.map(
            lambda x: self._mul(x, self._get_value(other)),
            width=new_width,
            signed=self.signed,
        )

    def __rmul__(self, other: WaveformOrScalar) -> Waveform:
        return self.__mul__(other)

    @staticmethod
    # @jit
    def _truediv(a, b):
        return a / b

    def __truediv__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_value = self._truediv(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=None,
            signed=self.signed,
        )

    def __rtruediv__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_value = self._truediv(self._get_value(other), self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=None,
            signed=self.signed,
        )

    @staticmethod
    # @jit
    def _floordiv(a, b):
        return a // b

    def __floordiv__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(lambda: self.width)
        new_value = self._floordiv(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rfloordiv__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(
            lambda: self._get_width(other))
        new_value = self._floordiv(self._get_value(other), self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    @staticmethod
    # @jit
    def _mod(a, b):
        return a % b

    def __mod__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(lambda: self.width)

        new_value = self._mod(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rmod__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(
            lambda: self._get_width(other))

        new_value = self._mod(self._get_value(other), self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    @staticmethod
    # @jit
    def _pow(a, b):
        return a**b

    def __pow__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(
            lambda: self.width * self._get_width(other)
        )

        new_value = self._pow(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __pow__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(lambda: 64)

        new_value = self._pow(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rpow__(self, other: WaveformOrScalar) -> Waveform:
        return other.__pow__(self)

    def _check_logical_op_type(self, other):
        if self.value.dtype not in (np.int64, np.uint64, np.object_):
            raise TypeError(
                "Can only perform logical operations on 64-bit integers")

        if isinstance(other, Waveform):
            if other.value.dtype not in (np.int64, np.uint64, np.object_):
                raise TypeError(
                    "Can only perform logical operations on 64-bit integers"
                )
        elif isinstance(other, float):
            raise TypeError(
                "Can only perform logical operations on 64-bit integers")

    def _infer_logical_op_width(
        self, other: WaveformOrScalar, inferred_width: int = None
    ):
        if inferred_width is not None:
            return inferred_width

        if isinstance(other, Waveform):
            if self.width != other.width:
                raise ValueError(
                    "width mismatch: {} and {}".format(self.width, other.width)
                )
            return self.width
        else:  # int
            if self.width < int.bit_length(other):
                raise ValueError(
                    "width mismatch: {} and {}".format(self.width, other))
            return self.width

    @staticmethod
    # @jit
    def _lshift(a, b):
        return a << b

    def __lshift__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        self._check_logical_op_type(other)
        new_width = self._infer_logical_op_width(
            other,
            inferred_width=(
                self.width + (other if isinstance(other, int)
                              else (1 << other.width))
            ),
        )

        new_value = self._lshift(
            self.value.astype(np.object_) if new_width > 64 else self.value,
            self._get_value(other),
        )

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rlshift__(self, other: WaveformOrScalar) -> Waveform:
        return self.__lshift__(other)

    @staticmethod
    # @jit
    def _rshift(a, b):
        return a >> b

    def __rshift__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        self._check_logical_op_type(other)

        new_width = self._infer_logical_op_width(
            self.width if isinstance(other, Waveform) else max(
                self.width - other, 0)
        )

        new_value = self._rshift(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rrshift__(self, other: WaveformOrScalar, width: int = None) -> Waveform:
        self._check_sign(other)
        self._check_logical_op_type(other)

        new_width = self._infer_logical_op_width(
            self.width if isinstance(other, Waveform) else max(
                self.width - other, 0)
        )

        new_value = self._rshift(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    @staticmethod
    # @jit
    def _and(a, b):
        return a & b

    def __and__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_logical_op_width(other)
        return self.map(
            lambda x: self._and(x, self._get_value(other)),
            width=new_width,
            signed=False,
        )

    def __rand__(self, other: WaveformOrScalar) -> Waveform:
        return self.__and__(other)

    @staticmethod
    # @jit
    def _or(a, b):
        return a | b

    def __or__(self, other: WaveformOrScalar, width: int = None) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_logical_op_width(other)
        return self.map(
            lambda x: self._or(x, self._get_value(other)), width=new_width, signed=False
        )

    def __ror__(self, other: WaveformOrScalar, width: int = None) -> Waveform:
        return self.__or__(other, width)

    @staticmethod
    # @jit
    def _xor(a, b):
        return a ^ b

    def __xor__(self, other: WaveformOrScalar, width: int = None) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_logical_op_width(other)
        return self.map(
            lambda x: self._xor(x, self._get_value(other)),
            width=new_width,
            signed=False,
        )

    def __rxor__(self, other: WaveformOrScalar, width: int = None) -> Waveform:
        return self.__xor__(other, width)

    @staticmethod
    # @jit
    def _invert(a, width: int):
        return (~a) & np.uint64((1 << width) - 1)

    def __invert__(self, width: int = None) -> Waveform:
        return self.map(
            lambda x: self._invert(x, self.width), width=self.width, signed=False
        )

    @staticmethod
    # @jit
    def _eq(a, b):
        return a == b

    def __eq__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        return self.map(
            lambda x: self._eq(x, self._get_value(other)), width=1, signed=False
        )

    @staticmethod
    # @jit
    def _fast_bitsel(value, start: int, width: int):
        return (value >> np.uint64(start)) & (
            (np.uint64(1) << np.uint64(width)) - np.uint64(1)
        )

    @staticmethod
    def _bitsel(value, start: int, width: int):
        if value.dtype == np.object_:
            return (value >> start) & ((1 << width) - 1)
        else:
            return Waveform._fast_bitsel(value, start, width)

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None:
                raise Exception("slice with step is not supported")

            if index.start < index.stop:
                raise Exception("only support little-endian slicing")

            start = index.stop
            width = (index.start - index.stop) + 1
        elif isinstance(index, int):
            start = index
            width = 1
        else:
            raise Exception("unsupported index type")

        new_value = self._bitsel(self.value, start, width)
        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=width,
            signed=False,
        )

    # 根据索引取值
    # TODO : 含义与numpy不一致，得改名
    def take(self, indices: Union[np.ndarray, list[int]]):

        return Waveform(
            value=self.value[indices],
            clock=self.clock[indices],
            time=self.time[indices],
            width=self.width,
            signed=self.signed,
        )

    def sample(self, chunk_size: int, func: [Callable[[np.ndarray], int]] = np.mean):
        def helper(arr: np.ndarray, func: Callable[[np.ndarray], int]):
            sampled_arr = [
                func(arr[i: i + chunk_size]) for i in range(0, len(arr), chunk_size)
            ]
            return np.array(sampled_arr)

        return Waveform(
            value=helper(self.value, func),
            clock=helper(self.clock, np.mean),
            time=helper(self.time, np.mean),
            width=None,
            signed=self.signed,
        )

    @staticmethod
    # @jit
    def _count_one_uint64(x: np.ndarray):
        m1 = np.uint64(0x5555555555555555)  # binary: 0101...
        m2 = np.uint64(0x3333333333333333)  # binary: 00110011..
        m4 = np.uint64(0x0F0F0F0F0F0F0F0F)  # binary:  4 zeros,  4 ones ...
        m8 = np.uint64(0x00FF00FF00FF00FF)  # binary:  8 zeros,  8 ones ...
        m16 = np.uint64(0x0000FFFF0000FFFF)  # binary: 16 zeros, 16 ones ...
        m32 = np.uint64(0x00000000FFFFFFFF)  # binary: 32 zeros, 32 ones ...
        hff = np.uint64(0xFFFFFFFFFFFFFFFF)  # binary: all ones
        h01 = np.uint64(0x0101010101010101)

        # put count of each 2 bits into those 2 bits
        x = x - ((x >> np.uint64(1)) & m1)
        x = (x & m2) + (
            (x >> np.uint64(2)) & m2
        )  # put count of each 4 bits into those 4 bits
        # put count of each 8 bits into those 8 bits
        x = (x + (x >> np.uint64(4))) & m4
        x = (x * h01) >> np.uint64(56)
        return x

    @staticmethod
    def _count_one(x, width: int):
        t = np.zeros(x.shape, dtype=np.uint64)
        for idx in range(0, width, 64):
            t = (
                Waveform._count_one_uint64(
                    ((x >> idx) & ((1 << 64) - 1)).astype(np.uint64)
                )
                + t
            )
        return t

    def map(
        self,
        func: Callable[[np.array], np.array],
        width: int = None,
        signed: bool = None,
    ) -> Waveform:
        new_value = func(self.value)
        return Waveform(
            value=new_value,
            clock=np.copy(self.clock),
            time=np.copy(self.time),
            width=width or self.width,
            signed=signed if signed is not None else self.signed,
        )

    def filter(self, func: Callable[[np.array], bool]) -> Waveform:
        new_indices = func(self.value)
        return self.take(new_indices)
        # return Waveform(
        #    value=self.value[new_indices],
        #    clock=self.clock[new_indices],
        #    time=self.time[new_indices],
        #    width=self.width,
        #    signed=self.signed
        # )

    def fall(self) -> Waveform:
        if self.width != 1:
            raise Exception("raising only support 1-bit waveform")
        one = self.value[:-1] == 1
        zero = self.value[1:] == 0
        new_indices = np.concatenate(([False], one & zero))
        return self.take(new_indices)

    def rise(self) -> Waveform:
        if self.width != 1:
            raise Exception("raising only support 1-bit waveform")
        zero = self.value[:-1] == 0
        one = self.value[1:] == 1
        new_indices = np.concatenate(([False], one & zero))
        return self.take(new_indices)

    def count_one(self) -> Waveform:
        return self.map(
            lambda v: Waveform._count_one(v, self.width), width=64, signed=False
        )

    def split_bits(
        self, bit_group_size: Union[int, list[int]], padding: bool = False
    ) -> List[Waveform]:
        if isinstance(bit_group_size, int):
            if (not padding) and (self.width % bit_group_size != 0):
                raise Exception(
                    "width must be a multiple of bit_group_size when padding is false"
                )
            return [
                self[min(i + bit_group_size - 1, self.width): i]
                for i in range(0, self.width, bit_group_size)
            ]
        else:
            if sum(bit_group_size) != self.width:
                raise Exception(
                    "the sum of the bit_group_size must be equal to the width when padding == False"
                )

            res = []
            start_bit = 0
            for s in bit_group_size:
                res.append(self[start_bit + s - 1: start_bit])
                start_bit += s
            return res

    @staticmethod
    def concat(waves: list[Waveform]) -> Waveform:
        if not (all([w.signed == False for w in waves])):
            raise Exception("all waveforms should be unsigned")

        concat_width = sum([w.width for w in waves])
        dtype = np.uint64 if concat_width <= 64 else np.object_

        new_value = 0
        for w in reversed(waves):
            new_value = ((new_value << w.width) | w.value).astype(dtype)

        return Waveform(
            value=new_value,
            clock=np.copy(waves[0].clock),
            time=np.copy(waves[0].time),
            width=concat_width,
            signed=False,
        )

    @staticmethod
    def merge(
        waves: list[Waveform],
        func: Callable[[list[any]], any],
        width: int,
        signed: bool,
    ) -> Waveform:
        wave_len = len(waves[0].value)
        assert all([wave_len == len(w.value) for w in waves])

        new_value = np.zeros(wave_len, dtype=np.object_)
        for idx in range(wave_len):
            value_lst = [w.value[idx] for w in waves]
            new_value[idx] = func(value_lst)

        return Waveform(
            value=new_value,
            clock=np.copy(waves[0].clock),
            time=np.copy(waves[0].time),
            width=width,
            signed=signed,
        )


"""
    @staticmethod
    def merge(waves:list[Waveform], func:Callable[[Waveform,Waveform],Waveform], width:int)->Waveform:

    @staticmethod
    #@jit
    def _ge(a,b):
        return a >= b

    def __ge__(self): ## gt le lt
        self._check_sign(other)

    def __getitem__():


    def resize(self, width: int)-> Waveform:
        pass

    def reduce()->Waveform:
        pass

    def apply()->Waveform:
        pass

    def sorted()->Waveform:
        pass

    @statisticmethod
    def concat(waves:list[Waveform])->Waveform:
        pass



"""
