from __future__ import annotations

from typing import Any, Callable, Union, cast

import numpy as np
import numpy.typing as npt

from .signal import Signal


class Waveform:
    WaveformOrScalar = Union['Waveform', int, float]

    def __init__(
        self,
        value: npt.NDArray[Any],
        clock: npt.NDArray[Any],
        time: npt.NDArray[Any],
        width: int | None,
        signed: bool,
        signal: str = '',
        width_resolver: Callable[[], int] | None = None,
    ):
        self.clock: npt.NDArray[Any] = clock
        self.time: npt.NDArray[Any] = time

        self._signal: Signal = Signal(
            name=signal,
            width=width,
            signed=signed,
            width_resolver=width_resolver,
        )

        if width is None:
            self.value = value
        elif width > 64:
            self.value = value.astype(np.object_)
        elif signed:
            self.value = value.astype(np.int64)
        else:
            self.value = value.astype(np.uint64)

    @property
    def width(self) -> int | None:
        return self._signal.width

    @width.setter
    def width(self, value: int | None):
        self._signal.width = value

    @property
    def signed(self) -> bool:
        return self._signal.signed

    @signed.setter
    def signed(self, value: bool):
        self._signal.signed = value

    @property
    def signal(self) -> Signal:
        return self._signal

    @property
    def name(self) -> str:
        return self._signal.name

    @name.setter
    def name(self, value: str):
        self._signal.name = value

    def __str__(self):
        return f"Waveform(signal='{self.name}', width={self.width}, signed={self.signed})"

    def set_signal(self, signal: str) -> Waveform:
        self.name = signal
        return self

    @property
    def data(self) -> Any:
        return cast(
            Any,
            np.rec.fromarrays(
                cast(Any, [self.time, self.clock, self.value]),
                names=cast(Any, 'time,clock,value'),
            ),
        )

    def unique_consecutive(self) -> Waveform:
        if len(self.value) <= 1:
            return self.copy()
        diff_mask = np.diff(self.value) != 0
        padded_diff_mask = np.concatenate(([1], diff_mask[:-1], [1]))
        diff_indices = np.where(padded_diff_mask)[0]
        return self.take(diff_indices)

    def compress(
        self,
        condition: Callable[[npt.NDArray[Any]], npt.NDArray[np.bool_]],
    ) -> Waveform:
        mask = condition(self.value)
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError('compress requires boolean numpy array')
        return self.mask(mask)

    def mask(self, mask: npt.NDArray[np.bool_]) -> Waveform:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError('mask requires boolean numpy array')
        return Waveform(
            value=self.value[mask],
            clock=self.clock[mask],
            time=self.time[mask],
            width=self.width,
            signed=self.signed,
        )

    def copy(self) -> Waveform:
        return self.vectorized_map(lambda x: np.copy(x))

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
        if self.width is None:
            raise ValueError('width is None')
        width = self.width
        return self.vectorized_map(lambda x: self._signed(x, width), signed=True)

    @staticmethod
    # @jit
    def _unsigned(value: np.ndarray, width: int):
        return value & ((1 << width) - 1)

    def as_unsigned(self) -> Waveform:
        if not self.signed:
            return self.copy()
        if self.width is None:
            raise ValueError('width is None')
        width = self.width
        return self.vectorized_map(lambda x: self._unsigned(x, width), signed=False)

    @staticmethod
    # @jit
    def _add(a, b):
        return a + b

    def _check_sign(self, other: WaveformOrScalar):
        if isinstance(other, Waveform) and self.signed != other.signed:
            raise ValueError('signedness mismatch')

    def _check_arithmetic_op_width(self, other: WaveformOrScalar):
        if self.width is not None and self.width > 64:
            raise ValueError('width too large')

        if isinstance(other, Waveform) and other.width is not None and other.width > 64:
            raise ValueError('width too large')

    def _infer_arithmetic_op_width(
        self,
        width_supplier: Callable[[], int | None],
    ) -> int | None:
        inferred_width = width_supplier()
        if inferred_width is None:
            return None
        return min(inferred_width, 64)

    @staticmethod
    def _get_width(other: WaveformOrScalar) -> int | None:
        if isinstance(other, Waveform):
            return other.width
        elif isinstance(other, int):
            return int.bit_length(other)
        elif isinstance(other, float):
            return None
        else:
            raise ValueError('unsupported type')

    def _optional_max_width(self, other: WaveformOrScalar) -> int | None:
        other_width = self._get_width(other)
        if self.width is None or other_width is None:
            return None
        return max(self.width, other_width)

    @staticmethod
    def _get_value(other: WaveformOrScalar) -> Any:
        if isinstance(other, Waveform):
            return other.value
        else:
            return other

    def __add__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        self._check_arithmetic_op_width(other)

        def inferred_width() -> int | None:
            max_width = self._optional_max_width(other)
            if max_width is None:
                return None
            return max_width + 1

        new_width = self._infer_arithmetic_op_width(inferred_width)

        return self.vectorized_map(
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

        def inferred_width() -> int | None:
            return self._optional_max_width(other)

        new_width = self._infer_arithmetic_op_width(inferred_width)

        return self.vectorized_map(
            lambda x: self._sub(x, self._get_value(other)),
            width=new_width,
            signed=self.signed,
        )

    def __rsub__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)

        def inferred_width() -> int | None:
            return self._optional_max_width(other)

        new_width = self._infer_arithmetic_op_width(inferred_width)

        return self.vectorized_map(
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

        def inferred_width() -> int | None:
            other_width = self._get_width(other)
            if self.width is None or other_width is None:
                return None
            return self.width + other_width

        new_width = self._infer_arithmetic_op_width(inferred_width)

        return self.vectorized_map(
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
        new_width = self._infer_arithmetic_op_width(lambda: self._get_width(other))
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
        new_width = self._infer_arithmetic_op_width(lambda: self._get_width(other))

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
        return cast(Any, other).__pow__(self)

    def _check_logical_op_type(self, other):
        if self.value.dtype not in (np.int64, np.uint64, np.object_):
            raise TypeError('Can only perform logical operations on 64-bit integers')

        if isinstance(other, Waveform):
            if other.value.dtype not in (np.int64, np.uint64, np.object_):
                raise TypeError('Can only perform logical operations on 64-bit integers')
        elif isinstance(other, float):
            raise TypeError('Can only perform logical operations on 64-bit integers')

    def _infer_logical_op_width(
        self,
        other: WaveformOrScalar,
        inferred_width: int | None = None,
    ) -> int:
        if inferred_width is not None:
            return inferred_width

        if isinstance(other, Waveform):
            if self.width is None or other.width is None:
                raise ValueError('width mismatch: None')
            if self.width != other.width:
                raise ValueError(f'width mismatch: {self.width} and {other.width}')
            return self.width
        if isinstance(other, float):
            raise TypeError('Can only perform logical operations on 64-bit integers')
        if self.width is None:
            raise ValueError('width mismatch: None')
        if self.width < int.bit_length(other):
            raise ValueError(f'width mismatch: {self.width} and {other}')
        return self.width

    @staticmethod
    # @jit
    def _lshift(a, b):
        return a << b

    def __lshift__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        self._check_logical_op_type(other)
        if isinstance(other, float):
            raise TypeError('Can only perform logical operations on 64-bit integers')
        base_width = self.width or 0
        if isinstance(other, Waveform):
            if other.width is None:
                raise ValueError('width mismatch: None')
            shift_width = 1 << other.width
        else:
            shift_width = other
        new_width = self._infer_logical_op_width(
            other,
            inferred_width=base_width + shift_width,
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
        if isinstance(other, float):
            raise TypeError('Can only perform logical operations on 64-bit integers')

        if isinstance(other, Waveform):
            new_width = self._infer_logical_op_width(other)
        else:
            new_width = self._infer_logical_op_width(
                other,
                inferred_width=max((self.width or 0) - other, 0),
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
        if isinstance(other, float):
            raise TypeError('Can only perform logical operations on 64-bit integers')

        if isinstance(other, Waveform):
            new_width = self._infer_logical_op_width(other)
        else:
            new_width = self._infer_logical_op_width(
                other,
                inferred_width=max((self.width or 0) - other, 0),
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
        return self.vectorized_map(
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
        return self.vectorized_map(
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
        return self.vectorized_map(
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
        if self.width is None:
            raise ValueError('width is None')
        width_value = self.width
        return self.vectorized_map(
            lambda x: self._invert(x, width_value),
            width=width_value,
            signed=False,
        )

    @staticmethod
    # @jit
    def _eq(a, b):
        return a == b

    def __eq__(self, other: object) -> Any:
        if not isinstance(other, (Waveform, int, float)):
            return NotImplemented
        self._check_sign(other)
        return self.vectorized_map(
            lambda x: self._eq(x, self._get_value(other)),
            width=1,
            signed=False,
        )

    @staticmethod
    # @jit
    def _ne(a, b):
        return a != b

    def __ne__(self, other: object) -> Any:
        if not isinstance(other, (Waveform, int, float)):
            return NotImplemented
        self._check_sign(other)
        return self.vectorized_map(
            lambda x: self._ne(x, self._get_value(other)),
            width=1,
            signed=False,
        )

    @staticmethod
    # @jit
    def _fast_bitsel(value, start: int, width: int):
        return (value >> np.uint64(start)) & ((np.uint64(1) << np.uint64(width)) - np.uint64(1))

    @staticmethod
    def _bitsel(value, start: int, width: int):
        if value.dtype == np.object_:
            return (value >> start) & ((1 << width) - 1)
        else:
            return Waveform._fast_bitsel(value, start, width)

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.step is not None:
                raise Exception('slice with step is not supported')

            if index.start < index.stop:
                raise Exception('only support little-endian slicing')

            start = index.stop
            width = (index.start - index.stop) + 1
        elif isinstance(index, int):
            start = index
            width = 1
        else:
            raise Exception('unsupported index type')

        new_value = self._bitsel(self.value, start, width)
        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=width,
            signed=False,
        )

    def take(self, indices: npt.NDArray[np.integer[Any]] | list[int]):
        if isinstance(indices, np.ndarray):
            if indices.dtype == np.bool_:
                raise TypeError('take requires integer indices')
            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError('take requires integer indices')
        else:
            if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices):
                raise TypeError('take requires integer indices')
        return Waveform(
            value=self.value[indices],
            clock=self.clock[indices],
            time=self.time[indices],
            width=self.width,
            signed=self.signed,
        )

    def downsample(
        self,
        chunk_size: int,
        func: Callable[[npt.NDArray[Any]], float] = np.mean,
    ) -> Waveform:
        def helper(
            arr: npt.NDArray[Any],
            func: Callable[[npt.NDArray[Any]], float],
        ) -> npt.NDArray[Any]:
            sampled_arr = [func(arr[i : i + chunk_size]) for i in range(0, len(arr), chunk_size)]
            return np.array(sampled_arr)

        return Waveform(
            value=helper(self.value, func),
            clock=helper(self.clock, np.mean),
            time=helper(self.time, np.mean),
            width=None,
            signed=self.signed,
        )

    @staticmethod
    def _count_one(x, width: int):
        t = np.zeros(x.shape, dtype=np.uint64)
        mask = (1 << 64) - 1
        for idx in range(0, width, 64):
            chunk = ((x >> idx) & mask).astype(np.uint64)
            t = np.bitwise_count(chunk).astype(np.uint64) + t
        return t

    def vectorized_map(
        self,
        func: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
        width: int | None = None,
        signed: bool | None = None,
    ) -> Waveform:
        new_value = func(self.value)
        return Waveform(
            value=new_value,
            clock=np.copy(self.clock),
            time=np.copy(self.time),
            width=width or self.width,
            signed=signed if signed is not None else self.signed,
        )

    def map(
        self,
        func: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
        width: int | None = None,
        signed: bool | None = None,
    ) -> Waveform:
        vectorized_func = np.vectorize(func)
        return self.vectorized_map(vectorized_func, width, signed)

    def falling_edge(self) -> Waveform:
        if self.width != 1:
            raise Exception('raising only support 1-bit waveform')
        one = self.value[:-1] == 1
        zero = self.value[1:] == 0
        new_value = np.concatenate(([False], one & zero))
        return Waveform(
            value=new_value,
            clock=np.copy(self.clock),
            time=np.copy(self.time),
            width=1,
            signed=False,
        )

    def rising_edge(self) -> Waveform:
        if self.width != 1:
            raise Exception('raising only support 1-bit waveform')
        zero = self.value[:-1] == 0
        one = self.value[1:] == 1
        new_value = np.concatenate(([False], one & zero))
        return Waveform(
            value=new_value,
            clock=np.copy(self.clock),
            time=np.copy(self.time),
            width=1,
            signed=False,
        )

    def bit_count(self) -> Waveform:
        if self.width is None:
            raise ValueError('width is None')
        width = self.width
        if self.value.dtype != np.object_ and self.width <= 64:
            return self.vectorized_map(lambda v: np.bitwise_count(v), width=64, signed=False)
        return self.vectorized_map(lambda v: Waveform._count_one(v, width), width=64, signed=False)

    def split_bits(self, bit_group_size: int | list[int], padding: bool = False) -> list[Waveform]:
        if self.width is None:
            raise ValueError('width is None')
        width = self.width
        if isinstance(bit_group_size, int):
            if (not padding) and (width % bit_group_size != 0):
                raise Exception('width must be a multiple of bit_group_size when padding is false')
            return [
                self[min(i + bit_group_size - 1, width) : i]
                for i in range(0, width, bit_group_size)
            ]
        else:
            if sum(bit_group_size) != width:
                raise Exception(
                    'the sum of the bit_group_size must be equal to the width when padding == False'
                )

            res = []
            start_bit = 0
            for s in bit_group_size:
                res.append(self[start_bit + s - 1 : start_bit])
                start_bit += s
            return res

    @staticmethod
    def concatenate(waves: list[Waveform]) -> Waveform:
        if not all(not w.signed for w in waves):
            raise Exception('all waveforms should be unsigned')

        widths = [w.width for w in waves]
        if any(w is None for w in widths):
            raise ValueError('width is None')
        widths_int = [w for w in widths if w is not None]
        concat_width = sum(widths_int)
        dtype = np.uint64 if concat_width <= 64 else np.object_

        new_value: Any = 0
        for w in reversed(waves):
            if w.width is None:
                raise ValueError('width is None')
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
        func: Callable[[list[Any]], Any],
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

    def time_slice(
        self,
        begin_time: int | None = None,
        end_time: int | None = None,
        include_end: bool = False,
    ) -> Waveform:
        # numpy有没有提供一个函数，根据范围，获取一个有序数组的索引
        if begin_time is None:
            begin_time = int(self.time[0])
        if end_time is None:
            end_time = int(self.time[-1]) + 1
        start_idx = np.searchsorted(self.time, begin_time, side='left')
        end_idx = np.searchsorted(self.time, end_time, side='right' if include_end else 'left')
        return Waveform(
            value=self.value[start_idx:end_idx],
            clock=self.clock[start_idx:end_idx],
            time=self.time[start_idx:end_idx],
            width=self.width,
            signed=self.signed,
            signal=self.name,
        )

    def slice(self, begin_idx: int, end_idx: int, include_end: bool = False) -> Waveform:
        if include_end:
            end_idx += 1
        return Waveform(
            value=self.value[begin_idx:end_idx],
            clock=self.clock[begin_idx:end_idx],
            time=self.time[begin_idx:end_idx],
            width=self.width,
            signed=self.signed,
            signal=self.name,
        )
