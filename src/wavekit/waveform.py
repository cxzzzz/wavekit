import numpy as np
from numba import jit
from typing import Union, Self, Callable


class Waveform:

    def __init__(
        self,
        value: np.ndarray,
        clock: np.ndarray,
        time: np.ndarray,
        width: int,
        signed: bool,
    ):

        self.value = value
        self.clock = clock
        self.time = time

        self.width = width
        self.signed = signed

    type WaveformOrScalar = Union[Self, int, float]

    @property
    def data(self):
        return np.rec.fromarrays(
            [self.time, self.clock, self.value],
            names="time,clock,value",
        )

    @staticmethod
    @jit
    def _signed(value: np.ndarray, offset: int):
        return np.where(
            value < (offset // 2),
            value.astype(np.uint64),
            value.astype(np.uint64) - offset,
        )

    def assigned(self) -> Self:
        offset = 1 << self.width
        new_value = self._signed(self.value, offset) if not self.signed else self.value
        return Waveform(
            new_value, clock=self.clock, time=self.time, width=self.width, signed=True
        )

    @staticmethod
    @jit
    def _unsigned(value: np.ndarray, offset: int):
        return np.where(value >= 0, value, value + offset)

    def asunsigned(self) -> Self:
        offset = 1 << self.width
        new_value = self._unsigned(self.value, offset) if self.signed else self.value
        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=self.width,
            signed=False,
        )

    @staticmethod
    @jit
    def _add(a, b):
        return a + b

    def _check_sign(self, other: WaveformOrScalar):
        if isinstance(other, Waveform) and self.signed != other.signed:
            raise ValueError("signedness mismatch")

    def _check_arithmetic_op_width(self, other: WaveformOrScalar):
        if self.width is not None and self.width > 64:
            raise ValueError("width too large")

        if other.isinstance(Waveform) and other.width is not None and other.width > 64:
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

    def __add__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)
        self._check_arithmetic_op_width(other)

        new_value = self._add(self.value, self._get_value(other.value))
        new_width = self._infer_arithmetic_op_width(
            lambda: max(self.width, self._get_width(other)) + 1
        )

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __radd__(self, other: WaveformOrScalar) -> Self:
        return self.__add__(other)

    @staticmethod
    @jit
    def _sub(a, b):
        return a - b

    def __sub__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)

        new_width = self._infer_arithmetic_op_width(
            lambda: max(self.width, self._get_width(other))
        )

        new_value = self._sub(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rsub__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)

        new_width = self._infer_arithmetic_op_width(
            lambda: max(self.width, self._get_width(other))
        )
        new_value = self._sub(self._get_value(other), self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    @staticmethod
    @jit
    def _mul(a, b):
        return a * b

    def __mul__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)

        new_width = self._infer_arithmetic_op_width(
            lambda: self.width + self._get_width(other)
        )
        new_value = self._mul(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rmul__(self, other: WaveformOrScalar) -> Self:
        return other.__mul__(self)

    @staticmethod
    @jit
    def _truediv(a, b):
        return a / b

    def __truediv__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)
        new_value = self._truediv(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=None,
            signed=self.signed,
        )

    def __rtruediv__(self, other: WaveformOrScalar) -> Self:
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
    @jit
    def _floordiv(a, b):
        return a // b

    def __floordiv__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(lambda: self.width)
        new_value = self._mul(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rfloordiv__(self, other: WaveformOrScalar) -> Self:
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
    @jit
    def _mod(a, b):
        return a % b

    def __mod__(self, other: WaveformOrScalar) -> Self:
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

    def __rmod__(self, other: WaveformOrScalar) -> Self:
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
    @jit
    def _pow(a, b):
        return a**b

    def __pow__(self, other: WaveformOrScalar) -> Self:
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

    def __pow__(self, other: WaveformOrScalar) -> Self:
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

    def __rpow__(self, other: WaveformOrScalar) -> Self:
        return other.__pow__(self)

    def _check_logical_op_type(self, other):
        if self.value.dtype not in (np.int64, np.uint64, np.object_):
            raise TypeError("Can only perform logical operations on 64-bit integers")

        if isinstance(other, Waveform):
            if other.value.dtype not in (np.int64, np.uint64, np.object_):
                raise TypeError(
                    "Can only perform logical operations on 64-bit integers"
                )
        elif isinstance(other, float):
            raise TypeError("Can only perform logical operations on 64-bit integers")

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
                raise ValueError("width mismatch: {} and {}".format(self.width, other))
            return self.width

    @staticmethod
    @jit
    def _lshift(a, b):
        return a << b

    def __lshift__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)
        self._check_logical_op_type(other)
        new_width = self._infer_logical_op_width(
            other,
            inferred_width=(
                self.width + (other if isinstance(other, int) else (1 << other.width))
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

    def __rlshift__(self, other: WaveformOrScalar) -> Self:
        return self.__lshift__(other)

    @staticmethod
    @jit
    def _rshift(a, b):
        return a >> b

    def __rshift__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)
        self._check_logical_op_type(other)

        new_width = self._infer_logical_op_width(
            self.width if isinstance(other, Waveform) else max(self.width - other, 0)
        )

        new_value = self._rshift(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rrshift__(self, other: WaveformOrScalar, width: int = None) -> Self:
        self._check_sign(other)
        self._check_logical_op_type(other)

        new_width = self._infer_logical_op_width(
            self.width if isinstance(other, Waveform) else max(self.width - other, 0)
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
    @jit
    def _and(a, b):
        return a & b

    def __and__(self, other: WaveformOrScalar, width: int = None) -> Self:
        self._check_sign(other)
        new_width = self._infer_logical_op_width(other, width)
        new_value = self._and(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rand__(self, other: WaveformOrScalar, width: int = None) -> Self:
        return other.__and__(self, width)

    @staticmethod
    @jit
    def _or(a, b):
        return a | b

    def __or__(self, other: WaveformOrScalar, width: int = None) -> Self:
        self._check_sign(other)
        new_width = self._infer_logical_op_width(other, width)
        new_value = self._or(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __ror__(self, other: WaveformOrScalar, width: int = None) -> Self:
        return other.__or__(self, width)

    @staticmethod
    @jit
    def _xor(a, b):
        return a ^ b

    def __xor__(self, other: WaveformOrScalar, width: int = None) -> Self:
        self._check_sign(other)
        new_width = self._infer_logical_op_width(other, width)
        new_value = self._xor(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    def __rxor__(self, other: WaveformOrScalar, width: int = None) -> Self:
        return other.__xor__(self, width)

    @staticmethod
    @jit
    def _invert(a):
        return ~a

    def __invert__(self, width: int = None) -> Self:
        new_width = self._infer_logical_op_width(width, self.width)
        new_value = self._invert(self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    @staticmethod
    @jit
    def _neg(a):
        return a != 0

    def __neg__(self, width: int = None) -> Self:
        new_width = self._infer_logical_op_width(width, self.width)
        new_value = self._neg(self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )

    @staticmethod
    @jit
    def _eq(a, b):
        return a == b

    def __eq__(self, other: WaveformOrScalar) -> Self:
        self._check_sign(other)
        new_width = self._infer_logical_op_width(other)
        new_value = self._eq(self.value, self._get_value(other))

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            width=new_width,
            signed=self.signed,
        )


"""
    @staticmethod
    @jit
    def _ge(a,b):
        return a >= b

    def __ge__(self): ## gt le lt
        self._check_sign(other)

    def __getitem__():


    def resize(self, width: int)-> Self:
        pass

    def map()->Self:
        pass

    def reduce()->Self:
        pass

    def filter( condition: Union[Union[np.ndarray,list[bool]],Callable[[Union[any]],bool]])->Self:
        pass

    def apply()->Self:
        pass

    def sorted()->Self:
        pass

    @statisticmethod
    def concat(waves:list[Self])->Self:
        pass

    @staticmethod
    def merge(waves:list[Self], func:Callable[[Self,Self],Self], width:int)->Self:
        pass


    # 根据索引取值
    def take():
        pass
"""
