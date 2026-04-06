from __future__ import annotations

import dataclasses
from typing import Any, Callable, Literal, Union, cast

import numpy as np
import numpy.typing as npt

from .signal import Signal


class Waveform:
    """A clock-synchronised, numpy-backed time series of a hardware signal.

    Every ``Waveform`` is a triple of parallel numpy arrays of equal length:

    * ``value``  — signal values sampled on every clock edge.
    * ``clock``  — clock edge counter (0 = negedge before first posedge, then
      increments by 1 on each sampled edge).
    * ``time``   — simulation timestamp (in the file's native time unit) of
      each sample.

    These three arrays are always kept in sync; any operation that filters or
    transforms values also transforms the corresponding ``clock`` and ``time``
    entries so positional alignment is preserved.

    The convenience property :attr:`data` returns them as a single numpy record
    array with fields ``("time", "clock", "value")`` for easy pandas / numpy
    interop.

    Signal metadata (name, bit-width, signedness) is stored in a companion
    :class:`~wavekit.signal.Signal` object accessible through the ``signal``
    attribute, or directly via the :attr:`width`, :attr:`signed`, and
    :attr:`name` properties.

    Bit-width rules
    ---------------
    * Widths ≤ 64 bits are stored as ``np.int64`` (signed) or ``np.uint64``
      (unsigned).
    * Widths > 64 bits are stored as Python ``object`` arrays (arbitrary
      precision integers).
    * Arithmetic operators automatically infer a result width (e.g. addition
      widens by 1 bit; multiplication sums both widths).  Width inference is
      capped at 64 bits for integer types.
    * Two :class:`Waveform` operands **must have the same signedness**; mixing
      signed and unsigned raises ``ValueError``.

    Typical usage
    -------------
    Waveforms are normally created by a :class:`~wavekit.readers.base.Reader`,
    not constructed directly::

        with VcdReader("sim.vcd") as r:
            data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
            valid = r.load_waveform("tb.dut.valid", clock="tb.clk")

        valid_data = data.mask(valid == 1)
        print(valid_data.value)
    """

    WaveformOrScalar = Union['Waveform', int, float]

    def __init__(
        self,
        value: npt.NDArray[Any],
        clock: npt.NDArray[np.number],
        time: npt.NDArray[np.number],
        signal: Signal | None = None,
    ):
        self.clock: npt.NDArray[np.number] = clock
        self.time: npt.NDArray[np.number] = time
        self.signal: Signal = signal if signal is not None else Signal('', '', None, None)

        if self.width is None or self.signed is None:
            self.value = value
        elif self.width > 64:
            self.value = value.astype(np.object_)
        elif self.signed:
            self.value = value.astype(np.int64)
        else:
            self.value = value.astype(np.uint64)

    @property
    def width(self) -> int | None:
        return self.signal.width

    @width.setter
    def width(self, value: int | None):
        self.signal.width = value

    @property
    def signed(self) -> bool:
        return self.signal.signed

    @signed.setter
    def signed(self, value: bool):
        self.signal.signed = value

    @property
    def name(self) -> str:
        return self.signal.name

    @name.setter
    def name(self, value: str):
        self.signal.name = value

    def __str__(self):
        return f'Waveform({self.signal})'

    @property
    def data(self) -> Any:
        """Return value/clock/time as a single numpy record array.

        Fields: ``"time"`` (simulation timestamp), ``"clock"`` (edge counter),
        ``"value"`` (signal value).  Useful for pandas conversion or bulk numpy
        operations that need all three columns together.
        """
        return cast(
            Any,
            np.rec.fromarrays(
                cast(Any, [self.time, self.clock, self.value]),
                names=cast(Any, 'time,clock,value'),
            ),
        )

    def unique_consecutive(self) -> Waveform:
        """Remove consecutive duplicate values, keeping the first occurrence.

        Equivalent to run-length encoding compression.  Useful for reducing a
        dense sampled waveform to just the value-change events.

        Returns a new :class:`Waveform` where no two adjacent entries have the
        same value.  Behavior matches :func:`numpy.unique_consecutive`.
        """
        if len(self.value) <= 1:
            return self.copy()
        # Keep first element, then keep elements where value changed from previous
        mask = np.concatenate(([True], np.diff(self.value) != 0))
        return self.take(np.where(mask)[0])

    def compress(self) -> Waveform:
        """Remove consecutive duplicate values, keeping the last occurrence.

        Unlike :meth:`unique_consecutive`, this preserves the final timestamp
        and value, which is useful for waveforms where the end time matters.

        Returns a new :class:`Waveform` where no two adjacent entries have the
        same value, except the last sample is always preserved.
        """
        if len(self.value) <= 1:
            return self.copy()
        diff_mask = np.diff(self.value) != 0
        # Keep where next value changed (which means current is last of its group)
        # plus always keep last element
        mask = np.concatenate((diff_mask, [True]))
        return self.take(np.where(mask)[0])

    def vectorized_filter(
        self,
        func: Callable[[npt.NDArray[Any]], npt.NDArray[np.bool_]],
    ) -> Waveform:
        """Return a new Waveform keeping only the samples where *func* returns True.

        *func* receives the entire ``value`` array at once and must return a
        boolean array of the same length.  Prefer this over :meth:`filter` for
        performance-critical paths.

        Parameters
        ----------
        func:
            Vectorized callable: ``(NDArray) -> NDArray[bool]``.
        """
        mask = func(self.value)
        return self.mask(mask)

    def filter(self, condition: Callable[[Any], bool]) -> Waveform:
        """Return a new Waveform keeping only the samples that satisfy *condition*.

        *condition* is called once per sample value (scalar, not vectorized).
        For large waveforms prefer :meth:`vectorized_filter`.

        Parameters
        ----------
        condition:
            A callable that accepts a single value and returns ``bool``.

        Example
        -------
        ::

            non_zero = wave.filter(lambda v: v != 0)
        """
        mask = np.vectorize(condition)(self.value)
        return self.mask(mask)

    def mask(self, mask: npt.NDArray[np.bool_] | Waveform) -> Waveform:
        """Return a new Waveform keeping only the samples where *mask* is True.

        Parameters
        ----------
        mask:
            Either a boolean ``np.ndarray`` or a 1-bit :class:`Waveform`
            (``width == 1`` or ``dtype == bool``).  Must have the same length
            as ``self``.

        Raises
        ------
        TypeError:
            If *mask* is a Waveform with width != 1, or is not a boolean array.

        Example
        -------
        ::

            valid_data = data.mask(valid == 1)   # keep only cycles where valid is high
        """
        if isinstance(mask, Waveform):
            if not (mask.width == 1 or mask.value.dtype == np.bool_):
                raise TypeError('mask requires waveform with width 1 or boolean dtype')
            mask = mask.value.astype(np.bool_)
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError('mask requires boolean numpy array')
        return Waveform(
            value=self.value[mask],
            clock=self.clock[mask],
            time=self.time[mask],
            signal=dataclasses.replace(self.signal),
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
        """Reinterpret the unsigned bit pattern as a two's-complement signed integer.

        Raises ``ValueError`` if ``width`` is unknown (``None``).
        Returns a copy if the waveform is already signed.
        """
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
        """Reinterpret the signed value as an unsigned bit pattern.

        Raises ``ValueError`` if ``width`` is unknown (``None``).
        Returns a copy if the waveform is already unsigned.
        """
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
            signal=Signal('', '', None, None, self.signed),
        )

    def __rtruediv__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_value = self._truediv(self._get_value(other), self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            signal=Signal('', '', None, None, self.signed),
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
            signal=Signal('', '', new_width, None, self.signed),
        )

    def __rfloordiv__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(lambda: self._get_width(other))
        new_value = self._floordiv(self._get_value(other), self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            signal=Signal('', '', new_width, None, self.signed),
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
            signal=Signal('', '', new_width, None, self.signed),
        )

    def __rmod__(self, other: WaveformOrScalar) -> Waveform:
        self._check_sign(other)
        new_width = self._infer_arithmetic_op_width(lambda: self._get_width(other))

        new_value = self._mod(self._get_value(other), self.value)

        return Waveform(
            value=new_value,
            clock=self.clock,
            time=self.time,
            signal=Signal('', '', new_width, None, self.signed),
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
            signal=Signal('', '', new_width, None, self.signed),
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
            signal=Signal('', '', new_width, None, self.signed),
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
            signal=Signal('', '', new_width, None, self.signed),
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
            signal=Signal('', '', new_width, None, self.signed),
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
        """Extract a bit field using **little-endian** slice notation.

        Parameters
        ----------
        index:
            * ``int``  — select a single bit; returns a 1-bit Waveform.
            * ``slice(high, low)`` — select bits ``[high:low]`` inclusive,
              i.e. ``wave[7:0]`` extracts the lower 8 bits.

        Notes
        -----
        The slice convention matches Verilog/SystemVerilog: ``high`` must be
        ≥ ``low`` (``start >= stop``); step is not supported.

        The result is always **unsigned** regardless of the source signedness.

        Example
        -------
        ::

            byte0 = wide_bus[7:0]    # bits 7 down to 0 → width=8
            msb   = wide_bus[31]     # single bit       → width=1
        """
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
            signal=Signal('', '', width, None, False),
        )

    def take(self, indices: npt.NDArray[np.integer] | list[int] | Waveform) -> Waveform:
        """Return a new Waveform selecting samples at the given integer positions.

        Parameters
        ----------
        indices:
            Integer index array, list of ints, or a :class:`Waveform` whose
            ``value`` array contains integer indices (e.g. the result of
            ``np.where``).  Boolean arrays are **not** accepted; use
            :meth:`mask` instead.

        Raises
        ------
        TypeError:
            If *indices* contains booleans or non-integer values.

        Example
        -------
        ::

            # Keep every other sample
            even = wave.take(list(range(0, len(wave.value), 2)))
        """
        if isinstance(indices, Waveform):
            indices = indices.value
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
            signal=dataclasses.replace(self.signal),
        )

    def downsample(
        self,
        chunk_size: int,
        func: Callable[[npt.NDArray[Any]], float] = np.mean,
    ) -> Waveform:
        """Reduce the sample rate by aggregating consecutive chunks.

        Splits ``value``, ``clock``, and ``time`` into non-overlapping windows
        of *chunk_size* and applies *func* to each window.  The result length
        is ``ceil(len / chunk_size)``.

        Parameters
        ----------
        chunk_size:
            Number of consecutive samples to aggregate into one.
        func:
            Aggregation function applied to each value chunk.  Defaults to
            ``np.mean``.  ``clock`` and ``time`` chunks are always averaged.

        Example
        -------
        ::

            # Average occupancy in 100-cycle windows
            avg = occupancy.downsample(100, np.mean)
        """

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
            signal=dataclasses.replace(self.signal, width=None),
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
        """Apply a vectorized function to the value array and return a new Waveform.

        *func* receives the entire ``value`` ndarray and must return an ndarray
        of the same length.  ``clock`` and ``time`` are deep-copied unchanged.

        Parameters
        ----------
        func:
            Vectorized callable: ``(NDArray) -> NDArray``.
        width:
            Bit-width of the result.  Defaults to the source width.
        signed:
            Signedness of the result.  Defaults to the source signedness.

        See Also
        --------
        map : element-wise (non-vectorized) variant.
        """
        new_value = func(self.value)
        return Waveform(
            value=new_value,
            clock=np.copy(self.clock),
            time=np.copy(self.time),
            signal=Signal(
                '', '', width or self.width, None, signed if signed is not None else self.signed
            ),
        )

    def map(
        self,
        func: Callable[[Any], Any],
        width: int | None = None,
        signed: bool | None = None,
    ) -> Waveform:
        """Apply a scalar function element-wise and return a new Waveform.

        Internally wraps *func* with ``np.vectorize``.  For large waveforms
        prefer :meth:`vectorized_map` with a native numpy operation.

        Parameters
        ----------
        func:
            Callable applied to each value element individually.
        width:
            Bit-width of the result.  Defaults to the source width.
        signed:
            Signedness of the result.  Defaults to the source signedness.

        Example
        -------
        ::

            upper_nibble = wave.map(lambda v: (v >> 4) & 0xF, width=4, signed=False)
        """
        vectorized_func = np.vectorize(func)
        return self.vectorized_map(vectorized_func, width, signed)

    def falling_edge(self) -> Waveform:
        """Detect 1→0 transitions in a 1-bit waveform.

        Returns a new 1-bit Waveform where ``value[i] == True`` if and only if
        ``self.value[i-1] == 1`` and ``self.value[i] == 0``.  The first sample
        is always ``False``.

        Raises
        ------
        Exception:
            If ``self.width != 1``.

        See Also
        --------
        rising_edge : detect 0→1 transitions.
        """
        if self.width != 1:
            raise Exception('raising only support 1-bit waveform')
        one = self.value[:-1] == 1
        zero = self.value[1:] == 0
        new_value = np.concatenate(([False], one & zero))
        return Waveform(
            value=new_value,
            clock=np.copy(self.clock),
            time=np.copy(self.time),
            signal=Signal('', '', 1, None, False),
        )

    def rising_edge(self) -> Waveform:
        """Detect 0→1 transitions in a 1-bit waveform.

        Returns a new 1-bit Waveform where ``value[i] == True`` if and only if
        ``self.value[i-1] == 0`` and ``self.value[i] == 1``.  The first sample
        is always ``False``.

        Raises
        ------
        Exception:
            If ``self.width != 1``.

        See Also
        --------
        falling_edge : detect 1→0 transitions.
        """
        if self.width != 1:
            raise Exception('raising only support 1-bit waveform')
        zero = self.value[:-1] == 0
        one = self.value[1:] == 1
        new_value = np.concatenate(([False], one & zero))
        return Waveform(
            value=new_value,
            clock=np.copy(self.clock),
            time=np.copy(self.time),
            signal=Signal('', '', 1, None, False),
        )

    def bit_count(self) -> Waveform:
        """Count the number of set bits (population count) in each sample value.

        Returns a new unsigned Waveform with ``width=64`` where each value is
        the popcount of the corresponding source sample.  Supports arbitrarily
        wide signals (> 64 bits) by chunking.

        Raises
        ------
        ValueError:
            If ``self.width`` is ``None``.
        """
        if self.width is None:
            raise ValueError('width is None')
        width = self.width
        if self.value.dtype != np.object_ and self.width <= 64:
            return self.vectorized_map(lambda v: np.bitwise_count(v), width=64, signed=False)
        return self.vectorized_map(lambda v: Waveform._count_one(v, width), width=64, signed=False)

    def split_bits(self, bit_group_size: int | list[int], padding: bool = False) -> list[Waveform]:
        """Split the waveform into multiple narrower waveforms by bit groups.

        Parameters
        ----------
        bit_group_size:
            * ``int`` — split into equal-sized groups of this many bits,
              starting from bit 0 (LSB).  ``width`` must be a multiple of
              *bit_group_size* unless ``padding=True``.
            * ``list[int]`` — explicit widths for each group (LSB-first).
              The values must sum to ``self.width``; ``padding`` is ignored.
        padding:
            If ``True`` and an integer *bit_group_size* is given, the last
            group may be narrower than *bit_group_size*.

        Returns
        -------
        list[Waveform]:
            Waveforms ordered from LSB group to MSB group, each unsigned.

        Raises
        ------
        ValueError:
            If ``self.width`` is ``None``.
        Exception:
            If width is not divisible by *bit_group_size* when ``padding=False``,
            or if list sizes do not sum to ``self.width``.

        Example
        -------
        ::

            # Split a 32-bit bus into four 8-bit bytes (byte0 = bits[7:0])
            bytes_ = bus32.split_bits(8)
        """
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
        """Concatenate multiple waveforms into a single wider waveform.

        Bits are joined so that the *last* element in *waves* becomes the MSB
        group and the *first* becomes the LSB group — this is the inverse of
        :meth:`split_bits`.

        All waveforms in *waves* must be **unsigned** and have the same length.
        The result width is the sum of all input widths.

        Parameters
        ----------
        waves:
            List of waveforms to concatenate; at least one element required.

        Raises
        ------
        Exception:
            If any waveform is signed.
        ValueError:
            If any waveform has ``width=None``.

        Example
        -------
        ::

            # Recombine four 8-bit bytes into one 32-bit value (byte3 = MSB)
            bus32 = Waveform.concatenate([byte0, byte1, byte2, byte3])
        """
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
            signal=Signal('', '', concat_width, None, False),
        )

    @staticmethod
    def merge(
        waves: list[Waveform],
        func: Callable[[list[Any]], Any],
        width: int,
        signed: bool,
    ) -> Waveform:
        """Combine multiple same-length waveforms into one using a custom function.

        At each sample index *i*, ``func`` is called with
        ``[w.value[i] for w in waves]`` and the returned value becomes the
        result sample.  ``clock`` and ``time`` are taken from ``waves[0]``.

        All waveforms must have the **same number of samples**.

        Parameters
        ----------
        waves:
            Input waveforms; must be non-empty and all equal in length.
        func:
            Callable ``(list[scalar]) -> scalar`` applied per sample.
        width:
            Bit-width to assign to the result.
        signed:
            Signedness of the result.

        Example
        -------
        ::

            # Compute bitwise majority across three 1-bit signals
            majority = Waveform.merge(
                [a, b, c], lambda vs: int(sum(vs) >= 2), width=1, signed=False
            )
        """
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
            signal=Signal('', '', width, None, signed),
        )

    def time_slice(
        self,
        begin_time: int | None = None,
        end_time: int | None = None,
        include_end: bool = False,
    ) -> Waveform:
        """Return a new Waveform trimmed to the given simulation time range.

        Uses binary search on the sorted ``time`` array so the operation is
        O(log n) regardless of waveform length.

        Parameters
        ----------
        begin_time:
            Start of the time window (inclusive).  Defaults to the first
            sample's timestamp.
        end_time:
            End of the time window.  Exclusive by default; set
            ``include_end=True`` to make it inclusive.
        include_end:
            If ``True``, samples exactly at *end_time* are included.

        Example
        -------
        ::

            # Analyse only the first 1000 simulation time units
            early = wave.time_slice(0, 1000)
        """
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
            signal=dataclasses.replace(self.signal),
        )

    def cycle_slice(
        self,
        begin_cycle: int | None = None,
        end_cycle: int | None = None,
        include_end: bool = False,
    ) -> Waveform:
        """Return a new Waveform trimmed to the given absolute clock cycle range.

        Uses binary search on the sorted ``clock`` array so the operation is
        O(log n) regardless of waveform length.  The ``clock`` values are
        absolute cycle numbers from the start of simulation (not relative to
        this waveform's window), so the same cycle number means the same
        simulation instant across different waveforms.

        Parameters
        ----------
        begin_cycle:
            First clock cycle to include (inclusive).  Defaults to the first
            sample's cycle number.
        end_cycle:
            Last clock cycle.  Exclusive by default; set ``include_end=True``
            to make it inclusive.
        include_end:
            If ``True``, samples exactly at *end_cycle* are included.

        See Also
        --------
        time_slice : slice by simulation timestamp instead of cycle number.

        Example
        -------
        ::

            # Analyse cycles 100 to 199 (exclusive end)
            window = wave.cycle_slice(100, 200)
        """
        if begin_cycle is None:
            begin_cycle = int(self.clock[0])
        if end_cycle is None:
            end_cycle = int(self.clock[-1]) + 1
        start_idx = np.searchsorted(self.clock, begin_cycle, side='left')
        end_idx = np.searchsorted(self.clock, end_cycle, side='right' if include_end else 'left')
        return Waveform(
            value=self.value[start_idx:end_idx],
            clock=self.clock[start_idx:end_idx],
            time=self.time[start_idx:end_idx],
            signal=dataclasses.replace(self.signal),
        )

    def slice(self, begin_idx: int, end_idx: int, include_end: bool = False) -> Waveform:
        """Return a new Waveform trimmed to the given sample index range.

        Parameters
        ----------
        begin_idx:
            First sample index to include (inclusive).
        end_idx:
            Last sample index.  Exclusive by default; set ``include_end=True``
            to make it inclusive.
        include_end:
            If ``True``, the sample at *end_idx* is included.

        See Also
        --------
        time_slice : slice by simulation timestamp instead of array index.
        """
        if include_end:
            end_idx += 1
        return Waveform(
            value=self.value[begin_idx:end_idx],
            clock=self.clock[begin_idx:end_idx],
            time=self.time[begin_idx:end_idx],
            signal=dataclasses.replace(self.signal),
        )

    def relative(
        self,
        offset: int,
        pad: Literal['none', 'repeat', 'value'] = 'repeat',
        pad_value: Any = None,
    ) -> Waveform:
        """Return a new Waveform shifted by *offset* cycles.

        This is the core method for relative time access. Use :meth:`next` and
        :meth:`prev` for more readable positive/negative offsets.

        Parameters
        ----------
        offset:
            Number of cycles to shift. Positive looks forward (future),
            negative looks backward (past).
        pad:
            Boundary handling strategy:

            - ``'repeat'`` (default): pad with boundary value (first/last element).
            - ``'none'``: truncate, return shorter array.
            - ``'value'``: pad with *pad_value* (must be provided).

        pad_value:
            Value to use when ``pad='value'``. Ignored otherwise.

        Returns
        -------
        Waveform
            A new waveform shifted by *offset* cycles.

        Raises
        ------
        ValueError:
            If ``pad='value'`` but *pad_value* is not provided.
        ValueError:
            If *pad* is not one of ``'none'``, ``'repeat'``, ``'value'``.

        Example
        -------
        ::

            # Rising edge detection
            rising = (wave == 0) & wave.next()

            # Look back 3 cycles
            past = wave.relative(-3)

        See Also
        --------
        next : Shift forward (positive offset).
        prev : Shift backward (negative offset).
        """
        if pad not in ('none', 'repeat', 'value'):
            raise ValueError(f"pad must be 'none', 'repeat', or 'value', got {pad!r}")
        if pad == 'value' and pad_value is None:
            raise ValueError("pad_value is required when pad='value'")

        n = len(self.value)
        if n == 0:
            return self.copy()

        if offset == 0:
            return self.copy()

        if pad == 'none':
            # Truncate: shift and lose boundary elements
            if offset > 0:
                # Forward shift: drop first offset elements, keep clock/time from start
                return Waveform(
                    value=self.value[offset:],
                    clock=self.clock[:-offset],
                    time=self.time[:-offset],
                    signal=dataclasses.replace(self.signal),
                )
            else:
                # Backward shift: drop last |offset| elements
                return Waveform(
                    value=self.value[:offset],  # offset is negative
                    clock=self.clock[-offset:],  # -offset is positive
                    time=self.time[-offset:],
                    signal=dataclasses.replace(self.signal),
                )
        elif pad == 'repeat':
            # Pad with boundary value
            if offset > 0:
                # Forward shift: result[i] = original[i+offset], pad end with last value
                value_padded = np.concatenate([self.value[offset:], [self.value[-1]] * offset])
            else:
                # Backward shift: result[i] = original[i-offset], pad start with first value
                value_padded = np.concatenate([[self.value[0]] * (-offset), self.value[:offset]])
        else:  # pad == 'value'
            if offset > 0:
                value_padded = np.concatenate([self.value[offset:], [pad_value] * offset])
            else:
                value_padded = np.concatenate([[pad_value] * (-offset), self.value[:offset]])

        return Waveform(
            value=value_padded,
            clock=self.clock.copy(),
            time=self.time.copy(),
            signal=dataclasses.replace(self.signal),
        )

    def next(
        self,
        n: int = 1,
        pad: Literal['none', 'repeat', 'value'] = 'repeat',
        pad_value: Any = None,
    ) -> Waveform:
        """Return a new Waveform looking *n* cycles into the future.

        Convenience wrapper around :meth:`relative` for positive offsets.

        Parameters
        ----------
        n:
            Number of cycles to look ahead. Default is 1.
        pad:
            See :meth:`relative` for options.
        pad_value:
            See :meth:`relative` for usage.

        Returns
        -------
        Waveform
            A new waveform shifted forward by *n* cycles.

        Example
        -------
        ::

            # Rising edge detection
            rising = (wave == 0) & wave.next()
        """
        return self.relative(n, pad, pad_value)

    def prev(
        self,
        n: int = 1,
        pad: Literal['none', 'repeat', 'value'] = 'repeat',
        pad_value: Any = None,
    ) -> Waveform:
        """Return a new Waveform looking *n* cycles into the past.

        Convenience wrapper around :meth:`relative` for negative offsets.

        Parameters
        ----------
        n:
            Number of cycles to look back. Default is 1.
        pad:
            See :meth:`relative` for options.
        pad_value:
            See :meth:`relative` for usage.

        Returns
        -------
        Waveform
            A new waveform shifted backward by *n* cycles.

        Example
        -------
        ::

            # Check if current value equals previous
            same = wave == wave.prev()
        """
        return self.relative(-n, pad, pad_value)
