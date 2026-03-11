from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalCompositeType(Enum):
    """Composite (non-leaf) signal type as reported by the waveform backend.

    Not all backends support composite signals.  When a backend does not
    distinguish composite types the field is ``None``.
    """

    ARRAY = 'array'
    STRUCT = 'struct'
    UNION = 'union'
    TAGGED_UNION = 'tagged_union'
    RECORD = 'record'


@dataclass
class Signal:
    """Metadata descriptor for a single hardware signal.

    Stores the signal's local name, full hierarchical path, bit-width,
    declared bit-range, and signedness.  For composite signals (structs,
    unions, arrays) the ``composite_type`` and ``children`` fields carry
    the internal structure.

    Attributes
    ----------
    name:
        Local signal identifier as it appears within its parent scope,
        matching the form used in the waveform file.  May include a range
        suffix when the file stores the signal with one, e.g. ``"data[7:0]"``
        or ``"mem[3][7:0]"``.  Scalar signals have no suffix, e.g. ``"clk"``.
        Invariant: ``full_name == parent_scope_path + "." + name``.
    full_name:
        Complete hierarchical signal path, e.g. ``"tb.dut.data[7:0]"`` or
        ``"tb.dut.mem[3][7:0]"``.  Equal to ``parent_scope_path + "." + name``.
        Pass this directly to :meth:`~wavekit.readers.base.Reader.load_waveform`.
    width:
        Bit-width of the signal, e.g. ``8`` for ``[7:0]``.  ``None`` if not
        yet resolved.
    range:
        The innermost (last) bit-range of the signal as a ``(high, low)``
        integer tuple.  For a plain vector ``data[7:0]`` this is ``(7, 0)``;
        for a multi-dimensional signal ``mem[3][7:0]`` this is ``(7, 0)``
        (the ``[3]`` dimension index is encoded in ``name``/``full_name`` only).
        For a single-bit index ``[n]`` this is ``(n, n)``.
        ``None`` if the signal is scalar, composite, or the format does not
        expose range information.
    signed:
        Whether the signal value should be interpreted as a two's-complement
        signed integer.  Defaults to ``False``.
    composite_type:
        ``None`` for leaf (non-composite) signals.  For composite signals
        (struct, union, array, …) this holds the :class:`SignalCompositeType`
        value describing the kind of composite.  Not all backends populate
        this field; backends that do not support composite introspection leave
        it as ``None`` (e.g. VCD).
    children:
        ``None`` for leaf signals.  For composite signals this is the list of
        direct member :class:`Signal` objects, populated in the same order the
        backend reports them.  Always ``None`` when ``composite_type`` is
        ``None``, and always a list (possibly empty) when ``composite_type``
        is set.
    """

    name: str
    full_name: str
    width: int | None
    range: tuple[int, int] | None
    signed: bool = False
    composite_type: SignalCompositeType | None = None
    children: list[Signal] | None = None

    def __str__(self) -> str:
        return (
            f"Signal(name='{self.name}', full_name='{self.full_name}', "
            f'width={self.width}, signed={self.signed}, range={self.range})'
        )
