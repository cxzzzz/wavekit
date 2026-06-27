from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property


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

    Stores the signal's bare local name, parent scope path, bit-width, and
    requested/view bit range.  ``full_name`` is derived from those fields so a
    native whole signal and a requested bit-sliced view share one consistent
    representation.

    Attributes
    ----------
    name:
        Bare local signal identifier inside ``parent_path``.  For VCD/FST this
        does not include the native value bit range; earlier array/index groups
        that are part of the identifier remain, e.g. ``"mem[3]"``.
    parent_path:
        Complete parent scope path.  Empty for anonymous/internal waveforms.
    width:
        Bit-width of the signal, e.g. ``8`` for ``[7:0]``.  ``None`` if not
        yet resolved.
    range:
        Requested/view bit range as ``(high, low)`` for non-composite signals.
        ``None`` means the whole native signal.  For composite arrays, backends
        may still use ``range`` for array bounds; callers must check
        ``composite_type`` before interpreting it as bit coordinates.
    composite_type:
        ``None`` for leaf (non-composite) signals.  For composite signals
        (struct, union, array, …) this holds the :class:`SignalCompositeType`
        value describing the kind of composite.  Not all backends populate
        this field; backends that do not support composite introspection leave
        it as ``None`` (e.g. VCD).
    member_list:
        ``None`` for leaf signals.  For composite signals this is the list of
        direct member :class:`Signal` objects, populated in the same order the
        backend reports them.  Always ``None`` when ``composite_type`` is
        ``None``, and always a list (possibly empty) when ``composite_type``
        is set.  Lazily evaluated on first access; backend-specific subclasses
        override the ``member_list`` cached property to provide the actual loading logic.
    """

    name: str
    parent_path: str
    width: int | None
    range: tuple[int, int] | None
    composite_type: SignalCompositeType | None = None

    @property
    def full_name(self) -> str:
        base = self.name if self.parent_path == '' else f'{self.parent_path}.{self.name}'
        if self.range is None or self.composite_type is not None:
            return base
        high, low = self.range
        if high == low:
            return f'{base}[{high}]'
        return f'{base}[{high}:{low}]'

    @cached_property
    def member_list(self) -> list[Signal] | None:
        return None

    def __str__(self) -> str:
        return f"Signal(full_name='{self.full_name}', width={self.width})"
