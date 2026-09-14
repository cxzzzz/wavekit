"""Wellen-backed VCD/FST waveform reader."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pywellen

from ..base import Reader
from ..hierarchy import Node, Scope, Signal, SignalCompositeType
from ..range import Range


@dataclass(frozen=True, eq=False)
class WellenSignal(Signal):
    """Wellen-backed signal: leaves hold a wellen ``Var``, composites a scope."""

    _wellen_var: Any = field(default=None, repr=False, compare=False)
    _wellen_scope: Any = field(default=None, repr=False, compare=False)

    @cached_property
    def children(self) -> tuple[Node, ...]:
        """Return composite member nodes, or an empty tuple for leaves."""
        if self._wellen_scope is None:
            return ()
        return tuple(_make_children(self._wellen_scope, self))

    @property
    def definition(self) -> None:
        """Return no definition name; wellen does not expose one for signals."""
        return None


@dataclass(frozen=True, eq=False)
class WellenScope(Scope):
    """Wellen-backed hierarchy scope."""

    _wellen_scope: Any = field(default=None, repr=False, compare=False)

    @cached_property
    def children(self) -> tuple[Node, ...]:
        """Return direct child scopes and signals from this wellen scope."""
        return tuple(_make_children(self._wellen_scope, self))


def _make_children(wellen_scope: Any, parent: Node) -> list[Node]:
    """Build the child nodes of a wellen scope or composite signal.

    Wellen names array members as bare indices (``[0]``), so each member's
    name gets the array's base name prepended: ``unpacked_arr`` + ``[0]``
    → ``unpacked_arr[0]``.
    """
    parent_reader = parent.reader
    name_prefix = (
        parent.base_name
        if isinstance(parent, Signal) and parent.composite_type == SignalCompositeType.ARRAY
        else ''
    )
    # Leaf vars: bit range from Var.index (width-1..0 when only length is known).
    nodes: list[Node] = []
    for wellen_var in wellen_scope.vars():
        base_name = f'{name_prefix}{wellen_var.name}'
        index = getattr(wellen_var, 'index', None)
        if index is not None:
            native_range = Range(index[0], index[1])
        elif (length := getattr(wellen_var, 'length', None)) is not None and length > 1:
            native_range = Range(length - 1, 0)
        else:
            native_range = None
        nodes.append(
            WellenSignal(
                base_name=base_name,
                parent=parent,
                range=native_range,
                native_range=native_range,
                composite_type=None,
                reader=parent_reader,
                _wellen_var=wellen_var,
            )
        )
    # Child scopes: composite scope types become Signal nodes, others stay
    # Scope nodes. Arrays get their element-index range from member names.
    for wellen_child in wellen_scope.scopes():
        base_name = f'{name_prefix}{wellen_child.name}'
        composite_types = {
            'sv_array': SignalCompositeType.ARRAY,
            'vhdl_array': SignalCompositeType.ARRAY,
            'struct': SignalCompositeType.STRUCT,
            'vhdl_record': SignalCompositeType.RECORD,
            'union': SignalCompositeType.UNION,
        }
        composite_type = composite_types.get(wellen_child.scope_type)
        if composite_type is None:
            nodes.append(
                WellenScope(
                    base_name=base_name,
                    parent=parent,
                    reader=parent_reader,
                    _wellen_scope=wellen_child,
                )
            )
            continue
        array_range = None
        if composite_type == SignalCompositeType.ARRAY:
            members = [*wellen_child.vars(), *wellen_child.scopes()]
            indices = sorted(
                int(m.group(1))
                for m in (re.match(r'\[(\d+)]', member.name) for member in members)
                if m
            )
            if indices:
                if not all(b - a == 1 for a, b in zip(indices, indices[1:])):
                    raise ValueError(
                        f"array members of '{wellen_child.name}' have "
                        f'non-consecutive indices: {indices}'
                    )
                array_range = Range(indices[-1], indices[0])
        nodes.append(
            WellenSignal(
                base_name=base_name,
                parent=parent,
                range=array_range,
                native_range=array_range,
                composite_type=composite_type,
                reader=parent_reader,
                _wellen_scope=wellen_child,
            )
        )
    return nodes


class WellenReader(Reader):
    """Read VCD/FST waveform files via the wellen Rust backend."""

    _EXPECTED_FORMAT: str | None = None

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = pywellen.Waveform(file)
        fmt = self.file_handle.file_format
        if self._EXPECTED_FORMAT is not None and fmt != self._EXPECTED_FORMAT:
            raise RuntimeError(
                f'{type(self).__name__} cannot open a {fmt} file; '
                f'expected {self._EXPECTED_FORMAT} ({file!r})'
            )

    @cached_property
    def top_scopes(self) -> tuple[Scope, ...]:
        """Return immutable top-level scopes in the wellen hierarchy."""
        return tuple(
            WellenScope(
                base_name=wellen_scope.name, parent=None, reader=self, _wellen_scope=wellen_scope
            )
            for wellen_scope in self.file_handle.scopes()
        )

    @cached_property
    def start_time(self) -> int:
        """Return the first timestamp stored in the file."""
        times = self.file_handle.time_table()
        return int(times[0]) if times else 0

    @cached_property
    def end_time(self) -> int:
        """Return the last timestamp stored in the file."""
        times = self.file_handle.time_table()
        return int(times[-1]) if times else 0

    def _load_value_changes(
        self,
        signal: Signal,
        value_mapping: dict[str, int],
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> np.ndarray:
        """Load mapped wellen value changes with an optional time window.

        ``start_time`` retains the last value change at or before the window
        start so the caller can reconstruct the signal value at that time.
        ``end_time`` is exclusive. Leaf bit-range selection is applied during
        decoding.
        """
        if not isinstance(signal, WellenSignal):
            raise TypeError('WellenReader requires a WellenSignal')

        if signal.composite_type is not None:
            raise NotImplementedError(
                f"Loading composite signal '{signal.full_name}' directly is not "
                'supported; load one of its members instead'
            )

        if not signal._wellen_var.is_bit_vector:
            raise ValueError(
                f"signal '{signal.full_name}' is a {signal._wellen_var.var_type} "
                'variable; only bit vectors can be loaded as waveforms'
            )

        wellen_signal = signal._wellen_var.signal
        changes = list(wellen_signal)

        native_range = signal.native_range or Range(0, 0)
        selected_range = signal.range or native_range

        def hdl_index_to_raw_offset(index: int) -> int:
            if native_range.end >= native_range.start:
                return index - native_range.start
            return native_range.start - index

        start_pos = hdl_index_to_raw_offset(selected_range.start)
        end_pos = hdl_index_to_raw_offset(selected_range.end)
        raw_start, raw_stop = start_pos, end_pos + 1

        native_width = signal.native_width
        dtype = np.object_ if signal.width > 64 else np.uint64

        def decode(value: Any) -> int:
            # Normalize to an MSB-first text form: X/Z values arrive as bit
            # strings; numeric values (any width) are formatted to the
            # native width so both paths share identical slice semantics.
            if isinstance(value, str):
                text = value.lower()
            else:
                text = f'{int(value):0{native_width}b}'
            out = 0
            for char in text[raw_start:raw_stop]:
                out = (out << 1) | value_mapping.get(char, 0)
            return out

        times = [time for time, _ in changes]
        start_index = 0 if start_time is None else max(0, bisect_right(times, start_time) - 1)
        end_index = len(changes) if end_time is None else bisect_left(times, end_time)
        windowed = changes[start_index:end_index]

        pairs = [(int(time), decode(value)) for time, value in windowed]
        if pairs:
            return np.array(pairs, dtype=dtype)
        return np.empty((0, 2), dtype=dtype)

    def close(self):
        """Close this wellen reader.

        The wellen backend releases file handles when the ``Waveform`` object
        is garbage collected and does not expose an explicit close operation,
        so this method is a no-op.
        """
        pass
