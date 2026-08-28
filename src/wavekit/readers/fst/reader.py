from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pylibfst

from ..base import Reader
from ..hierarchy import Node, Scope, Signal
from ..range import Range


@dataclass(frozen=True, eq=False)
class FstSignal(Signal):
    """FST-backed signal descriptor carrying the FST facility handle."""

    _handle: int = field(default=0, repr=False, compare=False)

    @property
    def children(self) -> tuple[Node, ...]:
        """Return no children; FST signals are leaf hierarchy nodes."""
        return ()


@dataclass(frozen=True, eq=False)
class FstScope(Scope):
    """Scope node from an FST hierarchy."""

    _children: tuple[Node, ...] = field(default_factory=tuple, repr=False)

    @property
    def children(self) -> tuple[Node, ...]:
        """Return direct child scopes and signals from this FST scope."""
        return self._children


class FstReader(Reader[FstSignal]):
    """Read FST waveform files via ``pylibfst``.

    Supports the same high-level APIs as ``VcdReader``, including
    context-manager usage, hierarchy traversal, pattern matching, expression
    evaluation, and clock-synchronised ``load_waveform`` sampling.
    """

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = pylibfst.lib.fstReaderOpen(file.encode('UTF-8'))
        if self.file_handle == pylibfst.ffi.NULL:
            raise RuntimeError(f"Unable to open FST file '{file}'")
        self._closed = False

        def build_scope_tree() -> tuple[FstScope, ...]:
            def normalize_name(name: str) -> str:
                return re.sub(r'\s+(?=\[)', '', name)

            range_re = re.compile(r'\[(\d+):(\d+)\]$')
            top_scopes: list[FstScope] = []
            scope_by_name: dict[str, FstScope] = {}
            children_by_scope: dict[FstScope, list[Node]] = {}
            _, signals = pylibfst.get_scopes_signals2(self.file_handle)
            for raw_full_name, raw_signal in signals.by_name.items():
                full_name = normalize_name(raw_full_name)
                path_parts = full_name.split('.')
                if len(path_parts) < 2:
                    raise ValueError(f"FST signal '{full_name}' is missing a scope path")

                parent: FstScope | None = None
                for index, scope_name in enumerate(path_parts[:-1]):
                    scope_full_name = '.'.join(path_parts[: index + 1])
                    scope = scope_by_name.get(scope_full_name)
                    if scope is None:
                        scope = FstScope(base_name=scope_name, parent=parent)
                        scope_by_name[scope_full_name] = scope
                        children_by_scope[scope] = []
                        if parent is None:
                            top_scopes.append(scope)
                        else:
                            children_by_scope[parent].append(scope)
                    parent = scope

                signal_name = path_parts[-1]
                width = int(raw_signal.length)
                signal_range: Range | None
                if match := range_re.search(signal_name):
                    signal_range = Range(int(match.group(1)), int(match.group(2)))
                    if abs(signal_range.start - signal_range.end) + 1 != width:
                        raise ValueError(
                            f'range {signal_range} does not match width {width} '
                            f"for signal '{full_name}'"
                        )
                    base_name = signal_name[: match.start()]
                else:
                    signal_range = None if width == 1 else Range(width - 1, 0)
                    base_name = signal_name

                assert parent is not None
                children_by_scope[parent].append(
                    FstSignal(
                        base_name=base_name,
                        parent=parent,
                        range=signal_range,
                        native_range=signal_range,
                        _handle=int(raw_signal.handle),
                    )
                )

            for scope, children in children_by_scope.items():
                object.__setattr__(scope, '_children', tuple(children))
            return tuple(top_scopes)

        self._top_scopes = build_scope_tree()

    def _load_value_changes(
        self,
        signal: FstSignal,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> np.ndarray:
        """Load mapped FST value changes with an optional time window.

        ``begin_time`` retains the last value change at or before the window
        start so the caller can reconstruct the signal value at that time.
        ``end_time`` is exclusive. Range-to-raw mapping is calculated once
        before iterating over value changes.
        """

        assert signal.composite_type is None

        native_range = signal.native_range or Range(0, 0)
        selected_range = signal.range or native_range

        def hdl_index_to_raw_offset(index: int) -> int:
            if native_range.end >= native_range.start:
                return index - native_range.start
            return native_range.start - index

        start_pos = hdl_index_to_raw_offset(selected_range.start)
        end_pos = hdl_index_to_raw_offset(selected_range.end)
        raw_start, raw_stop = start_pos, end_pos + 1

        def decode(raw: str) -> int:
            raw = raw.lower()
            assert (
                len(raw) == signal.native_width
            ), f'FST value {raw!r} does not match width {signal.native_width}'
            value = 0
            for char in raw[raw_start:raw_stop]:
                value = (value << 1) | value_mapping.get(char, 0)
            return value

        changes: list[tuple[int, str]] = []

        def value_change_callback(_data, time, _facidx, value):
            changes.append((int(time), pylibfst.string(value)))

        def value_change_callback_varlen(_data, time, _facidx, _value, length):
            raise ValueError(
                f"unsupported variable-length FST value for signal '{signal.full_name}' "
                f'at time {int(time)} with length {int(length)}'
            )

        pylibfst.lib.fstReaderClrFacProcessMaskAll(self.file_handle)
        pylibfst.lib.fstReaderSetFacProcessMask(self.file_handle, signal._handle)
        pylibfst.lib.fstReaderSetLimitTimeRange(self.file_handle, 0, self.end_time + 1)
        pylibfst.fstReaderIterBlocks2(
            self.file_handle,
            value_change_callback,
            value_change_callback_varlen,
        )

        times = [time for time, _ in changes]
        begin_index = 0 if begin_time is None else max(0, bisect_right(times, begin_time) - 1)
        end_index = len(changes) if end_time is None else bisect_left(times, end_time)
        windowed_changes = changes[begin_index:end_index]

        dtype = np.object_ if signal.width > 64 else np.uint64
        pairs = [(time, decode(raw)) for time, raw in windowed_changes]
        if pairs:
            return np.array(pairs, dtype=dtype)
        return np.empty((0, 2), dtype=dtype)

    @property
    def top_scopes(self) -> tuple[FstScope, ...]:
        """Return immutable top-level scopes in the FST hierarchy."""
        return self._top_scopes

    @cached_property
    def begin_time(self) -> int:
        """Return the first timestamp stored in the FST file."""
        return int(pylibfst.lib.fstReaderGetStartTime(self.file_handle))

    @cached_property
    def end_time(self) -> int:
        """Return the last timestamp stored in the FST file."""
        return int(pylibfst.lib.fstReaderGetEndTime(self.file_handle))

    def close(self):
        """Close the underlying FST reader handle."""
        if not self._closed:
            pylibfst.lib.fstReaderClose(self.file_handle)
            self._closed = True
