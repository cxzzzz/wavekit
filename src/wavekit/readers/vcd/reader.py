from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from vcdvcd import VCDVCD
from vcdvcd import Scope as VcdVcdScope

from ..base import Reader
from ..hierarchy import Node, Scope, Signal
from ..range import Range


@dataclass(frozen=True, eq=False)
class VcdSignal(Signal):
    """VCD-backed signal descriptor carrying the dumped reference name."""

    _ref: str = field(default='', repr=False, compare=False)

    @property
    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True, eq=False)
class VcdScope(Scope):
    vcdvcd_scope: VcdVcdScope = field(repr=False, compare=False)
    reader: VcdReader = field(repr=False, compare=False)

    @cached_property
    def children(self) -> tuple[Node, ...]:
        def signal_list() -> list[Signal]:
            range_re = re.compile(r'\[(\d+):(\d+)\]$')
            full_scope_name = self.full_name
            signals: list[Signal] = []
            for k, v in self.vcdvcd_scope.subElements.items():
                if isinstance(v, str):
                    signal_path = f'{full_scope_name}.{k}'
                    width = int(self.reader.file_handle[signal_path].size)
                    signal_range: Range | None
                    if m := range_re.search(k):
                        signal_range = Range(int(m.group(1)), int(m.group(2)))
                        assert abs(signal_range.start - signal_range.end) + 1 == width, (
                            f'range {signal_range} does not match width {width} '
                            f"for signal '{signal_path}'"
                        )
                        bare_name = k[: m.start()]
                    else:
                        signal_range = None if width == 1 else Range(width - 1, 0)
                        bare_name = k

                    signals.append(
                        VcdSignal(
                            base_name=bare_name,
                            parent=self,
                            range=signal_range,
                            native_range=signal_range,
                            _ref=signal_path,
                        )
                    )
            return signals

        def scope_list() -> list[Scope]:
            return [
                VcdScope(base_name=k, parent=self, vcdvcd_scope=v, reader=self.reader)
                for k, v in self.vcdvcd_scope.subElements.items()
                if isinstance(v, VcdVcdScope)
            ]

        children: list[Node] = []
        children.extend(scope_list())
        children.extend(signal_list())
        return tuple(children)


class VcdReader(Reader[VcdSignal]):
    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = VCDVCD(file, store_scopes=True)

    @cached_property
    def top_scopes(self) -> tuple[Scope, ...]:
        return tuple(
            VcdScope(base_name=k, parent=None, vcdvcd_scope=v, reader=self)
            for k, v in self.file_handle.scopes.items()
            if '.' not in k
        )

    @property
    def begin_time(self) -> int:
        return self.file_handle.begintime

    @property
    def end_time(self) -> int:
        return self.file_handle.endtime

    def _load_value_changes(
        self,
        signal: VcdSignal,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> np.ndarray:
        """Load mapped VCD value changes with an optional time window.

        ``begin_time`` retains the last value change at or before the window
        start so the caller can reconstruct the signal value at that time.
        ``end_time`` is exclusive. Range-to-raw mapping is calculated once
        before iterating over value changes.
        """

        assert signal.composite_type is None
        lookup_path = signal._ref
        signal_handle = self.file_handle[lookup_path]
        native_width = signal.native_width

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
            if len(raw) > native_width:
                raise ValueError(
                    f"VCD value {raw!r} exceeds width {native_width} for signal '{lookup_path}'"
                )
            padding = raw[0] if raw and raw[0] in {'x', 'z'} else '0'
            value = 0
            for char in raw.rjust(native_width, padding)[raw_start:raw_stop]:
                value = (value << 1) | value_mapping.get(char, 0)
            return value

        tv = signal_handle.tv
        times = [time for time, _ in tv]
        begin_index = 0 if begin_time is None else max(0, bisect_right(times, begin_time) - 1)
        end_index = len(tv) if end_time is None else bisect_left(times, end_time)
        windowed_tv = tv[begin_index:end_index]

        dtype = np.object_ if signal.width > 64 else np.uint64
        pairs = [(time, decode(raw)) for time, raw in windowed_tv]
        if pairs:
            return np.array(pairs, dtype=dtype)
        return np.empty((0, 2), dtype=dtype)

    def close(self):
        pass
