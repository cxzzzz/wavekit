from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cached_property

import numpy as np
from vcdvcd import VCDVCD
from vcdvcd import Scope as VcdVcdScope

from ...scope import Scope
from ...signal import Signal
from ...waveform import Waveform
from ..base import Reader
from ..pattern_parser import split_by_range_expr


class VcdScope(Scope):
    def __init__(
        self,
        vcdvcd_scope: VcdVcdScope,
        parent_scope: Scope | None,
        reader: Reader,
    ):
        super().__init__(name=vcdvcd_scope.name.split('.')[-1])
        self.vcdvcd_scope = vcdvcd_scope
        self.parent_scope = parent_scope
        self.reader = reader

    @cached_property
    def signal_list(self) -> Sequence[Signal]:
        full_scope_name = self.full_name()
        signals = []
        for k, v in self.vcdvcd_scope.subElements.items():
            if isinstance(v, str):
                signal_path = f'{full_scope_name}.{k}'
                width = int(self.reader.file_handle[signal_path].size)
                signals.append(
                    Signal(
                        name=k,
                        full_name=signal_path,
                        width=width,
                        range=None,
                        signed=False,
                    )
                )
        return signals

    @cached_property
    def child_scope_list(self) -> Sequence[Scope]:
        return [
            VcdScope(v, self, self.reader)
            for _, v in self.vcdvcd_scope.subElements.items()
            if isinstance(v, VcdVcdScope)
        ]



class VcdReader(Reader):
    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = VCDVCD(file, store_scopes=True)
        self._top_scope_list = [
            VcdScope(v, None, self) for k, v in self.file_handle.scopes.items() if '.' not in k
        ]

    def top_scope_list(self) -> Sequence[Scope]:
        return self._top_scope_list

    @property
    def begin_time(self) -> int:
        return self.file_handle.begintime

    @property
    def end_time(self) -> int:
        return self.file_handle.endtime

    def _get_signal_width(self, signal: str) -> int:
        return int(self.file_handle[signal].size)

    def load_waveform(
        self,
        signal: Signal | str,
        clock: Signal | str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        begin_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> Waveform:
        if begin_time is not None and begin_cycle is not None:
            raise ValueError("begin_time and begin_cycle are mutually exclusive")
        if end_time is not None and end_cycle is not None:
            raise ValueError("end_time and end_cycle are mutually exclusive")

        signal_path = signal.full_name if isinstance(signal, Signal) else signal
        clock_path = clock.full_name if isinstance(clock, Signal) else clock

        # Strip range suffix to get the bare signal path for lookup
        bare_signal_path, range_suffix = split_by_range_expr(signal_path)

        # VCD does not support multi-dimensional (more than one bracket pair)
        if range_suffix and len(re.findall(r'\[[\d:]+\]', range_suffix)) > 1:
            raise ValueError(
                f"VCD does not support multi-dimensional range access: '{signal_path}'. "
                "Use FSDB or load the full signal and slice manually."
            )

        # Resolve the actual VCD signal name (may include a range suffix in the file)
        lookup_path = bare_signal_path
        if lookup_path not in self.file_handle.references_to_ids:
            pattern = re.compile(rf'^{re.escape(lookup_path)}\[\d+(?::\d+)?\]$')
            matches = [
                ref for ref in self.file_handle.references_to_ids.keys() if pattern.fullmatch(ref)
            ]
            if len(matches) == 1:
                lookup_path = matches[0]
            elif len(matches) > 1:
                raise ValueError(f'pattern {lookup_path} matches more than one signal')

        signal_handle = self.file_handle[lookup_path]
        width = int(signal_handle.size)

        # Always load the full clock to compute absolute cycle numbers and clock_offset
        all_clock_changes = np.array(
            [(v[0], int(re.sub(r'[xXzZ]', '0', v[1]), 2)) for v in self.file_handle[clock_path].tv],
            dtype=np.uint64,
        )

        # Determine clock edge timestamps for the sampling edge
        sample_value = 1 if sample_on_posedge else 0
        clock_edge_times = all_clock_changes[all_clock_changes[:, 1] == sample_value, 0]

        # Convert begin_cycle/end_cycle to begin_time/end_time
        if begin_cycle is not None:
            begin_time = int(clock_edge_times[begin_cycle])
        if end_cycle is not None:
            end_time = int(clock_edge_times[end_cycle])

        # Compute clock_offset = number of sampling edges before begin_time
        begin_time_actual = begin_time if begin_time is not None else 0
        clock_offset = int(np.searchsorted(clock_edge_times, begin_time_actual, side='left'))

        # Trim clock to window [begin_time_actual, end_time] to reduce memory usage in value_change
        end_time_actual = end_time if end_time is not None else np.iinfo(np.uint64).max
        clock_mask = all_clock_changes[:, 0] >= begin_time_actual
        if end_time is not None:
            clock_mask &= all_clock_changes[:, 0] <= end_time_actual
        clock_value_change = all_clock_changes[clock_mask]

        signal_value_change = np.array(
            [(v[0], int(re.sub(r'[xXzZ]', str(xz_value), v[1]), 2)) for v in signal_handle.tv],
            dtype=np.object_ if width > 64 else np.uint64,
        )
        full_wave = self.value_change_to_waveform(
            signal_value_change,
            clock_value_change,
            width=width,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=lookup_path,
            clock_offset=clock_offset,
        )

        result = full_wave.time_slice(begin_time, end_time)

        # Apply sub-range slice if user specified a range
        if range_suffix:
            m = re.fullmatch(r'\[(\d+)(?::(\d+))?\]', range_suffix)
            if m:
                high = int(m.group(1))
                low = int(m.group(2)) if m.group(2) is not None else high
                if high >= width:
                    raise ValueError(
                        f"bit index {high} out of range for signal '{lookup_path}' "
                        f"with width {width}"
                    )
                if low < 0:
                    raise ValueError(f"bit index {low} cannot be negative")
                slice_width = high - low + 1
                if slice_width < width:
                    result = result[high:low]

        return result

    def close(self):
        pass
