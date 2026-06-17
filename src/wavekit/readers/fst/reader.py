from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pylibfst

from ...scope import Scope
from ...signal import Signal
from ...waveform import Waveform
from ..base import Reader
from ..pattern_parser import split_by_range_expr


@dataclass
class FstSignal(Signal):
    """FST-backed signal descriptor carrying the native FST handle."""

    handle: int = field(default=0, repr=False, compare=False)


class FstScope(Scope):
    """Scope node from an FST hierarchy."""

    def __init__(self, name: str, parent_scope: FstScope | None):
        super().__init__(name=name)
        self.parent_scope = parent_scope
        self.signal_list: list[FstSignal] = []
        self.child_scope_list: list[FstScope] = []


class FstReader(Reader):
    """Read FST waveform files via :mod:`pylibfst`.

    Supports the same high-level APIs as :class:`~wavekit.VcdReader`, including
    context-manager usage, hierarchy traversal, pattern matching, expression
    evaluation, and clock-synchronised ``load_waveform`` sampling.
    """

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = pylibfst.lib.fstReaderOpen(file.encode('UTF-8'))
        if self.file_handle == pylibfst.ffi.NULL:
            raise RuntimeError(f"Unable to open FST file '{file}'")
        self._signal_by_name: dict[str, FstSignal] = {}
        self._closed = False
        self._top_scope_list = self._build_scope_tree()

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r'\s+(?=\[)', '', name)

    @staticmethod
    def _range_from_name(name: str) -> tuple[int, int] | None:
        _, range_suffix = split_by_range_expr(name)
        match = re.search(r'\[(\d+)(?::(\d+))?\]$', range_suffix)
        if match is None:
            return None
        high = int(match.group(1))
        low = int(match.group(2)) if match.group(2) is not None else high
        return high, low

    def _build_scope_tree(self) -> list[FstScope]:
        top_scopes: list[FstScope] = []
        scope_by_name: dict[str, FstScope] = {}

        _, signals = pylibfst.get_scopes_signals2(self.file_handle)
        for raw_full_name, raw_signal in signals.by_name.items():
            full_name = self._normalize_name(raw_full_name)
            path_parts = full_name.split('.')
            if len(path_parts) < 2:
                raise ValueError(f"FST signal '{full_name}' is missing a scope path")

            parent: FstScope | None = None
            for index, scope_name in enumerate(path_parts[:-1]):
                scope_full_name = '.'.join(path_parts[: index + 1])
                if scope_full_name in scope_by_name:
                    parent = scope_by_name[scope_full_name]
                    continue

                scope = FstScope(scope_name, parent)
                scope_by_name[scope_full_name] = scope
                if parent is None:
                    top_scopes.append(scope)
                else:
                    parent.child_scope_list.append(scope)
                parent = scope

            signal_name = path_parts[-1]
            signal = FstSignal(
                name=signal_name,
                full_name=full_name,
                width=int(raw_signal.length),
                range=self._range_from_name(signal_name),
                signed=False,
                handle=int(raw_signal.handle),
            )
            assert parent is not None
            parent.signal_list.append(signal)
            self._signal_by_name[full_name] = signal

        return top_scopes

    def _resolve_signal(self, signal: Signal | str) -> tuple[FstSignal, str]:
        signal_path = signal.full_name if isinstance(signal, Signal) else signal
        bare_signal_path, range_suffix = split_by_range_expr(signal_path)
        lookup_path = bare_signal_path

        if lookup_path not in self._signal_by_name:
            pattern = re.compile(rf'^{re.escape(lookup_path)}\[\d+(?::\d+)?\]$')
            matches = [name for name in self._signal_by_name if pattern.fullmatch(name)]
            if len(matches) == 1:
                lookup_path = matches[0]
            elif len(matches) > 1:
                raise ValueError(f'pattern {lookup_path} matches more than one signal')

        if lookup_path not in self._signal_by_name:
            raise ValueError(f"signal '{signal_path}' not found")
        return self._signal_by_name[lookup_path], range_suffix

    def _load_value_change(
        self,
        signal: FstSignal,
        decoder,
        end_time: int | None = None,
    ) -> np.ndarray:
        changes: list[tuple[int, int]] = []

        def value_change_callback(_data, time, _facidx, value):
            if end_time is None or int(time) <= end_time:
                text = pylibfst.string(value)
                changes.append((int(time), decoder(text or '0')))

        def value_change_callback_varlen(_data, time, _facidx, _value, length):
            raise ValueError(
                f"unsupported variable-length FST value for signal '{signal.full_name}' "
                f'at time {int(time)} with length {int(length)}'
            )

        pylibfst.lib.fstReaderClrFacProcessMaskAll(self.file_handle)
        pylibfst.lib.fstReaderSetFacProcessMask(self.file_handle, signal.handle)
        if end_time is None:
            pylibfst.lib.fstReaderSetUnlimitedTimeRange(self.file_handle)
        else:
            pylibfst.lib.fstReaderSetLimitTimeRange(self.file_handle, 0, end_time)
        pylibfst.fstReaderIterBlocks2(
            self.file_handle,
            value_change_callback,
            value_change_callback_varlen,
        )

        dtype = np.object_ if (signal.width is not None and signal.width > 64) else np.uint64
        if not changes:
            raise ValueError(f"signal '{signal.full_name}' has no value changes")
        return np.array(changes, dtype=dtype)

    def _sample_on_clock(
        self,
        signal: Signal | str,
        clock: Signal | str,
        decoder,
        signed: bool,
        sample_on_posedge: bool,
        begin_time: int | None,
        end_time: int | None,
        begin_cycle: int | None,
        end_cycle: int | None,
    ) -> Waveform:
        if begin_time is not None and begin_cycle is not None:
            raise ValueError('begin_time and begin_cycle are mutually exclusive')
        if end_time is not None and end_cycle is not None:
            raise ValueError('end_time and end_cycle are mutually exclusive')

        fst_signal, requested_range = self._resolve_signal(signal)
        fst_clock, _ = self._resolve_signal(clock)

        all_clock_changes = self._load_value_change(
            fst_clock,
            decoder=lambda raw: int(re.sub(r'[xXzZ]', '0', raw), 2),
        )
        sample_value = 1 if sample_on_posedge else 0
        clock_edge_times = all_clock_changes[all_clock_changes[:, 1] == sample_value, 0]

        if begin_cycle is not None:
            begin_time = int(clock_edge_times[begin_cycle])
        if end_cycle is not None:
            end_time = int(clock_edge_times[end_cycle])

        begin_time_actual = begin_time if begin_time is not None else 0
        end_time_actual = end_time if end_time is not None else np.iinfo(np.uint64).max
        clock_offset = int(np.searchsorted(clock_edge_times, begin_time_actual, side='left'))

        clock_mask = all_clock_changes[:, 0] >= begin_time_actual
        if end_time is not None:
            clock_mask &= all_clock_changes[:, 0] <= end_time_actual
        windowed_clock_changes = all_clock_changes[clock_mask]

        # Load from the beginning for MVP correctness: pylibfst time-limit
        # iteration semantics do not guarantee an initial value at begin_time.
        signal_value_change = self._load_value_change(
            fst_signal,
            decoder=decoder,
            end_time=end_time,
        )

        full_wave = self.value_change_to_waveform(
            signal_value_change,
            windowed_clock_changes,
            width=fst_signal.width,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal='',
            clock_offset=clock_offset,
        )

        result = full_wave.time_slice(begin_time, end_time)

        if requested_range:
            match = re.fullmatch(r'\[(\d+)(?::(\d+))?\]', requested_range)
            if match is None:
                raise ValueError(
                    f"unsupported range access for signal '{fst_signal.full_name}': "
                    f'{requested_range}'
                )
            high = int(match.group(1))
            low = int(match.group(2)) if match.group(2) is not None else high
            if fst_signal.width is None:
                raise ValueError(f"width is unknown for signal '{fst_signal.full_name}'")
            if fst_signal.range is not None and fst_signal.range[1] != 0:
                raise ValueError(
                    f"sub-range access for signal '{fst_signal.full_name}' is only supported "
                    'when the stored signal range starts at bit 0'
                )
            if high >= fst_signal.width:
                raise ValueError(
                    f"bit index {high} out of range for signal '{fst_signal.full_name}' "
                    f'with width {fst_signal.width}'
                )
            if high - low + 1 < fst_signal.width:
                result = result[high:low]

        return result

    def top_scope_list(self) -> Sequence[Scope]:
        """Return top-level scopes from the FST hierarchy."""
        return self._top_scope_list

    @property
    def begin_time(self) -> int:
        return int(pylibfst.lib.fstReaderGetStartTime(self.file_handle))

    @property
    def end_time(self) -> int:
        return int(pylibfst.lib.fstReaderGetEndTime(self.file_handle))

    def close(self):
        """Close the underlying FST reader handle."""
        if not self._closed:
            pylibfst.lib.fstReaderClose(self.file_handle)
            self._closed = True
