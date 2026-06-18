from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pylibfst

from ...scope import Scope
from ...signal import Signal
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

    def _load_value_changes(
        self,
        path: str,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load raw value changes for an FST signal.

        Parameters
        ----------
        path:
            Full signal path (may include a range suffix).
        value_mapping:
            Character-to-bit mapping.
        begin_time, end_time:
            Time bounds.  *end_time* limits FST iteration; *begin_time* is
            ignored (FST always loads from the beginning for correct initial
            values).
        """
        fst_signal, range_suffix = self._resolve_signal(path)
        width = fst_signal.width or 1

        high = width - 1
        low = 0
        if range_suffix:
            range_match = re.fullmatch(r'\[(\d+)(?::(\d+))?\]', range_suffix)
            if range_match is None:
                raise ValueError(f"unsupported range access for signal '{path}': {range_suffix}")
            high = int(range_match.group(1))
            low = int(range_match.group(2)) if range_match.group(2) is not None else high
            if high < low:
                raise ValueError(
                    f"reversed range {range_suffix} is not supported for signal '{path}'"
                )
            if high >= width:
                raise ValueError(
                    f"bit index {high} out of range for signal '{path}' with width {width}"
                )

        def decoder(raw: str, high: int, low: int) -> int:
            raw = raw[width - 1 - high : width - low]
            decoded = 0
            for c in raw.lower():
                decoded = (decoded << 1) + value_mapping.get(c, 0)
            return decoded

        changes: list[tuple[int, int]] = []

        def value_change_callback(_data, time, _facidx, value):
            if end_time is None or int(time) <= end_time:
                text = pylibfst.string(value)
                changes.append((int(time), decoder(text or '0', high, low)))

        def value_change_callback_varlen(_data, time, _facidx, _value, length):
            raise ValueError(
                f"unsupported variable-length FST value for signal '{fst_signal.full_name}' "
                f'at time {int(time)} with length {int(length)}'
            )

        pylibfst.lib.fstReaderClrFacProcessMaskAll(self.file_handle)
        pylibfst.lib.fstReaderSetFacProcessMask(self.file_handle, fst_signal.handle)
        if end_time is None:
            pylibfst.lib.fstReaderSetUnlimitedTimeRange(self.file_handle)
        else:
            pylibfst.lib.fstReaderSetLimitTimeRange(self.file_handle, 0, end_time)
        pylibfst.fstReaderIterBlocks2(
            self.file_handle,
            value_change_callback,
            value_change_callback_varlen,
        )

        value_width = high - low + 1
        dtype = np.object_ if value_width > 64 else np.uint64
        if not changes:
            raise ValueError(f"signal '{fst_signal.full_name}' has no value changes")
        result = np.array(changes, dtype=dtype)
        return result, value_width

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
