from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pylibfst

from ...scope import Scope, map_range_to_offsets
from ...signal import Signal
from ..base import Reader


@dataclass
class FstSignal(Signal):
    """FST-backed signal descriptor carrying the native FST handle."""

    handle: int = field(default=0, repr=False, compare=False)
    native_range: tuple[int, int] | None = field(default=None, compare=False)
    native_width: int | None = field(default=None, compare=False)


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

    def _build_scope_tree(self) -> list[FstScope]:
        native_range_re = re.compile(r'\[(\d+):(\d+)\]$')
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
            width = int(raw_signal.length)
            if m := native_range_re.search(signal_name):
                high, low = int(m.group(1)), int(m.group(2))
                if abs(high - low) + 1 != width:
                    raise ValueError(
                        f'native range [{high}:{low}] does not match width {width} '
                        f"for signal '{full_name}'"
                    )
                native_range = (high, low)
                bare_signal_name = signal_name[: m.start()]
            else:
                if width != 1:
                    raise ValueError(f"width {width} mismatch for scalar signal '{full_name}'")
                native_range = None
                bare_signal_name = signal_name

            signal = FstSignal(
                name=bare_signal_name,
                parent_path='.'.join(path_parts[:-1]),
                width=width,
                range=native_range,
                handle=int(raw_signal.handle),
                native_range=native_range,
                native_width=width,
            )
            assert parent is not None
            parent.signal_list.append(signal)
            self._signal_by_name[signal.full_name] = signal
            self._signal_by_name[full_name] = signal

        return top_scopes

    def _load_value_changes(
        self,
        signal: Signal,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load mapped FST value changes with optional trailing range selection."""
        fst_signal = self._resolve_signal(signal)
        assert isinstance(fst_signal, FstSignal)
        width = fst_signal.native_width or fst_signal.width or 1
        high, low = map_range_to_offsets(
            fst_signal.full_name,
            width,
            fst_signal.native_range,
            fst_signal.range,
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
