from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np

from ...scope import Scope
from ...signal import Signal
from ...waveform import Waveform
from ..base import Reader
from .npi_fsdb_reader import NpiFsdbReader, NpiFsdbScope, NpiFsdbSignalScope


class FsdbScope(Scope):
    def __init__(self, handle: NpiFsdbScope, parent_scope: FsdbScope | None, reader: Reader):
        super().__init__(name=handle.name())
        self.handle = handle
        self.parent_scope = parent_scope
        self.reader = reader

    @cached_property
    def signal_list(self) -> Sequence[Signal]:
        full_scope_name = self.full_name()
        signals = []
        for s in self.handle.signal_list():
            signal_path = f'{full_scope_name}.{s}'
            width = self.reader._get_signal_width(signal_path)
            rng = self.reader._get_signal_range(signal_path)
            signals.append(
                Signal(
                    name=s,
                    full_name=signal_path,
                    width=width,
                    range=rng,
                    signed=False,
                )
            )
        return signals

    @cached_property
    def child_scope_list(self) -> Sequence[Scope]:
        return [FsdbScope(c, self, self.reader) for c in self.handle.child_scope_list()]

    @cached_property
    def child_normal_scope_list(self) -> Sequence[Scope]:
        return [
            FsdbScope(c, self, self.reader)
            for c in self.handle.child_scope_list(include_signal_scope=False)
        ]

    @property
    def type(self) -> str:
        if not hasattr(self, '_type'):
            self._type = self.handle.type()
        return self._type

    @property
    def def_name(self) -> str | None:
        if not hasattr(self, '_def_name'):
            self._def_name = self.handle.def_name()
        return self._def_name

    def find_scope_by_module(self, module_name: str, depth: int = 0) -> list[Scope]:
        if not hasattr(self, '_preloaded_module_scope'):
            self.preload_module_scope()
        return self._preloaded_module_scope[module_name]

    def preload_module_scope(self):
        if isinstance(self.handle, NpiFsdbSignalScope):
            return {}

        preloaded_module_scope = defaultdict(list)
        for c in self.child_normal_scope_list:
            for module_name, module_scope_list in c.preload_module_scope().items():
                preloaded_module_scope[module_name].extend(module_scope_list)

        if self.type == 'npiFsdbScopeSvModule':
            preloaded_module_scope[self.def_name].append(self)
        self._preloaded_module_scope = preloaded_module_scope
        return preloaded_module_scope


class FsdbReader(Reader):
    pynpi: dict[str, Any] = {}

    def __init__(self, file: str):
        super().__init__()

        if len(FsdbReader.pynpi) == 0:
            import os
            import sys

            rel_lib_path = os.environ['VERDI_HOME'] + '/share/NPI/python'
            sys.path.append(os.path.abspath(rel_lib_path))
            FsdbReader.pynpi['npisys'] = importlib.import_module('pynpi.npisys')
            FsdbReader.pynpi['waveform'] = importlib.import_module('pynpi.waveform')
            FsdbReader.pynpi['npisys'].init([''])

        self.file = file
        self.file_handle = NpiFsdbReader(file)

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

        # Always load the full clock to compute absolute cycle numbers
        all_clock_changes = self.file_handle.load_value_change(
            clock_path,
            begin_time=0,
            end_time=2**64 - 1,
            xz_value=0,
        )

        # Determine clock edge timestamps for the sampling edge
        sample_value = 1 if sample_on_posedge else 0
        clock_edge_times = all_clock_changes[all_clock_changes[:, 1] == sample_value, 0]

        # Convert begin_cycle/end_cycle to begin_time/end_time
        if begin_cycle is not None:
            begin_time = int(clock_edge_times[begin_cycle])
        if end_cycle is not None:
            end_time = int(clock_edge_times[end_cycle])

        begin_time_actual = begin_time if begin_time is not None else 0
        end_time_actual = end_time if end_time is not None else 2**64 - 1

        # Compute clock_offset = number of sampling edges before begin_time_actual
        clock_offset = int(np.searchsorted(clock_edge_times, begin_time_actual, side='left'))

        # Trim clock to window [begin_time_actual, end_time_actual] to reduce memory in value_change
        clock_mask = all_clock_changes[:, 0] >= begin_time_actual
        if end_time is not None:
            clock_mask &= all_clock_changes[:, 0] <= end_time_actual
        windowed_clock_changes = all_clock_changes[clock_mask]

        # Load signal within the requested window only (FSDB NPI provides the
        # correct initial value at begin_time even if the last change was earlier)
        signal_value_change = self.file_handle.load_value_change(
            signal_path,
            begin_time=begin_time_actual,
            end_time=end_time_actual,
            xz_value=xz_value,
        )

        full_wave = self.value_change_to_waveform(
            signal_value_change,
            windowed_clock_changes,
            width=self.file_handle.get_signal_width(signal_path),
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=signal_path,
            clock_offset=clock_offset,
        )

        # time_slice trims the garbage samples produced by clock edges before
        # begin_time (where the windowed signal data hasn't started yet)
        return full_wave.time_slice(
            begin_time_actual if begin_time is not None else None,
            end_time if end_time is not None else None,
        )

    def top_scope_list(self) -> Sequence[Scope]:
        if not hasattr(self, '_top_scope_list'):
            self._top_scope_list = [
                FsdbScope(s, None, self) for s in self.file_handle.top_scope_list()
            ]
        return self._top_scope_list

    def _get_signal_width(self, signal: str) -> int:
        return self.file_handle.get_signal_width(signal)

    def _get_signal_range(self, signal: str) -> tuple[int, int]:
        return self.file_handle.get_signal_range(signal)

    @property
    def begin_time(self) -> str:
        return self.file_handle.min_time()

    @property
    def end_time(self) -> str:
        return self.file_handle.max_time()

    def close(self):
        self.file_handle.close()
