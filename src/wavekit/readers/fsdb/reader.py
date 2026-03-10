from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property
from typing import Any

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

    @property
    def begin_time(self) -> int:
        return self.parent_scope.begin_time if self.parent_scope else 0

    @property
    def end_time(self) -> int:
        return self.parent_scope.end_time if self.parent_scope else 0

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
    ) -> Waveform:
        signal_path = signal.full_name if isinstance(signal, Signal) else signal
        clock_path = clock.full_name if isinstance(clock, Signal) else clock

        begin_time = begin_time or 0
        end_time = end_time or 2**64 - 1

        signal_value_change = self.file_handle.load_value_change(
            signal_path,
            begin_time=begin_time,
            end_time=end_time,
            xz_value=xz_value,
        )

        clock_value_change = self.file_handle.load_value_change(
            clock_path,
            begin_time=begin_time,
            end_time=end_time,
            xz_value=0,
        )

        return self.value_change_to_waveform(
            signal_value_change,
            clock_value_change,
            width=self.file_handle.get_signal_width(signal_path),
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=signal_path,
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
