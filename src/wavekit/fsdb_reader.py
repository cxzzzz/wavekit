from __future__ import annotations
import importlib
from collections import defaultdict
from typing import Optional
from .waveform import Waveform
from .reader import Reader, Scope
from .npi_fsdb_reader import NpiFsdbReader, NpiFsdbScope

class FsdbScope(Scope):

    def __init__(self, handle: NpiFsdbScope, parent_scope: FsdbScope):
        super().__init__(name=handle.name())
        self.handle = handle
        self.parent_scope = parent_scope
        self.child_scope_list = [FsdbScope(c, self) for c in self.handle.child_scope_list()]

    @property
    def signal_list(self) -> list[str]:
        return [s for s in self.handle.signal_list()]

    @property
    def type(self) -> str:
        if not hasattr(self, '_type'):
            self._type = self.handle.type()
        return self._type

    @property
    def def_name(self) -> Optional[str]:
        if not hasattr(self, '_def_name'):
            self._def_name = self.handle.def_name()
        return self._def_name

    def find_module_scope(self, module_name: str, depth: int = 0) -> list[Scope]:
        #if self.type == 'npiFsdbScopeSvModule' and self.def_name == module_name:
        #    return [self]
        #elif depth == 1:
        #    return []
        #else:  # depth == 0 or depth > 1
        #    return list(reduce(lambda a, b: a + b, [c.find_module_scope(module_name, depth - 1) for c in self.child_scope_list], []))
        if not hasattr(self, '_preloaded_module_scope'):
            self.preload_module_scope()
        return self._preloaded_module_scope[module_name]

    def preload_module_scope(self):
        preloaded_module_scope = defaultdict(list)
        for c in self.child_scope_list:
            for module_name, module_scope_list in c.preload_module_scope().items():
                preloaded_module_scope[module_name].extend(module_scope_list)

        if self.type == 'npiFsdbScopeSvModule':
            #assert self.def_name not in preloaded_module_scope , (self.def_name, preloaded_module_scope)
            preloaded_module_scope[self.def_name].append(self)
        self._preloaded_module_scope = preloaded_module_scope
        return preloaded_module_scope

class FsdbReader(Reader):

    pynpi = {}

    def __init__(self, file: str):
        super().__init__()

        if len(FsdbReader.pynpi) == 0:
            import sys
            import os
            rel_lib_path = os.environ["VERDI_HOME"] + "/share/NPI/python"
            sys.path.append(os.path.abspath(rel_lib_path))
            FsdbReader.pynpi['npisys'] = importlib.import_module(
                "pynpi.npisys")
            FsdbReader.pynpi['waveform'] = importlib.import_module(
                "pynpi.waveform")
            FsdbReader.pynpi['npisys'].init([''])

        self.file = file
        self.file_handle = NpiFsdbReader(file)

    def load_wave(
        self,
        signal: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Waveform:

        format = FsdbReader.pynpi['waveform'].VctFormat_e.BinStrVal
        begin_time = begin_time or 0
        end_time = end_time or 2**64-1

        signal_value_change = self.file_handle.load_value_change(signal,
            begin_time = begin_time,
            end_time = end_time,
            xz_value = xz_value
        )

        clock_value_change = self.file_handle.load_value_change(clock,
            begin_time = begin_time,
            end_time = end_time,
            xz_value = 0
        )

        return self.value_change_to_waveform(
            signal_value_change,
            clock_value_change,
            width=self.file_handle.get_signal_width(signal),
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=signal
        )

    def top_scope_list(self) -> list[Scope]:
        if not hasattr(self, "_top_scope_list"):
            self._top_scope_list = [FsdbScope(s, None) for s in self.file_handle.top_scope_list()]
        return self._top_scope_list

    @property
    def begin_time(self) -> str:
        return self.file_handle.min_time()

    @property
    def end_time(self) -> str:
        return self.file_handle.max_time()

    def close(self):
        self.file_handle.close()