from __future__ import annotations
import re
import numpy as np
from vcdvcd import VCDVCD, Scope as VcdVcdScope, Signal as VcdVcdSignal
from typing import Optional
from .waveform import Waveform
from .reader import Reader, Scope

class VcdScope(Scope):
    @staticmethod
    def from_signal_list(signal_list: list, scope_list: list) -> list[VcdScope]:
        scopes = {}
        for scope in scope_list:
            ancestors = scope.split(".")
            full_name = ""
            parent_scope = None
            for scope_name in ancestors:
                full_name = full_name + scope_name
                if full_name not in scopes:
                    if full_name not in scopes:
                        new_scope = VcdScope(
                            scope_name, parent_scope=parent_scope)
                        if parent_scope is not None:
                            parent_scope._child_scopes[scope_name] = new_scope
                        scopes[full_name] = new_scope

                parent_scope = scopes[full_name]
                full_name = full_name + "."

        for signal in signal_list:
            scope_name = ".".join(signal.split(".")[:-1])
            scopes[scope_name]._signals.add(signal.split(".")[-1])

        return [scope for scope in scopes.values() if scope.parent_scope is None]

    def __init__(self, name: str, parent_scope: VcdScope):
        super().__init__(name=name)
        self.parent_scope = parent_scope
        self._child_scopes = {}
        self._signals = set()
        self.child_scope_list =  list(self._child_scopes.values())

    @property
    def signal_list(self) -> list[str]:
        return list(self._signals)

class VcdReader(Reader):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = VCDVCD(file, store_scopes=True)
        self._top_scope_list = VcdScope.from_signal_list(
            self.file_handle.signals, self.file_handle.scopes
        )

    def top_scope_list(self) -> list[Scope]:
        return self._top_scope_list

    @property
    def begin_time(self) -> str:
        return self.file_handle.begintime

    @property
    def end_time(self) -> str:
        return self.file_handle.endtime

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

        if begin_time is not None:
            raise NotImplementedError("begin_time is not supported")
        if end_time is not None:
            raise NotImplementedError("end_time is not supported")

        width = int(signal_handle.size)
        signal_handle = self.file_handle[signal]
        signal_value_change = np.array([
            (v[0], int(re.sub(r"[xXzZ]", str(xz_value), v[1]), 2))
            for v in signal_handle.tv
        ], dtype=np.object_ if width > 64 else np.uint64)
        clock_value_change = np.array([
            (v[0], int(re.sub(r"[xXzZ]", "0", v[1]), 2)) for v in self.file_handle[clock].tv
        ], dtype = np.uint64)

        return self.value_change_to_waveform(
            signal_value_change,
            clock_value_change,
            width=int(signal_handle.size),
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=signal
        )

    def close(self):
        pass
