from __future__ import annotations
import re
import numpy as np
from functools import cached_property
from vcdvcd import VCDVCD, Scope as VcdVcdScope, Signal as VcdVcdSignal
from typing import Optional
from .waveform import Waveform
from .reader import Reader, Scope

class VcdScope(Scope):
    def __init__(self, vcdvcd_scope: VcdVcdScope, parent_scope: Scope):
        super().__init__(name = vcdvcd_scope.name.split(".")[-1])
        self.vcdvcd_scope = vcdvcd_scope
        self.parent_scope = parent_scope
        self._child_scopes = {}
        self._signals = set()

    @cached_property
    def signal_list(self) -> list[str]:
        return [k for k,v in self.vcdvcd_scope.subElements.items() if isinstance(v, str)]

    @cached_property
    def child_scope_list(self) -> list[Scope]:
        return [VcdScope(v, self) for k,v in self.vcdvcd_scope.subElements.items() if isinstance(v, VcdVcdScope)]

class VcdReader(Reader):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = VCDVCD(file, store_scopes=True)
        self._top_scope_list = [VcdScope(v, None) for k,v in self.file_handle.scopes.items() if '.' not in k]

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

        signal_handle = self.file_handle[signal]
        width = int(signal_handle.size)
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
