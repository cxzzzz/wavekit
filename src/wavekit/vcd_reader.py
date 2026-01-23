from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cached_property

import numpy as np
from vcdvcd import VCDVCD
from vcdvcd import Scope as VcdVcdScope

from .reader import Reader, Scope
from .waveform import Waveform


class VcdScope(Scope):
    def __init__(self, vcdvcd_scope: VcdVcdScope, parent_scope: Scope | None):
        super().__init__(name=vcdvcd_scope.name.split('.')[-1])
        self.vcdvcd_scope = vcdvcd_scope
        self.parent_scope = parent_scope

    @cached_property
    def signal_list(self) -> Sequence[str]:
        return [k for k, v in self.vcdvcd_scope.subElements.items() if isinstance(v, str)]

    @cached_property
    def child_scope_list(self) -> Sequence[Scope]:
        return [
            VcdScope(v, self)
            for _, v in self.vcdvcd_scope.subElements.items()
            if isinstance(v, VcdVcdScope)
        ]

    @property
    def begin_time(self) -> int:
        return self.parent_scope.begin_time if self.parent_scope else 0

    @property
    def end_time(self) -> int:
        return self.parent_scope.end_time if self.parent_scope else 0


class VcdReader(Reader):
    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = VCDVCD(file, store_scopes=True)
        self._top_scope_list = [
            VcdScope(v, None) for k, v in self.file_handle.scopes.items() if '.' not in k
        ]

    def top_scope_list(self) -> Sequence[Scope]:
        return self._top_scope_list

    @property
    def begin_time(self) -> int:
        return self.file_handle.begintime

    @property
    def end_time(self) -> int:
        return self.file_handle.endtime

    def get_width(self, signal: str) -> int:
        return int(self.file_handle[signal].size)

    def load_wave(
        self,
        signal: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> Waveform:
        signal_handle = self.file_handle[signal]
        width = int(signal_handle.size)

        #  TODO: opt performance
        signal_value_change = np.array(
            [(v[0], int(re.sub(r'[xXzZ]', str(xz_value), v[1]), 2)) for v in signal_handle.tv],
            dtype=np.object_ if width > 64 else np.uint64,
        )
        clock_value_change = np.array(
            [(v[0], int(re.sub(r'[xXzZ]', '0', v[1]), 2)) for v in self.file_handle[clock].tv],
            dtype=np.uint64,
        )

        full_wave = self.value_change_to_waveform(
            signal_value_change,
            clock_value_change,
            width=width,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=signal,
        )

        return full_wave.time_slice(begin_time, end_time)

    def close(self):
        pass
