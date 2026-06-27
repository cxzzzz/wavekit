from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from vcdvcd import VCDVCD
from vcdvcd import Scope as VcdVcdScope

from ...scope import Scope, map_range_to_offsets
from ...signal import Signal
from ..base import Reader


@dataclass
class VcdSignal(Signal):
    """VCD-backed signal descriptor carrying the dumped reference name."""

    ref: str = field(default='', repr=False, compare=False)
    native_range: tuple[int, int] | None = field(default=None, compare=False)
    native_width: int | None = field(default=None, compare=False)


class VcdScope(Scope):
    def __init__(
        self,
        vcdvcd_scope: VcdVcdScope,
        parent_scope: Scope | None,
        reader: VcdReader,
    ):
        super().__init__(name=vcdvcd_scope.name.split('.')[-1])
        self.vcdvcd_scope = vcdvcd_scope
        self.parent_scope = parent_scope
        self.reader = reader

    @cached_property
    def signal_list(self) -> Sequence[Signal]:
        native_range_re = re.compile(r'\[(\d+):(\d+)\]$')
        full_scope_name = self.full_name()
        signals = []
        for k, v in self.vcdvcd_scope.subElements.items():
            if isinstance(v, str):
                signal_path = f'{full_scope_name}.{k}'
                width = int(self.reader.file_handle[signal_path].size)
                if m := native_range_re.search(k):
                    high, low = int(m.group(1)), int(m.group(2))
                    if abs(high - low) + 1 != width:
                        raise ValueError(
                            f'native range [{high}:{low}] does not match width {width} '
                            f"for signal '{signal_path}'"
                        )
                    native_range = (high, low)
                    bare_name = k[: m.start()]
                else:
                    if width != 1:
                        raise ValueError(
                            f"width {width} mismatch for scalar signal '{signal_path}'"
                        )
                    native_range = None
                    bare_name = k

                signals.append(
                    VcdSignal(
                        name=bare_name,
                        parent_path=full_scope_name,
                        width=width,
                        range=native_range,
                        ref=signal_path,
                        native_range=native_range,
                        native_width=width,
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

    def _load_value_changes(
        self,
        signal: Signal,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load mapped VCD value changes with optional trailing range selection."""

        vcd_signal = self._resolve_signal(signal)
        assert isinstance(vcd_signal, VcdSignal)
        lookup_path = vcd_signal.ref

        signal_handle = self.file_handle[lookup_path]
        width = vcd_signal.native_width or int(signal_handle.size)
        high, low = map_range_to_offsets(
            vcd_signal.full_name,
            width,
            vcd_signal.native_range,
            vcd_signal.range,
        )

        def decode(raw: str, high: int, low: int) -> int:
            decoded = 0
            # VCD binary values may be shorter than the signal width when leading
            # bits are zero, so map bit indexes to raw-string positions manually.
            raw = raw.lower()
            for bit_index in range(min(high, len(raw) - 1), low - 1, -1):
                raw_index = len(raw) - 1 - bit_index
                decoded = (decoded << 1) + value_mapping.get(raw[raw_index], 0)
            return decoded

        value_width = high - low + 1
        dtype = np.object_ if value_width > 64 else np.uint64
        pairs = [(v[0], decode(v[1], high, low)) for v in signal_handle.tv]
        result = np.array(pairs, dtype=dtype)
        if len(result) == 0:
            raise ValueError(f"signal '{lookup_path}' has no value changes")
        return result, value_width

    def close(self):
        pass
