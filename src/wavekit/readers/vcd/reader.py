from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cached_property

import numpy as np
from vcdvcd import VCDVCD
from vcdvcd import Scope as VcdVcdScope

from ...scope import Scope
from ...signal import Signal
from ..base import Reader
from ..pattern_parser import split_by_range_expr


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
        full_scope_name = self.full_name()
        signals = []
        for k, v in self.vcdvcd_scope.subElements.items():
            if isinstance(v, str):
                signal_path = f'{full_scope_name}.{k}'
                width = int(self.reader.file_handle[signal_path].size)
                signals.append(
                    Signal(
                        name=k,
                        full_name=signal_path,
                        width=width,
                        range=None,
                        signed=False,
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
        path: str,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load raw value changes for a VCD signal.

        Parameters
        ----------
        path:
            Full signal path (may include a range suffix, e.g. ``"tb.sig[3:0]"``).
        value_mapping:
            Character-to-bit mapping, e.g. ``{'0': 0, '1': 1, 'x': 0, 'z': 0}``.
        begin_time, end_time:
            Time bounds (ignored — VCD always loads all value changes).
        """
        bare_signal_path, range_suffix = split_by_range_expr(path)

        # VCD does not support multi-dimensional (more than one bracket pair)
        if range_suffix and len(re.findall(r'\[[\d:]+\]', range_suffix)) > 1:
            raise ValueError(
                f"VCD does not support multi-dimensional range access: '{path}'. "
                'Use FSDB or load the full signal and slice manually.'
            )

        # Resolve the actual VCD signal name (may include a range suffix in the file)
        lookup_path = bare_signal_path
        if lookup_path not in self.file_handle.references_to_ids:
            pattern = re.compile(rf'^{re.escape(lookup_path)}\[\d+(?::\d+)?\]$')
            matches = [
                ref for ref in self.file_handle.references_to_ids.keys() if pattern.fullmatch(ref)
            ]
            if len(matches) == 1:
                lookup_path = matches[0]
            elif len(matches) > 1:
                raise ValueError(f'pattern {lookup_path} matches more than one signal')

        signal_handle = self.file_handle[lookup_path]
        width = int(signal_handle.size)
        _, file_range_suffix = split_by_range_expr(lookup_path)

        # Check file-range compatibility for sub-range access
        if range_suffix and file_range_suffix:
            file_range_match = re.fullmatch(r'\[(\d+)(?::(\d+))?\]', file_range_suffix)
            if file_range_match is not None:
                file_low = (
                    int(file_range_match.group(2))
                    if file_range_match.group(2) is not None
                    else int(file_range_match.group(1))
                )
            if file_low != 0:
                raise ValueError(
                    f"sub-range access for signal '{lookup_path}' is only supported "
                    'when the stored signal range starts at bit 0'
                )

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
