from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from ..base import Reader
from ..hierarchy import Node, Range, Scope, Signal, SignalCompositeType
from .npi_fsdb_reader import (
    NPI_FSDB_CT_ARRAY,
    NPI_FSDB_CT_RECORD,
    NPI_FSDB_CT_STRUCT,
    NPI_FSDB_CT_TAGGED_UNION,
    NPI_FSDB_CT_UNION,
    NpiFsdbReader,
    NpiFsdbScope,
    NpiFsdbSignal,
)

# Mapping from (val_0, val_1, val_x, val_z) to FSDB decode mode integer.
_MAPPING_TO_FSDB_MODE: dict[tuple[int, int, int, int], int] = {
    (0, 1, 0, 0): 0,  # xz_value=0  (value decode, X/Z→0)
    (0, 1, 1, 1): 1,  # xz_value=1  (value decode, X/Z→1)
    (0, 0, 1, 0): 2,  # X-only mask
    (0, 0, 0, 1): 3,  # Z-only mask
    (0, 0, 1, 1): 4,  # X-or-Z mask
    (0, 0, 0, 0): 5,  # mask none (both false)
}


@dataclass(frozen=True, eq=False)
class FsdbSignal(Signal):
    """FSDB-backed signal descriptor with lazy composite-member loading."""

    _npi_signal: NpiFsdbSignal | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_handle(cls, npi_signal: NpiFsdbSignal, parent: Node) -> FsdbSignal:
        """Build an FSDB signal node from an NPI handle and its parent node."""
        npi_ct_to_composite_type = {
            NPI_FSDB_CT_ARRAY: SignalCompositeType.ARRAY,
            NPI_FSDB_CT_STRUCT: SignalCompositeType.STRUCT,
            NPI_FSDB_CT_UNION: SignalCompositeType.UNION,
            NPI_FSDB_CT_TAGGED_UNION: SignalCompositeType.TAGGED_UNION,
            NPI_FSDB_CT_RECORD: SignalCompositeType.RECORD,
        }
        composite_type_raw = npi_signal.composite_type()
        if composite_type_raw is None:
            composite_type = None
        elif composite_type_raw in npi_ct_to_composite_type:
            composite_type = npi_ct_to_composite_type[composite_type_raw]
        else:
            raise ValueError(
                f'Unknown NPI composite type value: {composite_type_raw} '
                f"for signal '{npi_signal.name()}'"
            )

        raw_range = npi_signal.range()
        native_range = Range(*raw_range) if raw_range is not None else None
        return cls(
            base_name=npi_signal.name(),
            parent=parent,
            range=native_range,
            native_range=native_range,
            composite_type=composite_type,
            _npi_signal=npi_signal,
        )

    @cached_property
    def children(self) -> tuple[Node, ...]:
        if self.composite_type is None:
            return ()

        assert self._npi_signal is not None
        # NPI array members include their array base in their local names, e.g.
        # ``arr[0]`` and ``arr[0][1]``. Node.full_name intentionally skips ARRAY
        # parents, yielding ``scope.arr[0]`` rather than ``scope.arr.arr[0]``.
        return tuple(
            FsdbSignal.from_handle(member, self) for member in self._npi_signal.member_list()
        )


@dataclass(frozen=True, eq=False)
class FsdbScope(Scope):
    """FSDB-backed hierarchy scope with lazy direct-child loading."""

    _npi_scope: NpiFsdbScope = field(repr=False, compare=False)

    @cached_property
    def children(self) -> tuple[Node, ...]:
        scopes = tuple(
            FsdbScope(base_name=scope.name(), parent=self, _npi_scope=scope)
            for scope in self._npi_scope.child_scope_list()
        )
        signals = tuple(
            FsdbSignal.from_handle(signal, self) for signal in self._npi_scope.signal_list()
        )
        return scopes + signals

    @cached_property
    def def_name(self) -> str | None:
        """Return the module definition name, if this scope is a module."""
        return self._npi_scope.def_name()


class FsdbReader(Reader[FsdbSignal]):
    pynpi: dict[str, Any] = {}

    @classmethod
    def _maybe_init_pynpi(cls) -> Exception | None:
        if cls.pynpi:
            return None

        import os
        import sys

        verdi_home = os.environ.get('VERDI_HOME')
        if verdi_home is None:
            return None

        rel_lib_path = os.path.abspath(os.path.join(verdi_home, 'share', 'NPI', 'python'))
        if rel_lib_path not in sys.path:
            sys.path.append(rel_lib_path)

        try:
            cls.pynpi['npisys'] = importlib.import_module('pynpi.npisys')
            cls.pynpi['waveform'] = importlib.import_module('pynpi.waveform')
            cls.pynpi['npisys'].init([''])
        except Exception as exc:
            cls.pynpi.clear()
            return exc
        return None

    @staticmethod
    def _runtime_error(init_error: Exception | None, open_error: Exception) -> RuntimeError:
        details = [
            'Failed to initialize FSDB runtime.',
            'FsdbReader requires the Verdi runtime library (libNPI.so). Configure via:',
            '  - WAVEKIT_NPI_LIB — direct path to libNPI.so',
            '  - VERDI_HOME — Verdi installation directory',
            '  - LD_LIBRARY_PATH — system library search path',
            f'Open error: {open_error}',
        ]
        if init_error is not None:
            details.append(f'Optional pynpi bootstrap error: {init_error}')
        return RuntimeError('\n'.join(details))

    def __init__(self, file: str):
        super().__init__()
        init_error = self._maybe_init_pynpi()

        self.file = file
        try:
            self.file_handle = NpiFsdbReader(file)
        except Exception as exc:
            raise self._runtime_error(init_error, exc) from exc

    def _load_value_changes(
        self,
        signal: FsdbSignal,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> np.ndarray:
        """Load mapped FSDB value changes through the NPI reader."""
        # FSDB/NPI resolves the selected trailing range in signal.full_name.
        npi_signal = self.file_handle.get_signal(signal.full_name)
        mapping_key = (
            value_mapping['0'],
            value_mapping['1'],
            value_mapping['x'],
            value_mapping['z'],
        )
        mode = _MAPPING_TO_FSDB_MODE[mapping_key]
        begin = begin_time if begin_time is not None else 0
        end = end_time if end_time is not None else 2**64 - 1
        return self.file_handle.load_value_change_mode(npi_signal, begin, end, mode)

    @cached_property
    def top_scopes(self) -> tuple[FsdbScope, ...]:
        return tuple(
            FsdbScope(base_name=scope.name(), parent=None, _npi_scope=scope)
            for scope in self.file_handle.top_scope_list()
        )

    @property
    def begin_time(self) -> int:
        return self.file_handle.min_time()

    @property
    def end_time(self) -> int:
        return self.file_handle.max_time()

    def close(self) -> None:
        self.file_handle.close()
