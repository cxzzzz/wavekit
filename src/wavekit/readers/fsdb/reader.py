from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from ...scope import Scope
from ...signal import Signal, SignalCompositeType
from ..base import Reader
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


@dataclass
class FsdbSignal(Signal):
    """FSDB-backed :class:`~wavekit.signal.Signal` with lazy member loading.

    Extends :class:`~wavekit.signal.Signal` by holding an NPI signal handle
    directly, enabling on-demand loading of composite members without any
    reader reference.  Construct via :meth:`from_handle`.
    """

    _npi_signal: NpiFsdbSignal | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_handle(cls, npi_sig: NpiFsdbSignal, full_name: str) -> FsdbSignal:
        """Build an :class:`FsdbSignal` from an NPI signal handle and its full path."""
        _npi_ct_to_enum = {
            NPI_FSDB_CT_ARRAY: SignalCompositeType.ARRAY,
            NPI_FSDB_CT_STRUCT: SignalCompositeType.STRUCT,
            NPI_FSDB_CT_UNION: SignalCompositeType.UNION,
            NPI_FSDB_CT_TAGGED_UNION: SignalCompositeType.TAGGED_UNION,
            NPI_FSDB_CT_RECORD: SignalCompositeType.RECORD,
        }
        ct_raw = npi_sig.composite_type()
        if ct_raw is None:
            composite_type = None
        elif ct_raw in _npi_ct_to_enum:
            composite_type = _npi_ct_to_enum[ct_raw]
        else:
            raise ValueError(f"Unknown NPI composite type value: {ct_raw} for signal '{full_name}'")
        return cls(
            name=npi_sig.name(),
            full_name=full_name,
            width=npi_sig.width(),
            range=npi_sig.range(),
            composite_type=composite_type,
            _npi_signal=npi_sig,
        )

    @cached_property
    def member_list(self) -> list[Signal] | None:
        if self.composite_type is None:
            return None
        assert self._npi_signal is not None
        npi_members = self._npi_signal.member_list()
        if self.composite_type == SignalCompositeType.ARRAY:
            # Array members' NPI names already include the array base name at every
            # level, e.g. parent "a" has members "a[0]", "a[1]", and "a[0]" further
            # has members "a[0][0]", "a[0][1]", etc.  The member name is therefore an
            # extension of the parent name with no extra "." separator, so we must NOT
            # prepend full_name (which would produce "tb.dut.a.a[0]").  Instead we use
            # the parent scope path (full_name minus the signal's own local name) as
            # the base, giving "tb.dut.a[0]", "tb.dut.a[0][1]", etc.
            scope_path = self.full_name[: -(len(self.name) + 1)]
            return [FsdbSignal.from_handle(m, f'{scope_path}.{m.name()}') for m in npi_members]
        else:
            return [FsdbSignal.from_handle(m, f'{self.full_name}.{m.name()}') for m in npi_members]


class FsdbScope(Scope):
    def __init__(self, handle: NpiFsdbScope, parent_scope: FsdbScope | None, reader: FsdbReader):
        super().__init__(name=handle.name())
        self._npi_scope = handle
        self.parent_scope = parent_scope
        self.reader = reader

    @cached_property
    def signal_list(self) -> Sequence[Signal]:
        full_scope_name = self.full_name()
        return [
            FsdbSignal.from_handle(npi_sig, f'{full_scope_name}.{npi_sig.name()}')
            for npi_sig in self._npi_scope.signal_list()
        ]

    @cached_property
    def child_scope_list(self) -> Sequence[Scope]:
        return [FsdbScope(c, self, self.reader) for c in self._npi_scope.child_scope_list()]

    @cached_property
    def type(self) -> str:
        """Return the NPI scope type string, e.g. 'npiFsdbScopeSvModule'."""
        return self._npi_scope.type()

    @cached_property
    def def_name(self) -> str | None:
        """Return the module definition name, or None if this scope is not a module."""
        return self._npi_scope.def_name()

    def find_scope_by_module(self, module_name: str, depth: int = 0) -> list[Scope]:
        if not hasattr(self, '_preloaded_module_scope'):
            self.preload_module_scope()
        return self._preloaded_module_scope[module_name]

    def preload_module_scope(self):
        preloaded_module_scope = defaultdict(list)
        for c in self.child_scope_list:
            for module_name, module_scope_list in c.preload_module_scope().items():
                preloaded_module_scope[module_name].extend(module_scope_list)

        if self.type == 'npiFsdbScopeSvModule':
            preloaded_module_scope[self.def_name].append(self)
        self._preloaded_module_scope = preloaded_module_scope
        return preloaded_module_scope


class FsdbReader(Reader):
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
        path: str,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Load mapped FSDB value changes through the NPI reader."""
        # FSDB/NPI resolves any trailing bit range and reports the effective width.
        npi_signal = self.file_handle.get_signal(path)
        c = value_mapping
        key = (c['0'], c['1'], c['x'], c['z'])
        mode = _MAPPING_TO_FSDB_MODE[key]
        begin = begin_time if begin_time is not None else 0
        end = end_time if end_time is not None else 2**64 - 1
        result = self.file_handle.load_value_change_mode(
            npi_signal,
            begin,
            end,
            mode,
        )
        return result, npi_signal.width()

    def top_scope_list(self) -> Sequence[Scope]:
        if not hasattr(self, '_top_scope_list'):
            self._top_scope_list = [
                FsdbScope(s, None, self) for s in self.file_handle.top_scope_list()
            ]
        return self._top_scope_list

    @property
    def begin_time(self) -> str:
        return self.file_handle.min_time()

    @property
    def end_time(self) -> str:
        return self.file_handle.max_time()

    def close(self):
        self.file_handle.close()
