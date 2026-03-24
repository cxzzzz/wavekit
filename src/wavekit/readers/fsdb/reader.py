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
from ...waveform import Waveform
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
            raise ValueError('begin_time and begin_cycle are mutually exclusive')
        if end_time is not None and end_cycle is not None:
            raise ValueError('end_time and end_cycle are mutually exclusive')

        signal_path = signal.full_name if isinstance(signal, Signal) else signal
        clock_path = clock.full_name if isinstance(clock, Signal) else clock

        # Resolve NPI signal handles — reuse the handle if already available
        npi_signal = (
            signal._npi_signal
            if isinstance(signal, FsdbSignal) and signal._npi_signal is not None
            else self.file_handle.get_signal(signal_path)
        )
        npi_clock = (
            clock._npi_signal
            if isinstance(clock, FsdbSignal) and clock._npi_signal is not None
            else self.file_handle.get_signal(clock_path)
        )

        # Always load the full clock to compute absolute cycle numbers
        all_clock_changes = self.file_handle.load_value_change(
            npi_clock,
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
            npi_signal,
            begin_time=begin_time_actual,
            end_time=end_time_actual,
            xz_value=xz_value,
        )

        full_wave = self.value_change_to_waveform(
            signal_value_change,
            windowed_clock_changes,
            width=npi_signal.width(),
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

    @property
    def begin_time(self) -> str:
        return self.file_handle.min_time()

    @property
    def end_time(self) -> str:
        return self.file_handle.max_time()

    def close(self):
        self.file_handle.close()
