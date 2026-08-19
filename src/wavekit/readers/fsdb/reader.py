from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..base import Reader
from ..hierarchy import Node, Scope, Signal, SignalCompositeType
from ..range import Range
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

    @property
    def definition(self) -> None:
        """Return no definition name; the FSDB API does not expose signal typedef names."""
        return None

    @cached_property
    def children(self) -> tuple[Node, ...]:
        """Return composite member nodes, or an empty tuple for scalar signals."""
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
        """Return direct child scopes and signals from this FSDB scope."""
        scopes = tuple(
            FsdbScope(base_name=scope.name(), parent=self, _npi_scope=scope)
            for scope in self._npi_scope.child_scope_list()
        )
        signals = tuple(
            FsdbSignal.from_handle(signal, self) for signal in self._npi_scope.signal_list()
        )
        return scopes + signals

    @cached_property
    def definition(self) -> str | None:
        """Return the module definition name, if this scope is a module."""
        return self._npi_scope.def_name()


class FsdbReader(Reader[FsdbSignal]):
    """Read FSDB waveform files through the Verdi NPI runtime.

    ``FsdbReader`` requires the Verdi runtime library (``libNPI.so``). Configure
    it with ``WAVEKIT_NPI_LIB``, ``VERDI_HOME``, or ``LD_LIBRARY_PATH`` before
    opening FSDB files.

    ``quiet`` (default ``True``) suppresses the NPI console banner via the
    ``-quiet`` initialization argument. Pass ``quiet=False`` to keep the banner.
    NPI initialization is process-global, so ``quiet`` only takes effect on the
    first ``FsdbReader`` created in the process; later readers cannot reliably
    re-enable the banner.
    """

    def __init__(self, file: str, *, quiet: bool = True):
        super().__init__()
        self.file = file
        try:
            self.file_handle = NpiFsdbReader(file, quiet=quiet)
        except Exception as exc:
            details = [
                'Failed to initialize FSDB runtime.',
                'FsdbReader requires the Verdi runtime library (libNPI.so). Configure via:',
                '  - WAVEKIT_NPI_LIB — direct path to libNPI.so',
                '  - VERDI_HOME — Verdi installation directory',
                '  - LD_LIBRARY_PATH — system library search path',
                f'Open error: {exc}',
            ]
            raise RuntimeError('\n'.join(details)) from exc

    def _load_value_changes(
        self,
        signal: FsdbSignal,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> np.ndarray:
        """Load mapped FSDB value changes through the NPI reader."""
        if signal.composite_type in (
            SignalCompositeType.UNION,
            SignalCompositeType.TAGGED_UNION,
        ):
            raise NotImplementedError(
                f"Loading {signal.composite_type.value} signal '{signal.full_name}' "
                'as a waveform is not supported; load one of its members instead'
            )

        if (
            signal.composite_type == SignalCompositeType.ARRAY
            and signal.range != signal.native_range
        ):
            raise NotImplementedError(
                f"Loading partial range of FSDB array '{signal.full_name}' is not supported; "
                'load the complete array or individual array elements instead'
            )

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
        return self.file_handle.load_value_change_mode(
            npi_signal,
            begin,
            end,
            mode,
            signal.width,
        )

    @cached_property
    def top_scopes(self) -> tuple[FsdbScope, ...]:
        """Return immutable top-level scopes in the FSDB hierarchy."""
        return tuple(
            FsdbScope(base_name=scope.name(), parent=None, _npi_scope=scope)
            for scope in self.file_handle.top_scope_list()
        )

    @property
    def begin_time(self) -> int:
        """Return the first timestamp stored in the FSDB file."""
        return self.file_handle.min_time()

    @property
    def end_time(self) -> int:
        """Return the last timestamp stored in the FSDB file."""
        return self.file_handle.max_time()

    def close(self) -> None:
        """Close the underlying FSDB/NPI reader handle."""
        self.file_handle.close()
