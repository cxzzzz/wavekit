from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np

from ..expression import evaluate_expression, parse_expression
from ..waveform import Waveform
from .hierarchy import Node, Scope, Signal
from .matcher import CaptureKey
from .value_change import value_change_to_value_array

SignalT = TypeVar('SignalT', bound=Signal)


@dataclass(frozen=True, eq=False)
class _SearchRoot(Node):
    """Private search-only parent for the reader's top-level scopes."""

    base_name: str = field(default='', init=False, repr=False)
    parent: Node | None = field(default=None, init=False, repr=False)
    top_scopes: tuple[Scope, ...] = field(default_factory=tuple)

    @property
    def children(self) -> tuple[Scope, ...]:
        return self.top_scopes


class Reader(Generic[SignalT]):
    """Abstract base class for waveform file readers.

    Concrete subclasses (``VcdReader``,
    ``FstReader``, and ``FsdbReader``) implement
    the file-format-specific I/O;
    all high-level analysis APIs are provided here.

    Supports the context-manager protocol.

        with VcdReader("sim.vcd") as r:
            wave = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

    Signal path format
    ------------------
    All signal paths use dotted hierarchical notation matching the scope tree
    in the waveform file, e.g. ``"tb.dut.sub.signal_name[7:0]"``.
    If the bit-range suffix is omitted and the file stores the signal with a
    range, the range is appended automatically.

    Unknown-mask API
    -------------------------------
    ``load_unknown_mask`` and ``load_matched_unknown_masks`` return
    source X/Z bit presence as ordinary unsigned ``Waveform``
    bitmasks, so users can detect unknown bits without changing the two-state
    value model.

    Pattern syntax (used by ``get_matched_signals``, ``get_matched_scopes``,
    ``load_matched_waveforms``, ``load_matched_unknown_masks``, ``eval``)
    -------------------------------------------------
    * ``{a,b,c}``     — matches ``a``, ``b``, or ``c``; captures each as a key.
    * ``{0..7}``       — integer range 0 to 7 inclusive; step defaults to 1.
    * ``{0..7..2}``    — integer range with explicit step (0, 2, 4, 6).
    * ``/<regex>/``     — use a Python regex instead of exact matching; capture
      groups ``(...)`` are retained in a ``RegexCapture`` key.
    * ``@<regex>``      — legacy-compatible regex spelling accepted by the parser.
    * ``*`` / ``**``    — match one hierarchy level or recursively match levels;
      matches are retained as ``WildcardCapture`` keys.
    * ``$<module>`` / ``$$<module>`` — match direct or recursive FSDB module
      definitions; module captures are retained as ``ExactCapture``.

    Matching APIs use ``CaptureKey`` dictionary keys. Ordinary exact-name components
    are omitted from keys; binding matchers retain typed ``Capture`` objects. The
    dictionary value type depends on the API.
    """

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # exception wont be suppressed
        return False

    def load_waveform(
        self,
        signal: SignalT | str,
        clock: SignalT | str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        begin_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> Waveform:
        """Load a single signal as a clock-synchronised ``Waveform``.

        The signal is sampled on every **negedge** of *clock* by default
        (i.e. the value is captured at each falling edge of the clock, which
        reflects the value that was stable during the preceding high phase).
        Set ``sample_on_posedge=True`` to sample on rising edges instead.

        Parameters
        ----------
        signal:
            Full dotted path of the signal as a ``Signal``
            object or a string.  When a ``Signal`` is passed, ``signal.full_name``
            is used as the path (which may include bit-range suffixes).
            When a string is passed, the value is used verbatim as the full
            hierarchical path, e.g. ``"tb.dut.data[7:0]"`` or ``"tb.dut.data"``.
        clock:
            Clock signal as a ``Signal`` or full dotted
            path string, e.g. ``"tb.clk"``.
        xz_value:
            Integer substituted for ``X`` and ``Z`` values in the file.
            Defaults to ``0``.
        signed:
            If ``True``, the loaded values are interpreted as two's-complement
            signed integers.
        sample_on_posedge:
            If ``True``, sample on rising clock edges; otherwise on falling
            edges (default).
        begin_time:
            Simulation time to start loading from (inclusive).  ``None`` means
            start of simulation.  Mutually exclusive with *begin_cycle*.
        end_time:
            Simulation time to stop loading at (exclusive).  ``None`` means
            end of simulation.  Mutually exclusive with *end_cycle*.
        begin_cycle:
            Absolute clock cycle number to start loading from (inclusive).
            ``None`` means start of simulation.  Mutually exclusive with
            *begin_time*.  The clock is always loaded from time 0 so cycle
            numbers are absolute and comparable across different waveforms.
        end_cycle:
            Absolute clock cycle number to stop loading at (exclusive).
            ``None`` means end of simulation.  Mutually exclusive with
            *end_time*.

        Returns
        -------
        Waveform:
            One sample per clock edge within the requested window.  The
            ``.clock`` array contains absolute cycle numbers from the start
            of simulation.  ``waveform.signal.full_name`` records the resolved
            signal path when source metadata is available.

        Raises
        ------
        ValueError:
            If both *begin_time* and *begin_cycle* (or both *end_time* and
            *end_cycle*) are provided simultaneously.
        """
        self._validate_xz_value(xz_value)
        resolved_signal = self._resolve_signal(signal)
        resolved_clock = self._resolve_signal(clock)
        value_mapping = {'0': 0, '1': 1, 'x': xz_value, 'z': xz_value}
        wf = self._sample_on_clock(
            resolved_signal,
            resolved_clock,
            value_mapping=value_mapping,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
            begin_cycle=begin_cycle,
            end_cycle=end_cycle,
        )
        wf.signal = resolved_signal
        wf.width = wf.width
        wf.signed = signed
        return wf

    def load_unknown_mask(
        self,
        signal: SignalT | str,
        clock: SignalT | str,
        include_x: bool = True,
        include_z: bool = True,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        begin_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> Waveform:
        """Load source X/Z presence as an unsigned bitmask waveform.

        The returned ``Waveform`` is sampled on the same clock
        edges and supports the same time/cycle windowing as
        ``load_waveform``, but its values are masks instead of substituted
        two-state signal values.  A mask bit is ``1`` when the corresponding
        source bit is selected by *include_x* and/or *include_z*.

        Parameters
        ----------
        signal:
            Full dotted signal path or ``Signal`` object.
        clock:
            Clock signal path or ``Signal`` object.
        include_x:
            If ``True`` (default), mark source ``X``/``x`` bits.
        include_z:
            If ``True`` (default), mark source ``Z``/``z`` bits.
        sample_on_posedge, begin_time, end_time, begin_cycle, end_cycle:
            Same sampling/windowing semantics as ``load_waveform``.

        Returns
        -------
        Waveform:
            Unsigned mask waveform.
        """
        resolved_signal = self._resolve_signal(signal)
        resolved_clock = self._resolve_signal(clock)

        value_mapping = {
            '0': 0,
            '1': 0,
            'x': 1 if include_x else 0,
            'z': 1 if include_z else 0,
        }
        wf = self._sample_on_clock(
            resolved_signal,
            resolved_clock,
            value_mapping=value_mapping,
            signed=False,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
            begin_cycle=begin_cycle,
            end_cycle=end_cycle,
        )
        wf.signal = resolved_signal
        wf.signed = False
        return wf

    @property
    @abstractmethod
    def top_scopes(self) -> tuple[Scope, ...]:
        """Return the real top-level scopes in the waveform hierarchy."""
        pass

    @staticmethod
    def _value_change_to_waveform(
        value_change: np.ndarray,
        clock_changes: np.ndarray,
        width: int | None,
        signed: bool,
        sample_on_posedge: bool = False,
        clock_offset: int = 0,
    ) -> Waveform:
        value, clock, time = value_change_to_value_array(
            value_change,
            clock_changes,
            sample_on_posedge=sample_on_posedge,
            clock_offset=clock_offset,
        )

        return Waveform(
            value=value,
            clock=clock,
            time=time,
            width=width,
            signed=signed,
        )

    @abstractmethod
    def _load_value_changes(
        self,
        signal: SignalT,
        value_mapping: dict[str, int],
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> np.ndarray:
        """Load raw value changes for a signal.

        Subclasses implement file-format-specific loading. Returns an array
        with shape ``(N, 2)`` whose columns are ``[time, value]``. The
        effective width is defined by ``signal.width``.

        Parameters
        ----------
        signal:
            Resolved signal descriptor. Backend-specific subclasses may carry
            native handles or dumped references required for loading.
        value_mapping:
            Character-to-bit mapping, e.g. ``{'0': 0, '1': 1, 'x': 0, 'z': 0}``.
        begin_time:
            Optional earliest time to include (inclusive).
        end_time:
            Optional latest time to include (exclusive).
        """  # noqa: E501

    def _sample_on_clock(
        self,
        signal: SignalT,
        clock: SignalT,
        value_mapping: dict[str, int],
        signed: bool,
        sample_on_posedge: bool,
        begin_time: int | None,
        end_time: int | None,
        begin_cycle: int | None,
        end_cycle: int | None,
    ) -> Waveform:
        """Sample *signal* on every *clock* edge and return a raw Waveform.

        Subclasses provide ``_load_value_changes`` for format-specific I/O.
        The returned Waveform is a simple value array — naming is handled by
        the caller.
        """
        if begin_time is not None and begin_cycle is not None:
            raise ValueError('begin_time and begin_cycle are mutually exclusive')
        if end_time is not None and end_cycle is not None:
            raise ValueError('end_time and end_cycle are mutually exclusive')

        # Load clock value changes for absolute cycle computation
        clock_mapping = {'0': 0, '1': 1, 'x': 0, 'z': 0}
        all_clock_changes = self._load_value_changes(clock, clock_mapping)

        # Find sampling edge timestamps
        sample_value = 1 if sample_on_posedge else 0
        clock_edge_times = all_clock_changes[all_clock_changes[:, 1] == sample_value, 0]

        if len(clock_edge_times) == 0:
            edge_kind = 'pos' if sample_on_posedge else 'neg'
            raise ValueError(f'no {edge_kind}edges found in clock signal')

        # Convert begin_cycle/end_cycle to begin_time/end_time
        if begin_cycle is not None:
            if begin_cycle >= len(clock_edge_times):
                raise ValueError(
                    f'begin_cycle {begin_cycle} out of range (max {len(clock_edge_times) - 1})'
                )
            begin_time = int(clock_edge_times[begin_cycle])
        if end_cycle is not None:
            if end_cycle > len(clock_edge_times):
                raise ValueError(
                    f'end_cycle {end_cycle} out of range (max {len(clock_edge_times)})'
                )
            if end_cycle < len(clock_edge_times):
                end_time = int(clock_edge_times[end_cycle])

        # Compute clock_offset = number of sampling edges before begin_time
        begin_time = begin_time if begin_time is not None else 0
        clock_offset = int(
            np.searchsorted(
                clock_edge_times,
                begin_time,
                side='left',
            )
        )

        # Trim clock to window [begin_time, end_time)
        clock_mask = all_clock_changes[:, 0] >= begin_time
        if end_time is not None:
            clock_mask &= all_clock_changes[:, 0] < end_time
        windowed_clock_changes = all_clock_changes[clock_mask]

        # Load signal value changes (backend handles selection during decoding).
        signal_value_change = self._load_value_changes(
            signal,
            value_mapping,
            begin_time=begin_time,
            end_time=end_time,
        )

        if len(signal_value_change) == 0:
            raise ValueError(f"signal '{signal.full_name}' has no value changes")

        # Convert to Waveform via sampling and trim
        result = self._value_change_to_waveform(
            signal_value_change,
            windowed_clock_changes,
            width=signal.width,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            clock_offset=clock_offset,
        )

        return result

    def _search_root(self, root_scope: Scope | None) -> Node:
        """Return a real root or an internal search container for all top-level scopes."""
        if root_scope is not None:
            return root_scope
        return _SearchRoot(top_scopes=self.top_scopes)

    def get_matched_signals(
        self,
        path: str,
        root_scope: Scope | None = None,
    ) -> dict[CaptureKey, SignalT]:
        """Return all signals whose paths match *path*, keyed by captures.

        Traverses the scope tree starting from *root_scope* (or the file's
        top-level scopes if *root_scope* is ``None``) and applies the query
        path to each level.  See the class docstring for query path syntax.

        Parameters
        ----------
        path:
            Signal query path, e.g. ``"tb.dut.fifo_{0..3}.w_ptr[2:0]"`` or
            ``r"tb.dut./([a-z]+)_valid/"``.
        root_scope:
            If provided, search only within this scope instead of starting
            from the file's top-level scopes.

        Returns
        -------
        dict[CaptureKey, Signal]:
            Maps each capture key to the matched ``Signal``
            object (carrying name, width, range, signed).
            Ordinary exact-name matches are omitted from the key, so a query
            without binding matchers uses ``()``.

        Raises
        ------
        ValueError:
            If two different signals resolve to the same key, or if using
            module matchers on a backend without ``definition`` support (VCD/FST).
        """
        search_root = self._search_root(root_scope)
        return cast(dict[CaptureKey, SignalT], search_root.get_matched_signals(path))

    def get_matched_scopes(
        self,
        path: str,
        root_scope: Scope | None = None,
    ) -> dict[CaptureKey, Scope]:
        """Return all scopes whose paths match *path*, keyed by captures.

        Similar to ``get_matched_signals`` but stops at the scope level —
        the last component of *path* must match a scope name, not a signal.
        Useful for enumerating module instances before loading their signals.

        Parameters
        ----------
        path:
            Scope query path using the same syntax as signal paths.  The last
            component must match a scope (module) name, e.g.
            ``"tb.dut.fifo_{0..3}"`` or ``r"tb./([a-z]+)_core/"``.
        root_scope:
            If provided, search only within this scope instead of starting
            from the file's top-level scopes.

        Returns
        -------
        dict[CaptureKey, Scope]:
            Maps each capture key to the matched ``Scope``.
            Ordinary exact-name matches are omitted from the key, so a query
            without binding matchers uses ``()``.

        Raises
        ------
        ValueError:
            If two different scopes resolve to the same key, or if using
            module matchers on a backend without ``definition`` support (VCD/FST),
            or if the path contains a terminal signal bit-range suffix.
        """
        search_root = self._search_root(root_scope)
        return search_root.get_matched_scopes(path)

    def load_matched_waveforms(
        self,
        signal_path: str,
        clock_path: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        begin_cycle: int | None = None,
        end_cycle: int | None = None,
        root_scope: Scope | None = None,
    ) -> dict[CaptureKey, Waveform]:
        """Batch-load all signals matching *signal_path*, each paired with its clock.

        Internally calls ``get_matched_signals`` for both *signal_path* and
        *clock_path*, then dispatches ``load_waveform`` for every match.

        Clock assignment rules:

        * **Single clock** — if *clock_path* matches exactly one signal, that
          clock is broadcast to all matched signals.
        * **Multiple clocks** — for each signal key, the clock whose key is the
          longest prefix of the signal key is selected. If no clock key is a
          prefix, raises ``ValueError``.

        Parameters
        ----------
        signal_path:
            Signal query path.  See class docstring.
        clock_path:
            Clock signal query path.  Must match at least one signal.
        xz_value, signed, sample_on_posedge, begin_time, end_time, begin_cycle, end_cycle:
            Forwarded to ``load_waveform`` for every loaded signal.
        root_scope:
            If provided, both *signal_path* and *clock_path* are searched within
            this scope instead of the file's top-level scopes.

        Returns
        -------
        dict[CaptureKey, Waveform]:
            Same keys as ``get_matched_signals`` on *signal_path*.

        Raises
        ------
        ValueError:
            If *clock_path* matches no signals, or if no clock key is a prefix
            of a signal key.
        """
        self._validate_xz_value(xz_value)
        clock_pairing = self._resolve_clock_pairing(signal_path, clock_path, root_scope)
        matched_signals = self.get_matched_signals(signal_path, root_scope=root_scope)
        load_kwargs: dict[str, Any] = dict(
            xz_value=xz_value,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
            begin_cycle=begin_cycle,
            end_cycle=end_cycle,
        )
        return {
            k: self.load_waveform(sig, clock_pairing[k], **load_kwargs)
            for k, sig in matched_signals.items()
        }

    def load_matched_unknown_masks(
        self,
        signal_path: str,
        clock_path: str,
        include_x: bool = True,
        include_z: bool = True,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        begin_cycle: int | None = None,
        end_cycle: int | None = None,
        root_scope: Scope | None = None,
    ) -> dict[CaptureKey, Waveform]:
        """Batch-load X/Z mask waveforms for all signals matching *signal_path*.

        Clock assignment follows ``load_matched_waveforms``: a single
        matched clock is broadcast to all signals; otherwise the longest-prefix
        clock key is selected for each signal key.

        Parameters
        ----------
        signal_path:
            Signal query path.  See class docstring.
        clock_path:
            Clock signal query path.  Must match at least one signal.
        include_x:
            If ``True`` (default), mark source ``X``/``x`` bits.
        include_z:
            If ``True`` (default), mark source ``Z``/``z`` bits.
        sample_on_posedge, begin_time, end_time, begin_cycle, end_cycle:
            Same sampling/windowing semantics as ``load_waveform``.
        root_scope:
            If provided, both *signal_path* and *clock_path* are searched within
            this scope instead of the file's top-level scopes.

        Returns
        -------
        dict[CaptureKey, Waveform]:
            Same keys as ``get_matched_signals`` on *signal_path*.
        """
        clock_pairing = self._resolve_clock_pairing(signal_path, clock_path, root_scope)
        load_kwargs: dict[str, Any] = dict(
            include_x=include_x,
            include_z=include_z,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
            begin_cycle=begin_cycle,
            end_cycle=end_cycle,
        )
        matched_signals = self.get_matched_signals(signal_path, root_scope=root_scope)
        return {
            k: self.load_unknown_mask(sig, clock_pairing[k], **load_kwargs)
            for k, sig in matched_signals.items()
        }

    @staticmethod
    def _validate_xz_value(xz_value: int) -> None:
        if xz_value not in (0, 1):
            raise ValueError('xz_value must be 0 or 1')

    def _resolve_signal(self, signal: SignalT | str) -> SignalT:
        if isinstance(signal, Signal):
            return signal
        matched = self.get_matched_signals(signal)
        if len(matched) == 0:
            raise ValueError(f"signal '{signal}' not found")
        if len(matched) > 1:
            raise ValueError(f"signal '{signal}' matches more than one signal")
        return matched[()]

    def _resolve_clock_pairing(
        self,
        signal_path: str,
        clock_path: str,
        root_scope: Scope | None,
    ) -> dict[CaptureKey, SignalT]:
        """Resolve signal/clock query paths into a {signal_key: clock_signal} map.

        Rules:
        - Single clock match: broadcast to all signals.
        - Multiple clock matches: longest-prefix clock key per signal key.
        - No prefix match for a signal: raise ValueError.
        """
        matched_clocks = self.get_matched_signals(clock_path, root_scope=root_scope)
        if not matched_clocks:
            raise ValueError(f'clock path {clock_path!r} matched no signals')

        matched_signals = self.get_matched_signals(signal_path, root_scope=root_scope)

        if len(matched_clocks) == 1:
            clock_signal = next(iter(matched_clocks.values()))
            return {k: clock_signal for k in matched_signals}

        clock_keys = list(matched_clocks.keys())
        pairing: dict[CaptureKey, SignalT] = {}
        for sig_key in matched_signals:
            best_len = -1
            best_clock_key: CaptureKey | None = None
            for ck in clock_keys:
                if sig_key[: len(ck)] == ck and len(ck) > best_len:
                    best_len = len(ck)
                    best_clock_key = ck
            if best_clock_key is None:
                raise ValueError(
                    f'no clock key is a prefix of signal key {sig_key!r}; '
                    f'available clock keys: {clock_keys!r}'
                )
            pairing[sig_key] = matched_clocks[best_clock_key]
        return pairing

    @abstractmethod
    def close(self):
        """Close the underlying waveform file handle.

        Subclasses should make this method idempotent when the backing library
        exposes an explicit close operation. Prefer using readers as context
        managers so ``close()`` is called automatically.
        """
        pass

    # ------------------------------------------------------------------
    # High-level expression APIs
    # ------------------------------------------------------------------

    def eval(
        self,
        expr: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        begin_cycle: int | None = None,
        end_cycle: int | None = None,
        mode: Literal['single', 'zip'] = 'single',
        root_scope: Scope | None = None,
    ) -> Waveform | dict[CaptureKey, Waveform]:
        """Evaluate a waveform expression containing physical signal paths.

        Parameters
        ----------
        expr:
            Expression string. Signal paths may be used as operands or as
            arguments to registered expression functions.
        clock:
            Clock signal used for all waveform loads.
        xz_value, signed, sample_on_posedge, begin_time, end_time, begin_cycle,
        end_cycle:
            Forwarded to ``load_matched_waveforms`` for every path.
        mode:
            ``'single'`` requires every path to match one signal. ``'zip'``
            evaluates once per shared multi-match key and broadcasts singleton
            paths.
        root_scope:
            If provided, resolve paths within this scope.

        Returns
        -------
        Waveform or dict[CaptureKey, Waveform]
            The evaluated waveform, or one waveform per zip key.
        """
        self._validate_xz_value(xz_value)
        substituted, path_entries = parse_expression(expr)
        load_kwargs: dict[str, Any] = dict(
            xz_value=xz_value,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
            begin_cycle=begin_cycle,
            end_cycle=end_cycle,
            root_scope=root_scope,
        )

        loaded_per_path: list[tuple[str, str, dict[CaptureKey, Waveform]]] = []
        for placeholder, path in path_entries:
            matched = self.load_matched_waveforms(
                signal_path=path,
                clock_path=clock,
                **load_kwargs,
            )
            if not matched:
                raise ValueError(f"path '{path}' matched no signals")
            loaded_per_path.append((placeholder, path, matched))

        if mode == 'single':
            for _placeholder, path, matched in loaded_per_path:
                if len(matched) > 1:
                    matched_names = [
                        wave.signal.full_name if wave.signal is not None else None
                        for wave in matched.values()
                    ]
                    raise ValueError(
                        f"path '{path}' matched {len(matched)} signals in mode='single',"
                        f" use mode='zip'. Matched: "
                        f'{matched_names}'
                    )
            namespace = {
                placeholder: next(iter(matched.values()))
                for placeholder, _, matched in loaded_per_path
            }
            try:
                return evaluate_expression(substituted, namespace)
            except Exception as exc:
                raise ValueError(
                    f"failed to evaluate expression '{expr}' " f"(substituted: '{substituted}')"
                ) from exc

        if mode == 'zip':
            multi_paths = [
                (placeholder, path, matched)
                for placeholder, path, matched in loaded_per_path
                if len(matched) > 1
            ]
            single_paths = [
                (placeholder, path, matched)
                for placeholder, path, matched in loaded_per_path
                if len(matched) == 1
            ]

            if multi_paths:
                _ref_placeholder, ref_path, ref_matched = multi_paths[0]
                ref_keys = set(ref_matched)
                for _placeholder, path, matched in multi_paths[1:]:
                    if set(matched) != ref_keys:
                        raise ValueError(
                            'inconsistent match keys between paths: '
                            f"'{ref_path}' has keys {ref_keys!r}, "
                            f"'{path}' has keys {set(matched)!r}"
                        )
                zip_keys = list(ref_keys)
            else:
                zip_keys = [()]

            broadcast_namespace = {
                placeholder: next(iter(matched.values()))
                for placeholder, _, matched in single_paths
            }
            result: dict[CaptureKey, Waveform] = {}
            for key in zip_keys:
                namespace = dict(broadcast_namespace)
                for placeholder, _path, matched in multi_paths:
                    namespace[placeholder] = matched[key]
                try:
                    result[key] = evaluate_expression(substituted, namespace)
                except Exception as exc:
                    raise ValueError(
                        f'failed to evaluate expression {expr!r} for key {key!r}'
                    ) from exc
            return result

        raise ValueError(f"unknown mode '{mode}', expected 'single' or 'zip'")
