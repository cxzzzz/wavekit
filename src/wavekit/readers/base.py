from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from ..expression import evaluate_expression, parse_expression
from ..waveform import Waveform
from .clock_domain import ClockDomain
from .hierarchy import Node, Scope, Signal
from .matcher import Capture, ExactMatcher, parse_query_path
from .value_change import value_change_to_value_array


@dataclass(frozen=True, eq=False)
class _SearchRoot(Node):
    """Private search-only parent for the reader's top-level scopes."""

    base_name: str = field(default='', init=False, repr=False)
    parent: Node | None = field(default=None, init=False, repr=False)
    top_scopes: tuple[Scope, ...] = field(default_factory=tuple)

    @property
    def children(self) -> tuple[Scope, ...]:
        return self.top_scopes


class Reader:
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

    Matching APIs use ``tuple[Capture, ...]`` dictionary keys. Ordinary exact-name components
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

    def __getitem__(self, path: str) -> Node:
        """Return the ``Signal`` or ``Scope`` at an exact dotted *path*."""
        matched = self.get_matched_nodes(path)
        if not matched:
            raise KeyError(f'{path!r} not found in hierarchy')
        if list(matched) != [()]:
            raise KeyError(
                f'{path!r}: dict lookup requires an exact path; '
                'use get_matched_nodes() for pattern queries'
            )
        return matched[()]

    def clock_domain(
        self,
        clock: Signal | str,
        *,
        sample_on_posedge: bool = False,
        start_time: int | None = None,
        end_time: int | None = None,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> ClockDomain:
        """Build a reusable sampling recipe: a clock plus sampling parameters.

        Use as a context manager (or decorator) so ``Signal.waveform()``/
        ``Signal.unknown_mask()`` (and their ``.w``/``.m`` shorthands) load
        against it.

        Parameters
        ----------
        clock:
            Clock signal as a ``Signal`` or full dotted path string.
        sample_on_posedge, start_time, end_time, start_cycle, end_cycle:
            Same sampling/windowing semantics as ``load_waveform``.

        Returns
        -------
        ClockDomain:
            The constructed sampling recipe.
        """
        resolved_clock = clock if isinstance(clock, Signal) else self.get_signal(clock)
        return ClockDomain(
            clock=resolved_clock,
            sample_on_posedge=sample_on_posedge,
            start_time=start_time,
            end_time=end_time,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
        )

    @staticmethod
    def _resolve_ambient_clock(
        clock: Signal | str | None,
        **call_sampling: Any,
    ) -> tuple[Signal | str, dict[str, Any]]:
        """Resolve *clock*, falling back to the ambient domain when ``None``.

        Explicit *clock* wins outright and keeps *call_sampling* as given;
        omitted *clock* reads the ambient ``ClockDomain`` for both the
        clock and every sampling field, raising ``RuntimeError`` if none
        is active.
        """
        if clock is not None:
            return clock, call_sampling
        domain = ClockDomain.current()
        return domain.clock, domain.sampling_kwargs()

    def load_waveform(
        self,
        signal: Signal | str,
        clock: Signal | str | None = None,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        start_time: int | None = None,
        end_time: int | None = None,
        start_cycle: int | None = None,
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
            Clock signal as a ``Signal`` or full dotted path string.  If
            omitted (``None``), uses the ambient ``ClockDomain`` — including
            its edge and window — and raises ``RuntimeError`` if none is
            active. An explicit *clock* always takes the sampling parameters
            below from this call, ignoring any active domain.
        xz_value:
            Integer substituted for ``X`` and ``Z`` values in the file.
            Defaults to ``0``.
        signed:
            If ``True``, the loaded values are interpreted as two's-complement
            signed integers.
        sample_on_posedge:
            If ``True``, sample on rising clock edges; otherwise on falling
            edges (default).  Ignored when *clock* is omitted.
        start_time:
            Simulation time to start loading from (inclusive).  ``None`` means
            start of simulation.  Mutually exclusive with *start_cycle*.
            Ignored when *clock* is omitted.
        end_time:
            Simulation time to stop loading at (exclusive).  ``None`` means
            end of simulation.  Mutually exclusive with *end_cycle*.  Ignored
            when *clock* is omitted.
        start_cycle:
            Absolute clock cycle number to start loading from (inclusive).
            ``None`` means start of simulation.  Mutually exclusive with
            *start_time*.  The clock is always loaded from time 0 so cycle
            numbers are absolute and comparable across different waveforms.
            Ignored when *clock* is omitted.
        end_cycle:
            Absolute clock cycle number to stop loading at (exclusive).
            ``None`` means end of simulation.  Mutually exclusive with
            *end_time*.  Ignored when *clock* is omitted.

        Returns
        -------
        Waveform:
            One sample per clock edge within the requested window.  The
            ``.cycle`` array contains absolute cycle numbers from the start
            of simulation.  ``waveform.signal.full_name`` records the resolved
            signal path when source metadata is available.

        Raises
        ------
        RuntimeError:
            If *clock* is omitted and no ``ClockDomain`` is active.
        ValueError:
            If both *start_time* and *start_cycle* (or both *end_time* and
            *end_cycle*) are provided simultaneously.
        """
        self._validate_xz_value(xz_value)
        resolved_signal = signal if isinstance(signal, Signal) else self.get_signal(signal)
        resolved_clock, sampling = self._resolve_ambient_clock(
            clock,
            sample_on_posedge=sample_on_posedge,
            start_time=start_time,
            end_time=end_time,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
        )
        if not isinstance(resolved_clock, Signal):
            resolved_clock = self.get_signal(resolved_clock)
        value_mapping = {'0': 0, '1': 1, 'x': xz_value, 'z': xz_value}
        wf = self._sample_on_clock(
            resolved_signal,
            resolved_clock,
            value_mapping=value_mapping,
            signed=signed,
            **sampling,
        )
        wf.signal = resolved_signal
        wf.width = wf.width
        wf.signed = signed
        return wf

    def load_unknown_mask(
        self,
        signal: Signal | str,
        clock: Signal | str | None = None,
        include_x: bool = True,
        include_z: bool = True,
        sample_on_posedge: bool = False,
        start_time: int | None = None,
        end_time: int | None = None,
        start_cycle: int | None = None,
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
            Clock signal path or ``Signal`` object.  If omitted (``None``),
            uses the ambient ``ClockDomain`` (same rules as ``load_waveform``).
        include_x:
            If ``True`` (default), mark source ``X``/``x`` bits.
        include_z:
            If ``True`` (default), mark source ``Z``/``z`` bits.
        sample_on_posedge, start_time, end_time, start_cycle, end_cycle:
            Same sampling/windowing semantics as ``load_waveform``.

        Returns
        -------
        Waveform:
            Unsigned mask waveform.

        Raises
        ------
        RuntimeError:
            If *clock* is omitted and no ``ClockDomain`` is active.
        """
        resolved_signal = signal if isinstance(signal, Signal) else self.get_signal(signal)
        resolved_clock, sampling = self._resolve_ambient_clock(
            clock,
            sample_on_posedge=sample_on_posedge,
            start_time=start_time,
            end_time=end_time,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
        )
        if not isinstance(resolved_clock, Signal):
            resolved_clock = self.get_signal(resolved_clock)

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
            **sampling,
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
            cycle=clock,
            time=time,
            width=width,
            signed=signed,
        )

    @abstractmethod
    def _load_value_changes(
        self,
        signal: Signal,
        value_mapping: dict[str, int],
        start_time: int | None = None,
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
        start_time:
            Optional earliest time to include (inclusive).
        end_time:
            Optional latest time to include (exclusive).
        """  # noqa: E501

    def _sample_on_clock(
        self,
        signal: Signal,
        clock: Signal,
        value_mapping: dict[str, int],
        signed: bool,
        sample_on_posedge: bool,
        start_time: int | None,
        end_time: int | None,
        start_cycle: int | None,
        end_cycle: int | None,
    ) -> Waveform:
        """Sample *signal* on every *clock* edge and return a raw Waveform.

        Subclasses provide ``_load_value_changes`` for format-specific I/O.
        The returned Waveform is a simple value array — naming is handled by
        the caller.
        """
        if start_time is not None and start_cycle is not None:
            raise ValueError('start_time and start_cycle are mutually exclusive')
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

        # Convert start_cycle/end_cycle to start_time/end_time
        if start_cycle is not None:
            if start_cycle >= len(clock_edge_times):
                raise ValueError(
                    f'start_cycle {start_cycle} out of range (max {len(clock_edge_times) - 1})'
                )
            start_time = int(clock_edge_times[start_cycle])
        if end_cycle is not None:
            if end_cycle > len(clock_edge_times):
                raise ValueError(
                    f'end_cycle {end_cycle} out of range (max {len(clock_edge_times)})'
                )
            if end_cycle < len(clock_edge_times):
                end_time = int(clock_edge_times[end_cycle])

        # Compute clock_offset = number of sampling edges before start_time
        start_time = start_time if start_time is not None else 0
        clock_offset = int(
            np.searchsorted(
                clock_edge_times,
                start_time,
                side='left',
            )
        )

        # Trim clock to window [start_time, end_time)
        clock_mask = all_clock_changes[:, 0] >= start_time
        if end_time is not None:
            clock_mask &= all_clock_changes[:, 0] < end_time
        windowed_clock_changes = all_clock_changes[clock_mask]

        # Load signal value changes (backend handles selection during decoding).
        signal_value_change = self._load_value_changes(
            signal,
            value_mapping,
            start_time=start_time,
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

    def get_signal(
        self,
        path: str,
        root_scope: Scope | None = None,
    ) -> Signal:
        """Return the signal at an exact hierarchy path.

        Parameters
        ----------
        path:
            Exact dotted signal path, with an optional terminal bit selection
            or range. Matcher expressions are not accepted.
        root_scope:
            If provided, resolve *path* within this scope instead of starting
            from the file's top-level scopes.

        Returns
        -------
        Signal
            The exact matched signal, including any requested range view.

        Raises
        ------
        ValueError
            If *path* contains matcher syntax or the signal does not exist.
        """
        steps = parse_query_path(path)
        if any(
            not isinstance(step.matcher, ExactMatcher)
            or step.matcher.target != 'name'
            or step.recursive
            or step.native_recursive
            for step in steps
        ):
            raise ValueError(
                'get_signal() requires an exact path; '
                'use get_matched_signals() for pattern queries'
            )
        search_root = root_scope or _SearchRoot(reader=self, top_scopes=self.top_scopes)
        matched = search_root._match_path(
            steps,
            lambda node, remaining: len(remaining) > 1 or isinstance(node, Signal),
        )
        if not matched:
            raise ValueError(f"signal '{path}' not found")
        return cast(Signal, next(iter(matched.values())))

    def get_scope(
        self,
        path: str,
        root_scope: Scope | None = None,
    ) -> Scope:
        """Return the scope at an exact hierarchy path.

        Parameters
        ----------
        path:
            Exact dotted scope path. Matcher expressions and terminal signal
            ranges are not accepted.
        root_scope:
            If provided, resolve *path* within this scope instead of starting
            from the file's top-level scopes.

        Returns
        -------
        Scope
            The exact matched scope.

        Raises
        ------
        ValueError
            If *path* contains matcher syntax, contains a terminal range, or
            the scope does not exist.
        """
        steps = parse_query_path(path)
        if any(
            not isinstance(step.matcher, ExactMatcher)
            or step.matcher.target != 'name'
            or step.recursive
            or step.native_recursive
            for step in steps
        ):
            raise ValueError(
                'get_scope() requires an exact path; '
                'use get_matched_scopes() for pattern queries'
            )
        search_root = root_scope or _SearchRoot(reader=self, top_scopes=self.top_scopes)
        matched = search_root._match_path(
            steps,
            lambda node, _remaining: isinstance(node, Scope),
        )
        if not matched:
            raise ValueError(f"scope '{path}' not found")
        return cast(Scope, next(iter(matched.values())))

    def get_matched_nodes(
        self,
        path: str,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Capture, ...], Node]:
        """Return all nodes whose paths match *path*, keyed by captures.

        Like ``get_matched_signals`` / ``get_matched_scopes`` but returns
        every matching node regardless of kind — signals, scopes, and
        composite signals alike.
        """
        search_root = root_scope or _SearchRoot(reader=self, top_scopes=self.top_scopes)
        return search_root.get_matched_nodes(path)

    def get_matched_signals(
        self,
        path: str,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Capture, ...], Signal]:
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
        dict[tuple[Capture, ...], Signal]:
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
        search_root = root_scope or _SearchRoot(reader=self, top_scopes=self.top_scopes)
        return search_root.get_matched_signals(path)

    def get_matched_scopes(
        self,
        path: str,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Capture, ...], Scope]:
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
        dict[tuple[Capture, ...], Scope]:
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
        search_root = root_scope or _SearchRoot(reader=self, top_scopes=self.top_scopes)
        return search_root.get_matched_scopes(path)

    def load_matched_waveforms(
        self,
        signal_path: str,
        clock_path: str | None = None,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        start_time: int | None = None,
        end_time: int | None = None,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Capture, ...], Waveform]:
        """Batch-load all signals matching *signal_path*, each paired with its clock.

        Internally calls ``get_matched_signals`` for both *signal_path* and
        *clock_path*, then dispatches ``load_waveform`` for every match.

        Clock assignment rules:

        * **Ambient** — if *clock_path* is omitted (``None``), every matched
          signal uses the ambient ``ClockDomain`` (including its edge and
          window); raises ``RuntimeError`` if none is active.
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
            Clock signal query path.  Must match at least one signal. If
            omitted, uses the ambient ``ClockDomain`` for every match — see
            ``load_waveform``.
        xz_value, signed, sample_on_posedge, start_time, end_time, start_cycle, end_cycle:
            Forwarded to ``load_waveform`` for every loaded signal. Ignored
            when *clock_path* is omitted (the ambient domain's fields apply
            instead).
        root_scope:
            If provided, both *signal_path* and *clock_path* are searched within
            this scope instead of the file's top-level scopes.

        Returns
        -------
        dict[tuple[Capture, ...], Waveform]:
            Same keys as ``get_matched_signals`` on *signal_path*.

        Raises
        ------
        RuntimeError:
            If *clock_path* is omitted and no ``ClockDomain`` is active.
        ValueError:
            If *clock_path* matches no signals, or if no clock key is a prefix
            of a signal key.
        """
        self._validate_xz_value(xz_value)
        resolved_clock, sampling = self._resolve_ambient_clock(
            clock_path,
            sample_on_posedge=sample_on_posedge,
            start_time=start_time,
            end_time=end_time,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
        )
        clock_pairing = self._resolve_clock_pairing(signal_path, resolved_clock, root_scope)
        matched_signals = self.get_matched_signals(signal_path, root_scope=root_scope)
        load_kwargs: dict[str, Any] = dict(xz_value=xz_value, signed=signed, **sampling)
        return {
            k: self.load_waveform(sig, clock_pairing[k], **load_kwargs)
            for k, sig in matched_signals.items()
        }

    def load_matched_unknown_masks(
        self,
        signal_path: str,
        clock_path: str | None = None,
        include_x: bool = True,
        include_z: bool = True,
        sample_on_posedge: bool = False,
        start_time: int | None = None,
        end_time: int | None = None,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Capture, ...], Waveform]:
        """Batch-load X/Z mask waveforms for all signals matching *signal_path*.

        Clock assignment follows ``load_matched_waveforms``: an omitted
        *clock_path* uses the ambient domain for every match; otherwise a
        single matched clock is broadcast to all signals, or the
        longest-prefix clock key is selected for each signal key.

        Parameters
        ----------
        signal_path:
            Signal query path.  See class docstring.
        clock_path:
            Clock signal query path.  Must match at least one signal. If
            omitted, uses the ambient ``ClockDomain`` for every match.
        include_x:
            If ``True`` (default), mark source ``X``/``x`` bits.
        include_z:
            If ``True`` (default), mark source ``Z``/``z`` bits.
        sample_on_posedge, start_time, end_time, start_cycle, end_cycle:
            Same sampling/windowing semantics as ``load_waveform``. Ignored
            when *clock_path* is omitted.
        root_scope:
            If provided, both *signal_path* and *clock_path* are searched within
            this scope instead of the file's top-level scopes.

        Returns
        -------
        dict[tuple[Capture, ...], Waveform]:
            Same keys as ``get_matched_signals`` on *signal_path*.

        Raises
        ------
        RuntimeError:
            If *clock_path* is omitted and no ``ClockDomain`` is active.
        """
        resolved_clock, sampling = self._resolve_ambient_clock(
            clock_path,
            sample_on_posedge=sample_on_posedge,
            start_time=start_time,
            end_time=end_time,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
        )
        clock_pairing = self._resolve_clock_pairing(signal_path, resolved_clock, root_scope)
        matched_signals = self.get_matched_signals(signal_path, root_scope=root_scope)
        mask_kwargs: dict[str, Any] = dict(include_x=include_x, include_z=include_z, **sampling)
        return {
            k: self.load_unknown_mask(sig, clock_pairing[k], **mask_kwargs)
            for k, sig in matched_signals.items()
        }

    @staticmethod
    def _validate_xz_value(xz_value: int) -> None:
        if xz_value not in (0, 1):
            raise ValueError('xz_value must be 0 or 1')

    def _resolve_clock_pairing(
        self,
        signal_path: str,
        clock: Signal | str,
        root_scope: Scope | None,
    ) -> dict[tuple[Capture, ...], Signal]:
        """Resolve signal/clock query paths into a {signal_key: clock_signal} map.

        Rules:
        - A concrete clock ``Signal``: broadcast to all signals.
        - Single clock match: broadcast to all signals.
        - Multiple clock matches: longest-prefix clock key per signal key.
        - No prefix match for a signal: raise ValueError.
        """
        if isinstance(clock, Signal):
            matched_signals = self.get_matched_signals(signal_path, root_scope=root_scope)
            return {k: clock for k in matched_signals}
        matched_clocks = self.get_matched_signals(clock, root_scope=root_scope)
        if not matched_clocks:
            raise ValueError(f'clock path {clock!r} matched no signals')

        matched_signals = self.get_matched_signals(signal_path, root_scope=root_scope)

        if len(matched_clocks) == 1:
            clock_signal = next(iter(matched_clocks.values()))
            return {k: clock_signal for k in matched_signals}

        clock_keys = list(matched_clocks.keys())
        pairing: dict[tuple[Capture, ...], Signal] = {}
        for sig_key in matched_signals:
            best_len = -1
            best_clock_key: tuple[Capture, ...] | None = None
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
        start_time: int | None = None,
        end_time: int | None = None,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
        mode: Literal['single', 'zip'] = 'single',
        root_scope: Scope | None = None,
    ) -> Waveform | dict[tuple[Capture, ...], Waveform]:
        """Evaluate a waveform expression containing physical signal paths.

        Parameters
        ----------
        expr:
            Expression string. Signal paths may be used as operands or as
            arguments to registered expression functions.
        clock:
            Clock signal used for all waveform loads.
        xz_value, signed, sample_on_posedge, start_time, end_time, start_cycle,
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
        Waveform or dict[tuple[Capture, ...], Waveform]
            The evaluated waveform, or one waveform per zip key.
        """
        self._validate_xz_value(xz_value)
        substituted, path_entries = parse_expression(expr)
        load_kwargs: dict[str, Any] = dict(
            xz_value=xz_value,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            start_time=start_time,
            end_time=end_time,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            root_scope=root_scope,
        )

        loaded_per_path: list[tuple[str, str, dict[tuple[Capture, ...], Waveform]]] = []
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
            result: dict[tuple[Capture, ...], Waveform] = {}
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
