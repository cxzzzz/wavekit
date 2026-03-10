from __future__ import annotations

import ast
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from ..scope import Scope, match_scopes, match_signals
from ..signal import Signal
from ..waveform import Waveform
from .expr_parser import extract_wave_paths
from .pattern_parser import (
    PatternMap,
    expand_brace_pattern,
    split_by_hierarchy,
)
from .value_change import value_change_to_value_array


class Reader:
    """Abstract base class for waveform file readers.

    Concrete subclasses (:class:`~wavekit.VcdReader`,
    :class:`~wavekit.FsdbReader`) implement the file-format-specific I/O;
    all high-level analysis APIs are provided here.

    Supports the context-manager protocol::

        with VcdReader("sim.vcd") as r:
            wave = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

    Signal path format
    ------------------
    All signal paths use dotted hierarchical notation matching the scope tree
    in the waveform file, e.g. ``"tb.dut.sub.signal_name[7:0]"``.
    If the bit-range suffix is omitted and the file stores the signal with a
    range, the range is appended automatically.

    Pattern syntax (used by :meth:`get_matched_signals`,
    :meth:`load_matched_waveforms`, :meth:`eval`)
    -------------------------------------------------
    * ``{a,b,c}``     — matches ``a``, ``b``, or ``c``; captures each as a key.
    * ``{0..7}``       — integer range 0 to 7 inclusive; step defaults to 1.
    * ``{0..7..2}``    — integer range with explicit step (0, 2, 4, 6).
    * ``@<regex>``     — prefix a path component with ``@`` to use a Python
      regex instead of exact matching; capture groups ``(...)`` become tuple
      elements in the result key.

    All pattern expansions produce ``dict[tuple, str]`` where the tuple key
    encodes the captured values and the value is the full signal path.
    """

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # exception wont be suppressed
        return False

    @abstractmethod
    def load_waveform(
        self,
        signal: Signal | str,
        clock: Signal | str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> Waveform:
        """Load a single signal as a clock-synchronised :class:`~wavekit.Waveform`.

        The signal is sampled on every **negedge** of *clock* by default
        (i.e. the value is captured at each falling edge of the clock, which
        reflects the value that was stable during the preceding high phase).
        Set ``sample_on_posedge=True`` to sample on rising edges instead.

        Parameters
        ----------
        signal:
            Full dotted path of the signal as a :class:`~wavekit.signal.Signal`
            object or a string.  When a ``Signal`` is passed, ``signal.full_name``
            is used as the path (which may include bit-range suffixes).
            When a string is passed, the value is used verbatim as the full
            hierarchical path, e.g. ``"tb.dut.data[7:0]"`` or ``"tb.dut.data"``.
        clock:
            Clock signal as a :class:`~wavekit.signal.Signal` or full dotted
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
            start of simulation.
        end_time:
            Simulation time to stop loading at (exclusive).  ``None`` means
            end of simulation.

        Returns
        -------
        Waveform:
            One sample per clock edge within the requested time range.
        """

    @abstractmethod
    def top_scope_list(self) -> Sequence[Scope]:
        """Return the top-level :class:`~wavekit.scope.Scope` nodes of the file.

        Each element corresponds to one root module in the waveform hierarchy
        (typically just one, e.g. the testbench).  Traverse the tree via
        :attr:`~wavekit.scope.Scope.child_scope_list` and
        :attr:`~wavekit.scope.Scope.signal_list` for custom scope inspection.
        """

    @staticmethod
    def value_change_to_waveform(
        value_change: np.ndarray,
        clock_changes: np.ndarray,
        width: int | None,
        signed: bool,
        sample_on_posedge: bool = False,
        signal: str = '',
    ) -> Waveform:
        value, clock, time = value_change_to_value_array(
            value_change, clock_changes, sample_on_posedge=sample_on_posedge
        )

        return Waveform(
            value=value,
            clock=clock,
            time=time,
            signal=Signal(name=signal, full_name=signal, width=width, range=None, signed=signed),
        )

    def _search_roots(self, root_scope: Scope | None) -> Sequence[Scope]:
        """Return the list of scopes to start a search from."""
        return [root_scope] if root_scope is not None else self.top_scope_list()

    def get_matched_signals(
        self,
        pattern: str,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Any, ...], Signal]:
        """Return all signals whose paths match *pattern*, keyed by captured values.

        Traverses the scope tree starting from *root_scope* (or the file's
        top-level scopes if *root_scope* is ``None``) and applies the pattern
        to each level.  See the class docstring for pattern syntax details.

        Parameters
        ----------
        pattern:
            Signal path pattern, e.g. ``"tb.dut.fifo_{0..3}.w_ptr[2:0]"`` or
            ``r"tb.dut.@([a-z]+)_valid"``.
        root_scope:
            If provided, search only within this scope instead of starting
            from the file's top-level scopes.

        Returns
        -------
        dict[tuple, Signal]:
            Maps each captured key tuple to the matched :class:`~wavekit.signal.Signal`
            object (carrying name, width, range, signed).
            For patterns without capture groups the key is ``()``.

        Raises
        ------
        Exception:
            If two different signals resolve to the same key.
        """
        def combine_dict(
            dict1: dict[tuple[Any, ...], Signal],
            dict2: dict[tuple[Any, ...], Signal],
        ) -> dict[tuple[Any, ...], Signal]:
            common_keys = set(dict1.keys()).intersection(dict2.keys())

            if common_keys:
                signal1s = [dict1[k].name for k in common_keys]
                signal2s = [dict2[k].name for k in common_keys]
                raise Exception(
                    'found more than one signal with the same keys: '
                    f'keys:{list(common_keys)} , signals:{signal1s + signal2s}'
                )

            return {**dict1, **dict2}

        pattern = pattern.strip()
        expanded_pattern_list: list[PatternMap] = [
            expand_brace_pattern(p) for p in split_by_hierarchy(pattern)
        ]

        matched_signals: dict[tuple[Any, ...], Signal] = {}
        for scope in self._search_roots(root_scope):
            matched_signals = combine_dict(
                matched_signals,
                match_signals(scope, expanded_pattern_list),
            )
        return matched_signals

    def get_matched_scope(
        self,
        pattern: str,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Any, ...], Scope]:
        """Return all scopes whose paths match *pattern*, keyed by captured values.

        Similar to :meth:`get_matched_signals` but stops at the scope level —
        the last component of *pattern* must match a scope name, not a signal.
        Useful for enumerating module instances before loading their signals.

        Parameters
        ----------
        pattern:
            Scope path pattern using the same brace/regex syntax as signal
            patterns.  The last component must match a scope (module) name,
            e.g. ``"tb.dut.fifo_{0..3}"`` or ``r"tb.@([a-z]+)_core"``.
        root_scope:
            If provided, search only within this scope instead of starting
            from the file's top-level scopes.

        Returns
        -------
        dict[tuple, Scope]:
            Maps each captured key tuple to the matched :class:`~wavekit.scope.Scope`.
            For patterns without capture groups the key is ``()``.

        Raises
        ------
        Exception:
            If two different scopes resolve to the same key.

        Example
        -------
        ::

            # Find all fifo_N sub-scopes under tb.dut
            scopes = reader.get_matched_scope("tb.dut.fifo_{0..3}")
            for (idx,), scope in scopes.items():
                waves = reader.load_matched_waveforms(
                    "w_ptr[2:0]", clock_pattern="clk", root_scope=scope
                )
        """
        def combine_dict(
            dict1: dict[tuple[Any, ...], Scope],
            dict2: dict[tuple[Any, ...], Scope],
        ) -> dict[tuple[Any, ...], Scope]:
            common_keys = set(dict1.keys()).intersection(dict2.keys())
            if common_keys:
                raise Exception(
                    'found more than one scope with the same keys: '
                    f'keys:{list(common_keys)}'
                )
            return {**dict1, **dict2}

        pattern = pattern.strip()
        expanded_pattern_list: list[PatternMap] = [
            expand_brace_pattern(p) for p in split_by_hierarchy(pattern)
        ]

        matched_scopes: dict[tuple[Any, ...], Scope] = {}
        for scope in self._search_roots(root_scope):
            matched_scopes = combine_dict(
                matched_scopes,
                match_scopes(scope, expanded_pattern_list),
            )
        return matched_scopes

    def load_matched_waveforms(
        self,
        pattern: str,
        clock_pattern: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: int | None = None,
        end_time: int | None = None,
        root_scope: Scope | None = None,
    ) -> dict[tuple[Any, ...], Waveform]:
        """Batch-load all signals matching *pattern*, each paired with its clock.

        Internally calls :meth:`get_matched_signals` for both *pattern* and
        *clock_pattern*, then dispatches :meth:`load_waveform` for every match.

        Clock assignment rules:

        * **Single clock** — if *clock_pattern* matches exactly one signal, that
          clock is broadcast to all matched signals.
        * **Per-signal clock** — if *clock_pattern* matches multiple signals,
          its key set must equal the key set of *pattern* exactly; each signal
          is paired with the clock sharing the same key.

        Parameters
        ----------
        pattern:
            Signal path pattern (brace/regex).  See class docstring.
        clock_pattern:
            Clock signal path or pattern.  Must match at least one signal.
        xz_value, signed, sample_on_posedge, begin_time, end_time:
            Forwarded to :meth:`load_waveform` for every loaded signal.
        root_scope:
            If provided, both *pattern* and *clock_pattern* are searched within
            this scope instead of the file's top-level scopes.

        Returns
        -------
        dict[tuple, Waveform]:
            Same keys as :meth:`get_matched_signals` on *pattern*.

        Raises
        ------
        Exception:
            If *clock_pattern* matches no signals, or if per-signal clock keys
            do not match signal keys.

        Example
        -------
        ::

            # Load J_state and J_next sharing a single clock
            waves = reader.load_matched_waveforms(
                "tb.u0.J_{state,next}[3:0]",
                clock_pattern="tb.tck",
            )
            # waves == { ('state',): Waveform, ('next',): Waveform }
        """
        matched_clocks = self.get_matched_signals(clock_pattern, root_scope=root_scope)
        if not matched_clocks:
            raise Exception(f'clock pattern {clock_pattern} can not match any signal')

        matched_signals = self.get_matched_signals(pattern, root_scope=root_scope)

        load_kwargs: dict[str, Any] = dict(
            xz_value=xz_value,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
        )

        if len(matched_clocks) == 1:
            # Broadcast: single clock shared by all matched signals
            clock_full_name = next(iter(matched_clocks.values())).full_name
            return {
                k: self.load_waveform(sig.full_name, clock_full_name, **load_kwargs)
                for k, sig in matched_signals.items()
            }
        else:
            # Per-signal clock: keys must match exactly
            if set(matched_clocks.keys()) != set(matched_signals.keys()):
                raise Exception(
                    f'clock pattern {clock_pattern!r} matched keys {sorted(matched_clocks.keys())} '
                    f'which do not match signal pattern keys {sorted(matched_signals.keys())}'
                )
            return {
                k: self.load_waveform(sig.full_name, matched_clocks[k].full_name, **load_kwargs)
                for k, sig in matched_signals.items()
            }

    @abstractmethod
    def close(self):
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
        mode: Literal['single', 'zip'] = 'single',
        root_scope: Scope | None = None,
    ) -> Waveform | dict[tuple[Any, ...], Waveform]:
        """Evaluate a waveform expression string.

        Wave signal paths embedded in *expr* are automatically extracted,
        loaded, and substituted before evaluating the resulting Python
        arithmetic expression.

        Parameters
        ----------
        expr:
            Expression string.
            Single mode example: ``"tb.dut.w_ptr[3:0] - tb.dut.r_ptr[3:0]"``.
            Zip mode example: ``"tb.dut.fifo_{0..3}.w_ptr[3:0] - tb.dut.fifo_{0..3}.r_ptr[3:0]"``.
        clock:
            Clock signal used for all waveform loads.
        mode:
            ``'single'`` (default) — every path must match exactly one
            signal; returns a single :class:`Waveform`.
            ``'zip'`` — paths matching multiple signals must all share the
            same set of pattern keys; single-match paths are broadcast;
            returns ``dict[tuple, Waveform]``.
        root_scope:
            If provided, all signal paths in *expr* are resolved within this
            scope instead of the file's top-level scopes.
        """
        substituted, path_entries = extract_wave_paths(expr)

        load_kwargs: dict[str, Any] = dict(
            clock=clock,
            xz_value=xz_value,
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            begin_time=begin_time,
            end_time=end_time,
        )

        # Resolve each path to its matched signal(s)
        matched_per_path: list[tuple[str, str, dict[tuple[Any, ...], Signal]]] = []
        for placeholder, path in path_entries:
            matched = self.get_matched_signals(path, root_scope=root_scope)
            if not matched:
                raise ValueError(f"path '{path}' matched no signals")
            matched_per_path.append((placeholder, path, matched))

        if mode == 'single':
            for _placeholder, path, matched in matched_per_path:
                if len(matched) > 1:
                    raise ValueError(
                        f"path '{path}' matched {len(matched)} signals in mode='single',"
                        f" use mode='zip'. Matched: {[sig.full_name for sig in matched.values()]}"
                    )
            ns: dict[str, Any] = {
                placeholder: self.load_waveform(next(iter(matched.values())).full_name, **load_kwargs)
                for placeholder, _, matched in matched_per_path
            }
            try:
                code = compile(ast.parse(substituted, mode='eval'), '<eval_expr>', 'eval')
                return eval(code, {'__builtins__': {}}, ns)  # noqa: S307
            except Exception as exc:
                raise ValueError(
                    f"failed to evaluate expression '{expr}' (substituted: '{substituted}')"
                ) from exc

        elif mode == 'zip':
            multi_paths = [(ph, p, m) for ph, p, m in matched_per_path if len(m) > 1]
            single_paths = [(ph, p, m) for ph, p, m in matched_per_path if len(m) == 1]

            if multi_paths:
                # All multi-match paths must share identical key sets
                ref_ph, ref_p, ref_matched = multi_paths[0]
                ref_keys = set(ref_matched.keys())
                for _ph, p, matched in multi_paths[1:]:
                    if set(matched.keys()) != ref_keys:
                        raise ValueError(
                            f'inconsistent match keys between paths: '
                            f"'{ref_p}' has keys {sorted(ref_keys)}, "
                            f"'{p}' has keys {sorted(matched.keys())}"
                        )
                zip_keys: list[tuple[Any, ...]] = list(ref_keys)
            else:
                # All paths are single-match; degenerate to single behaviour
                zip_keys = [()]

            # Pre-load broadcast waveforms (single-match paths)
            broadcast_ns: dict[str, Waveform] = {
                placeholder: self.load_waveform(next(iter(matched.values())).full_name, **load_kwargs)
                for placeholder, _, matched in single_paths
            }

            result: dict[tuple[Any, ...], Waveform] = {}
            try:
                code = compile(ast.parse(substituted, mode='eval'), '<eval_expr>', 'eval')
            except SyntaxError as exc:
                raise ValueError(
                    f"invalid expression '{expr}' (substituted: '{substituted}')"
                ) from exc

            for key in zip_keys:
                ns = dict(broadcast_ns)
                for placeholder, _, matched in multi_paths:
                    ns[placeholder] = self.load_waveform(matched[key].full_name, **load_kwargs)
                try:
                    result[key] = eval(code, {'__builtins__': {}}, ns)  # noqa: S307
                except Exception as exc:
                    raise ValueError(
                        f"failed to evaluate expression '{expr}' for key {key}"
                    ) from exc

            return result

        else:
            raise ValueError(f"unknown mode '{mode}', expected 'single' or 'zip'")
