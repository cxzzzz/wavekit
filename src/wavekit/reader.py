from __future__ import annotations
import importlib
import numpy as np
import re
from functools import reduce
from abc import abstractmethod
from numba import njit
from dataclasses import dataclass
from typing import Union, Optional
from vcdvcd import VCDVCD, Scope as VcdVcdScope, Signal as VcdVcdSignal
from .waveform import Waveform
from .utils import expand_brace_pattern, split_by_range_expr, split_by_hierarchy


class Scope:

    def __init__(self, name: str):
        self.name = name
        self._module_cache = dict()

    @property
    @abstractmethod
    def child_scope_list(self) -> list[Scope]:
        pass

    @property
    @abstractmethod
    def parent_scope(self) -> Optional[Scope]:
        pass

    @property
    @abstractmethod
    def signal_list(self) -> list[str]:
        pass

    def full_name(self, root: Scope = None) -> list[str]:
        ancestors, parent = [], self
        while parent is not None:
            ancestors.append(parent)
            if parent == root:
                break
            parent = parent.parent_scope
        return ".".join([x.name for x in reversed(ancestors)])

    def find_module_scope(self, module_name: str, depth: int = 0) -> list[Scope]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def begin_time(self) -> int:
        pass

    @property
    @abstractmethod
    def end_time(self) -> int:
        pass


class Reader:

    def __init__(self):

        pass

    @abstractmethod
    def load_wave(
        self,
        signal: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Waveform:
        pass

    @staticmethod
    @njit
    def _fast_value_changes_to_value_list(
        value: np.ndarray,
        value_time: np.ndarray,
        clock: np.ndarray,
        clock_time: np.ndarray,
        sample_on_posedge: bool,
    ) -> (np.ndarray, np.ndarray, np.ndarray):

        vidx = 0
        cidx = 0
        ccnt = 0

        value_res = []
        clock_res = []
        time_res = []

        sample_clock_value = 1 if sample_on_posedge else 0
        for cidx in range(clock.size):
            cvalue = clock[cidx]
            if cvalue == sample_clock_value:
                ctime = clock_time[cidx]
                while vidx + 1 < value.size and value_time[vidx + 1] <= ctime:
                    vidx += 1
                value_res.append(value[vidx])
                clock_res.append(ccnt)
                time_res.append(ctime)

                ccnt += 1

        return (
            np.array(value_res, dtype=np.uint64),
            np.array(clock_res, dtype=np.uint64),
            np.array(time_res, dtype=np.uint64),
        )

    @staticmethod
    def _value_changes_to_value_list(
        value: np.ndarray,
        value_time: np.ndarray,
        clock: np.ndarray,
        clock_time: np.ndarray,
        sample_on_posedge: bool,
    ) -> (np.ndarray, np.ndarray, np.ndarray):

        vidx = 0
        cidx = 0
        ccnt = 0

        value_res = []
        clock_res = []
        time_res = []

        sample_clock_value = 1 if sample_on_posedge else 0
        for cidx in range(clock.size):
            cvalue = clock[cidx]
            if cvalue == sample_clock_value:
                ctime = clock_time[cidx]
                while vidx + 1 < value.size and value_time[vidx + 1] <= ctime:
                    vidx += 1
                value_res.append(value[vidx])
                clock_res.append(ccnt)
                time_res.append(ctime)

                ccnt += 1

        return (
            np.array(value_res, dtype=np.object_),
            np.array(clock_res, dtype=np.uint64),
            np.array(time_res, dtype=np.uint64),
        )

    @staticmethod
    def value_changes_to_waveform(
        value_changes: list[tuple[int, int]],
        clock_changes: list[tuple[int, int]],
        width: str,
        signed: bool,
        sample_on_posedge: bool = False,
        signal: str = "",
    ):
        valye_type = np.object_ if (width > 64) else (
            np.int64 if signed else np.uint64)
        value = np.array([x[1] for x in value_changes], dtype=valye_type)
        value_time = np.array([x[0] for x in value_changes], dtype=np.uint64)

        clock = np.array([x[1] for x in clock_changes], dtype=np.uint64)
        clock_time = np.array([x[0] for x in clock_changes], dtype=np.uint64)

        if width > 64:
            value, clock, time = Reader._value_changes_to_value_list(
                value=value,
                value_time=value_time,
                clock=clock,
                clock_time=clock_time,
                sample_on_posedge=sample_on_posedge,
            )
        else:
            value, clock, time = Reader._fast_value_changes_to_value_list(
                value=value,
                value_time=value_time,
                clock=clock,
                clock_time=clock_time,
                sample_on_posedge=sample_on_posedge,
            )
            if signed:
                value = np.where(
                    value >= (2 ** (width - 1)),
                    ((2 ** (64 - width) - 1) << width) | value,
                    value,
                ).view(dtype=np.int64)

        return Waveform(value=value, clock=clock, time=time, width=width, signed=signed, signal=signal)

    @abstractmethod
    def top_scope_list(self) -> list[Scope]:
        pass

    def load_waves(
        self,
        pattern: str,
        clock_pattern: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> dict[tuple, Waveform]:

        pattern = pattern.strip()
        expanded_pattern_list: list[dict[any, str]] = [
            expand_brace_pattern(p) for p in split_by_hierarchy(pattern)
        ]

        expanded_clock_pattern_list: list[dict[any, str]] = [
            expand_brace_pattern(p) for p in split_by_hierarchy(clock_pattern)
        ]

        def traverse_signal(
            scope: Scope, descendant_scope_pattern_list: dict[tuple[any], str]
        ) -> dict[tuple, str]:

            res = {}
            if len(descendant_scope_pattern_list) == 1:
                for signal in scope.signal_list:
                    for k, p in descendant_scope_pattern_list[0].items():
                        p, range_expr = split_by_range_expr(p)
                        '''
                        if match := re.fullmatch(p,signal):
                            match_groups = iter(match.groups())
                            new_signal_k = tuple(
                                next(match_groups) if item is None else item
                                for item in k
                            )
                        '''
                        # doesnt support regex
                        # regex
                        if p[0] == '@':
                            if match := re.fullmatch(p[1:], signal):
                                assert len(k) == 0
                                # signal must be the last element of the key, to avoid overwriting among signals when pattern does has any group
                                key = (match.groups(),)
                                if key in res:
                                    raise Exception(
                                        f"pattern {pattern} match more than one signal")
                                res[key] = f"{signal}{range_expr}"
                        elif p == signal:
                            key = k
                            assert key not in res
                            res[key] = f"{signal}{range_expr}"
                            break
            return res

        def traverse_scope(
            scope: Scope, descendant_scope_pattern_list: dict[tuple[any], str]
        ) -> dict[tuple, str]:

            res = dict()

            for k, p in descendant_scope_pattern_list[0].items():

                if len(p) >= 2 and p[0] == '$':

                    if p[1] == '$':
                        module_name = p[2:]
                        depth = 0
                    else:
                        module_name = p[1:]
                        depth = 1

                    module_scopes = scope._module_cache[p] if (
                        p in scope._module_cache) else scope.find_module_scope(module_name=module_name, depth=depth)
                    scope._module_cache[p] = module_scopes

                    if len(module_scopes) == 1 and module_scopes[0] == scope:
                        for sk, ss in traverse_signal(scope, descendant_scope_pattern_list[1:]).items():
                            key = (scope.name,) + sk
                            assert key not in res
                            res[key] = f"{scope.name}.{ss}"

                        for child_scope in scope.child_scope_list:
                            for ck, cs in traverse_scope(child_scope, descendant_scope_pattern_list[1:]).items():
                                res[(scope.name,) + ck] = f"{scope.name}.{cs}"
                    else:
                        for child_scope in module_scopes:
                            for ck, cs in traverse_scope(child_scope, descendant_scope_pattern_list[0:]).items():
                                res[(f"{child_scope.parent_scope.full_name(scope)}.{
                                     ck[0]}",) + ck[1:]] = f"{child_scope.parent_scope.full_name(scope)}.{cs}"

                # elif match := re.fullmatch(p,scope.name):
                #    match_groups = iter(match.groups())
                #    new_k = tuple(
                #        next(match_groups) if item is None else item for item in k
                #    )
                else:
                    matched = False
                    if p[0] == '@':  # regex
                        if match := re.fullmatch(p[1:], scope.name):
                            matched = True
                            assert len(k) == 0
                            new_k = (match.groups(),)
                    else:
                        if p == scope.name:
                            matched = True
                            new_k = k

                    if matched:
                        for sk, ss in traverse_signal(scope, descendant_scope_pattern_list[1:]).items():
                            key = new_k + sk
                            if key in res:
                                raise Exception(
                                    f"pattern {pattern} match more than one signal")
                            res[key] = f"{scope.name}.{ss}"

                        for child_scope in scope.child_scope_list:
                            for ck, cs in traverse_scope(child_scope, descendant_scope_pattern_list[1:]).items():
                                key = new_k + ck
                                if key in res:
                                    raise Exception(
                                        f"pattern {pattern} match more than one signal")
                                res[key] = f"{scope.name}.{cs}"
            return res

        clock_patterns = reduce(
            lambda a, b: a + b,
            [
                traverse_scope(scope, expanded_clock_pattern_list)
                for scope in self.top_scope_list()
            ],
        )
        # if len(clock_patterns) > 1:
        #    raise Exception(f"clock pattern {clock_pattern} match more than one signal")
        if len(clock_patterns) == 0:
            raise Exception(f"clock pattern {
                            clock_pattern} can not match any signal")
        clock_full_name = list(clock_patterns.values())[0]

        def combine_dict(dict1, dict2):
            common_keys = set(dict1.keys()).intersection(dict2.keys())

            if common_keys:
                signal1s = [dict1[k] for k in common_keys]
                signal2s = [dict2[k] for k in common_keys]
                raise Exception(f"found more than one signal with the same keys: keys:{
                                list(common_keys.keys())} , signals:{signal1s+signal2s}")

            return {**dict1, **dict2}

        return {
            k: self.load_wave(
                s,
                clock_full_name,
                xz_value=xz_value,
                signed=signed,
                sample_on_posedge=sample_on_posedge,
                begin_time=begin_time,
                end_time=end_time,
            )
            for k, s in reduce(
                combine_dict,
                [
                    traverse_scope(scope, expanded_pattern_list)
                    for scope in self.top_scope_list()
                ],
            ).items()
        }

    @abstractmethod
    def close(self):
        pass


class VcdScope(Scope):
    @staticmethod
    def from_signal_list(signal_list: list, scope_list: list) -> list[VcdScope]:
        scopes = {}
        for scope in scope_list:
            ancestors = scope.split(".")
            full_name = ""
            parent_scope = None
            for scope_name in ancestors:
                full_name = full_name + scope_name
                if full_name not in scopes:
                    if full_name not in scopes:
                        new_scope = VcdScope(
                            scope_name, parent_scope=parent_scope)
                        if parent_scope is not None:
                            parent_scope._child_scopes[scope_name] = new_scope
                        scopes[full_name] = new_scope

                parent_scope = scopes[full_name]
                full_name = full_name + "."

        for signal in signal_list:
            scope_name = ".".join(signal.split(".")[:-1])
            scopes[scope_name]._signals.add(signal.split(".")[-1])

        return [scope for scope in scopes.values() if scope._parent_scope is None]

    def __init__(self, name: str, parent_scope: VcdScope):
        super().__init__(name=name)
        self._parent_scope = parent_scope
        self._child_scopes = {}
        self._signals = set()

    @property
    def child_scope_list(self) -> list[Scope]:
        return list(self._child_scopes.values())

    @property
    def signal_list(self) -> list[str]:
        return list(self._signals)

    @property
    def parent_scope(self) -> Optional[Scope]:
        return self._parent_scope


class VcdReader(Reader):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.file_handle = VCDVCD(file, store_scopes=True)
        self._top_scope_list = VcdScope.from_signal_list(
            self.file_handle.signals, self.file_handle.scopes
        )

    def top_scope_list(self) -> list[Scope]:
        return self._top_scope_list

    @property
    def begin_time(self) -> str:
        return self.file_handle.begintime

    @property
    def end_time(self) -> str:
        return self.file_handle.endtime

    def load_wave(
        self,
        signal: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Waveform:

        if begin_time is not None:
            raise NotImplementedError("begin_time is not supported")
        if end_time is not None:
            raise NotImplementedError("end_time is not supported")

        signal_handle = self.file_handle[signal]
        signal_value_changes = [
            (v[0], int(re.sub(r"[xXzZ]", str(xz_value), v[1]), 2))
            for v in signal_handle.tv
        ]
        clock_value_changes = [
            (v[0], int(re.sub(r"[xXzZ]", "0", v[1]), 2)) for v in self.file_handle[clock].tv
        ]

        return self.value_changes_to_waveform(
            signal_value_changes,
            clock_value_changes,
            width=int(signal_handle.size),
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=signal
        )

    def close(self):
        pass


class FsdbScope(Scope):
    def __init__(self, handle, parent_scope: FsdbScope):
        super().__init__(name=handle.name())
        self.handle = handle
        self._parent_scope = parent_scope

    @property
    def child_scope_list(self) -> list[Scope]:
        return [FsdbScope(c, self) for c in self.handle.child_scope_list()]

    @property
    def signal_list(self) -> list[str]:
        return [s.name() for s in self.handle.sig_list()]

    @property
    def parent_scope(self) -> Optional[Scope]:
        return self._parent_scope

    @property
    def type(self) -> str:
        return self.handle.type(isEnum=False)

    @property
    def def_name(self) -> Optional[str]:
        return self.handle.def_name()

    def find_module_scope(self, module_name: str, depth: int = 0) -> list[Scope]:
        if self.type == 'npiFsdbScopeSvModule' and self.def_name == module_name:
            return [self]
        elif depth == 1:
            return []
        else:  # depth == 0 or depth > 1
            return list(reduce(lambda a, b: a + b, [c.find_module_scope(module_name, depth - 1) for c in self.child_scope_list], []))


class FsdbReader(Reader):
    pynpi = {}

    def __init__(self, file: str):
        super().__init__()
        if len(FsdbReader.pynpi) == 0:
            import sys
            import os
            rel_lib_path = os.environ["VERDI_HOME"] + "/share/NPI/python"
            sys.path.append(os.path.abspath(rel_lib_path))
            FsdbReader.pynpi['npisys'] = importlib.import_module(
                "pynpi.npisys")
            FsdbReader.pynpi['waveform'] = importlib.import_module(
                "pynpi.waveform")
            FsdbReader.pynpi['npisys'].init([''])

        self.file = file
        self.file_handle = FsdbReader.pynpi['waveform'].open(file)

    def load_wave(
        self,
        signal: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
        begin_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Waveform:

        format = FsdbReader.pynpi['waveform'].VctFormat_e.BinStrVal
        begin_time = begin_time or 0
        end_time = end_time or 2**64-1

        signal_handle = self.file_handle.sig_by_name(signal)
        signal_value_changes = FsdbReader.pynpi['waveform'].sig_hdl_value_between(
            signal_handle, begin_time, end_time, format=format)
        signal_value_changes = [
            (v[0], int(re.sub(r"[xXzZ]", str(xz_value), v[1]), 2))
            for v in signal_value_changes
        ]

        clock_handle = self.file_handle.sig_by_name(clock)
        clock_value_changes = FsdbReader.pynpi['waveform'].sig_hdl_value_between(
            clock_handle, begin_time, end_time, format=format)
        clock_value_changes = [
            (v[0], int(re.sub(r"[xXzZ]", "0", v[1]), 2)) for v in clock_value_changes
        ]

        return self.value_changes_to_waveform(
            signal_value_changes,
            clock_value_changes,
            width=signal_handle.range_size(),
            signed=signed,
            sample_on_posedge=sample_on_posedge,
            signal=signal
        )

    def top_scope_list(self) -> list[Scope]:
        return [FsdbScope(s, None) for s in self.file_handle.top_scope_list()]

    @property
    def begin_time(self) -> str:
        return self.file_handle.min_time()

    @property
    def end_time(self) -> str:
        return self.file_handle.max_time()

    def close(self):
        FsdbReader.pynpi['waveform'].close(self.file_handle)
