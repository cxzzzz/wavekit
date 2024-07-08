import numpy as np
import re
from functools import reduce
from abc import abstractmethod
from numba import njit
from dataclasses import dataclass
from typing import Self, Union
from vcdvcd import VCDVCD, Scope as VcdVcdScope, Signal as VcdVcdSignal
from .waveform import Waveform
from .utils import expand_pattern


@dataclass
class Scope:

    name: str

    @abstractmethod
    def child_scope_list(self) -> list[Self]:
        pass

    @abstractmethod
    def parent_scope(self) -> Union[Self, None]:
        pass

    @abstractmethod
    def signal_list(self) -> list[str]:
        pass

    def full_name(self) -> list[str]:
        ancestors, parent = [self], self
        while (parent := parent.parent_scope()) is not None:
            ancestors.append(parent)
        return ".".join(reversed(ancestors))


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
    ):
        valye_type = np.object_ if (width > 64) else (np.int64 if signed else np.uint64)
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

        return Waveform(value=value, clock=clock, time=time, width=width, signed=signed)

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
    ) -> dict[tuple, Waveform]:

        expanded_pattern_list: list[dict[any, str]] = [
            expand_pattern(p) for p in pattern.split(".")
        ]

        expanded_clock_pattern_list: list[dict[any, str]] = [
            expand_pattern(p) for p in clock_pattern.split(".")
        ]

        def helper(
            scope: Scope, descendant_scope_pattern_list: dict[tuple[any], str]
        ) -> dict[tuple, str]:

            res = dict()

            expanded_scope_patterns = {
                k: re.compile(p) for k, p in descendant_scope_pattern_list[0].items()
            }

            for k, p in expanded_scope_patterns.items():
                if match := p.match(scope.name):
                    match_groups = iter(match.groups())
                    new_k = tuple(
                        next(match_groups) if item is None else item for item in k
                    )

                    if len(descendant_scope_pattern_list) == 2:

                        expanded_signal_patterns = {
                            k: re.compile(p)
                            for k, p in descendant_scope_pattern_list[1].items()
                        }

                        for signal in scope.signal_list():
                            for k, p in expanded_signal_patterns.items():
                                if match := p.match(signal):
                                    match_groups = iter(match.groups())
                                    new_signal_k = tuple(
                                        next(match_groups) if item is None else item
                                        for item in k
                                    )
                                    res[new_k + new_signal_k] = f"{scope.name}.{signal}"
                                    break
                    else:
                        for ck, cs in reduce(
                            lambda a, b: {**a, **b},
                            [
                                helper(child_scope, descendant_scope_pattern_list[1:])
                                for child_scope in scope.child_scope_list()
                            ],
                        ).items():
                            res[new_k + ck] = f"{scope.name}.{cs}"
                    break
            return res

        clock_patterns = reduce(
            lambda a, b: a + b,
            [
                helper(scope, expanded_clock_pattern_list)
                for scope in self.top_scope_list()
            ],
        )
        if len(clock_patterns) > 1:
            raise Exception("clock pattern can not match more than one signal")
        if len(clock_patterns) == 0:
            raise Exception("clock pattern can not match any signal")
        clock_full_name = list(clock_patterns.values())[0]

        return {
            k: self.load_wave(
                s,
                clock_full_name,
                xz_value=xz_value,
                signed=signed,
                sample_on_posedge=sample_on_posedge,
            )
            for k, s in reduce(
                lambda a, b: {**a, **b},
                [
                    helper(scope, expanded_pattern_list)
                    for scope in self.top_scope_list()
                ],
            ).items()
        }


class FsdbReader(Reader):

    def __init__(self, file: str):
        pass

    def load_wave(self, signal: str) -> Waveform:
        pass


class VcdScope(Scope):
    @staticmethod
    def load_signal_list(signal_list: list, scope_list: list) -> list[Self]:
        scopes = {}
        for scope in scope_list:
            ancestors = scope.split(".")
            full_name = ""
            parent_scope = None
            for scope_name in ancestors:
                full_name = full_name + scope_name
                if full_name not in scopes:
                    if full_name not in scopes:
                        new_scope = VcdScope(scope_name, parent_scope=parent_scope)
                        if parent_scope is not None:
                            parent_scope._child_scopes[scope_name] = new_scope
                        scopes[full_name] = new_scope

                parent_scope = scopes[full_name]
                full_name = full_name + "."

        for signal in signal_list:
            scope_name = ".".join(signal.split(".")[:-1])
            scopes[scope_name]._signals.add(signal.split(".")[-1])

        return [scope for scope in scopes.values() if scope._parent_scope is None]

    def __init__(self, name: str, parent_scope: Self):
        super().__init__(name=name)
        self._parent_scope = parent_scope
        self._child_scopes = {}
        self._signals = set()

    def child_scope_list(self) -> list[Scope]:
        return list(self._child_scopes.values())

    def signal_list(self) -> list[str]:
        return list(self._signals)

    def parent_scope(self) -> Union[None, Scope]:
        return self._parent_scope

    def full_name(self) -> list[str]:
        ancestors, parent = [self], self
        while (parent := parent.parent_scope()) is not None:
            ancestors.append(parent)
        return ".".join(reversed(ancestors))


class VcdReader(Reader):

    def __init__(self, file: str):
        self.file = file
        self.vcd = VCDVCD(file, store_scopes=True)
        self._top_scope_list = VcdScope.load_signal_list(
            self.vcd.signals, self.vcd.scopes
        )

    def top_scope_list(self) -> list[Scope]:
        return self._top_scope_list

    def load_wave(
        self,
        signal: str,
        clock: str,
        xz_value: int = 0,
        signed: bool = False,
        sample_on_posedge: bool = False,
    ) -> Waveform:
        signal_handle = self.vcd[signal]
        signal_value_changes = [
            (v[0], int(re.sub(r"[xXzZ]", str(xz_value), v[1]), 2))
            for v in signal_handle.tv
        ]
        clock_value_changes = [
            (v[0], int(re.sub(r"[xXzZ]", "0", v[1]), 2)) for v in self.vcd[clock].tv
        ]

        return self.value_changes_to_waveform(
            signal_value_changes,
            clock_value_changes,
            width=int(signal_handle.size),
            signed=signed,
            sample_on_posedge=sample_on_posedge,
        )
