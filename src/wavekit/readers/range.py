"""HDL range value type used by hierarchy nodes and query matchers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Range:
    """An HDL index range whose direction is preserved as ``start:end``."""

    start: int
    end: int

    def __str__(self) -> str:
        if self.start == self.end:
            return f'[{self.start}]'
        return f'[{self.start}:{self.end}]'
