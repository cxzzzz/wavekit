from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    name: str
    width: Optional[int]
    signed: bool

    def __str__(self) -> str:
        return f"Signal(name='{self.name}', width={self.width}, signed={self.signed})"
