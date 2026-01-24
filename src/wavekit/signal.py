from __future__ import annotations

from collections.abc import Callable


class Signal:
    name: str
    signed: bool
    width_resolver: Callable[[], int] | None
    _width: int | None

    def __init__(
        self,
        name: str,
        width: int | None,
        signed: bool,
        width_resolver: Callable[[], int] | None = None,
    ):
        self.name = name
        self._width = width
        self.signed = signed
        self.width_resolver = width_resolver

    @property
    def width(self) -> int | None:
        if self._width is None and self.width_resolver is not None:
            self._width = self.width_resolver()
        return self._width

    @width.setter
    def width(self, value: int | None):
        self._width = value

    def __str__(self) -> str:
        return f"Signal(name='{self.name}', width={self.width}, signed={self.signed})"
