from __future__ import annotations

from collections.abc import Callable


class Signal:
    """Metadata descriptor for a single hardware signal.

    Stores the signal's full hierarchical name, bit-width, and signedness.
    Width resolution is lazy: if ``width`` is not provided at construction time,
    it is computed on first access via ``width_resolver`` (a callable that reads
    the actual width from the underlying waveform file).  This avoids eagerly
    querying file I/O for every signal in a large scope tree.

    Attributes
    ----------
    name:
        Full dotted signal path, e.g. ``"tb.dut.data_out[7:0]"``.
    signed:
        Whether the signal value should be interpreted as a two's-complement
        signed integer.
    width_resolver:
        Optional zero-argument callable that returns the bit-width.  Only
        called once; the result is cached in ``_width``.
    """

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
