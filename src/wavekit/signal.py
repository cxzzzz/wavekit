from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    """Metadata descriptor for a single hardware signal.

    Stores the signal's local name, full hierarchical path, bit-width,
    declared bit-range, and signedness.

    Attributes
    ----------
    name:
        Local signal identifier as it appears within its parent scope,
        matching the form used in the waveform file.  May include a range
        suffix when the file stores the signal with one, e.g. ``"data[7:0]"``
        or ``"mem[3][7:0]"``.  Scalar signals have no suffix, e.g. ``"clk"``.
        Invariant: ``full_name == parent_scope_path + "." + name``.
    full_name:
        Complete hierarchical signal path, e.g. ``"tb.dut.data[7:0]"`` or
        ``"tb.dut.mem[3][7:0]"``.  Equal to ``parent_scope_path + "." + name``.
        Pass this directly to :meth:`~wavekit.readers.base.Reader.load_waveform`.
    width:
        Bit-width of the signal, e.g. ``8`` for ``[7:0]``.  ``None`` if not
        yet resolved.
    range:
        The declared or user-requested bit-range of the signal as a
        ``(high, low)`` integer tuple, e.g. ``(7, 0)`` for ``[7:0]``.
        For single-bit selection ``[n]`` this is stored as ``(n, n)``.
        ``None`` if the signal is scalar or the format does not expose
        range information.
    signed:
        Whether the signal value should be interpreted as a two's-complement
        signed integer.  Defaults to ``False``.
    """

    name: str
    full_name: str
    width: int | None
    range: tuple[int, int] | None
    signed: bool = False

    def __str__(self) -> str:
        return (
            f"Signal(name='{self.name}', full_name='{self.full_name}', "
            f'width={self.width}, signed={self.signed}, range={self.range})'
        )
