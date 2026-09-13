"""VCD waveform reader."""

from __future__ import annotations

from ..wellen.reader import WellenReader


class VcdReader(WellenReader):
    """Read VCD waveform files via the wellen backend."""

    _EXPECTED_FORMAT = 'VCD'
