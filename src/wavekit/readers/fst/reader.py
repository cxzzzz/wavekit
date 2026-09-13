"""FST waveform reader."""

from __future__ import annotations

from ..wellen.reader import WellenReader


class FstReader(WellenReader):
    """Read FST waveform files via the wellen backend."""

    _EXPECTED_FORMAT = 'FST'
