"""Protocol pattern matching for waveform analysis."""

from .dsl import Pattern as Pattern
from .dsl import collect as collect
from .dsl import match as match
from .errors import PatternError as PatternError
from .result import MatchPoint as MatchPoint
from .result import MatchRecord as MatchRecord
from .result import MatchRecords as MatchRecords
from .result import MatchStatus as MatchStatus
from .steps import Channel as Channel

__all__ = [
    'Pattern',
    'match',
    'collect',
    'MatchPoint',
    'MatchRecord',
    'MatchRecords',
    'MatchStatus',
    'PatternError',
    'Channel',
]
