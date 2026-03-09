from __future__ import annotations

import re

# Matches a waveform signal path token in an expression.
# A valid path is one of:
#   - Starts with $$ or $ (module search)
#   - Starts with @ (regex pattern)
#   - Contains at least one . separator (hierarchical path)
#
# Path structure:
#   [$$|$|@]  first_ident  (.segment[{brace}])*  ([N:M]|[N])?
#
# The trailing [N:M] / [N] is consumed as the signal range (not a bit-slice).
# Any further [N:M] after the placeholder in the resulting expression will be
# handled as a Python subscript on the Waveform object (i.e. bit-slicing).
_WAVE_PATH_RE = re.compile(
    r'(?P<path>'
    r'(?:[$]{1,2}|@)?'  # optional prefix $$ / $ / @
    r'(?:[A-Za-z_][A-Za-z0-9_]*)'  # first identifier segment
    r'(?:[.][A-Za-z_][A-Za-z0-9_{}.,]*)*'  # dot-separated segments (allow braces, no [ ])
    r'(?:\[\d+(?::\d+)?\])?'  # optional trailing [N:M] or [N] range
    r')'
)


def _is_wave_path(token: str) -> bool:
    """Return True if token looks like a waveform signal path."""
    if token.startswith('$') or token.startswith('@'):
        return True
    # Must contain a dot (hierarchy) but not look like a floating-point literal
    if '.' in token:
        try:
            float(token)
            return False  # it's a numeric literal like "1.0"
        except ValueError:
            return True
    return False


def extract_wave_paths(expr: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse *expr* and replace wave path tokens with placeholder identifiers.

    Returns
    -------
    substituted_expr : str
        The expression with every wave path replaced by ``__wave_N__``.
    paths : list of (placeholder, original_path)
        The ordered list of substitutions made.
    """
    paths: list[tuple[str, str]] = []
    counter = 0

    def replacer(m: re.Match) -> str:
        nonlocal counter
        token = m.group('path')
        if not _is_wave_path(token):
            return token
        placeholder = f'__wave_{counter}__'
        paths.append((placeholder, token))
        counter += 1
        return placeholder

    substituted = _WAVE_PATH_RE.sub(replacer, expr)
    return substituted, paths
