"""Parsing and evaluating WaveKit waveform expressions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lark import Lark, Transformer

_EXPRESSION_GRAMMAR = r"""
?start: expr
?expr: bit_or
?bit_or: bit_xor | bit_or VBAR bit_xor
?bit_xor: bit_and | bit_xor CIRCUMFLEX bit_and
?bit_and: comparison | bit_and AMPERSAND comparison
?comparison: shift
           | comparison EQEQ shift
           | comparison NEQ shift
           | comparison LE shift
           | comparison LT shift
           | comparison GE shift
           | comparison GT shift
?shift: sum | shift LSHIFT sum | shift RSHIFT sum
?sum: product | sum PLUS product | sum MINUS product
?product: unary
        | product STAR unary
        | product SLASH unary
        | product FLOOR unary
        | product PERCENT unary
        | product POWER unary
?unary: TILDE unary | PLUS unary | MINUS unary | atom
?atom: call | path | NUMBER | group
group: LPAR expr RPAR
call: NAME LPAR [expr (COMMA expr)*] RPAR
path: segment (DOT segment)* SUBSCRIPT*
segment: segment_text | WILDCARD
segment_text: segment_piece+
?segment_piece: NAME | MODULE | BRACE | REGEX

MODULE: /\${1,2}/
BRACE: /\{[^{}]*\}/
WILDCARD.3: "**" | "*"
REGEX: /\/(?:\\.|[^\/\\])*\//
SUBSCRIPT: /\[\d+(?::\d+)?\]/
NAME: /[A-Za-z_][A-Za-z0-9_]*/
HEX_NUMBER.2: /0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*/
BIN_NUMBER.2: /0[bB][01](?:_?[01])*/
OCT_NUMBER.2: /0[oO][0-7](?:_?[0-7])*/
DECIMAL_FLOAT_NUMBER.2: /(?:\d(?:_?\d)*)?\.(?:\d(?:_?\d)*)?(?:[eE][+-]?\d(?:_?\d)*)?/
EXP_NUMBER.2: /\d(?:_?\d)*[eE][+-]?\d(?:_?\d)*/
DEC_NUMBER.1: /\d(?:_?\d)*/
NUMBER: HEX_NUMBER | BIN_NUMBER | OCT_NUMBER | DECIMAL_FLOAT_NUMBER | EXP_NUMBER | DEC_NUMBER

VBAR: "|"
CIRCUMFLEX: "^"
AMPERSAND: "&"
EQEQ: "=="
NEQ: "!="
LE: "<="
LT: "<"
GE: ">="
GT: ">"
LSHIFT: "<<"
RSHIFT: ">>"
PLUS: "+"
MINUS: "-"
STAR: "*"
SLASH: "/"
FLOOR: "//"
PERCENT: "%"
POWER: "**"
TILDE: "~"
LPAR: "("
RPAR: ")"
COMMA: ","
DOT: "."

%import common.WS_INLINE
%ignore WS_INLINE
"""

_EXPRESSION_PARSER = Lark(
    _EXPRESSION_GRAMMAR,
    parser='lalr',
    lexer='contextual',
)
_EXPRESSION_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def expression_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register a Waveform method as callable from an expression string."""
    _EXPRESSION_FUNCTIONS[func.__name__] = func
    return func


class ExpressionTransformer(Transformer):
    def __init__(self, *, function_names: set[str]):
        super().__init__()
        self.function_names = function_names
        self.path_entries: list[tuple[str, str]] = []

    @staticmethod
    def join(items: list[Any]) -> str:
        return ''.join(str(item) for item in items)

    def call(self, items: list[Any]) -> str:
        name = str(items[0])
        if name not in self.function_names:
            raise ValueError(f'unknown expression function {name!r}')
        return self.join(items)

    def path(self, items: list[Any]) -> str:
        path = self.join(items)
        placeholder = f'__wave_{len(self.path_entries)}__'
        self.path_entries.append((placeholder, path))
        return placeholder

    def __default__(self, data: str, children: list[Any], meta: Any) -> Any:
        return self.join(children)

    def __default_token__(self, token: Any) -> str:
        return str(token)


def parse_expression(expr: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse an expression and replace each signal path with a placeholder.

    Parameters
    ----------
    expr:
        WaveKit expression containing physical signal paths.

    Returns
    -------
    tuple[str, list[tuple[str, str]]]
        The substituted expression and ordered ``(placeholder, path)`` entries.
    """
    tree = _EXPRESSION_PARSER.parse(expr)
    transformer = ExpressionTransformer(
        function_names=set(_EXPRESSION_FUNCTIONS),
    )
    result = transformer.transform(tree)
    return str(result), transformer.path_entries


def evaluate_expression(expr: str, namespace: dict[str, Any]) -> Any:
    """Evaluate an already-parsed expression against preloaded Waveforms.

    Parameters
    ----------
    expr:
        The substituted expression returned by ``parse_expression``.
        Physical paths must be resolved and replaced before calling this
        function.
    namespace:
        Names available to the expression, normally Waveform objects.

    Returns
    -------
    Any
        The result produced by the Waveform expression.
    """
    evaluation_namespace = {**_EXPRESSION_FUNCTIONS, **namespace}
    code = compile(expr, '<wavekit-expression>', 'eval')
    return eval(code, {'__builtins__': {}}, evaluation_namespace)  # noqa: S307
