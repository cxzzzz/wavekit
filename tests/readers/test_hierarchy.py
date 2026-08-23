from dataclasses import dataclass, field

from wavekit.readers.hierarchy import Node, Scope
from wavekit.readers.matcher import ExactCapture, ExactMatcher, WildcardCapture


@dataclass(frozen=True, eq=False)
class DefinitionScope(Scope):
    module: str = ''
    _children: tuple[Node, ...] = field(default_factory=tuple)

    @property
    def definition(self) -> str:
        return self.module

    @property
    def children(self) -> tuple[Node, ...]:
        return self._children


def test_native_recursive_steps_restore_their_own_capture_paths():
    root = DefinitionScope(base_name='root', parent=None, module='Top')
    module_a = DefinitionScope(base_name='a', parent=root, module='A')
    module_b = DefinitionScope(base_name='b', parent=module_a, module='B')
    object.__setattr__(root, '_children', (module_a,))
    object.__setattr__(module_a, '_children', (module_b,))

    matched = root.get_matched_nodes('$$A.$$B')

    assert len(matched) == 1
    (captures,) = matched.keys()
    assert [(capture.path, capture.definition) for capture in captures] == [
        ('a', 'A'),
        ('b', 'B'),
    ]


def test_lowered_recursive_definition_match_restores_wildcard_capture():
    root = DefinitionScope(base_name='root', parent=None, module='Top')
    module_a = DefinitionScope(base_name='a', parent=root, module='A')
    module_b = DefinitionScope(base_name='b', parent=module_a, module='B')
    object.__setattr__(root, '_children', (module_a,))
    object.__setattr__(module_a, '_children', (module_b,))

    nested = root.get_matched_nodes('**.$B')
    (nested_captures,) = nested.keys()
    assert isinstance(nested_captures[0], WildcardCapture)
    assert isinstance(nested_captures[1], ExactCapture)
    assert [(capture.path, capture.definition) for capture in nested_captures] == [
        ('a', None),
        ('b', 'B'),
    ]

    direct = root.get_matched_nodes('a.**.$B')
    (direct_captures,) = direct.keys()
    assert [(capture.path, capture.definition) for capture in direct_captures] == [
        ('', None),
        ('b', 'B'),
    ]


def test_recursive_definition_match_reuses_value_semantic_cache():
    root = DefinitionScope(base_name='root', parent=None, module='Top')
    module_a = DefinitionScope(base_name='a', parent=root, module='A')
    module_b = DefinitionScope(base_name='b', parent=module_a, module='B')
    object.__setattr__(root, '_children', (module_a,))
    object.__setattr__(module_a, '_children', (module_b,))

    first = root.get_matched_nodes('**.$B')
    matcher = ExactMatcher(target='definition', pattern='B')
    cached_parents = root._recursive_match_cache[matcher]

    second = root.get_matched_nodes('**.$B')

    assert first == second
    assert cached_parents == (module_a,)
    assert root._recursive_match_cache[matcher] is cached_parents
