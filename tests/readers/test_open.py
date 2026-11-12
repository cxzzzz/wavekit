import pytest

import wavekit
from wavekit import FstReader, VcdReader, has_fsdb_support

GENERATED = 'tests/readers/fixtures/generated'


@pytest.mark.parametrize(
    ('name', 'reader_class'), [('compare.vcd', VcdReader), ('compare.fst', FstReader)]
)
def test_open_picks_reader_by_extension(name, reader_class):
    with wavekit.open(f'{GENERATED}/{name}') as reader:
        assert type(reader) is reader_class
        assert reader.get_signal('compare_tb.clk').width == 1


def test_open_rejects_unknown_extension():
    with pytest.raises(ValueError, match=r'expected one of \.vcd, \.fst, \.fsdb'):
        wavekit.open('pyproject.toml')


def test_open_is_not_star_exported():
    assert 'open' not in wavekit.__all__


@pytest.mark.parametrize('reader_class', [VcdReader, FstReader])
def test_missing_file_raises_file_not_found(reader_class):
    with pytest.raises(FileNotFoundError):
        reader_class('missing.vcd')


@pytest.mark.skipif(not has_fsdb_support(), reason='FSDB tests require the Verdi NPI runtime')
def test_missing_fsdb_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        wavekit.open('missing.fsdb')
