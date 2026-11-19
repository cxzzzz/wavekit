import numpy as np
import pytest

from wavekit import VcdReader, Waveform
from wavekit.expression import evaluate_expression, parse_expression


def build_waveform(values, width, signed=False):
    value = np.array(values, dtype=object if width is not None and width > 64 else None)
    cycle = np.arange(len(value))
    return Waveform(value, cycle=cycle, time=cycle * 10, width=width, signed=signed)


UNSIGNED = build_waveform([3, 1, 4, 1, 5, 9, 2, 6], width=8)
SIGNED = build_waveform([-3, 1, -4, 1, 5, -9, 2, 6], width=8, signed=True)
WIDE = build_waveform([1 << 70, 3, (1 << 79) + 5, 7], width=80)
FLOAT = build_waveform([0.5, 1.5, 2.5, 4.0], width=None)

NUMPY_REDUCTIONS = [
    ('sum', np.sum),
    ('prod', np.prod),
    ('min', np.min),
    ('max', np.max),
    ('mean', np.mean),
    ('std', np.std),
    ('var', np.var),
    ('median', np.median),
    ('any', np.any),
    ('all', np.all),
]


REDUCTION_NAMES = [name for name, _ in NUMPY_REDUCTIONS]


@pytest.mark.parametrize('wave', [UNSIGNED, SIGNED, WIDE, FLOAT], ids=['u', 's', 'wide', 'f'])
@pytest.mark.parametrize(('name', 'reference'), NUMPY_REDUCTIONS, ids=REDUCTION_NAMES)
def test_reductions_match_numpy(wave, name, reference):
    result = getattr(wave, name)()
    expected = reference(wave.value)
    expected = expected.item() if isinstance(expected, np.generic) else expected
    assert result == pytest.approx(expected) if isinstance(expected, float) else result == expected
    assert not isinstance(result, np.generic)


@pytest.mark.parametrize('wave', [UNSIGNED, SIGNED, FLOAT], ids=['u', 's', 'f'])
def test_percentile_and_quantile_match_numpy(wave):
    assert wave.percentile(90) == pytest.approx(np.percentile(wave.value, 90))
    assert wave.quantile(0.9) == pytest.approx(np.quantile(wave.value, 0.9))
    assert isinstance(wave.percentile(50), float)


def test_ddof_is_forwarded():
    assert UNSIGNED.std(ddof=1) == pytest.approx(np.std(UNSIGNED.value, ddof=1))
    assert UNSIGNED.var(ddof=1) == pytest.approx(np.var(UNSIGNED.value, ddof=1))


def test_return_types_are_python_scalars():
    assert type(UNSIGNED.sum()) is int
    assert type(SIGNED.min()) is int
    assert type(UNSIGNED.mean()) is float
    assert type(UNSIGNED.any()) is bool
    assert type(WIDE.max()) is int
    assert WIDE.max() == (1 << 79) + 5


def test_len_and_at():
    assert UNSIGNED.len() == 8
    assert type(UNSIGNED.len()) is int
    assert UNSIGNED.at(0) == 3
    assert UNSIGNED.at(-1) == 6
    assert type(UNSIGNED.at(0)) is int
    assert WIDE.at(2) == (1 << 79) + 5
    with pytest.raises(IndexError):
        UNSIGNED.at(8)
    with pytest.raises(IndexError):
        UNSIGNED.at(1.5)


EMPTY = build_waveform([], width=8)


def test_empty_input_follows_numpy():
    assert EMPTY.sum() == 0
    assert EMPTY.prod() == 1
    assert EMPTY.any() is False
    assert EMPTY.all() is True
    assert EMPTY.len() == 0
    with pytest.raises(ValueError):
        EMPTY.max()
    with pytest.warns(RuntimeWarning):
        assert np.isnan(EMPTY.mean())
    with pytest.raises(IndexError):
        EMPTY.at(0)


def test_scalar_result_composes_with_waveform():
    centered = UNSIGNED - UNSIGNED.min()
    assert list(centered.value) == [2, 0, 3, 0, 4, 8, 1, 5]
    above = UNSIGNED > UNSIGNED.mean()
    assert list(above.value) == [0, 0, 1, 0, 1, 1, 0, 1]  # mean is 3.875


def test_aggregations_are_expression_functions():
    expr, paths = parse_expression('x - mean(x)')
    assert [path for _, path in paths] == ['x', 'x']
    namespace = {placeholder: UNSIGNED for placeholder, _ in paths}
    result = evaluate_expression(expr, namespace)
    assert np.allclose(result.value, UNSIGNED.value - UNSIGNED.value.mean())
    expr, paths = parse_expression('percentile(x, 50) + len(x) + at(x, -1)')
    namespace = {placeholder: UNSIGNED for placeholder, _ in paths}
    assert evaluate_expression(expr, namespace) == pytest.approx(3.5 + 8 + 6)


def test_reader_eval_with_aggregation():
    with VcdReader('tests/readers/fixtures/generated/compare.vcd') as reader:
        counter = reader.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        assert reader.eval('max(compare_tb.dut.counter)', clock='compare_tb.clk') == counter.max()
