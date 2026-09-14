import numpy as np
import pytest

from wavekit import ClockDomain, FstReader, VcdReader


@pytest.fixture()
def vcd_path():
    return 'tests/readers/fixtures/vcd/compare.vcd'


@pytest.fixture()
def xz_vcd_path():
    return 'tests/readers/fixtures/vcd/compare_xz.vcd'


def _open(path):
    return VcdReader(path)


def test_clock_domain_exported():
    assert ClockDomain.__name__ == 'ClockDomain'


def test_signal_accessors_require_domain(vcd_path):
    with _open(vcd_path) as r:
        sig = r['compare_tb.dut.counter']
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            _ = sig.w
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            sig.waveform()
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            _ = sig.m
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            sig.unknown_mask()


def test_ambient_access_matches_load_waveform(vcd_path):
    with _open(vcd_path) as r:
        expected = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain(clock='compare_tb.clk'):
            ambient = r['compare_tb.dut.counter'].w

        assert np.array_equal(ambient.value, expected.value)
        assert np.array_equal(ambient.cycle, expected.cycle)
        assert ambient.width == expected.width


def test_domain_is_reusable_value_and_context(vcd_path):
    with _open(vcd_path) as r:
        cd = r.clock_domain(clock='compare_tb.clk')
        assert isinstance(cd, ClockDomain)

        explicit = r.load_waveform('compare_tb.dut.counter', clock=cd)
        with cd:
            ambient = r['compare_tb.dut.counter'].w

        assert np.array_equal(explicit.value, ambient.value)


def test_nested_domains_restore_outer(vcd_path):
    with _open(vcd_path) as r:
        with r.clock_domain(clock='compare_tb.clk', start_time=0, end_time=100):
            outer = r['compare_tb.dut.counter'].w
            with r.clock_domain(clock='compare_tb.clk'):
                inner = r['compare_tb.dut.counter'].w
            after = r['compare_tb.dut.counter'].w

        assert len(outer.value) < len(inner.value)
        assert len(after.value) == len(outer.value)


def test_domain_exits_restore_state(vcd_path):
    with _open(vcd_path) as r:
        sig = r['compare_tb.dut.counter']
        with r.clock_domain(clock='compare_tb.clk'):
            pass
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            _ = sig.w


def test_domain_as_decorator(vcd_path):
    with _open(vcd_path) as r:

        @r.clock_domain(clock='compare_tb.clk')
        def grab():
            return r['compare_tb.dut.counter'].w

        wave = grab()
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            _ = r['compare_tb.dut.counter'].w
        assert len(wave.value) > 0


def test_domain_eager_clock_validation(vcd_path):
    with _open(vcd_path) as r:
        with pytest.raises(ValueError, match="signal 'compare_tb.missing' not found"):
            r.clock_domain(clock='compare_tb.missing')


def test_domain_window_applies_to_ambient(vcd_path):
    with _open(vcd_path) as r:
        full = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain(clock='compare_tb.clk', start_time=0, end_time=100):
            windowed = r['compare_tb.dut.counter'].w

        assert len(windowed.value) < len(full.value)


def test_domain_window_cycle_variant(vcd_path):
    with _open(vcd_path) as r:
        with r.clock_domain(clock='compare_tb.clk', start_cycle=0, end_cycle=5):
            windowed = r['compare_tb.dut.counter'].w

        assert len(windowed.value) == 5


def test_domain_window_exclusive_pair_validation(vcd_path):
    with _open(vcd_path) as r:
        with pytest.raises(ValueError, match='start_time and start_cycle are mutually exclusive'):
            r.load_waveform(
                'compare_tb.dut.counter',
                clock=r.clock_domain(clock='compare_tb.clk', start_time=0, start_cycle=0),
            )
        with pytest.raises(ValueError, match='end_time and end_cycle are mutually exclusive'):
            r.load_waveform(
                'compare_tb.dut.counter',
                clock=r.clock_domain(clock='compare_tb.clk', end_time=0, end_cycle=0),
            )


def test_waveform_and_unknown_mask_signal_level_options(xz_vcd_path):
    with _open(xz_vcd_path) as r:
        with r.clock_domain(clock='compare_xz_tb.clk'):
            sig = r['compare_xz_tb.data_0']
            signed = sig.waveform(signed=True)
            assert signed.signed is True

            expected = r.load_unknown_mask(
                'compare_xz_tb.data_0', clock='compare_xz_tb.clk', include_x=False
            )
            mask_x_off = sig.unknown_mask(include_x=False)
            assert np.array_equal(mask_x_off.value, expected.value)

            default_mask = sig.m
            expected_default = r.load_unknown_mask(
                'compare_xz_tb.data_0', clock='compare_xz_tb.clk'
            )
            assert np.array_equal(default_mask.value, expected_default.value)


def test_resolve_sampling_params_merges_domain_and_call(vcd_path):
    with _open(vcd_path) as r:
        cd = r.clock_domain(clock='compare_tb.clk')
        merged = r.load_waveform('compare_tb.dut.counter', clock=cd, start_time=0, end_time=100)
        expected = r.load_waveform(
            'compare_tb.dut.counter', clock='compare_tb.clk', start_time=0, end_time=100
        )
        assert np.array_equal(merged.value, expected.value)

        # domain value wins over the call-site default
        cd_win = r.clock_domain(clock='compare_tb.clk', start_time=0, end_time=100)
        domain_window = r.load_waveform('compare_tb.dut.counter', clock=cd_win)
        assert len(domain_window.value) == len(expected.value)


def test_resolve_sampling_params_conflict(vcd_path):
    with _open(vcd_path) as r:
        cd = r.clock_domain(clock='compare_tb.clk', start_time=0, end_time=100)
        with pytest.raises(ValueError, match='start_time=5 conflicts with the clock domain'):
            r.load_waveform('compare_tb.dut.counter', clock=cd, start_time=5)

        # domain posedge wins over the unset call site; an explicit opposite is a conflict
        cd_pos = r.clock_domain(clock='compare_tb.clk', sample_on_posedge=True)
        posedge = r.load_waveform('compare_tb.dut.counter', clock=cd_pos)
        assert len(posedge.value) > 0
        cd_neg = r.clock_domain(clock='compare_tb.clk')
        with pytest.raises(ValueError, match='sample_on_posedge=True conflicts'):
            r.load_waveform('compare_tb.dut.counter', clock=cd_neg, sample_on_posedge=True)


def test_load_matched_waveforms_accepts_domain(vcd_path):
    with _open(vcd_path) as r:
        cd = r.clock_domain(clock='compare_tb.clk')
        matched = r.load_matched_waveforms('compare_tb.dut.{counter,status}', clock_path=cd)
        by_path = r.load_matched_waveforms(
            'compare_tb.dut.{counter,status}', clock_path='compare_tb.clk'
        )

        assert set(matched) == set(by_path)
        for key in matched:
            assert np.array_equal(matched[key].value, by_path[key].value)


def test_load_matched_unknown_masks_accepts_domain(xz_vcd_path):
    with _open(xz_vcd_path) as r:
        cd = r.clock_domain(clock='compare_xz_tb.clk')
        matched = r.load_matched_unknown_masks('compare_xz_tb.{data_0,data_1}', clock_path=cd)
        by_path = r.load_matched_unknown_masks(
            'compare_xz_tb.{data_0,data_1}', clock_path='compare_xz_tb.clk'
        )

        assert set(matched) == set(by_path)
        for key in matched:
            assert np.array_equal(matched[key].value, by_path[key].value)


def test_clock_domain_on_fst_reader():
    with FstReader('tests/readers/fixtures/fst/compare.fst') as r:
        expected = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain(clock='compare_tb.clk'):
            ambient = r['compare_tb.dut.counter'].w

        assert np.array_equal(ambient.value, expected.value)


def test_bit_selected_signal_uses_domain(vcd_path):
    with _open(vcd_path) as r:
        expected = r.load_waveform('compare_tb.dut.counter[1:0]', clock='compare_tb.clk')
        with r.clock_domain(clock='compare_tb.clk'):
            ambient = r['compare_tb.dut.counter'][1:0].w

        assert np.array_equal(ambient.value, expected.value)
        assert ambient.width == 2


def test_resolve_sampling_params_passes_through_non_domain(vcd_path):
    with _open(vcd_path) as r:
        clock, options = r._resolve_sampling_params(
            'compare_tb.clk', sample_on_posedge=False, start_time=None, end_time=None
        )
    assert clock == 'compare_tb.clk'
    assert options == dict(sample_on_posedge=False, start_time=None, end_time=None)


def test_domain_field_names_stay_load_api_parameters(vcd_path):
    """Drift lock: every mergeable domain field must be a load API parameter."""
    import inspect

    with _open(vcd_path) as r:
        load_params = set(inspect.signature(r.load_waveform).parameters)
        domain_fields = {
            f for f in vars(r.clock_domain(clock='compare_tb.clk')) if not f.startswith('_')
        } - {'reader', 'clock'}

    assert domain_fields <= load_params
