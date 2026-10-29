import numpy as np
import pytest

from wavekit import ClockDomain, FstReader, VcdReader


@pytest.fixture()
def vcd_path():
    return 'tests/readers/fixtures/generated/compare.vcd'


@pytest.fixture()
def xz_vcd_path():
    return 'tests/readers/fixtures/generated/compare_xz.vcd'


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

        with cd:
            first = r['compare_tb.dut.counter'].w
        with cd:
            second = r['compare_tb.dut.counter'].w

        assert np.array_equal(first.value, second.value)


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
            with r.clock_domain(clock='compare_tb.clk', start_time=0, start_cycle=0):
                _ = r['compare_tb.dut.counter'].w
        with pytest.raises(ValueError, match='end_time and end_cycle are mutually exclusive'):
            with r.clock_domain(clock='compare_tb.clk', end_time=0, end_cycle=0):
                _ = r['compare_tb.dut.counter'].w


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


def test_clock_domain_on_fst_reader():
    with FstReader('tests/readers/fixtures/generated/compare.fst') as r:
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


def test_load_waveform_omitted_clock_uses_ambient(vcd_path):
    with _open(vcd_path) as r:
        expected = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain(clock='compare_tb.clk'):
            ambient = r.load_waveform('compare_tb.dut.counter')

        assert np.array_equal(ambient.value, expected.value)


def test_load_waveform_omitted_clock_uses_ambient_window(vcd_path):
    with _open(vcd_path) as r:
        full = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain(clock='compare_tb.clk', start_cycle=0, end_cycle=5):
            windowed = r.load_waveform('compare_tb.dut.counter')

        assert len(windowed.value) == 5
        assert len(windowed.value) < len(full.value)


def test_load_waveform_omitted_clock_raises_outside_domain(vcd_path):
    with _open(vcd_path) as r:
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            r.load_waveform('compare_tb.dut.counter')


def test_load_waveform_explicit_clock_ignores_ambient(vcd_path):
    with _open(vcd_path) as r:
        expected = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain(clock='compare_tb.clk', start_cycle=0, end_cycle=5):
            explicit = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')

        assert np.array_equal(explicit.value, expected.value)
        assert len(explicit.value) != 5


def test_load_unknown_mask_omitted_clock_uses_ambient(xz_vcd_path):
    with _open(xz_vcd_path) as r:
        expected = r.load_unknown_mask('compare_xz_tb.data_0', clock='compare_xz_tb.clk')
        with r.clock_domain(clock='compare_xz_tb.clk'):
            ambient = r.load_unknown_mask('compare_xz_tb.data_0')

        assert np.array_equal(ambient.value, expected.value)


def test_load_unknown_mask_omitted_clock_raises_outside_domain(xz_vcd_path):
    with _open(xz_vcd_path) as r:
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            r.load_unknown_mask('compare_xz_tb.data_0')


def test_load_matched_waveforms_omitted_clock_uses_ambient(vcd_path):
    with _open(vcd_path) as r:
        by_path = r.load_matched_waveforms(
            'compare_tb.dut.{counter,status}', clock_path='compare_tb.clk'
        )
        with r.clock_domain(clock='compare_tb.clk'):
            ambient = r.load_matched_waveforms('compare_tb.dut.{counter,status}')

        assert set(ambient) == set(by_path)
        for key in ambient:
            assert np.array_equal(ambient[key].value, by_path[key].value)


def test_load_matched_waveforms_omitted_clock_raises_outside_domain(vcd_path):
    with _open(vcd_path) as r:
        with pytest.raises(RuntimeError, match='requires an active clock domain'):
            r.load_matched_waveforms('compare_tb.dut.{counter,status}')


def test_load_matched_unknown_masks_omitted_clock_uses_ambient(xz_vcd_path):
    with _open(xz_vcd_path) as r:
        by_path = r.load_matched_unknown_masks(
            'compare_xz_tb.{data_0,data_1}', clock_path='compare_xz_tb.clk'
        )
        with r.clock_domain(clock='compare_xz_tb.clk'):
            ambient = r.load_matched_unknown_masks('compare_xz_tb.{data_0,data_1}')

        assert set(ambient) == set(by_path)
        for key in ambient:
            assert np.array_equal(ambient[key].value, by_path[key].value)


def _assert_same_waveform(actual, expected):
    assert np.array_equal(actual.value, expected.value)
    assert np.array_equal(actual.cycle, expected.cycle)
    assert np.array_equal(actual.time, expected.time)
    assert actual.width == expected.width
    assert actual.signed == expected.signed


def test_domain_reset_active_low_drops_reset_samples(vcd_path):
    with _open(vcd_path) as r:
        rst_n = r.load_waveform('compare_tb.rst_n', clock='compare_tb.clk')
        counter = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain('compare_tb.clk', reset='compare_tb.rst_n', reset_active_level='low'):
            filtered = r['compare_tb.dut.counter'].w

        assert len(filtered.value) < len(counter.value)
        _assert_same_waveform(filtered, counter.mask(rst_n == 1))


def test_domain_reset_active_high_drops_reset_samples(vcd_path):
    with _open(vcd_path) as r:
        rst_n = r.load_waveform('compare_tb.rst_n', clock='compare_tb.clk')
        counter = r.load_waveform('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain('compare_tb.clk', reset='compare_tb.rst_n', reset_active_level='high'):
            filtered = r['compare_tb.dut.counter'].w

        assert len(filtered.value) < len(counter.value)
        _assert_same_waveform(filtered, counter.mask(rst_n == 0))


def test_domain_reset_filters_unknown_mask(vcd_path):
    with _open(vcd_path) as r:
        rst_n = r.load_waveform('compare_tb.rst_n', clock='compare_tb.clk')
        mask = r.load_unknown_mask('compare_tb.dut.counter', clock='compare_tb.clk')
        with r.clock_domain('compare_tb.clk', reset='compare_tb.rst_n', reset_active_level='low'):
            filtered_mask = r['compare_tb.dut.counter'].m
            filtered_value = r['compare_tb.dut.counter'].w

        _assert_same_waveform(filtered_mask, mask.mask(rst_n == 1))
        assert len(filtered_mask.value) == len(filtered_value.value)


def test_domain_reset_applies_to_load_matched_waveforms(vcd_path):
    with _open(vcd_path) as r:
        rst_n = r.load_waveform('compare_tb.rst_n', clock='compare_tb.clk')
        by_path = r.load_matched_waveforms(
            'compare_tb.dut.{counter,status}', clock_path='compare_tb.clk'
        )
        with r.clock_domain('compare_tb.clk', reset='compare_tb.rst_n', reset_active_level='low'):
            ambient = r.load_matched_waveforms('compare_tb.dut.{counter,status}')

        assert set(ambient) == set(by_path)
        for key in ambient:
            _assert_same_waveform(ambient[key], by_path[key].mask(rst_n == 1))


def test_domain_reset_keeps_signals_on_one_axis_for_pattern(vcd_path):
    from wavekit.pattern import Pattern, match

    with _open(vcd_path) as r:
        with r.clock_domain('compare_tb.clk', reset='compare_tb.rst_n', reset_active_level='low'):
            counter = r['compare_tb.dut.counter'].w
            status = r['compare_tb.dut.status'].w
            result = match(Pattern().consume(counter == counter).capture('status', status))

        assert np.array_equal(counter.cycle, status.cycle)
        assert set(result.start.cycle) <= set(counter.cycle)


def test_domain_reset_unknown_counts_as_active(tmp_path):
    vcd = tmp_path / 'reset_x.vcd'
    vcd.write_text(
        '$timescale 1ns $end\n'
        '$scope module tb $end\n'
        '$var wire 1 ! clk $end\n'
        '$var wire 1 " rst $end\n'
        '$var wire 8 # data $end\n'
        '$upscope $end\n'
        '$enddefinitions $end\n'
        '#0\n0!\nx"\nb00000001 #\n'
        '#5\n1!\n#10\n0!\nb00000010 #\n'
        '#15\n1!\n#20\n0!\n0"\nb00000011 #\n'
        '#25\n1!\n#30\n0!\nb00000100 #\n'
        '#35\n1!\n#40\n0!\n1"\nb00000101 #\n'
        '#45\n1!\n#50\n0!\nb00000110 #\n'
        '#55\n1!\n#60\n0!\n'
    )
    with VcdReader(str(vcd)) as r:
        # reset samples (posedges): x, x, 0, 0, 1, 1
        with r.clock_domain(
            'tb.clk', sample_on_posedge=True, reset='tb.rst', reset_active_level='low'
        ):
            low = r['tb.data'].w
        with r.clock_domain(
            'tb.clk', sample_on_posedge=True, reset='tb.rst', reset_active_level='high'
        ):
            high = r['tb.data'].w

    assert list(low.value) == [5, 6]
    assert list(high.value) == [3, 4]


def test_domain_reset_requires_valid_level(vcd_path):
    with _open(vcd_path) as r:
        with r.clock_domain('compare_tb.clk', reset='compare_tb.rst_n'):
            with pytest.raises(ValueError, match='reset_active_level'):
                _ = r['compare_tb.dut.counter'].w

        with pytest.raises(ValueError, match='reset_active_level'):
            r.load_waveform(
                'compare_tb.dut.counter',
                clock='compare_tb.clk',
                reset='compare_tb.rst_n',
                reset_active_level='sideways',
            )


def test_domain_eager_reset_validation(vcd_path):
    with _open(vcd_path) as r:
        with pytest.raises(ValueError, match="signal 'compare_tb.missing' not found"):
            r.clock_domain('compare_tb.clk', reset='compare_tb.missing', reset_active_level='low')


@pytest.fixture()
def two_domain_vcd(tmp_path):
    """Two instances sharing one timeline, each with its own reset, plus a global reset.

    Posedge samples (cycle 0..5): u0.rst = 0 0 1 1 1 1, u1.rst = 0 0 0 0 1 1,
    grst = 0 1 1 1 1 1; u0.data = 10..15, u1.data = 20..25.
    """
    u0_rst, u1_rst, grst = '001111', '000011', '011111'
    lines = [
        '$timescale 1ns $end',
        '$scope module tb $end',
        '$scope module u0 $end',
        '$var wire 1 a clk $end',
        '$var wire 1 b rst $end',
        '$var wire 8 c data $end',
        '$upscope $end',
        '$scope module u1 $end',
        '$var wire 1 d clk $end',
        '$var wire 1 e rst $end',
        '$var wire 8 f data $end',
        '$upscope $end',
        '$var wire 1 g grst $end',
        '$upscope $end',
        '$enddefinitions $end',
    ]
    for i in range(6):
        lines += [
            f'#{10 * i}',
            '0a',
            '0d',
            f'{u0_rst[i]}b',
            f'{u1_rst[i]}e',
            f'{grst[i]}g',
            f'b{10 + i:08b} c',
            f'b{20 + i:08b} f',
            f'#{10 * i + 5}',
            '1a',
            '1d',
        ]
    lines.append('#65')
    path = tmp_path / 'two_domain.vcd'
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def _by_name(waves):
    return {w.signal.full_name.split('[')[0]: w for w in waves.values()}


def test_load_matched_waveforms_pairs_reset_per_instance(two_domain_vcd):
    with VcdReader(two_domain_vcd) as r:
        waves = r.load_matched_waveforms(
            'tb.u{0,1}.data',
            clock_path='tb.u{0,1}.clk',
            reset_path='tb.u{0,1}.rst',
            reset_active_level='low',
            sample_on_posedge=True,
        )

    by_name = _by_name(waves)
    assert list(by_name['tb.u0.data'].value) == [12, 13, 14, 15]
    assert list(by_name['tb.u1.data'].value) == [24, 25]
    assert list(by_name['tb.u1.data'].cycle) == [4, 5]


def test_load_matched_waveforms_broadcasts_single_reset(two_domain_vcd):
    with VcdReader(two_domain_vcd) as r:
        waves = r.load_matched_waveforms(
            'tb.u{0,1}.data',
            clock_path='tb.u{0,1}.clk',
            reset_path='tb.grst',
            reset_active_level='low',
            sample_on_posedge=True,
        )

    by_name = _by_name(waves)
    assert list(by_name['tb.u0.data'].value) == [11, 12, 13, 14, 15]
    assert list(by_name['tb.u1.data'].value) == [21, 22, 23, 24, 25]


def test_load_matched_waveforms_reset_without_prefix_raises(two_domain_vcd):
    with VcdReader(two_domain_vcd) as r:
        with pytest.raises(ValueError, match='no reset key is a prefix'):
            r.load_matched_waveforms(
                'tb.grst',
                clock_path='tb.u0.clk',
                reset_path='tb.u{0,1}.rst',
                reset_active_level='low',
                sample_on_posedge=True,
            )


def test_load_matched_unknown_masks_pairs_reset_per_instance(two_domain_vcd):
    with VcdReader(two_domain_vcd) as r:
        masks = r.load_matched_unknown_masks(
            'tb.u{0,1}.data',
            clock_path='tb.u{0,1}.clk',
            reset_path='tb.u{0,1}.rst',
            reset_active_level='low',
            sample_on_posedge=True,
        )

    by_name = _by_name(masks)
    assert len(by_name['tb.u0.data'].value) == 4
    assert len(by_name['tb.u1.data'].value) == 2


def test_eval_zip_pairs_reset_per_key(two_domain_vcd):
    with VcdReader(two_domain_vcd) as r:
        result = r.eval(
            'tb.u{0,1}.data + 1',
            clock='tb.u{0,1}.clk',
            reset='tb.u{0,1}.rst',
            reset_active_level='low',
            sample_on_posedge=True,
            mode='zip',
        )

    assert sorted(list(w.value) for w in result.values()) == [[13, 14, 15, 16], [25, 26]]


def test_reset_path_ignored_when_clock_path_omitted(two_domain_vcd):
    with VcdReader(two_domain_vcd) as r:
        with r.clock_domain('tb.u0.clk', sample_on_posedge=True):
            waves = r.load_matched_waveforms(
                'tb.u{0,1}.data',
                reset_path='tb.grst',
                reset_active_level='low',
            )

    assert all(len(w.value) == 6 for w in waves.values())
