"""Declarative Pattern builder/API and compiler control-flow tests."""

import signal

import numpy as np
import pytest

from helpers import bool_wf as _bool_wf
from helpers import wf as _wf
from wavekit import Waveform
from wavekit.pattern import (
    MatchRecord,
    MatchRecords,
    MatchStatus,
    Pattern,
    PatternError,
    collect,
    match,
)


class TestSingleWait:
    def test_basic_trigger(self):
        sig = _bool_wf([0, 1, 0, 1, 1, 0])
        result = match(Pattern().wait(sig))
        assert len(result) == 3
        np.testing.assert_array_equal(result.start.value, [1, 3, 4])
        np.testing.assert_array_equal(result.status.value, [MatchStatus.OK()] * 3)

    def test_no_matches(self):
        sig = _bool_wf([0, 0, 0])
        result = match(Pattern().wait(sig))
        assert len(result) == 0

    def test_all_ones(self):
        sig = _bool_wf([1, 1, 1])
        result = match(Pattern().wait(sig))
        assert len(result) == 3


class TestWaitCapture:
    def test_capture_value(self):
        valid = _bool_wf([0, 1, 0, 1, 0])
        data = _wf([10, 20, 30, 40, 50], width=8)
        result = match(Pattern().wait(valid).capture('data', data))
        assert len(result) == 2
        np.testing.assert_array_equal(result.captures['data'].value, [20, 40])

    def test_two_phase_handshake(self):
        req = _bool_wf([0, 1, 0, 0, 0, 1, 0, 0])
        ack = _bool_wf([0, 0, 0, 1, 0, 0, 1, 0])
        data = _wf([0, 0, 0, 99, 0, 0, 77, 0], width=8)
        result = match(Pattern().wait(req).wait(ack).capture('data', data))
        assert len(result) == 2
        np.testing.assert_array_equal(result.captures['data'].value, [99, 77])
        np.testing.assert_array_equal(result.start.value, [1, 5])
        np.testing.assert_array_equal(result.end.value, [3, 6])


class TestLoopUntil:
    def test_simple_burst(self):
        """Collect data beats until last=1."""
        start = _bool_wf([1, 0, 0, 0, 0, 0])
        beat = _bool_wf([0, 1, 0, 1, 0, 1])
        last = _bool_wf([0, 0, 0, 0, 0, 1])
        data = _wf([0, 10, 0, 20, 0, 30], width=8)
        result = match(
            Pattern()
            .wait(start)
            .loop(
                Pattern()
                .wait(beat)
                .capture('d', data, mode='list')
                .branch(last == 0, true_body=Pattern().delay(1)),
                until=last,
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert list(ok.captures['d'].value[0]) == [10, 20, 30]

    def test_single_beat(self):
        """Burst with one beat (last=1 on first beat)."""
        start = _bool_wf([1, 0, 0])
        beat = _bool_wf([0, 1, 0])
        last = _bool_wf([0, 1, 0])
        data = _wf([0, 99, 0], width=8)
        result = match(
            Pattern()
            .wait(start)
            .loop(Pattern().wait(beat).capture('d', data, mode='list'), until=last)
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert list(ok.captures['d'].value[0]) == [99]


class TestLoopWhile:
    def test_capture_while_high(self):
        """Capture data each cycle while enable is high."""
        trigger = _bool_wf([0, 1, 0, 0, 0, 0])
        enable = _bool_wf([0, 1, 1, 1, 0, 0])
        data = _wf([0, 10, 20, 30, 40, 50], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .loop(Pattern().capture('d', data, mode='list').delay(1), when=enable)
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert list(ok.captures['d'].value[0]) == [10, 20, 30]

    def test_while_false_immediately(self):
        """When condition is False at entry, loop is skipped (0 iterations)."""
        trigger = _bool_wf([1, 0, 0])
        cond = _bool_wf([0, 0, 0])
        data = _wf([10, 20, 30], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .loop(Pattern().delay(1).capture('d', data, mode='list'), when=cond)
            .capture('after', data)
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['after'].value[0] == 10
        assert 'd' not in ok.captures or list(ok.captures['d'].value[0]) == []


class TestStallDetection:
    def test_stall_interval(self):
        """Find stall intervals using loop-until."""
        stall = _bool_wf([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
        trigger = stall.rising_edge()
        result = match(Pattern().wait(trigger).loop(Pattern().delay(1), until=stall == 0))
        ok = result.filter_ok()
        assert len(ok) == 2
        np.testing.assert_array_equal(ok.start.value, [1, 7])
        np.testing.assert_array_equal(ok.end.value, [5, 9])
        np.testing.assert_array_equal(ok.duration.value, [5, 3])


class TestRepeat:
    def test_repeat_static(self):
        trigger = _bool_wf([1, 0, 0, 0, 0, 0])
        beat = _bool_wf([0, 1, 1, 1, 0, 0])
        data = _wf([0, 10, 20, 30, 0, 0], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .repeat(Pattern().wait(beat).capture('d', data, mode='list').delay(1), n=3)
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert list(ok.captures['d'].value[0]) == [10, 20, 30]


class TestRepeatDynamic:
    def test_dynamic_n_from_capture(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])
        len_sig = _wf([2, 0, 0, 0, 0], width=4)
        beat = _bool_wf([0, 1, 1, 0, 0])
        data = _wf([0, 10, 20, 0, 0], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .capture('len', len_sig)
            .repeat(
                Pattern().wait(beat).capture('d', data, mode='list').delay(1),
                n=lambda idx, cap: int(cap['len']),
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert list(ok.captures['d'].value[0]) == [10, 20]


class TestCaptureDynamic:
    def test_capture_lambda(self):
        trigger = _bool_wf([0, 1, 0])
        sig_a = _wf([0, 100, 0], width=8)
        sig_b = _wf([0, 200, 0], width=8)
        mode = _wf([0, 1, 0], width=1)
        result = match(
            Pattern()
            .wait(trigger)
            .capture('mode', mode)
            .capture(
                'val', lambda idx, cap: sig_a.value[idx] if cap['mode'] == 0 else sig_b.value[idx]
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == 200


class TestBranch:
    @pytest.mark.parametrize(('cond_value', 'expected'), [(1, 10), (0, 20)])
    def test_branch_selects_body(self, cond_value, expected):
        trigger = _bool_wf([0, 1, 0])
        cond = _bool_wf([0, cond_value, 0])
        data_a = _wf([0, 10, 0], width=8)
        data_b = _wf([0, 20, 0], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .branch(
                cond,
                true_body=Pattern().capture('val', data_a),
                false_body=Pattern().capture('val', data_b),
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == expected

    def test_branch_none_body(self):
        """Branch with no false_body → skip."""
        trigger = _bool_wf([0, 1, 0])
        cond = _bool_wf([0, 0, 0])
        data = _wf([0, 42, 0], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .branch(cond, true_body=Pattern().capture('optional', data))
            .capture('always', data)
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['always'].value[0] == 42
        assert 'optional' not in ok.captures


class TestPatternRegressionEdges:
    def test_zero_time_loop_is_guarded(self):
        trigger = _bool_wf([1])
        pattern = Pattern().wait(trigger).loop(Pattern().delay(0), when=True)

        def fail_on_alarm(_signum, _frame):
            raise AssertionError('zero-time loop did not stop')

        old_handler = signal.signal(signal.SIGALRM, fail_on_alarm)
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try:
            with pytest.raises(PatternError, match='same-cycle'):
                match(pattern)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)

    def test_first_unguarded_wait_callback_runs_once_per_start(self):
        fire = _bool_wf([1, 0, 1])
        calls = []

        def trigger(index, _captures):
            calls.append(index)
            return bool(fire.value[index])

        result = match(Pattern().wait(trigger), axis=fire)
        assert len(result) == 2
        assert calls == [0, 1, 2]

    @pytest.mark.parametrize('timeout', [0, -1, 1.5])
    def test_timeout_requires_positive_integer(self, timeout):
        fire = _bool_wf([1])
        with pytest.raises(PatternError, match='timeout'):
            match(Pattern().wait(fire), timeout=timeout)
        with pytest.raises(PatternError, match='timeout'):

            def collect_body(ctx):
                return ctx.OK if ctx.value(fire) else None

            collect(collect_body, timeout=timeout)

    def test_declarative_delay_dynamic_count_is_not_coerced(self):
        fire = _bool_wf([1, 0])
        with pytest.raises(PatternError, match='integer value'):
            match(Pattern().wait(fire).delay(lambda _idx, _cap: '1'))

    def test_declarative_repeat_dynamic_count_is_not_coerced(self):
        fire = _bool_wf([1])
        with pytest.raises(PatternError, match='integer value'):
            match(Pattern().wait(fire).repeat(Pattern(), lambda _idx, _cap: '2'))


class TestMatchRecords:
    def _mixed_status_result(self):
        start = Waveform(
            np.array([0, 1, 2], dtype=np.int64),
            np.array([10, 20, 30], dtype=np.int64),
            np.array([100, 200, 300], dtype=np.int64),
            width=64,
        )
        end = Waveform(
            np.array([0, 4, 3], dtype=np.int64),
            np.array([10, 23, 31], dtype=np.int64),
            np.array([100, 230, 310], dtype=np.int64),
            width=64,
        )
        duration = Waveform(
            np.array([1, 4, 2], dtype=np.int64), start.clock.copy(), start.time.copy(), width=64
        )
        status = Waveform(
            np.array(
                [MatchStatus.OK(), MatchStatus.Timeout(), MatchStatus.RequireViolated()],
                dtype=object,
            ),
            start.clock.copy(),
            start.time.copy(),
        )
        samples = np.empty(3, dtype=object)
        samples[:] = [[1], [2, 3], [4, 5, 6]]
        captures = {
            'data': Waveform(
                np.array([100, 200, 300], dtype=np.int64),
                start.clock.copy(),
                start.time.copy(),
                width=16,
            ),
            'samples': Waveform(samples, start.clock.copy(), start.time.copy()),
        }
        return MatchRecords(start, end, duration, status, captures)

    def test_filter_ok(self):
        sometimes = _bool_wf([0, 0, 1, 0, 0])
        data = _wf([0, 0, 42, 0, 0], width=8)
        trig2 = _bool_wf([1, 0, 1, 0, 0])
        result = match(Pattern().wait(trig2).wait(sometimes).capture('val', data), timeout=3)
        assert len(result) >= 1
        ok = result.filter_ok()
        assert all(s == MatchStatus.OK() for s in ok.status.value)

    def test_ok_replaces_valid_aliases(self):
        trigger = _bool_wf([1, 0])
        result = match(Pattern().wait(trigger))
        np.testing.assert_array_equal(result.ok.value, [True])
        assert not hasattr(result, 'valid')
        assert not hasattr(result, 'filter_valid')

    def test_failed_preserves_status_axis(self):
        result = self._mixed_status_result()
        np.testing.assert_array_equal(result.failed.value, [False, True, True])
        np.testing.assert_array_equal(result.failed.clock, result.status.clock)
        np.testing.assert_array_equal(result.failed.time, result.status.time)
        assert result.failed.width == 1
        assert result.failed.signed is False

    def test_filter_status_rejects_concrete_status_object(self):
        result = self._mixed_status_result()
        with pytest.raises(TypeError, match='status class'):
            result.filter_status(MatchStatus.Timeout())

    def test_filter_status_rejects_unrelated_class(self):
        result = self._mixed_status_result()
        with pytest.raises(TypeError, match='MatchStatus'):
            result.filter_status(object)

    def test_filter_status_accepts_status_class(self):
        result = self._mixed_status_result()
        timeout = result.filter_status(MatchStatus.Timeout)
        assert len(timeout) == 1
        assert timeout[0].status == MatchStatus.Timeout()
        np.testing.assert_array_equal(timeout.start.value, [1])
        np.testing.assert_array_equal(timeout.end.value, [4])
        np.testing.assert_array_equal(timeout.duration.value, [4])
        np.testing.assert_array_equal(timeout.captures['data'].value, [200])
        np.testing.assert_array_equal(timeout.captures['data'].clock, timeout.start.clock)
        assert list(timeout.captures['samples'].value[0]) == [2, 3]
        np.testing.assert_array_equal(timeout.captures['samples'].clock, timeout.start.clock)

    def test_integer_index_too_negative_raises_index_error(self):
        result = self._mixed_status_result()
        with pytest.raises(IndexError):
            _ = result[-len(result) - 1]

    def test_filter_failed_keeps_non_ok_statuses(self):
        result = self._mixed_status_result()
        failed = result.filter_failed()
        np.testing.assert_array_equal(
            failed.status.value, [MatchStatus.Timeout(), MatchStatus.RequireViolated()]
        )
        np.testing.assert_array_equal(failed.start.value, [1, 2])
        np.testing.assert_array_equal(failed.captures['data'].value, [200, 300])
        assert [list(value) for value in failed.captures['samples'].value] == [[2, 3], [4, 5, 6]]
        np.testing.assert_array_equal(failed.captures['samples'].clock, failed.start.clock)

    def test_ok_and_filter_ok_preserve_result_axes_and_list_captures(self):
        trigger = _bool_wf([1, 0, 1, 0, 0])
        ready = _bool_wf([0, 1, 0, 0, 0])
        data = _wf([10, 20, 30, 40, 50], width=8)
        result = match(Pattern().wait(trigger).capture('samples', data, mode='list').wait(ready))
        np.testing.assert_array_equal(result.ok.clock, result.status.clock)
        np.testing.assert_array_equal(result.ok.time, result.status.time)
        assert result.ok.width == 1
        ok = result.filter_ok()
        assert len(ok) == 1
        np.testing.assert_array_equal(ok.start.clock, [0])
        np.testing.assert_array_equal(ok.captures['samples'].clock, ok.start.clock)
        assert list(ok.captures['samples'].value[0]) == [10]

    def test_start_end_points_store_index_cycle_and_time(self):
        trigger = Waveform(
            np.array([1, 0, 0], dtype=np.int64),
            np.array([10, 20, 30], dtype=np.int64),
            np.array([100, 200, 300], dtype=np.int64),
            width=1,
        )
        result = match(Pattern().wait(trigger).delay(1))
        assert len(result) == 1
        np.testing.assert_array_equal(result.start.value, [0])
        np.testing.assert_array_equal(result.start.clock, [10])
        np.testing.assert_array_equal(result.start.time, [100])
        np.testing.assert_array_equal(result.end.value, [1])
        np.testing.assert_array_equal(result.end.clock, [20])
        np.testing.assert_array_equal(result.end.time, [200])
        np.testing.assert_array_equal(result.duration.value, [2])
        record = result[0]
        assert record.start.index == 0
        assert record.start.cycle == 10
        assert record.start.time == 100
        assert record.end.index == 1
        assert record.end.cycle == 20
        assert record.end.time == 200
        assert record.duration == 2

    def test_row_access_and_slicing(self):
        result = self._mixed_status_result()
        first = result[0]
        assert isinstance(first, MatchRecord)
        assert first.start.index == 0
        assert first.end.cycle == 10

        sliced = result[1:]
        assert isinstance(sliced, MatchRecords)
        np.testing.assert_array_equal(sliced.start.value, [1, 2])
        np.testing.assert_array_equal(sliced.end.value, [4, 3])
        assert isinstance(list(sliced)[0], MatchRecord)

    def test_repr(self):
        trigger = _bool_wf([1, 0])
        result = match(Pattern().wait(trigger))
        assert 'MatchRecords' in repr(result)

    def test_len(self):
        trigger = _bool_wf([1, 1, 1])
        result = match(Pattern().wait(trigger))
        assert len(result) == 3


class TestEveryCycleFork:
    def test_capture_delay_capture(self):
        """Pattern without wait → fork every cycle, delay creates pairing."""
        a = _wf([10, 20, 30, 40, 50], width=8)
        b = _wf([100, 200, 300, 400, 500], width=8)
        result = match(Pattern().capture('a', a).delay(2).capture('b', b))
        assert len(result) == 5
        ok = result.filter_ok()
        assert len(ok) == 3
        np.testing.assert_array_equal(ok.captures['a'].value, [10, 20, 30])
        np.testing.assert_array_equal(ok.captures['b'].value, [300, 400, 500])

    def test_pure_capture(self):
        """Pure epsilon pattern → every cycle completes immediately."""
        sig = _wf([1, 2, 3], width=8)
        result = match(Pattern().capture('v', sig))
        assert len(result) == 3
        np.testing.assert_array_equal(result.captures['v'].value, [1, 2, 3])

    def test_with_start_end_cycle(self):
        """start_cycle/end_cycle limits fork range."""
        sig = _wf([10, 20, 30, 40, 50], width=8)
        result = match(Pattern().capture('v', sig), start_cycle=1, end_cycle=4)
        assert len(result) == 3
        np.testing.assert_array_equal(result.captures['v'].value, [20, 30, 40])


class TestCaptureModes:
    def test_mode_first_in_loop(self):
        """mode='first' keeps only the first write inside a loop body."""
        trigger = _bool_wf([1, 0, 0, 0, 0])
        beat = _bool_wf([0, 1, 1, 1, 0])
        last = _bool_wf([0, 0, 0, 0, 1])
        data = _wf([0, 10, 20, 30, 0], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .loop(
                Pattern()
                .wait(beat)
                .capture('first_d', data, mode='first')
                .branch(last == 0, true_body=Pattern().delay(1)),
                until=last,
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['first_d'].value[0] == 10

    def test_mode_last_default(self):
        """mode='last' (default) overwrites — last write wins."""
        trigger = _bool_wf([1, 0, 0, 0, 0])
        beat = _bool_wf([0, 1, 1, 1, 0])
        last = _bool_wf([0, 0, 0, 1, 0])
        data = _wf([0, 10, 20, 30, 0], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .loop(
                Pattern()
                .wait(beat)
                .capture('last_d', data)
                .branch(last == 0, true_body=Pattern().delay(1)),
                until=last,
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['last_d'].value[0] == 20

    def test_mode_invalid_raises(self):
        with pytest.raises(ValueError, match='mode'):
            Pattern().capture('x', _wf([1]), mode='whatever')
