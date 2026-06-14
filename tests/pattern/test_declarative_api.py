"""Declarative Pattern builder/API and compiler control-flow tests."""

import numpy as np
import pytest
from helpers import bool_wf as _bool_wf
from helpers import wf as _wf

from wavekit.pattern import MatchStatus, Pattern

# ---------------------------------------------------------------------------


class TestSingleWait:
    def test_basic_trigger(self):
        sig = _bool_wf([0, 1, 0, 1, 1, 0])
        result = Pattern().wait(sig).match()
        assert len(result) == 3
        np.testing.assert_array_equal(result.start.value, [1, 3, 4])
        np.testing.assert_array_equal(result.status.value, [0, 0, 0])  # all OK

    def test_no_matches(self):
        sig = _bool_wf([0, 0, 0])
        result = Pattern().wait(sig).match()
        assert len(result) == 0

    def test_all_ones(self):
        sig = _bool_wf([1, 1, 1])
        result = Pattern().wait(sig).match()
        assert len(result) == 3


class TestWaitCapture:
    def test_capture_value(self):
        valid = _bool_wf([0, 1, 0, 1, 0])
        data = _wf([10, 20, 30, 40, 50], width=8)
        result = Pattern().wait(valid).capture('data', data).match()
        assert len(result) == 2
        np.testing.assert_array_equal(result.captures['data'].value, [20, 40])

    def test_two_phase_handshake(self):
        req = _bool_wf([0, 1, 0, 0, 0, 1, 0, 0])
        ack = _bool_wf([0, 0, 0, 1, 0, 0, 1, 0])
        data = _wf([0, 0, 0, 99, 0, 0, 77, 0], width=8)
        result = Pattern().wait(req).wait(ack).capture('data', data).match()
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
        result = (
            Pattern()
            .wait(start)
            .loop(
                Pattern()
                .wait(beat)
                .capture('d', data, mode='list')
                .branch(last == 0, true_body=Pattern().delay(1)),
                until=last,
            )
            .match()
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
        result = (
            Pattern()
            .wait(start)
            .loop(
                Pattern().wait(beat).capture('d', data, mode='list'),
                until=last,
            )
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert list(ok.captures['d'].value[0]) == [99]


class TestLoopWhile:
    def test_capture_while_high(self):
        """Capture data each cycle while enable is high."""
        # Use a single trigger point, then loop while high
        trigger = _bool_wf([0, 1, 0, 0, 0, 0])
        enable = _bool_wf([0, 1, 1, 1, 0, 0])
        data = _wf([0, 10, 20, 30, 40, 50], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .loop(
                Pattern().capture('d', data, mode='list').delay(1),
                when=enable,
            )
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert list(ok.captures['d'].value[0]) == [10, 20, 30]

    def test_while_false_immediately(self):
        """When condition is False at entry, loop is skipped (0 iterations)."""
        trigger = _bool_wf([1, 0, 0])
        cond = _bool_wf([0, 0, 0])
        data = _wf([10, 20, 30], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .loop(
                Pattern().delay(1).capture('d', data, mode='list'),
                when=cond,
            )
            .capture('after', data)
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        # Loop skipped, capture happens at trigger cycle
        assert ok.captures['after'].value[0] == 10
        # No loop captures
        assert 'd' not in ok.captures or list(ok.captures['d'].value[0]) == []


class TestStallDetection:
    def test_stall_interval(self):
        """Find stall intervals using loop-until."""
        #                      0  1  2  3  4  5  6  7  8  9
        stall = _bool_wf([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
        trigger = stall.rising_edge()  # rising edge at cycles 1 and 7
        result = Pattern().wait(trigger).loop(Pattern().delay(1), until=stall == 0).match()
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
        result = (
            Pattern()
            .wait(trigger)
            .repeat(
                Pattern().wait(beat).capture('d', data, mode='list').delay(1),
                n=3,
            )
            .match()
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
        result = (
            Pattern()
            .wait(trigger)
            .capture('len', len_sig)
            .repeat(
                Pattern().wait(beat).capture('d', data, mode='list').delay(1),
                n=lambda idx, cap: cap['len'],
            )
            .match()
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
        result = (
            Pattern()
            .wait(trigger)
            .capture('mode', mode)
            .capture(
                'val', lambda idx, cap: sig_a.value[idx] if cap['mode'] == 0 else sig_b.value[idx]
            )
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == 200  # mode=1 → sig_b


class TestBranch:
    @pytest.mark.parametrize(
        ('cond_value', 'expected'),
        [
            (1, 10),
            (0, 20),
        ],
    )
    def test_branch_selects_body(self, cond_value, expected):
        trigger = _bool_wf([0, 1, 0])
        cond = _bool_wf([0, cond_value, 0])
        data_a = _wf([0, 10, 0], width=8)
        data_b = _wf([0, 20, 0], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .branch(
                cond,
                true_body=Pattern().capture('val', data_a),
                false_body=Pattern().capture('val', data_b),
            )
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == expected

    def test_branch_none_body(self):
        """Branch with no false_body → skip."""
        trigger = _bool_wf([0, 1, 0])
        cond = _bool_wf([0, 0, 0])
        data = _wf([0, 42, 0], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .branch(cond, true_body=Pattern().capture('optional', data))
            .capture('always', data)
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['always'].value[0] == 42
        assert 'optional' not in ok.captures


# ---------------------------------------------------------------------------
# MatchResult helpers
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_filter_ok(self):
        sometimes = _bool_wf([0, 0, 1, 0, 0])
        data = _wf([0, 0, 42, 0, 0], width=8)
        trig2 = _bool_wf([1, 0, 1, 0, 0])
        result = Pattern().wait(trig2).wait(sometimes).capture('val', data).timeout(3).match()
        # Instance 1: fork at 0, wait sometimes → cycle 2 → OK, capture val=42
        # Instance 2: fork at 2, wait sometimes → needs next True... cycle 2 already past
        #   for this instance, it starts at step_idx=1 (skip first wait), wait sometimes
        #   at cycle 3,4 → never True → TIMEOUT
        assert len(result) >= 1
        ok = result.filter_ok()
        assert all(s == MatchStatus.OK for s in ok.status.value)

    def test_ok_replaces_valid_aliases(self):
        trigger = _bool_wf([1, 0])
        result = Pattern().wait(trigger).match()

        np.testing.assert_array_equal(result.ok.value, [True])
        assert not hasattr(result, 'valid')
        assert not hasattr(result, 'filter_valid')

    def test_ok_and_filter_ok_preserve_result_axes_and_list_captures(self):
        trigger = _bool_wf([1, 0, 1, 0, 0])
        ready = _bool_wf([0, 1, 0, 0, 0])
        data = _wf([10, 20, 30, 40, 50], width=8)

        result = (
            Pattern()
            .wait(trigger)
            .capture('samples', data, mode='list')
            .wait(ready)
            .timeout(3)
            .match()
        )

        np.testing.assert_array_equal(result.ok.clock, result.status.clock)
        np.testing.assert_array_equal(result.ok.time, result.status.time)
        assert result.ok.width == 1

        ok = result.filter_ok()
        assert len(ok) == 1
        np.testing.assert_array_equal(ok.start.clock, [0])
        np.testing.assert_array_equal(ok.captures['samples'].clock, ok.start.clock)
        assert list(ok.captures['samples'].value[0]) == [10]

    def test_repr(self):
        trigger = _bool_wf([1, 0])
        result = Pattern().wait(trigger).match()
        assert 'MatchResult' in repr(result)

    def test_len(self):
        trigger = _bool_wf([1, 1, 1])
        result = Pattern().wait(trigger).match()
        assert len(result) == 3


class TestEveryCycleFork:
    def test_capture_delay_capture(self):
        """Pattern without wait → fork every cycle, delay creates pairing."""
        a = _wf([10, 20, 30, 40, 50], width=8)
        b = _wf([100, 200, 300, 400, 500], width=8)
        result = Pattern().capture('a', a).delay(2).capture('b', b).match()
        # Forks at cycles 0,1,2,3,4. Only 0,1,2 complete (need 2 more cycles).
        # Cycles 3,4 are incomplete → TIMEOUT.
        assert len(result) == 5
        ok = result.filter_ok()
        assert len(ok) == 3
        np.testing.assert_array_equal(ok.captures['a'].value, [10, 20, 30])
        np.testing.assert_array_equal(ok.captures['b'].value, [300, 400, 500])

    def test_pure_capture(self):
        """Pure epsilon pattern → every cycle completes immediately."""
        sig = _wf([1, 2, 3], width=8)
        result = Pattern().capture('v', sig).match()
        assert len(result) == 3
        np.testing.assert_array_equal(result.captures['v'].value, [1, 2, 3])

    def test_with_start_end_cycle(self):
        """start_cycle/end_cycle limits fork range."""
        sig = _wf([10, 20, 30, 40, 50], width=8)
        result = Pattern().capture('v', sig).match(start_cycle=1, end_cycle=4)
        assert len(result) == 3
        np.testing.assert_array_equal(result.captures['v'].value, [20, 30, 40])


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestCaptureModes:
    def test_mode_first_in_loop(self):
        """mode='first' keeps only the first write inside a loop body."""
        trigger = _bool_wf([1, 0, 0, 0, 0])
        beat = _bool_wf([0, 1, 1, 1, 0])
        last = _bool_wf([0, 0, 0, 0, 1])
        data = _wf([0, 10, 20, 30, 0], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .loop(
                Pattern()
                .wait(beat)
                .capture('first_d', data, mode='first')
                .branch(last == 0, true_body=Pattern().delay(1)),
                until=last,
            )
            .match()
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
        result = (
            Pattern()
            .wait(trigger)
            .loop(
                Pattern()
                .wait(beat)
                .capture('last_d', data)
                .branch(last == 0, true_body=Pattern().delay(1)),
                until=last,
            )
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['last_d'].value[0] == 20

    def test_mode_invalid_raises(self):
        with pytest.raises(ValueError, match='mode'):
            Pattern().capture('x', _wf([1]), mode='whatever')
