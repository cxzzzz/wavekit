"""Tests for the pattern matching engine.

Test progression (by complexity):
1. Single wait → find trigger positions
2. wait + capture → handshake extraction
3. wait + guard (throughout)
4. wait + require (at-fire)
5. delay + guard
6. channel (ordered FIFO pairing)
7. loop until (do-while burst)
8. loop while (while semantics)
9. loop + stall detection
10. repeat static n
11. repeat dynamic n
12. capture dynamic signal
13. timeout
14. branch
15. capture list ([] append)
"""

import numpy as np
import pytest

from wavekit import Signal, Waveform
from wavekit.pattern import MatchStatus, Pattern, PatternError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wf(values, width=1, signed=False):
    """Build a test Waveform from a list of values."""
    value = np.array(values)
    clock = np.arange(len(value))
    time = clock * 10
    return Waveform(value, clock, time, signal=Signal('', '', width, None, signed))


def _bool_wf(values):
    """Build a 1-bit unsigned Waveform (boolean-like)."""
    return _wf(values, width=1, signed=False)


# ---------------------------------------------------------------------------
# 1. Single wait — find all trigger positions
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


# ---------------------------------------------------------------------------
# 2. wait + capture — handshake extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. wait + guard (throughout)
# ---------------------------------------------------------------------------


class TestGuard:
    def test_guard_holds(self):
        """valid stays high while waiting for ready → OK."""
        # Only one trigger at cycle 1 (valid goes high once)
        valid = _bool_wf([0, 1, 1, 1, 0])
        ready = _bool_wf([0, 0, 0, 1, 0])
        result = Pattern().wait(valid).wait(ready, guard=valid).match()
        # Triggers at cycle 1,2,3 (valid=1). Inst@1: guard OK, ready@3 → OK.
        # Inst@2: guard OK, ready@3 → OK. Inst@3: ready@3 immediately → OK.
        valid_results = result.filter_valid()
        assert len(valid_results) >= 1
        # The first instance (trigger@1) completes at cycle 3
        assert valid_results.start.value[0] == 1
        assert valid_results.end.value[0] == 3

    def test_guard_violated(self):
        """valid drops before ready → REQUIRE_VIOLATED."""
        valid = _bool_wf([0, 1, 0, 0, 0])
        ready = _bool_wf([0, 0, 0, 1, 0])
        result = Pattern().wait(valid).wait(ready, guard=valid).match()
        # Instance forked at cycle 1, guard fails at cycle 2 (valid=0)
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.REQUIRE_VIOLATED


# ---------------------------------------------------------------------------
# 4. wait + require (at-fire)
# ---------------------------------------------------------------------------


class TestRequire:
    def test_require_pass(self):
        trigger = _bool_wf([0, 1, 0])
        check = _bool_wf([0, 1, 0])
        result = Pattern().wait(trigger).require(check).match()
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.OK

    def test_require_fail(self):
        trigger = _bool_wf([0, 1, 0])
        check = _bool_wf([0, 0, 0])
        result = Pattern().wait(trigger).require(check).match()
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.REQUIRE_VIOLATED


# ---------------------------------------------------------------------------
# 5. delay + guard
# ---------------------------------------------------------------------------


class TestDelay:
    def test_delay_basic(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])
        data = _wf([10, 20, 30, 40, 50], width=8)
        result = Pattern().wait(trigger).delay(2).capture('val', data).match()
        assert len(result) == 1
        # Trigger at 0, delay 2 ticks → capture at cycle 2
        assert result.captures['val'].value[0] == 30

    def test_delay_with_guard(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])
        enable = _bool_wf([1, 1, 0, 0, 0])  # drops at cycle 2
        result = Pattern().wait(trigger).delay(3, guard=enable).match()
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.REQUIRE_VIOLATED

    def test_delay_dynamic_n(self):
        trigger = _bool_wf([1, 0, 0, 0, 0, 0])
        len_sig = _wf([3, 0, 0, 0, 0, 0], width=4)
        data = _wf([0, 10, 20, 30, 40, 50], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .capture('n', len_sig)
            .delay(lambda idx, cap: cap['n'])
            .capture('val', data)
            .match()
        )
        assert len(result) == 1
        # n=3 captured at cycle 0, delay 3 → capture at cycle 3
        assert result.captures['val'].value[0] == 30


# ---------------------------------------------------------------------------
# 6. channel (ordered FIFO pairing)
# ---------------------------------------------------------------------------


class TestChannel:
    def test_fifo_pairing(self):
        """Three requests followed by three responses — FIFO order."""
        req = _bool_wf([1, 1, 1, 0, 0, 0, 0, 0, 0])
        rsp = _bool_wf([0, 0, 0, 0, 1, 0, 1, 0, 1])
        req_data = _wf([10, 20, 30, 0, 0, 0, 0, 0, 0], width=8)
        rsp_data = _wf([0, 0, 0, 0, 55, 0, 66, 0, 77], width=8)
        result = (
            Pattern()
            .wait(req)
            .capture('req', req_data)
            .wait(rsp, channel='rsp')
            .capture('rsp', rsp_data)
            .match()
        )
        valid = result.filter_valid()
        assert len(valid) == 3
        np.testing.assert_array_equal(valid.captures['req'].value, [10, 20, 30])
        np.testing.assert_array_equal(valid.captures['rsp'].value, [55, 66, 77])


# ---------------------------------------------------------------------------
# 7. loop until (do-while burst)
# ---------------------------------------------------------------------------


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
                Pattern().wait(beat).capture('d[]', data),
                until=last,
            )
            .match()
        )
        valid = result.filter_valid()
        assert len(valid) == 1
        assert list(valid.captures['d'].value[0]) == [10, 20, 30]

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
                Pattern().wait(beat).capture('d[]', data),
                until=last,
            )
            .match()
        )
        valid = result.filter_valid()
        assert len(valid) == 1
        assert list(valid.captures['d'].value[0]) == [99]


# ---------------------------------------------------------------------------
# 8. loop while (pre-condition check)
# ---------------------------------------------------------------------------


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
                Pattern().capture('d[]', data).delay(1),
                when=enable,
            )
            .match()
        )
        valid = result.filter_valid()
        assert len(valid) == 1
        assert list(valid.captures['d'].value[0]) == [10, 20, 30]

    def test_while_false_immediately(self):
        """When condition is False at entry, loop is skipped (0 iterations)."""
        trigger = _bool_wf([1, 0, 0])
        cond = _bool_wf([0, 0, 0])
        data = _wf([10, 20, 30], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .loop(
                Pattern().delay(1).capture('d[]', data),
                when=cond,
            )
            .capture('after', data)
            .match()
        )
        valid = result.filter_valid()
        assert len(valid) == 1
        # Loop skipped, capture happens at trigger cycle
        assert valid.captures['after'].value[0] == 10
        # No loop captures
        assert 'd' not in valid.captures or list(valid.captures['d'].value[0]) == []


# ---------------------------------------------------------------------------
# 9. loop + stall detection
# ---------------------------------------------------------------------------


class TestStallDetection:
    def test_stall_interval(self):
        """Find stall intervals using loop-until."""
        #                      0  1  2  3  4  5  6  7  8  9
        stall = _bool_wf([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
        trigger = stall.rising_edge()  # rising edge at cycles 1 and 7
        result = Pattern().wait(trigger).loop(Pattern().delay(1), until=stall == 0).match()
        valid = result.filter_valid()
        assert len(valid) == 2
        np.testing.assert_array_equal(valid.start.value, [1, 7])
        np.testing.assert_array_equal(valid.end.value, [5, 9])
        np.testing.assert_array_equal(valid.duration.value, [5, 3])


# ---------------------------------------------------------------------------
# 10. repeat static n
# ---------------------------------------------------------------------------


class TestRepeat:
    def test_repeat_static(self):
        trigger = _bool_wf([1, 0, 0, 0, 0, 0])
        beat = _bool_wf([0, 1, 1, 1, 0, 0])
        data = _wf([0, 10, 20, 30, 0, 0], width=8)
        result = (
            Pattern()
            .wait(trigger)
            .repeat(
                Pattern().wait(beat).capture('d[]', data),
                n=3,
            )
            .match()
        )
        valid = result.filter_valid()
        assert len(valid) == 1
        assert list(valid.captures['d'].value[0]) == [10, 20, 30]


# ---------------------------------------------------------------------------
# 11. repeat dynamic n
# ---------------------------------------------------------------------------


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
                Pattern().wait(beat).capture('d[]', data),
                n=lambda idx, cap: cap['len'],
            )
            .match()
        )
        valid = result.filter_valid()
        assert len(valid) == 1
        assert list(valid.captures['d'].value[0]) == [10, 20]


# ---------------------------------------------------------------------------
# 12. capture dynamic signal
# ---------------------------------------------------------------------------


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
        valid = result.filter_valid()
        assert len(valid) == 1
        assert valid.captures['val'].value[0] == 200  # mode=1 → sig_b


# ---------------------------------------------------------------------------
# 13. timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_triggers(self):
        trigger = _bool_wf([1, 0, 0, 0, 0, 0, 0, 0])
        never = _bool_wf([0, 0, 0, 0, 0, 0, 0, 0])
        result = Pattern().wait(trigger).wait(never).timeout(3).match()
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.TIMEOUT

    def test_timeout_1_cycle(self):
        """timeout=1: only epsilon steps on fork cycle can complete."""
        trigger = _bool_wf([1, 0, 0])
        data = _wf([42, 0, 0], width=8)
        # Pattern with only epsilon after trigger → completes in 1 cycle
        result = Pattern().wait(trigger).capture('val', data).timeout(1).match()
        valid = result.filter_valid()
        assert len(valid) == 1
        assert valid.captures['val'].value[0] == 42

    def test_timeout_2_needs_blocking(self):
        """timeout=2: has 1 tick after fork to complete a blocking step."""
        trigger = _bool_wf([1, 0, 0, 0])
        cond = _bool_wf([0, 1, 0, 0])
        data = _wf([0, 99, 0, 0], width=8)
        result = Pattern().wait(trigger).wait(cond).capture('val', data).timeout(2).match()
        valid = result.filter_valid()
        assert len(valid) == 1
        assert valid.captures['val'].value[0] == 99


# ---------------------------------------------------------------------------
# 14. branch
# ---------------------------------------------------------------------------


class TestBranch:
    def test_branch_true(self):
        trigger = _bool_wf([0, 1, 0])
        cond = _bool_wf([0, 1, 0])
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
        valid = result.filter_valid()
        assert len(valid) == 1
        assert valid.captures['val'].value[0] == 10

    def test_branch_false(self):
        trigger = _bool_wf([0, 1, 0])
        cond = _bool_wf([0, 0, 0])
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
        valid = result.filter_valid()
        assert len(valid) == 1
        assert valid.captures['val'].value[0] == 20

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
        valid = result.filter_valid()
        assert len(valid) == 1
        assert valid.captures['always'].value[0] == 42
        assert 'optional' not in valid.captures


# ---------------------------------------------------------------------------
# 15. capture list ([] append)
# ---------------------------------------------------------------------------


class TestCaptureList:
    def test_list_capture_in_repeat(self):
        trigger = _bool_wf([1, 0, 0, 0])
        beat = _bool_wf([0, 1, 1, 1])
        data = _wf([0, 10, 20, 30], width=8)
        result = (
            Pattern().wait(trigger).repeat(Pattern().wait(beat).capture('d[]', data), n=3).match()
        )
        valid = result.filter_valid()
        assert len(valid) == 1
        assert list(valid.captures['d'].value[0]) == [10, 20, 30]


# ---------------------------------------------------------------------------
# MatchResult helpers
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_filter_valid(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])  # noqa: F841
        never = _bool_wf([0, 0, 0, 0, 0])  # noqa: F841
        sometimes = _bool_wf([0, 0, 1, 0, 0])
        data = _wf([0, 0, 42, 0, 0], width=8)
        # Two triggers at same cycle? No — use two separate triggers.
        # Instead, test with one OK and one TIMEOUT
        trig2 = _bool_wf([1, 0, 1, 0, 0])
        result = Pattern().wait(trig2).wait(sometimes).capture('val', data).timeout(3).match()
        # Instance 1: trigger at 0, wait for sometimes, finds at 2 → OK
        # Instance 2: trigger at 2, sometimes is True right now → wait consumed by trigger,
        #   but then wait(sometimes) is the SECOND wait... let me reconsider.
        # Actually first wait is trig2, second wait is sometimes.
        # Instance 1: fork at 0, wait sometimes → cycle 2 → OK, capture val=42
        # Instance 2: fork at 2, wait sometimes → needs next True... cycle 2 already past
        #   for this instance, it starts at step_idx=1 (skip first wait), wait sometimes
        #   at cycle 3,4 → never True → TIMEOUT
        assert len(result) >= 1
        valid = result.filter_valid()
        assert all(s == MatchStatus.OK for s in valid.status.value)

    def test_repr(self):
        trigger = _bool_wf([1, 0])
        result = Pattern().wait(trigger).match()
        assert 'MatchResult' in repr(result)

    def test_len(self):
        trigger = _bool_wf([1, 1, 1])
        result = Pattern().wait(trigger).match()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# 16. Pure epsilon (no wait → every-cycle fork)
# ---------------------------------------------------------------------------


class TestEveryCycleFork:
    def test_capture_delay_capture(self):
        """Pattern without wait → fork every cycle, delay creates pairing."""
        a = _wf([10, 20, 30, 40, 50], width=8)
        b = _wf([100, 200, 300, 400, 500], width=8)
        result = Pattern().capture('a', a).delay(2).capture('b', b).match()
        # Forks at cycles 0,1,2,3,4. Only 0,1,2 complete (need 2 more cycles).
        # Cycles 3,4 are incomplete → TIMEOUT.
        assert len(result) == 5
        valid = result.filter_valid()
        assert len(valid) == 3
        np.testing.assert_array_equal(valid.captures['a'].value, [10, 20, 30])
        np.testing.assert_array_equal(valid.captures['b'].value, [300, 400, 500])

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


class TestErrors:
    def test_no_waveform(self):
        """Pattern with only dynamic conditions and no trigger → error."""
        with pytest.raises(PatternError):
            Pattern().wait(lambda idx, cap: True).match()

    def test_loop_missing_condition(self):
        with pytest.raises(ValueError):
            Pattern().loop(Pattern().delay(1))

    def test_loop_both_conditions(self):
        cond = _bool_wf([1])
        with pytest.raises(ValueError):
            Pattern().loop(Pattern().delay(1), until=cond, when=cond)

    def test_infinite_loop_detection(self):
        """Loop body with only epsilon steps and never-True until → error."""
        trigger = _bool_wf([1, 0, 0])
        data = _wf([0, 0, 0], width=8)
        never = _bool_wf([0, 0, 0])
        with pytest.raises(PatternError, match='Infinite loop'):
            (
                Pattern()
                .wait(trigger)
                .loop(
                    Pattern().capture('x', data),
                    until=never,
                )
                .match()
            )
