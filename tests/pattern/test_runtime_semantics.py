"""Core Pattern runtime semantics shared by declarative and programmable APIs."""

from collections import defaultdict

import numpy as np
import pytest

from helpers import bool_wf as _bool_wf
from helpers import wf as _wf
from wavekit import Waveform
from wavekit.pattern import Channel, MatchStatus, Pattern, PatternError, match


class TestWaitRequire:
    def test_require_holds(self):
        """valid stays high while waiting for ready → OK."""
        valid = _bool_wf([0, 1, 1, 1, 0])
        ready = _bool_wf([0, 0, 0, 1, 0])
        result = match(Pattern().wait(valid).wait(ready, require=valid))
        ok = result.filter_ok()
        assert len(ok) >= 1
        assert ok.start.value[0] == 1
        assert ok.end.value[0] == 3

    def test_require_violated(self):
        """valid drops before ready → REQUIRE_VIOLATED."""
        valid = _bool_wf([0, 1, 0, 0, 0])
        ready = _bool_wf([0, 0, 0, 1, 0])
        result = match(Pattern().wait(valid).wait(ready, require=valid))
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.RequireViolated()


class TestRequire:
    def test_require_pass(self):
        trigger = _bool_wf([0, 1, 0])
        check = _bool_wf([0, 1, 0])
        result = match(Pattern().wait(trigger).require(check))
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.OK()

    def test_require_fail(self):
        trigger = _bool_wf([0, 1, 0])
        check = _bool_wf([0, 0, 0])
        result = match(Pattern().wait(trigger).require(check))
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.RequireViolated()


class TestDelay:
    def test_delay_basic(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])
        data = _wf([10, 20, 30, 40, 50], width=8)
        result = match(Pattern().wait(trigger).delay(2).capture('val', data))
        assert len(result) == 1
        assert result.captures['val'].value[0] == 30

    def test_delay_with_require(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])
        enable = _bool_wf([1, 1, 0, 0, 0])
        result = match(Pattern().wait(trigger).delay(3, require=enable))
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.RequireViolated()

    def test_delay_dynamic_n(self):
        trigger = _bool_wf([1, 0, 0, 0, 0, 0])
        len_sig = _wf([3, 0, 0, 0, 0, 0], width=4)
        data = _wf([0, 10, 20, 30, 40, 50], width=8)
        result = match(
            Pattern()
            .wait(trigger)
            .capture('n', len_sig)
            .delay(lambda idx, cap: cap['n'])
            .capture('val', data)
        )
        assert len(result) == 1
        assert result.captures['val'].value[0] == 30

    def test_delay_zero_does_not_check_require(self):
        trigger = _bool_wf([1, 0])
        guard = _bool_wf([0, 0])
        result = match(Pattern().wait(trigger).delay(0, require=guard)).filter_ok()
        assert len(result) == 1
        assert result.start.value[0] == 0
        assert result.end.value[0] == 0


class TestConsumeWithChannel:
    """Tests for consume(channel=...) FIFO consumption semantics."""

    def test_fifo_pairing(self):
        """Three requests followed by three responses — FIFO order."""
        req = _bool_wf([1, 1, 1, 0, 0, 0, 0, 0, 0])
        rsp = _bool_wf([0, 0, 0, 0, 1, 0, 1, 0, 1])
        req_data = _wf([10, 20, 30, 0, 0, 0, 0, 0, 0], width=8)
        rsp_data = _wf([0, 0, 0, 0, 55, 0, 66, 0, 77], width=8)
        rsp_chan = Channel()
        result = match(
            Pattern()
            .wait(req)
            .capture('req', req_data)
            .consume(rsp, channel=rsp_chan)
            .capture('rsp', rsp_data)
        )
        ok = result.filter_ok()
        assert len(ok) == 3
        np.testing.assert_array_equal(ok.captures['req'].value, [10, 20, 30])
        np.testing.assert_array_equal(ok.captures['rsp'].value, [55, 66, 77])

    def test_multiple_channels(self):
        """Different Channels within the SAME Pattern via branching.

        When a Pattern uses branch() to select different wait channels,
        each channel maintains its own independent FIFO state.
        """
        req = _bool_wf([1, 1, 1, 1, 0, 0, 0, 0])
        req_type = _wf([0, 1, 0, 1, 0, 0, 0, 0], width=4)
        req_data = _wf([10, 20, 30, 40, 0, 0, 0, 0], width=8)
        rd_rsp = _bool_wf([0, 0, 0, 1, 0, 1, 0, 0])
        wr_rsp = _bool_wf([0, 0, 0, 0, 1, 0, 1, 0])
        rd_rsp_data = _wf([0, 0, 0, 111, 0, 333, 0, 0], width=8)
        wr_rsp_data = _wf([0, 0, 0, 0, 222, 0, 444, 0], width=8)
        rd_chan = Channel()
        wr_chan = Channel()
        result = match(
            Pattern()
            .wait(req)
            .capture('req_type', req_type)
            .capture('req_data', req_data)
            .branch(
                lambda idx, cap: cap['req_type'] == 0,
                Pattern().consume(rd_rsp, channel=rd_chan).capture('rsp_data', rd_rsp_data),
                Pattern().consume(wr_rsp, channel=wr_chan).capture('rsp_data', wr_rsp_data),
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 4
        rd_matches = [i for i in range(len(ok)) if ok.captures['req_type'].value[i] == 0]
        wr_matches = [i for i in range(len(ok)) if ok.captures['req_type'].value[i] == 1]
        rd_req = [ok.captures['req_data'].value[i] for i in rd_matches]
        rd_rsp_val = [ok.captures['rsp_data'].value[i] for i in rd_matches]
        wr_req = [ok.captures['req_data'].value[i] for i in wr_matches]
        wr_rsp_val = [ok.captures['rsp_data'].value[i] for i in wr_matches]
        assert rd_req == [10, 30], f'rd channel FIFO: {rd_req}'
        assert rd_rsp_val == [111, 333], f'rd rsp FIFO: {rd_rsp_val}'
        assert wr_req == [20, 40], f'wr channel FIFO: {wr_req}'
        assert wr_rsp_val == [222, 444], f'wr rsp FIFO: {wr_rsp_val}'

    def test_multi_id_multi_match_per_id(self):
        """Multiple IDs with multiple transactions per ID.

        Each channel maintains FIFO order independently.

        This tests the key scenario: each ID has its own FIFO channel, and multiple
        transactions with the same ID are matched in FIFO order within that channel,
        while different IDs are independent.
        """
        req = _bool_wf([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        req_id = _wf([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], width=4)
        req_data = _wf([10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0], width=8)
        rsp = _bool_wf([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0])
        rsp_id = _wf([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], width=4)
        rsp_data = _wf([0, 0, 0, 0, 0, 111, 0, 222, 0, 333, 444, 0], width=8)

        def match_rsp_with_id(idx, cap):
            """Condition: rsp is True AND rsp_id matches the captured req_id."""
            return bool(rsp.value[idx]) and int(rsp_id.value[idx]) == int(cap['req_id'])

        chans = defaultdict(Channel)
        result = match(
            Pattern()
            .wait(req)
            .capture('req_id', req_id)
            .capture('req_data', req_data)
            .consume(match_rsp_with_id, channel=lambda idx, cap: chans[int(cap['req_id'])])
            .capture('rsp_id', rsp_id)
            .capture('rsp_data', rsp_data)
        )
        ok = result.filter_ok()
        assert len(ok) == 4
        id_0_matches = [i for i in range(len(ok)) if ok.captures['req_id'].value[i] == 0]
        id_1_matches = [i for i in range(len(ok)) if ok.captures['req_id'].value[i] == 1]
        assert len(id_0_matches) == 2
        assert len(id_1_matches) == 2
        id_0_req_data = [ok.captures['req_data'].value[i] for i in id_0_matches]
        id_0_rsp_data = [ok.captures['rsp_data'].value[i] for i in id_0_matches]
        assert id_0_req_data == [10, 30], f'ID 0 req_data should be FIFO: {id_0_req_data}'
        assert id_0_rsp_data == [222, 444], f'ID 0 rsp_data should be FIFO: {id_0_rsp_data}'
        id_1_req_data = [ok.captures['req_data'].value[i] for i in id_1_matches]
        id_1_rsp_data = [ok.captures['rsp_data'].value[i] for i in id_1_matches]
        assert id_1_req_data == [20, 40], f'ID 1 req_data should be FIFO: {id_1_req_data}'
        assert id_1_rsp_data == [111, 333], f'ID 1 rsp_data should be FIFO: {id_1_rsp_data}'

    def test_dynamic_channel_per_id(self):
        """Dynamic channel based on captured transaction ID (AXI-like routing).

        Key insight: Dynamic channel creates independent FIFOs per channel value.
        For AXI-style ID routing, combine:
        1. Dynamic channel to separate channels per ID
        2. Condition that checks response ID matches request ID
        """
        req = _bool_wf([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        req_id = _wf([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], width=4)
        req_data = _wf([100, 0, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0], width=8)
        rsp = _bool_wf([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
        rsp_id = _wf([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], width=4)
        rsp_data = _wf([0, 0, 0, 0, 0, 222, 0, 0, 111, 0, 0, 0], width=8)

        def match_id(idx, cap):
            """Condition: rsp is True AND rsp_id matches the captured req_id."""
            return bool(rsp.value[idx]) and int(rsp_id.value[idx]) == int(cap['req_id'])

        chans = defaultdict(Channel)
        result = match(
            Pattern()
            .wait(req)
            .capture('req_id', req_id)
            .capture('req_data', req_data)
            .consume(match_id, channel=lambda idx, cap: chans[int(cap['req_id'])])
            .capture('rsp_id', rsp_id)
            .capture('rsp_data', rsp_data)
        )
        ok = result.filter_ok()
        assert len(ok) == 2
        for i in range(len(ok)):
            req_id_val = ok.captures['req_id'].value[i]
            rsp_id_val = ok.captures['rsp_id'].value[i]
            assert req_id_val == rsp_id_val, f'ID mismatch: req={req_id_val}, rsp={rsp_id_val}'

    def test_falsy_hashable_channel_key(self):
        """Falsy channel keys such as 0 still select a stable explicit channel."""
        req = _bool_wf([1, 1, 0, 0])
        rsp = _bool_wf([0, 0, 1, 1])
        req_data = _wf([10, 20, 0, 0], width=8)
        rsp_data = _wf([0, 0, 111, 222], width=8)
        result = match(
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .consume(rsp, channel=lambda _idx, _cap: 0)
            .capture('rsp_data', rsp_data)
        ).filter_ok()
        np.testing.assert_array_equal(result.captures['req_data'].value, [10, 20])
        np.testing.assert_array_equal(result.captures['rsp_data'].value, [111, 222])

    def test_dynamic_channel_fifo_per_channel(self):
        """Each dynamic channel value maintains its own FIFO order.

        When two instances have the same channel key, they share a FIFO.
        When they have different keys, they are independent.
        """
        req = _bool_wf([1, 1, 0, 0, 0, 0, 0, 0])
        req_data = _wf([10, 11, 0, 0, 0, 0, 0, 0], width=8)
        rsp = _bool_wf([0, 0, 0, 1, 0, 1, 0, 0])
        rsp_data = _wf([0, 0, 0, 111, 0, 222, 0, 0], width=8)
        chans = defaultdict(Channel)
        result = match(
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .consume(rsp, channel=lambda idx, cap: chans[int(cap['req_data']) // 10])
            .capture('rsp_data', rsp_data)
        )
        ok = result.filter_ok()
        assert len(ok) == 2
        np.testing.assert_array_equal(ok.captures['req_data'].value, [10, 11])
        np.testing.assert_array_equal(ok.captures['rsp_data'].value, [111, 222])

    def test_consume_as_first_step(self):
        """consume as first step: no trigger optimization,
        channel is consumed.

        When the first step consumes a channel, the engine must not use it as a
        trigger (which would skip channel consumption).  Instead, instances are
        forked every cycle and the consume step processes normally.
        """
        rsp = _bool_wf([0, 0, 1, 0, 1, 0])
        rsp_data = _wf([0, 0, 111, 0, 222, 0], width=8)
        rsp_chan = Channel()
        result = match(Pattern().consume(rsp, channel=rsp_chan).capture('rsp_data', rsp_data))
        ok = result.filter_ok()
        assert len(ok) == 2
        np.testing.assert_array_equal(ok.captures['rsp_data'].value, [111, 222])

    def test_require_with_channel(self):
        """require violation while consuming with channel → REQUIRE_VIOLATED."""
        req = _bool_wf([1, 1, 0, 0, 0, 0])
        req_data = _wf([10, 20, 0, 0, 0, 0], width=8)
        rsp = _bool_wf([0, 0, 0, 0, 1, 0])
        enable = _bool_wf([1, 1, 1, 0, 0, 0])
        rsp_chan = Channel()
        result = match(
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .consume(rsp, channel=rsp_chan, require=enable)
        )
        assert len(result) == 2
        assert all(s == MatchStatus.RequireViolated() for s in result.status.value)

    def test_require_with_channel_not_checked_on_success_cycle(self):
        req = _bool_wf([1, 0, 0])
        rsp = _bool_wf([0, 1, 0])
        guard = _bool_wf([1, 0, 0])
        result = match(Pattern().wait(req).consume(rsp, channel='rsp', require=guard)).filter_ok()
        assert len(result) == 1
        assert result.start.value[0] == 0
        assert result.end.value[0] == 1

    def test_timeout_with_channel(self):
        """Instance consuming with channel times out → next instance can consume."""
        req = _bool_wf([1, 1, 0, 0, 0, 0])
        req_data = _wf([10, 20, 0, 0, 0, 0], width=8)
        rsp = _bool_wf([0, 0, 0, 0, 0, 1])
        rsp_data = _wf([0, 0, 0, 0, 0, 99], width=8)
        rsp_chan = Channel()
        pattern = (
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .consume(rsp, channel=rsp_chan)
            .capture('rsp_data', rsp_data)
        )
        result = match(pattern, timeout=5)
        assert len(result) == 2
        statuses = list(result.status.value)
        assert MatchStatus.Timeout() in statuses
        assert MatchStatus.OK() in statuses
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['rsp_data'].value[0] == 99

    def test_different_channels_same_cycle(self):
        """Multiple consume steps with different Channels on the same cycle
        can all consume independently."""
        req = _bool_wf([1, 1, 0, 0, 0, 0])
        req_data = _wf([10, 20, 0, 0, 0, 0], width=8)
        rd_rsp = _bool_wf([0, 0, 0, 1, 0, 0])
        wr_rsp = _bool_wf([0, 0, 0, 1, 0, 0])
        rd_data = _wf([0, 0, 0, 111, 0, 0], width=8)
        wr_data = _wf([0, 0, 0, 222, 0, 0], width=8)
        rd_chan = Channel()
        wr_chan = Channel()
        result = match(
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .branch(
                lambda idx, cap: cap['req_data'] == 10,
                Pattern().consume(rd_rsp, channel=rd_chan).capture('rsp_data', rd_data),
                Pattern().consume(wr_rsp, channel=wr_chan).capture('rsp_data', wr_data),
            )
        )
        ok = result.filter_ok()
        assert len(ok) == 2
        rd_idx = [i for i in range(len(ok)) if ok.captures['req_data'].value[i] == 10]
        wr_idx = [i for i in range(len(ok)) if ok.captures['req_data'].value[i] == 20]
        assert ok.captures['rsp_data'].value[rd_idx[0]] == 111
        assert ok.captures['rsp_data'].value[wr_idx[0]] == 222


class TestTimeout:
    def test_timeout_triggers(self):
        trigger = _bool_wf([1, 0, 0, 0, 0, 0, 0, 0])
        never = _bool_wf([0, 0, 0, 0, 0, 0, 0, 0])
        result = match(Pattern().wait(trigger).wait(never), timeout=3)
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.Timeout()

    def test_timeout_1_cycle(self):
        """timeout=1: only epsilon steps on fork cycle can complete."""
        trigger = _bool_wf([1, 0, 0])
        data = _wf([42, 0, 0], width=8)
        result = match(Pattern().wait(trigger).capture('val', data), timeout=1)
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == 42

    def test_timeout_2_needs_blocking(self):
        """timeout=2: has 1 cycle after fork to complete a blocking step."""
        trigger = _bool_wf([1, 0, 0, 0])
        cond = _bool_wf([0, 1, 0, 0])
        data = _wf([0, 99, 0, 0], width=8)
        result = match(Pattern().wait(trigger).wait(cond).capture('val', data), timeout=2)
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == 99


class TestErrors:
    def test_no_waveform(self):
        """Pattern with only dynamic conditions and no trigger → error."""
        with pytest.raises(PatternError, match='could not determine scan axis'):
            match(Pattern().wait(lambda idx, cap: True))

    def test_unobserved_static_waveform_is_not_eagerly_validated(self):
        """Static waveforms in an untaken branch do not define or validate the axis."""
        ok = _wf([10, 20, 30], width=8)
        unused_misaligned = Waveform(
            np.array([1, 2, 3]), np.array([10, 11, 12]), np.array([100, 110, 120]), width=8
        )
        result = match(
            Pattern().branch(
                lambda idx, cap: False,
                true_body=Pattern().capture('unused', unused_misaligned),
                false_body=Pattern().capture('ok', ok),
            )
        )
        assert len(result) == 3
        np.testing.assert_array_equal(result.captures['ok'].value, [10, 20, 30])

    def test_start_end_cycle_infers_axis_from_nested_static_waveform(self):
        """Cycle bounds still work when the first observed waveform is nested."""
        sig = _wf([10, 20, 30, 40], width=8)
        result = match(
            Pattern().branch(
                lambda idx, cap: False, false_body=Pattern().repeat(Pattern().capture('v', sig), 1)
            ),
            start_cycle=1,
            end_cycle=3,
        )
        assert len(result) == 2
        np.testing.assert_array_equal(result.captures['v'].value, [20, 30])

    def test_loop_missing_condition(self):
        with pytest.raises(ValueError):
            Pattern().loop(Pattern().delay(1))

    def test_loop_both_conditions(self):
        cond = _bool_wf([1])
        with pytest.raises(ValueError):
            Pattern().loop(Pattern().delay(1), until=cond, when=cond)

    def test_dynamic_wait_condition_exception_propagates(self):
        class CustomConditionError(Exception):
            pass

        axis = _bool_wf([1])

        def bad_condition(_idx, _cap):
            raise CustomConditionError('condition boom')

        with pytest.raises(CustomConditionError, match='condition boom'):
            match(Pattern().wait(axis).wait(bad_condition))

    def test_dynamic_consume_condition_exception_propagates(self):
        class CustomConditionError(Exception):
            pass

        axis = _bool_wf([1])

        def bad_condition(_idx, _cap):
            raise CustomConditionError('consume condition boom')

        with pytest.raises(CustomConditionError, match='consume condition boom'):
            match(Pattern().wait(axis).consume(bad_condition, channel='rsp'))

    def test_dynamic_consume_channel_exception_propagates(self):
        class CustomChannelError(Exception):
            pass

        req = _bool_wf([1])
        rsp = _bool_wf([1])

        def bad_channel(_idx, _cap):
            raise CustomChannelError('channel boom')

        with pytest.raises(CustomChannelError, match='channel boom'):
            match(Pattern().wait(req).consume(rsp, channel=bad_channel))

    def test_dynamic_consume_channel_invalid_unhashable_raises_pattern_error(self):
        req = _bool_wf([1])
        rsp = _bool_wf([1])
        with pytest.raises(PatternError, match='channel must be a Channel or hashable key'):
            match(Pattern().wait(req).consume(rsp, channel=lambda _idx, _cap: []))


class TestZeroCycleWait:
    def test_zero_cycle_same_cycle_match(self):
        """wait(valid).wait(valid & ready) can complete on the same cycle."""
        valid = _bool_wf([0, 1, 1, 1, 0])
        ready = _bool_wf([0, 0, 1, 1, 0])
        result = match(Pattern().wait(valid).wait(valid & ready))
        ok = result.filter_ok()
        np.testing.assert_array_equal(ok.start.value[:2], [1, 2])
        np.testing.assert_array_equal(ok.end.value[:2], [2, 2])

    def test_zero_cycle_blocked_continues(self):
        """Same-cycle wait with cond False on current cycle waits until a later cycle."""
        trigger = _bool_wf([1, 0, 0, 0])
        cond = _bool_wf([0, 0, 1, 0])
        result = match(Pattern().wait(trigger).wait(cond))
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.start.value[0] == 0
        assert ok.end.value[0] == 2

    def test_zero_cycle_chained(self):
        """Multiple waits in a row can collapse to the same cycle."""
        a = _bool_wf([0, 1, 0])
        b = _bool_wf([0, 1, 0])
        c = _bool_wf([0, 1, 0])
        result = match(Pattern().wait(a).wait(b).wait(c))
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.start.value[0] == 1
        assert ok.end.value[0] == 1

    def test_zero_cycle_at_first_step(self):
        """First wait trigger optimization still completes captures on the same cycle."""
        sig = _bool_wf([0, 1, 0, 1, 0])
        data = _wf([0, 11, 0, 33, 0], width=8)
        result = match(Pattern().wait(sig).capture('d', data))
        ok = result.filter_ok()
        assert len(ok) == 2
        np.testing.assert_array_equal(ok.captures['d'].value, [11, 33])

    def test_default_waits_continue_same_cycle(self):
        """Default wait semantics are same-cycle."""
        a = _bool_wf([0, 1, 0])
        b = _bool_wf([0, 1, 0])
        data = _wf([0, 42, 0], width=8)
        result = match(Pattern().wait(a).wait(b).capture('d', data)).filter_ok()
        assert len(result) == 1
        assert result.end.value[0] == 1
        assert result.captures['d'].value[0] == 42

    def test_delay_one_resumes_next_cycle(self):
        """Use explicit delay(1) for next-cycle continuation."""
        a = _bool_wf([1, 0, 0])
        data = _wf([10, 20, 30], width=8)
        result = match(Pattern().wait(a).delay(1).capture('d', data)).filter_ok()
        assert len(result) == 1
        assert result.start.value[0] == 0
        assert result.end.value[0] == 1
        assert result.captures['d'].value[0] == 20

    def test_wait_require_not_checked_on_match_cycle(self):
        ready = _bool_wf([1, 0])
        guard = _bool_wf([0, 0])
        result = match(Pattern().wait(ready, require=guard)).filter_ok()
        assert len(result) == 1
        assert result.start.value[0] == 0

    def test_delay_require_does_not_check_resume_cycle(self):
        trigger = _bool_wf([1, 0, 0])
        guard = _bool_wf([1, 1, 0])
        result = match(Pattern().wait(trigger).delay(2, require=guard)).filter_ok()
        assert len(result) == 1
        assert result.end.value[0] == 2

    def test_delay_require_checks_blocked_cycles(self):
        trigger = _bool_wf([1, 0, 0])
        guard = _bool_wf([1, 0, 1])
        result = match(Pattern().wait(trigger).delay(2, require=guard))
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.RequireViolated()
        assert result.end.value[0] == 1

    def test_negative_dynamic_repeat_count_raises_pattern_error(self):
        trigger = _bool_wf([1])
        with pytest.raises(PatternError, match='repeat count'):
            match(Pattern().wait(trigger).repeat(Pattern().delay(0), n=lambda _i, _c: -1))

    def test_first_wait_trigger_respects_start_end_cycle_bounds(self):
        trigger = _bool_wf([1, 1, 1, 1])
        result = match(Pattern().wait(trigger), start_cycle=1, end_cycle=3)
        np.testing.assert_array_equal(result.start.value, [1, 2])
        np.testing.assert_array_equal(result.end.value, [1, 2])


class TestChannelReset:
    def test_static_channel_reset_between_runs(self):
        """Same Pattern instance, two match() calls → second run not polluted by first."""
        req = _bool_wf([1, 0, 0, 0])
        rsp = _bool_wf([0, 0, 1, 0])
        rsp_chan = Channel()
        p = Pattern().wait(req).consume(rsp, channel=rsp_chan)
        r1 = match(p)
        r2 = match(p)
        v1 = r1.filter_ok()
        v2 = r2.filter_ok()
        assert len(v1) == 1
        assert len(v2) == 1
        np.testing.assert_array_equal(v1.start.value, v2.start.value)
        np.testing.assert_array_equal(v1.end.value, v2.end.value)

    def test_dynamic_channel_reused_across_runs(self):
        """User-managed dynamic channels (defaultdict(Channel)) must also reset
        between runs without explicit user intervention."""
        from collections import defaultdict

        req = _bool_wf([1, 0, 0, 0])
        rsp = _bool_wf([0, 0, 1, 0])
        chans = defaultdict(Channel)
        p = Pattern().wait(req).consume(rsp, channel=lambda i, cap: chans['only'])
        r1 = match(p)
        r2 = match(p)
        v1 = r1.filter_ok()
        v2 = r2.filter_ok()
        assert len(v1) == 1
        assert len(v2) == 1
        np.testing.assert_array_equal(v1.start.value, v2.start.value)
        np.testing.assert_array_equal(v1.end.value, v2.end.value)

    def test_plain_wait_is_observational_across_instances(self):
        """Multiple in-flight instances can observe the same plain wait event."""
        req = _bool_wf([1, 1, 1, 0, 0, 0])
        rsp = _bool_wf([0, 0, 0, 1, 0, 0])
        result = match(Pattern().wait(req).wait(rsp))
        ok = result.filter_ok()
        assert len(ok) == 3
        np.testing.assert_array_equal(ok.start.value, [0, 1, 2])
        np.testing.assert_array_equal(ok.end.value, [3, 3, 3])

    def test_consume_serializes_instances_fifo(self):
        """Explicit consume preserves one-owner FIFO event matching."""
        req = _bool_wf([1, 1, 1, 0, 0, 0])
        rsp = _bool_wf([0, 0, 0, 1, 0, 0])
        result = match(Pattern().wait(req).consume(rsp, channel='rsp'))
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.start.value[0] == 0
        assert ok.end.value[0] == 3


class TestChannelAPI:
    def test_channel_is_distinct_identity(self):
        """Two Channel() instances are distinct objects (default identity semantics)."""
        a = Channel()
        b = Channel()
        assert a is not b
        d = {a: 'x', b: 'y'}
        assert d[a] == 'x' and d[b] == 'y'

    def test_dynamic_channel_callable_may_return_hashable_key(self):
        start = _bool_wf([1, 0])
        rsp = _bool_wf([0, 1])
        result = match(Pattern().wait(start).consume(rsp, channel=lambda i, c: 'hashable-key'))
        assert len(result.filter_ok()) == 1

    def test_dynamic_channel_resolved_once_for_successful_consume(self):
        """A dynamic channel chosen during ready is reused for commit."""
        start = _bool_wf([1, 0])
        rsp = _bool_wf([0, 1])
        calls = []

        def channel(idx, cap):
            calls.append((idx, dict(cap)))
            return 'shared'

        result = match(Pattern().wait(start).consume(rsp, channel=channel))
        assert len(result.filter_ok()) == 1
        assert calls == [(1, {})]


def test_declarative_consume_and_timeout_builder():
    req = _bool_wf([1, 1, 0, 0, 0])
    rsp = _bool_wf([0, 0, 1, 0, 1])
    data = _wf([10, 20, 111, 0, 222], width=8)
    result = match(Pattern().wait(req).consume(rsp, channel='rsp').capture('rsp', data))
    np.testing.assert_array_equal(result.filter_ok().captures['rsp'].value, [111, 222])
    result = match(Pattern().wait(req), timeout=3)
    assert len(result) >= 1
