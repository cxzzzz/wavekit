"""Core Pattern runtime semantics shared by declarative and programmable APIs."""

from collections import defaultdict

import numpy as np
import pytest
from helpers import bool_wf as _bool_wf
from helpers import wf as _wf

from wavekit import Channel, Signal, Waveform
from wavekit.pattern import MatchStatus, Pattern, PatternError

# ---------------------------------------------------------------------------


class TestWaitRequire:
    def test_require_holds(self):
        """valid stays high while waiting for ready → OK."""
        # Only one trigger at cycle 1 (valid goes high once)
        valid = _bool_wf([0, 1, 1, 1, 0])
        ready = _bool_wf([0, 0, 0, 1, 0])
        result = Pattern().wait(valid).wait(ready, require=valid).match()
        # Triggers at cycle 1,2,3 (valid=1). Inst@1: require OK, ready@3 → OK.
        # Inst@2: require OK, ready@3 → OK. Inst@3: ready@3 immediately → OK.
        ok = result.filter_ok()
        assert len(ok) >= 1
        # The first instance (trigger@1) completes at cycle 3
        assert ok.start.value[0] == 1
        assert ok.end.value[0] == 3

    def test_require_violated(self):
        """valid drops before ready → REQUIRE_VIOLATED."""
        valid = _bool_wf([0, 1, 0, 0, 0])
        ready = _bool_wf([0, 0, 0, 1, 0])
        result = Pattern().wait(valid).wait(ready, require=valid).match()
        # Instance forked at cycle 1, require fails at cycle 2 (valid=0)
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.REQUIRE_VIOLATED


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


class TestDelay:
    def test_delay_basic(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])
        data = _wf([10, 20, 30, 40, 50], width=8)
        result = Pattern().wait(trigger).delay(2).capture('val', data).match()
        assert len(result) == 1
        # Trigger at 0, delay 2 cycles → capture at cycle 2
        assert result.captures['val'].value[0] == 30

    def test_delay_with_require(self):
        trigger = _bool_wf([1, 0, 0, 0, 0])
        enable = _bool_wf([1, 1, 0, 0, 0])  # drops at cycle 2
        result = Pattern().wait(trigger).delay(3, require=enable).match()
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

    def test_delay_zero_does_not_check_require(self):
        trigger = _bool_wf([1, 0])
        guard = _bool_wf([0, 0])

        result = Pattern().wait(trigger).delay(0, require=guard).match().filter_ok()

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
        result = (
            Pattern()
            .wait(req)
            .capture('req', req_data)
            .consume(rsp, channel=rsp_chan)
            .capture('rsp', rsp_data)
            .match()
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
        # Two transaction types: read (type=0) and write (type=1)
        # Cycle:    0  1  2  3  4  5  6  7
        req = _bool_wf([1, 1, 1, 1, 0, 0, 0, 0])  # reqs at 0, 1, 2, 3
        req_type = _wf([0, 1, 0, 1, 0, 0, 0, 0], width=4)  # types: rd, wr, rd, wr
        req_data = _wf([10, 20, 30, 40, 0, 0, 0, 0], width=8)

        # Responses: rd_rsp and wr_rsp are independent
        rd_rsp = _bool_wf([0, 0, 0, 1, 0, 1, 0, 0])  # rd rsps at 3, 5
        wr_rsp = _bool_wf([0, 0, 0, 0, 1, 0, 1, 0])  # wr rsps at 4, 6
        rd_rsp_data = _wf([0, 0, 0, 111, 0, 333, 0, 0], width=8)
        wr_rsp_data = _wf([0, 0, 0, 0, 222, 0, 444, 0], width=8)

        rd_chan = Channel()
        wr_chan = Channel()
        result = (
            Pattern()
            .wait(req)
            .capture('req_type', req_type)
            .capture('req_data', req_data)
            .branch(
                lambda idx, cap: cap['req_type'] == 0,  # read
                Pattern().consume(rd_rsp, channel=rd_chan).capture('rsp_data', rd_rsp_data),
                Pattern().consume(wr_rsp, channel=wr_chan).capture('rsp_data', wr_rsp_data),
            )
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 4

        # Check FIFO per channel:
        # rd channel: req_data [10, 30] → rsp_data [111, 333]
        # wr channel: req_data [20, 40] → rsp_data [222, 444]
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
        # Scenario: 4 requests with IDs [0, 1, 0, 1] (2 per ID)
        #           4 responses with IDs [1, 0, 1, 0] (out of order per-ID, but FIFO per channel)
        # Cycle:    0  1  2  3  4  5  6  7  8  9  10 11
        req = _bool_wf([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # reqs at 0, 1, 2, 3
        req_id = _wf([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], width=4)  # IDs: 0, 1, 0, 1
        req_data = _wf([10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0], width=8)  # data: 10, 20, 30, 40

        # Responses arrive out of order globally, but we want FIFO per ID channel
        # ID 0's FIFO: req=10@cycle0, req=30@cycle2 → rsp in FIFO order
        # ID 1's FIFO: req=20@cycle1, req=40@cycle3 → rsp in FIFO order
        # Response order: ID 1@5, ID 0@7, ID 1@9, ID 0@10
        rsp = _bool_wf([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0])  # rsps at 5, 7, 9, 10
        rsp_id = _wf([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], width=4)  # IDs: 1, 0, 1, 0
        rsp_data = _wf([0, 0, 0, 0, 0, 111, 0, 222, 0, 333, 444, 0], width=8)  # 111, 222, 333, 444

        def match_rsp_with_id(idx, cap):
            """Condition: rsp is True AND rsp_id matches the captured req_id."""
            return bool(rsp.value[idx]) and int(rsp_id.value[idx]) == int(cap['req_id'])

        chans = defaultdict(Channel)
        result = (
            Pattern()
            .wait(req)
            .capture('req_id', req_id)
            .capture('req_data', req_data)
            .consume(
                match_rsp_with_id,
                channel=lambda idx, cap: chans[int(cap['req_id'])],
            )
            .capture('rsp_id', rsp_id)
            .capture('rsp_data', rsp_data)
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 4

        # Verify FIFO order per ID:
        # ID 0 channel: req_data [10, 30] matched with rsp_data [222, 444] (responses at 7, 10)
        # ID 1 channel: req_data [20, 40] matched with rsp_data [111, 333] (responses at 5, 9)
        id_0_matches = [i for i in range(len(ok)) if ok.captures['req_id'].value[i] == 0]
        id_1_matches = [i for i in range(len(ok)) if ok.captures['req_id'].value[i] == 1]

        assert len(id_0_matches) == 2
        assert len(id_1_matches) == 2

        # Check ID 0: FIFO order should be [10, 30] → [222, 444]
        id_0_req_data = [ok.captures['req_data'].value[i] for i in id_0_matches]
        id_0_rsp_data = [ok.captures['rsp_data'].value[i] for i in id_0_matches]
        assert id_0_req_data == [10, 30], f'ID 0 req_data should be FIFO: {id_0_req_data}'
        assert id_0_rsp_data == [222, 444], f'ID 0 rsp_data should be FIFO: {id_0_rsp_data}'

        # Check ID 1: FIFO order should be [20, 40] → [111, 333]
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
        # Simulate AXI: requests with different IDs
        # Cycle:    0  1  2  3  4  5  6  7  8  9  10 11
        req = _bool_wf([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # req at 0, 2
        req_id = _wf([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], width=4)  # ID 0, 1
        req_data = _wf([100, 0, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0], width=8)

        # Responses arrive out of order (ID 1 first, then ID 0)
        rsp = _bool_wf([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])  # rsp at 5, 8
        rsp_id = _wf(
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], width=4
        )  # ID 1 at cycle 5, ID 0 at cycle 8
        rsp_data = _wf([0, 0, 0, 0, 0, 222, 0, 0, 111, 0, 0, 0], width=8)

        # Full AXI-style: dynamic channel + condition checks ID match
        def match_id(idx, cap):
            """Condition: rsp is True AND rsp_id matches the captured req_id."""
            return bool(rsp.value[idx]) and int(rsp_id.value[idx]) == int(cap['req_id'])

        chans = defaultdict(Channel)
        result = (
            Pattern()
            .wait(req)
            .capture('req_id', req_id)
            .capture('req_data', req_data)
            .consume(
                match_id,
                channel=lambda idx, cap: chans[int(cap['req_id'])],
            )
            .capture('rsp_id', rsp_id)
            .capture('rsp_data', rsp_data)
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 2

        # Check that request ID matches response ID (routing worked)
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

        result = (
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .consume(rsp, channel=lambda _idx, _cap: 0)
            .capture('rsp_data', rsp_data)
            .match()
            .filter_ok()
        )

        np.testing.assert_array_equal(result.captures['req_data'].value, [10, 20])
        np.testing.assert_array_equal(result.captures['rsp_data'].value, [111, 222])

    def test_dynamic_channel_fifo_per_channel(self):
        """Each dynamic channel value maintains its own FIFO order.

        When two instances have the same channel key, they share a FIFO.
        When they have different keys, they are independent.
        """
        # Two requests with same derived channel key
        req = _bool_wf([1, 1, 0, 0, 0, 0, 0, 0])
        # Use 10, 11 → both // 10 = 1 → same channel
        req_data = _wf([10, 11, 0, 0, 0, 0, 0, 0], width=8)

        rsp = _bool_wf([0, 0, 0, 1, 0, 1, 0, 0])  # responses at 3, 5
        rsp_data = _wf([0, 0, 0, 111, 0, 222, 0, 0], width=8)

        chans = defaultdict(Channel)
        result = (
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            # Same channel for both (both // 10 = 1), so they share a FIFO
            .consume(rsp, channel=lambda idx, cap: chans[int(cap['req_data']) // 10])
            .capture('rsp_data', rsp_data)
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 2
        # FIFO order preserved within same channel
        np.testing.assert_array_equal(ok.captures['req_data'].value, [10, 11])
        np.testing.assert_array_equal(ok.captures['rsp_data'].value, [111, 222])

    def test_consume_as_first_step(self):
        """consume as first step: no trigger optimization,
        channel is consumed.

        When the first step consumes a channel, the engine must not use it as a
        trigger (which would skip channel consumption).  Instead, instances are
        forked every cycle and the consume step processes normally.
        """
        # Two rsp events at cycles 2 and 4
        rsp = _bool_wf([0, 0, 1, 0, 1, 0])
        rsp_data = _wf([0, 0, 111, 0, 222, 0], width=8)

        rsp_chan = Channel()
        result = Pattern().consume(rsp, channel=rsp_chan).capture('rsp_data', rsp_data).match()
        ok = result.filter_ok()
        # Only 2 OK matches (at cycles where rsp=1 and channel is consumed)
        assert len(ok) == 2
        np.testing.assert_array_equal(ok.captures['rsp_data'].value, [111, 222])

    def test_require_with_channel(self):
        """require violation while consuming with channel → REQUIRE_VIOLATED."""
        req = _bool_wf([1, 1, 0, 0, 0, 0])
        req_data = _wf([10, 20, 0, 0, 0, 0], width=8)
        rsp = _bool_wf([0, 0, 0, 0, 1, 0])  # rsp arrives at cycle 4
        enable = _bool_wf([1, 1, 1, 0, 0, 0])  # drops at cycle 3

        rsp_chan = Channel()
        result = (
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .consume(rsp, channel=rsp_chan, require=enable)
            .match()
        )
        # Both instances should fail: require drops before rsp arrives
        assert len(result) == 2
        assert all(s == MatchStatus.REQUIRE_VIOLATED for s in result.status.value)

    def test_require_with_channel_not_checked_on_success_cycle(self):
        req = _bool_wf([1, 0, 0])
        rsp = _bool_wf([0, 1, 0])
        guard = _bool_wf([1, 0, 0])

        result = Pattern().wait(req).consume(rsp, channel='rsp', require=guard).match().filter_ok()

        assert len(result) == 1
        assert result.start.value[0] == 0
        assert result.end.value[0] == 1

    def test_timeout_with_channel(self):
        """Instance consuming with channel times out → next instance can consume."""
        # Cycle:     0  1  2  3  4  5
        req = _bool_wf([1, 1, 0, 0, 0, 0])
        req_data = _wf([10, 20, 0, 0, 0, 0], width=8)
        rsp = _bool_wf([0, 0, 0, 0, 0, 1])  # rsp at cycle 5
        rsp_data = _wf([0, 0, 0, 0, 0, 99], width=8)

        rsp_chan = Channel()
        result = (
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .consume(rsp, channel=rsp_chan)
            .capture('rsp_data', rsp_data)
            .timeout(5)  # Instance 0 (forked@0): elapsed=6 > 5 at cycle 5 → TIMEOUT
            .match()  # Instance 1 (forked@1): elapsed=5 at cycle 5, 5 > 5? No → advance
        )
        # Instance 0 times out; Instance 1 survives and consumes the channel
        assert len(result) == 2
        statuses = sorted(result.status.value)
        assert MatchStatus.TIMEOUT in statuses
        assert MatchStatus.OK in statuses
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['rsp_data'].value[0] == 99

    def test_different_channels_same_cycle(self):
        """Multiple consume steps with different Channels on the same cycle
        can all consume independently."""
        req = _bool_wf([1, 1, 0, 0, 0, 0])
        req_data = _wf([10, 20, 0, 0, 0, 0], width=8)
        # Both rd_rsp and wr_rsp arrive at cycle 3
        rd_rsp = _bool_wf([0, 0, 0, 1, 0, 0])
        wr_rsp = _bool_wf([0, 0, 0, 1, 0, 0])
        rd_data = _wf([0, 0, 0, 111, 0, 0], width=8)
        wr_data = _wf([0, 0, 0, 222, 0, 0], width=8)

        rd_chan = Channel()
        wr_chan = Channel()
        result = (
            Pattern()
            .wait(req)
            .capture('req_data', req_data)
            .branch(
                lambda idx, cap: cap['req_data'] == 10,  # first request → read
                Pattern().consume(rd_rsp, channel=rd_chan).capture('rsp_data', rd_data),
                Pattern().consume(wr_rsp, channel=wr_chan).capture('rsp_data', wr_data),
            )
            .match()
        )
        ok = result.filter_ok()
        assert len(ok) == 2
        # Both instances can consume on the same cycle (different channels)
        rd_idx = [i for i in range(len(ok)) if ok.captures['req_data'].value[i] == 10]
        wr_idx = [i for i in range(len(ok)) if ok.captures['req_data'].value[i] == 20]
        assert ok.captures['rsp_data'].value[rd_idx[0]] == 111
        assert ok.captures['rsp_data'].value[wr_idx[0]] == 222


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
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == 42

    def test_timeout_2_needs_blocking(self):
        """timeout=2: has 1 cycle after fork to complete a blocking step."""
        trigger = _bool_wf([1, 0, 0, 0])
        cond = _bool_wf([0, 1, 0, 0])
        data = _wf([0, 99, 0, 0], width=8)
        result = Pattern().wait(trigger).wait(cond).capture('val', data).timeout(2).match()
        ok = result.filter_ok()
        assert len(ok) == 1
        assert ok.captures['val'].value[0] == 99


class TestErrors:
    def test_no_waveform(self):
        """Pattern with only dynamic conditions and no trigger → error."""
        with pytest.raises(PatternError, match='could not determine scan axis'):
            Pattern().wait(lambda idx, cap: True).match()

    def test_unobserved_static_waveform_is_not_eagerly_validated(self):
        """Static waveforms in an untaken branch do not define or validate the axis."""
        ok = _wf([10, 20, 30], width=8)
        unused_misaligned = Waveform(
            np.array([1, 2, 3]),
            np.array([10, 11, 12]),
            np.array([100, 110, 120]),
            signal=Signal('', '', 8, None, False),
        )

        result = (
            Pattern()
            .branch(
                lambda idx, cap: False,
                true_body=Pattern().capture('unused', unused_misaligned),
                false_body=Pattern().capture('ok', ok),
            )
            .match()
        )

        assert len(result) == 3
        np.testing.assert_array_equal(result.captures['ok'].value, [10, 20, 30])

    def test_start_end_cycle_infers_axis_from_nested_static_waveform(self):
        """Cycle bounds still work when the first observed waveform is nested."""
        sig = _wf([10, 20, 30, 40], width=8)

        result = (
            Pattern()
            .branch(
                lambda idx, cap: False,
                false_body=Pattern().repeat(Pattern().capture('v', sig), 1),
            )
            .match(start_cycle=1, end_cycle=3)
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
            Pattern().wait(axis).wait(bad_condition).match()

    def test_dynamic_consume_condition_exception_propagates(self):
        class CustomConditionError(Exception):
            pass

        axis = _bool_wf([1])

        def bad_condition(_idx, _cap):
            raise CustomConditionError('consume condition boom')

        with pytest.raises(CustomConditionError, match='consume condition boom'):
            Pattern().wait(axis).consume(bad_condition, channel='rsp').match()

    def test_dynamic_consume_channel_exception_propagates(self):
        class CustomChannelError(Exception):
            pass

        req = _bool_wf([1])
        rsp = _bool_wf([1])

        def bad_channel(_idx, _cap):
            raise CustomChannelError('channel boom')

        with pytest.raises(CustomChannelError, match='channel boom'):
            Pattern().wait(req).consume(rsp, channel=bad_channel).match()

    def test_dynamic_consume_channel_invalid_unhashable_raises_pattern_error(self):
        req = _bool_wf([1])
        rsp = _bool_wf([1])

        with pytest.raises(PatternError, match='channel must be a Channel or hashable key'):
            Pattern().wait(req).consume(rsp, channel=lambda _idx, _cap: []).match()


class TestZeroCycleWait:
    def test_zero_cycle_same_cycle_match(self):
        """wait(valid).wait(valid & ready) can complete on the same cycle."""
        # cycle:        0  1  2  3  4
        valid = _bool_wf([0, 1, 1, 1, 0])
        ready = _bool_wf([0, 0, 1, 1, 0])
        # valid & ready first true at cycle 2; trigger forks every valid cycle.
        result = Pattern().wait(valid).wait(valid & ready).match()
        ok = result.filter_ok()
        # Instance@1: wait advances to cycle 2 (valid&ready) and stays at t=2,
        #   completes at cycle 2 → duration = 2 - 1 + 1 = 2
        # Instance@1 advances at cycle 2; instance@2 also observes the same true
        # condition and completes in the same cycle because wait is non-consuming.
        np.testing.assert_array_equal(ok.start.value[:2], [1, 2])
        np.testing.assert_array_equal(ok.end.value[:2], [2, 2])

    def test_zero_cycle_blocked_continues(self):
        """Same-cycle wait with cond False on current cycle waits until a later cycle."""
        # cycle:        0  1  2  3
        trigger = _bool_wf([1, 0, 0, 0])
        cond = _bool_wf([0, 0, 1, 0])  # only true at cycle 2
        result = Pattern().wait(trigger).wait(cond).match()
        ok = result.filter_ok()
        # Instance@0: first wait observed by trigger → step_idx=1 at t=0,
        # second wait: cond False at 0,1 → next cycle.
        # cond True at 2 → match without consuming a cycle → end_cycle=2.
        assert len(ok) == 1
        assert ok.start.value[0] == 0
        assert ok.end.value[0] == 2

    def test_zero_cycle_chained(self):
        """Multiple waits in a row can collapse to the same cycle."""
        # cycle:        0  1  2
        a = _bool_wf([0, 1, 0])
        b = _bool_wf([0, 1, 0])
        c = _bool_wf([0, 1, 0])
        result = Pattern().wait(a).wait(b).wait(c).match()
        ok = result.filter_ok()
        assert len(ok) == 1
        # All three fire on cycle 1
        assert ok.start.value[0] == 1
        assert ok.end.value[0] == 1

    def test_zero_cycle_at_first_step(self):
        """First wait trigger optimization still completes captures on the same cycle."""
        sig = _bool_wf([0, 1, 0, 1, 0])
        data = _wf([0, 11, 0, 33, 0], width=8)
        result = Pattern().wait(sig).capture('d', data).match()
        ok = result.filter_ok()
        # Two triggers; epsilon capture completes same cycle.
        assert len(ok) == 2
        np.testing.assert_array_equal(ok.captures['d'].value, [11, 33])

    def test_default_waits_continue_same_cycle(self):
        """Default wait semantics are same-cycle."""
        a = _bool_wf([0, 1, 0])
        b = _bool_wf([0, 1, 0])
        data = _wf([0, 42, 0], width=8)
        result = Pattern().wait(a).wait(b).capture('d', data).match().filter_ok()
        assert len(result) == 1
        assert result.end.value[0] == 1
        assert result.captures['d'].value[0] == 42

    def test_delay_one_resumes_next_cycle(self):
        """Use explicit delay(1) for next-cycle continuation."""
        a = _bool_wf([1, 0, 0])
        data = _wf([10, 20, 30], width=8)
        result = Pattern().wait(a).delay(1).capture('d', data).match().filter_ok()
        assert len(result) == 1
        assert result.start.value[0] == 0
        assert result.end.value[0] == 1
        assert result.captures['d'].value[0] == 20

    def test_wait_require_not_checked_on_match_cycle(self):
        ready = _bool_wf([1, 0])
        guard = _bool_wf([0, 0])
        result = Pattern().wait(ready, require=guard).match().filter_ok()
        assert len(result) == 1
        assert result.start.value[0] == 0

    def test_delay_require_does_not_check_resume_cycle(self):
        trigger = _bool_wf([1, 0, 0])
        guard = _bool_wf([1, 1, 0])
        result = Pattern().wait(trigger).delay(2, require=guard).match().filter_ok()
        assert len(result) == 1
        assert result.end.value[0] == 2

    def test_delay_require_checks_blocked_cycles(self):
        trigger = _bool_wf([1, 0, 0])
        guard = _bool_wf([1, 0, 1])
        result = Pattern().wait(trigger).delay(2, require=guard).match()
        assert len(result) == 1
        assert result.status.value[0] == MatchStatus.REQUIRE_VIOLATED
        assert result.end.value[0] == 1

    def test_negative_dynamic_repeat_count_raises_pattern_error(self):
        trigger = _bool_wf([1])
        with pytest.raises(PatternError, match='repeat count'):
            Pattern().wait(trigger).repeat(Pattern().delay(0), n=lambda _i, _c: -1).match()

    def test_first_wait_trigger_respects_start_end_cycle_bounds(self):
        trigger = _bool_wf([1, 1, 1, 1])

        result = Pattern().wait(trigger).match(start_cycle=1, end_cycle=3)

        np.testing.assert_array_equal(result.start.value, [1, 2])
        np.testing.assert_array_equal(result.end.value, [1, 2])


class TestChannelReset:
    def test_static_channel_reset_between_runs(self):
        """Same Pattern instance, two match() calls → second run not polluted by first."""
        req = _bool_wf([1, 0, 0, 0])
        rsp = _bool_wf([0, 0, 1, 0])
        rsp_chan = Channel()
        p = Pattern().wait(req).consume(rsp, channel=rsp_chan)
        r1 = p.match()
        r2 = p.match()
        # Both runs must produce the same OK result.
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
        chans = defaultdict(Channel)  # reused across both match() calls
        p = Pattern().wait(req).consume(rsp, channel=lambda i, cap: chans['only'])
        r1 = p.match()
        r2 = p.match()
        v1 = r1.filter_ok()
        v2 = r2.filter_ok()
        assert len(v1) == 1
        assert len(v2) == 1
        np.testing.assert_array_equal(v1.start.value, v2.start.value)
        np.testing.assert_array_equal(v1.end.value, v2.end.value)

    def test_plain_wait_is_observational_across_instances(self):
        """Multiple in-flight instances can observe the same plain wait event."""
        # Three triggers in a row, then one rsp event later.
        # cycle:    0  1  2  3  4  5
        req = _bool_wf([1, 1, 1, 0, 0, 0])
        rsp = _bool_wf([0, 0, 0, 1, 0, 0])
        # Each trigger forks an instance; the second wait observes rsp@3 without
        # consuming it, so all in-flight instances complete at cycle 3.
        result = Pattern().wait(req).wait(rsp).match()
        ok = result.filter_ok()
        assert len(ok) == 3
        np.testing.assert_array_equal(ok.start.value, [0, 1, 2])
        np.testing.assert_array_equal(ok.end.value, [3, 3, 3])

    def test_consume_serializes_instances_fifo(self):
        """Explicit consume preserves one-owner FIFO event matching."""
        req = _bool_wf([1, 1, 1, 0, 0, 0])
        rsp = _bool_wf([0, 0, 0, 1, 0, 0])

        result = Pattern().wait(req).consume(rsp, channel='rsp').match()
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
        # Used as dict keys
        d = {a: 'x', b: 'y'}
        assert d[a] == 'x' and d[b] == 'y'

    def test_dynamic_channel_callable_may_return_hashable_key(self):
        start = _bool_wf([1, 0])
        rsp = _bool_wf([0, 1])
        result = Pattern().wait(start).consume(rsp, channel=lambda i, c: 'hashable-key').match()
        assert len(result.filter_ok()) == 1

    def test_dynamic_channel_resolved_once_for_successful_consume(self):
        """A dynamic channel chosen during ready is reused for commit."""
        start = _bool_wf([1, 0])
        rsp = _bool_wf([0, 1])
        calls = []

        def channel(idx, cap):
            calls.append((idx, dict(cap)))
            return 'shared'

        result = Pattern().wait(start).consume(rsp, channel=channel).match()

        assert len(result.filter_ok()) == 1
        assert calls == [(1, {})]


# ---------------------------------------------------------------------------
# Declarative compatibility/runtime smoke
# ---------------------------------------------------------------------------


def test_declarative_consume_and_timeout_deprecation():
    req = _bool_wf([1, 1, 0, 0, 0])
    rsp = _bool_wf([0, 0, 1, 0, 1])
    data = _wf([10, 20, 111, 0, 222], width=8)
    result = Pattern().wait(req).consume(rsp, channel='rsp').capture('rsp', data).match()
    np.testing.assert_array_equal(result.filter_ok().captures['rsp'].value, [111, 222])

    with pytest.warns(DeprecationWarning, match='timeout'):
        Pattern().wait(req).timeout(3)
