"""Smoke tests for the example snippets in README.md / README_ZH.md.

We can't open the real VCDs the README mentions, so we synthesize Waveforms
that mimic the relevant handshakes and run the patterns end-to-end.
"""

from collections import defaultdict

import numpy as np
import pytest

from wavekit import Channel, Pattern, Signal, Waveform


def _wf(values, width=1, signed=False):
    value = np.asarray(values, dtype=np.int64)
    clock = np.arange(len(value), dtype=np.int64)
    time = clock * 10
    return Waveform(value, clock, time, signal=Signal('', '', width, None, signed))


# ---------------------------------------------------------------------------
# Example 1: AXI-lite Read Latency
# ---------------------------------------------------------------------------


def test_axi_read_latency():
    """README: Pattern().wait(arvalid&arready).wait(rvalid&rready).capture('rdata', rdata)."""
    # Two reads:
    #   read 0: AR@1, R@4  → latency = 4 - 1 + 1 = 4 cycles, rdata = 0xDEAD
    #   read 1: AR@6, R@8  → latency = 8 - 6 + 1 = 3 cycles, rdata = 0xBEEF
    arvalid = _wf([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    arready = _wf([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    rvalid = _wf([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    rready = _wf([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    rdata = _wf([0, 0, 0, 0, 0xDEAD, 0, 0, 0, 0xBEEF, 0], width=32)

    result = (
        Pattern()
        .wait(arvalid & arready)
        .wait(rvalid & rready)
        .capture('rdata', rdata)
        .timeout(256)
        .match()
    )

    valid = result.filter_ok()
    assert len(valid) == 2
    np.testing.assert_array_equal(valid.duration.value, [4, 3])
    np.testing.assert_array_equal(valid.captures['rdata'].value, [0xDEAD, 0xBEEF])


# ---------------------------------------------------------------------------
# Example 2: AXI Write Burst (multi-beat)
# ---------------------------------------------------------------------------


def test_axi_write_burst():
    """README: nested Pattern with loop(beat, until=wlast) and capture mode='list'."""
    # One burst of 3 beats: AW@0, beats at cycles 1,2,3 with wlast at 3.
    awvalid = _wf([1, 0, 0, 0, 0])
    awready = _wf([1, 0, 0, 0, 0])
    wvalid = _wf([0, 1, 1, 1, 0])
    wready = _wf([0, 1, 1, 1, 0])
    wdata = _wf([0, 0xA0, 0xA1, 0xA2, 0], width=8)
    wlast = _wf([0, 0, 0, 1, 0])

    beat = Pattern().consume(wvalid & wready, channel='w').capture('beats', wdata, mode='list')

    result = Pattern().wait(awvalid & awready).loop(beat, until=wlast).timeout(512).match()

    valid = result.filter_ok()
    assert len(valid) == 1
    assert list(valid.captures['beats'].value[0]) == [0xA0, 0xA1, 0xA2]


# ---------------------------------------------------------------------------
# Example 3: Stall Detection
# ---------------------------------------------------------------------------


def test_stall_detection():
    """README: .wait(stall.rising_edge()).loop(Pattern().delay(1), when=stall)."""
    # cycle:           0  1  2  3  4  5  6  7  8  9
    valid_sig = _wf([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    ready_sig = _wf([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    stall = valid_sig & (ready_sig == 0)

    result = Pattern().wait(stall.rising_edge()).loop(Pattern().delay(1), when=stall).match()

    stalls = result.filter_ok()
    assert len(stalls) == 2
    # First stall: rising edge at 1, stall stays high through 4 → ends when stall falls at 5
    # Second stall: rising at 7, stays through 8 → ends when stall falls at 9
    np.testing.assert_array_equal(stalls.start.value, [1, 7])
    np.testing.assert_array_equal(stalls.duration.value, [5, 3])


# ---------------------------------------------------------------------------
# Example 4: AXI Read Burst + Out-of-Order ID Matching
# ---------------------------------------------------------------------------


def test_axi_read_burst_ooo():
    """Each AR issues a multi-beat read; R beats from different IDs may
    interleave on the bus. Per-instance loop until rlast on the matching ID."""
    # AR0 (arid=0, 3 beats): issued @ 0
    # AR1 (arid=1, 2 beats): issued @ 1
    # R channel beats interleaved (single-issue per cycle):
    #   c=2  rid=0 data=0xA0 last=0
    #   c=3  rid=1 data=0xB0 last=0
    #   c=4  rid=0 data=0xA1 last=0
    #   c=5  rid=1 data=0xB1 last=1   ← arid=1 burst ends
    #   c=6  rid=0 data=0xA2 last=1   ← arid=0 burst ends
    arvalid = _wf([1, 1, 0, 0, 0, 0, 0, 0])
    arready = _wf([1, 1, 0, 0, 0, 0, 0, 0])
    arid = _wf([0, 1, 0, 0, 0, 0, 0, 0], width=4)

    rvalid = _wf([0, 0, 1, 1, 1, 1, 1, 0])
    rready = _wf([0, 0, 1, 1, 1, 1, 1, 0])
    rid = _wf([0, 0, 0, 1, 0, 1, 0, 0], width=4)
    rdata = _wf([0, 0, 0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0], width=32)
    rlast = _wf([0, 0, 0, 0, 0, 1, 1, 0])

    rfire = rvalid & rready

    # No explicit Channel needed here: rid==arid filters each beat to the
    # matching burst instance in this single-issue example.
    beat = (
        Pattern()
        .consume(
            lambda i, cap: rfire.value[i] and rid.value[i] == cap['arid'],
            channel=lambda i, cap: ('r', int(cap['arid'])),
        )
        .capture('beats', rdata, mode='list')
    )

    result = (
        Pattern()
        .wait(arvalid & arready)
        .capture('arid', arid)
        .loop(beat, until=lambda i, cap: bool(rlast.value[i]))
        .timeout(64)
        .match()
    )

    valid = result.filter_ok()
    assert len(valid) == 2
    by_id = {
        int(arid_val): list(beats)
        for arid_val, beats in zip(valid.captures['arid'].value, valid.captures['beats'].value)
    }
    assert by_id == {0: [0xA0, 0xA1, 0xA2], 1: [0xB0, 0xB1]}


# ---------------------------------------------------------------------------
# Example 5: Per-bank Channel partitioning (concurrent multi-port)
# ---------------------------------------------------------------------------


def test_multi_bank_concurrent_responses():
    """4-bank cache: each bank has an independent response port; multiple banks
    may fire on the same cycle. Demonstrates why per-bank Channel partitioning
    matters when responses are physically parallel."""
    # 4 requests routed to banks based on (addr & 1) — 2 banks for simplicity.
    # cycle:           0  1  2  3  4  5  6
    req_valid = _wf([1, 1, 0, 0, 0, 0, 0])
    req_addr = _wf([0, 1, 0, 0, 0, 0, 0], width=8)  # bank0, bank1

    # Bank 0 and Bank 1 BOTH fire at cycle 4 (concurrent multi-port).
    bank0_valid = _wf([0, 0, 0, 0, 1, 0, 0])
    bank1_valid = _wf([0, 0, 0, 0, 1, 0, 0])
    bank0_data = _wf([0, 0, 0, 0, 0xAA, 0, 0], width=32)
    bank1_data = _wf([0, 0, 0, 0, 0xBB, 0, 0], width=32)

    banks = defaultdict(Channel)

    def bank_resp_fire(i, cap):
        bank = cap['bank']
        return (bank0_valid if bank == 0 else bank1_valid).value[i]

    def bank_resp_data(i, cap):
        bank = cap['bank']
        return (bank0_data if bank == 0 else bank1_data).value[i]

    result = (
        Pattern()
        .wait(req_valid)
        .capture('bank', req_addr & 1)
        .consume(bank_resp_fire, channel=lambda i, cap: banks[cap['bank']])
        .capture('rdata', bank_resp_data)
        .match()
    )

    valid = result.filter_ok()
    # Both reqs complete at cycle 4 because their banks are independent.
    assert len(valid) == 2
    pairs = sorted(zip(valid.captures['bank'].value, valid.captures['rdata'].value))
    assert [(int(b), int(d)) for b, d in pairs] == [(0, 0xAA), (1, 0xBB)]
    # End cycle for both is 4 (concurrent).
    np.testing.assert_array_equal(valid.end.value, [4, 4])


def test_multi_bank_plain_wait_observes_concurrent_responses():
    """Same data as above but using observational wait (no consumption)."""
    req_valid = _wf([1, 1, 0, 0, 0, 0, 0])
    req_addr = _wf([0, 1, 0, 0, 0, 0, 0], width=8)
    bank0_valid = _wf([0, 0, 0, 0, 1, 0, 0])
    bank1_valid = _wf([0, 0, 0, 0, 1, 0, 0])

    def bank_resp_fire(i, cap):
        bank = cap['bank']
        return (bank0_valid if bank == 0 else bank1_valid).value[i]

    # Same pattern, plain wait → both instances observe the same response cycle.
    result = (
        Pattern()
        .wait(req_valid)
        .capture('bank', req_addr & 1)
        .wait(bank_resp_fire)
        .timeout(8)
        .match()
    )

    valid = result.filter_ok()
    assert len(valid) == 2
    np.testing.assert_array_equal(valid.end.value, [4, 4])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
