"""Smoke tests for the example snippets in README.md / README_ZH.md.

We can't open the real VCDs the README mentions, so we synthesize Waveforms
that mimic the relevant handshakes and run the patterns end-to-end.
"""

from collections import defaultdict

import numpy as np
import pytest

from helpers import wf as _wf
from wavekit.pattern import Channel, Pattern, match


def test_axi_read_latency():
    """README: Pattern().wait(arvalid&arready).wait(rvalid&rready).capture('rdata', rdata)."""
    arvalid = _wf([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    arready = _wf([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    rvalid = _wf([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    rready = _wf([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    rdata = _wf([0, 0, 0, 0, 57005, 0, 0, 0, 48879, 0], width=32)
    result = match(
        Pattern().wait(arvalid & arready).wait(rvalid & rready).capture('rdata', rdata),
        timeout=256,
    )
    ok = result.filter_ok()
    assert len(ok) == 2
    np.testing.assert_array_equal(ok.duration.value, [4, 3])
    np.testing.assert_array_equal(ok.captures['rdata'].value, [57005, 48879])


def test_axi_write_burst():
    """README: nested Pattern with loop(beat, until=wlast) and capture mode='list'."""
    awvalid = _wf([1, 0, 0, 0, 0])
    awready = _wf([1, 0, 0, 0, 0])
    wvalid = _wf([0, 1, 1, 1, 0])
    wready = _wf([0, 1, 1, 1, 0])
    wdata = _wf([0, 160, 161, 162, 0], width=8)
    wlast = _wf([0, 0, 0, 1, 0])
    beat = Pattern().consume(wvalid & wready, channel='w').capture('beats', wdata, mode='list')
    result = match(Pattern().wait(awvalid & awready).loop(beat, until=wlast), timeout=512)
    ok = result.filter_ok()
    assert len(ok) == 1
    assert list(ok.captures['beats'].value[0]) == [160, 161, 162]


def test_stall_detection():
    """README: .wait(stall.rising_edge()).loop(Pattern().delay(1), when=stall)."""
    valid_sig = _wf([0, 1, 1, 1, 1, 0, 0, 1, 1, 0])
    ready_sig = _wf([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    stall = valid_sig & (ready_sig == 0)
    result = match(Pattern().wait(stall.rising_edge()).loop(Pattern().delay(1), when=stall))
    stalls = result.filter_ok()
    assert len(stalls) == 2
    np.testing.assert_array_equal(stalls.start.value, [1, 7])
    np.testing.assert_array_equal(stalls.duration.value, [5, 3])


def test_axi_read_burst_ooo():
    """Each AR issues a multi-beat read; R beats from different IDs may
    interleave on the bus. Per-instance loop until rlast on the matching ID."""
    arvalid = _wf([1, 1, 0, 0, 0, 0, 0, 0])
    arready = _wf([1, 1, 0, 0, 0, 0, 0, 0])
    arid = _wf([0, 1, 0, 0, 0, 0, 0, 0], width=4)
    rvalid = _wf([0, 0, 1, 1, 1, 1, 1, 0])
    rready = _wf([0, 0, 1, 1, 1, 1, 1, 0])
    rid = _wf([0, 0, 0, 1, 0, 1, 0, 0], width=4)
    rdata = _wf([0, 0, 160, 176, 161, 177, 162, 0], width=32)
    rlast = _wf([0, 0, 0, 0, 0, 1, 1, 0])
    rfire = rvalid & rready
    beat = (
        Pattern()
        .consume(
            lambda i, cap: rfire.value[i] and rid.value[i] == cap['arid'],
            channel=lambda i, cap: ('r', int(cap['arid'])),
        )
        .capture('beats', rdata, mode='list')
    )
    result = match(
        Pattern()
        .wait(arvalid & arready)
        .capture('arid', arid)
        .loop(beat, until=lambda i, cap: bool(rlast.value[i])),
        timeout=64,
    )
    ok = result.filter_ok()
    assert len(ok) == 2
    by_id = {
        int(arid_val): list(beats)
        for arid_val, beats in zip(ok.captures['arid'].value, ok.captures['beats'].value)
    }
    assert by_id == {0: [160, 161, 162], 1: [176, 177]}


def test_multi_bank_concurrent_responses():
    """4-bank cache: each bank has an independent response port; multiple banks
    may fire on the same cycle. Demonstrates why per-bank Channel partitioning
    matters when responses are physically parallel."""
    req_valid = _wf([1, 1, 0, 0, 0, 0, 0])
    req_addr = _wf([0, 1, 0, 0, 0, 0, 0], width=8)
    bank0_valid = _wf([0, 0, 0, 0, 1, 0, 0])
    bank1_valid = _wf([0, 0, 0, 0, 1, 0, 0])
    bank0_data = _wf([0, 0, 0, 0, 170, 0, 0], width=32)
    bank1_data = _wf([0, 0, 0, 0, 187, 0, 0], width=32)
    banks = defaultdict(Channel)

    def bank_resp_fire(i, cap):
        bank = cap['bank']
        return (bank0_valid if bank == 0 else bank1_valid).value[i]

    def bank_resp_data(i, cap):
        bank = cap['bank']
        return (bank0_data if bank == 0 else bank1_data).value[i]

    result = match(
        Pattern()
        .wait(req_valid)
        .capture('bank', req_addr & 1)
        .consume(bank_resp_fire, channel=lambda i, cap: banks[cap['bank']])
        .capture('rdata', bank_resp_data)
    )
    ok = result.filter_ok()
    assert len(ok) == 2
    pairs = sorted(zip(ok.captures['bank'].value, ok.captures['rdata'].value))
    assert [(int(b), int(d)) for b, d in pairs] == [(0, 170), (1, 187)]
    np.testing.assert_array_equal(ok.end.value, [4, 4])


def test_multi_bank_plain_wait_observes_concurrent_responses():
    """Same data as above but using observational wait (no consumption)."""
    req_valid = _wf([1, 1, 0, 0, 0, 0, 0])
    req_addr = _wf([0, 1, 0, 0, 0, 0, 0], width=8)
    bank0_valid = _wf([0, 0, 0, 0, 1, 0, 0])
    bank1_valid = _wf([0, 0, 0, 0, 1, 0, 0])

    def bank_resp_fire(i, cap):
        bank = cap['bank']
        return (bank0_valid if bank == 0 else bank1_valid).value[i]

    result = match(
        Pattern().wait(req_valid).capture('bank', req_addr & 1).wait(bank_resp_fire), timeout=8
    )
    ok = result.filter_ok()
    assert len(ok) == 2
    np.testing.assert_array_equal(ok.end.value, [4, 4])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
