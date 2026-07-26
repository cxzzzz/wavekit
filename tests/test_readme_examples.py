"""Smoke tests for the example snippets in README.md / README_ZH.md.

We can't open the real VCDs the README mentions, so we synthesize Waveforms
that mimic the relevant handshakes and run the patterns end-to-end.
"""

from collections import defaultdict

import numpy as np
import pytest

from helpers import wf as _wf
from wavekit.pattern import Channel, Pattern, collect, match


def test_axi_read_latency():
    """README: Pattern().wait(arvalid&arready).wait(rvalid&rready).capture('rdata', rdata)."""
    arvalid = _wf([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    arready = _wf([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
    rvalid = _wf([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    rready = _wf([0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    rdata = _wf([0, 0, 0, 0, 57005, 0, 0, 0, 48879, 0], width=32)
    result = match(Pattern().wait(arvalid & arready).wait(rvalid & rready).capture('rdata', rdata))
    ok = result.filter_ok()
    assert len(ok) == 2
    np.testing.assert_array_equal(ok.end.clock - ok.start.clock, [3, 2])
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
    result = match(Pattern().wait(awvalid & awready).loop(beat, until=wlast))
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
    np.testing.assert_array_equal(stalls.duration.value - 1, [4, 2])


def test_dma_command_stream_collect():
    """README: programmable collect() example for opcode-dependent bursts."""
    cmd_fire = _wf([1, 1, 0, 0, 0, 0, 0, 0])
    w_fire = _wf([0, 0, 1, 1, 0, 0, 0, 0])
    rsp_fire = _wf([0, 0, 0, 0, 1, 1, 0, 0])
    r_fire = _wf([0, 0, 0, 0, 0, 0, 1, 1])

    cmd_op = _wf([1, 0, 0, 0, 0, 0, 0, 0], width=8)
    cmd_addr = _wf([0x100, 0x200, 0, 0, 0, 0, 0, 0], width=32)
    cmd_len = _wf([2, 2, 0, 0, 0, 0, 0, 0], width=8)
    w_data = _wf([0, 0, 160, 161, 0, 0, 0, 0], width=32)
    rsp_status = _wf([0, 0, 0, 0, 7, 0, 0, 0], width=8)
    r_data = _wf([0, 0, 0, 0, 0, 0, 176, 177], width=32)

    op_read = 0
    op_write = 1

    def read_dma_cmd(ctx):
        if not ctx.value(cmd_fire):
            return None

        op = int(ctx.value(cmd_op))
        addr = int(ctx.value(cmd_addr))
        length = int(ctx.value(cmd_len))

        if op == op_write:
            data = []
            for _ in range(length):
                ctx.consume(w_fire, channel='wdata')
                data.append(int(ctx.value(w_data)))

            ctx.consume(rsp_fire, channel='rsp')
            return {'op': 'write', 'addr': addr, 'data': data, 'status': int(ctx.value(rsp_status))}

        if op == op_read:
            ctx.consume(rsp_fire, channel='rsp')
            data = []
            for _ in range(length):
                ctx.consume(r_fire, channel='rdata')
                data.append(int(ctx.value(r_data)))

            return {'op': 'read', 'addr': addr, 'data': data}

        ctx.require(False, message=f'unknown DMA op {op}')
        return None

    commands = collect(read_dma_cmd)
    assert commands == [
        {'op': 'write', 'addr': 0x100, 'data': [160, 161], 'status': 7},
        {'op': 'read', 'addr': 0x200, 'data': [176, 177]},
    ]


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

    result = match(Pattern().wait(req_valid).capture('bank', req_addr & 1).wait(bank_resp_fire))
    ok = result.filter_ok()
    assert len(ok) == 2
    np.testing.assert_array_equal(ok.end.value, [4, 4])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
