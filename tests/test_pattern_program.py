import numpy as np
import pytest

from wavekit import Channel, Signal, Waveform
from wavekit.pattern import MatchStatus, Pattern, PatternError


def _wf(values, width=1, signed=False):
    value = np.array(values)
    clock = np.arange(len(value))
    time = clock * 10
    return Waveform(value, clock, time, signal=Signal('', '', width, None, signed))


def _bool_wf(values):
    return _wf(values, width=1, signed=False)


def test_non_async_program_body_raises():
    def tx(ctx):
        return None

    with pytest.raises(PatternError, match='async'):
        Pattern(tx)


def test_match_records_only_ctx_ok_and_ignores_none():
    fire = _bool_wf([1, 0, 1, 0])
    data = _wf([10, 20, 30, 40], width=8)

    async def tx(ctx):
        if ctx.value(fire):
            ctx.capture('data', data)
            return ctx.OK
        return None

    result = Pattern(tx).match()
    assert len(result) == 2
    np.testing.assert_array_equal(result.start.value, [0, 2])
    np.testing.assert_array_equal(result.captures['data'].value, [10, 30])


def test_match_rejects_other_return_value():
    fire = _bool_wf([1])

    async def tx(ctx):
        if ctx.value(fire):
            return {'bad': True}
        return None

    with pytest.raises(PatternError, match='ctx.OK or None'):
        Pattern(tx).match()


def test_collect_records_non_none_values_including_ok():
    fire = _bool_wf([1, 0, 1])
    data = _wf([10, 20, 30], width=8)

    async def tx(ctx):
        if ctx.value(fire):
            if ctx.index == 0:
                return {'data': int(ctx.value(data))}
            return ctx.OK
        return None

    records = Pattern(tx).collect()
    assert records[0] == {'data': 10}
    assert repr(records[1]) == 'ctx.OK'


def test_context_value_cycle_time_and_offsets():
    fire = _bool_wf([0, 1, 0])
    data = _wf([10, 20, 30], width=8)

    async def tx(ctx):
        if ctx.value(fire):
            ctx.capture('prev', ctx.value(data, offset=-1))
            ctx.capture('cycle', ctx.cycle(data))
            ctx.capture('time', ctx.time(data))
            ctx.capture('scalar', 123)
            return ctx.OK
        return None

    result = Pattern(tx).match()
    assert result.captures['prev'].value[0] == 10
    assert result.captures['cycle'].value[0] == 1
    assert result.captures['time'].value[0] == 10
    assert result.captures['scalar'].value[0] == 123


def test_condition_and_delay_validation():
    fire = _bool_wf([1])

    async def bad_condition(ctx):
        if ctx.value(fire):
            await ctx.wait(1)

    with pytest.raises(PatternError, match='condition must'):
        Pattern(bad_condition).match()

    async def bad_delay(ctx):
        if ctx.value(fire):
            await ctx.delay(True)

    with pytest.raises(PatternError, match='integer'):
        Pattern(bad_delay).match()


def test_wait_delay_and_capture_list_modes():
    fire = _bool_wf([1, 0, 0, 0])
    ready = _bool_wf([0, 0, 1, 0])
    data = _wf([10, 20, 30, 40], width=8)

    async def tx(ctx):
        if ctx.value(fire):
            ctx.capture('samples', data, mode='list')
            await ctx.delay(0)
            ctx.capture('samples', data, mode='list')
            await ctx.wait(ready, consume=False)
            ctx.capture('last', data)
            return ctx.OK
        return None

    result = Pattern(tx).match()
    assert list(result.captures['samples'].value[0]) == [10, 10]
    assert result.captures['last'].value[0] == 30
    assert result.end.value[0] == 2


def test_consuming_wait_requires_explicit_channel():
    fire = _bool_wf([1, 1, 0, 0])
    ready = _bool_wf([0, 0, 1, 0])

    async def tx(ctx):
        if ctx.value(fire):
            await ctx.wait(ready, channel='ready')
            return ctx.OK
        return None

    result = Pattern(tx).match().filter_valid()
    assert len(result) == 1
    assert result.start.value[0] == 0
    assert result.end.value[0] == 2


def test_channel_fifo_and_tuple_keys():
    req = _bool_wf([1, 1, 0, 0, 0, 0])
    req_id = _wf([0, 0, 0, 0, 0, 0], width=4)
    rsp = _bool_wf([0, 0, 0, 1, 0, 1])
    data = _wf([10, 20, 0, 111, 0, 222], width=8)

    async def tx(ctx):
        if ctx.value(req):
            ctx.capture('req', data)
            key = ('r', int(ctx.value(req_id)))
            await ctx.consume(rsp, channel=key)
            ctx.capture('rsp', data)
            return ctx.OK
        return None

    result = Pattern(tx).match().filter_valid()
    np.testing.assert_array_equal(result.captures['req'].value, [10, 20])
    np.testing.assert_array_equal(result.captures['rsp'].value, [111, 222])


def test_try_consume_polling_arbitration():
    req = _bool_wf([1, 0, 0, 0, 0])
    fast = _bool_wf([0, 0, 0, 1, 0])
    slow = _bool_wf([0, 0, 1, 0, 0])

    async def tx(ctx):
        if ctx.value(req):
            while True:
                if ctx.try_consume(fast, channel='rsp'):
                    ctx.capture('kind', 0)
                    return ctx.OK
                if ctx.try_consume(slow, channel='rsp'):
                    ctx.capture('kind', 1)
                    return ctx.OK
                await ctx.delay(1)
        return None

    result = Pattern(tx).match().filter_valid()
    assert result.captures['kind'].value[0] == 1
    assert result.end.value[0] == 2


def test_require_and_timeout_match_statuses_and_collect_raises():
    fire = _bool_wf([1, 0, 0, 0])
    never = _bool_wf([0, 0, 0, 0])

    async def timeout_tx(ctx):
        if ctx.value(fire):
            await ctx.wait(never, consume=False)
            return ctx.OK
        return None

    result = Pattern(timeout_tx, timeout=2).match()
    assert result.status.value[0] == MatchStatus.TIMEOUT
    with pytest.raises(Exception):
        Pattern(timeout_tx, timeout=2).collect()

    async def require_tx(ctx):
        if ctx.value(fire):
            ctx.require(False)
            return ctx.OK
        return None

    result = Pattern(require_tx).match()
    assert result.status.value[0] == MatchStatus.REQUIRE_VIOLATED
    with pytest.raises(Exception):
        Pattern(require_tx).collect()


def test_max_active_guard_guidance():
    never = _bool_wf([0, 0, 0, 0])

    async def tx(ctx):
        await ctx.wait(never, consume=False)

    with pytest.raises(PatternError, match=r'if ctx.value\(fire\)'):
        Pattern(tx, max_active=1).match()


def test_declarative_consume_and_timeout_deprecation():
    req = _bool_wf([1, 1, 0, 0, 0])
    rsp = _bool_wf([0, 0, 1, 0, 1])
    data = _wf([10, 20, 111, 0, 222], width=8)
    result = Pattern().wait(req).consume(rsp, channel='rsp').capture('rsp', data).match()
    np.testing.assert_array_equal(result.filter_valid().captures['rsp'].value, [111, 222])

    with pytest.warns(DeprecationWarning, match='timeout'):
        Pattern().wait(req).timeout(3)


def test_channel_object_works_programmable():
    fire = _bool_wf([1, 0])
    channel = Channel()

    async def tx(ctx):
        if ctx.value(fire):
            await ctx.consume(True, channel=channel)
            return ctx.OK
        return None

    assert len(Pattern(tx).match()) == 1
