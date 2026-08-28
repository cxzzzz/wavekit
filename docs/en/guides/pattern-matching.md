# Pattern matching

Pattern matching describes temporal relationships among signals across multiple
clock cycles, making complex multi-cycle behavior easier to analyze. It supports
higher-level analysis of protocol behavior and other temporal relationships,
including transaction extraction, latency measurement, and timeout detection.

It supports two ways to describe a transaction:

- **Declarative patterns** use a `Pattern` builder for a stable, mostly fixed
  flow. Choose this approach when the transaction structure is known in advance
  and benefits from a readable step description.
- **Programmable patterns** use a normal synchronous Python function for dynamic
  transaction shapes, such as value-dependent branches, dynamic lengths, or
  Python result structures.

Both approaches use the same event model and result concepts. All waveforms used
by one pattern must use the same clock source, sampling edge, and time or cycle
window.

## Declarative patterns

A declarative pattern is a sequence of steps chained on `Pattern`. Its steps
can wait for or consume events, advance time, capture values, check requirements,
and express loops or branches:

- `wait()` waits for a condition without claiming the event.
- `consume()` waits for and claims an event on a logical channel.
- `delay()` advances by a fixed number of cycles.
- `capture()` records a value without advancing time.
- `require()` checks a condition and reports a failed match when it is false.
- `loop()`, `repeat()`, and `branch()` express repeated or conditional flows.

```python
from wavekit.pattern import Pattern, match

request = req_valid & req_ready
response = rsp_valid & rsp_ready

pattern = (
    Pattern()
    .wait(request)
    .consume(response, channel='response')
    .capture('rsp_data', rsp_data)
)
result = match(pattern)
```

The first blocking step determines the cycles at which a pattern starts
matching. Later blocking steps describe the timing within that match. Once a
blocking step is satisfied, the following step is evaluated in the same cycle;
use `delay(1)` to advance to the next cycle explicitly.

For repeated or conditional flows, combine these steps with `loop()`, `repeat()`,
and `branch()`:

```python
beat = Pattern().consume(w_valid & w_ready, channel='w').capture(
    'data', w_data, mode='list'
)
pattern = Pattern().wait(aw_valid & aw_ready).loop(beat, until=w_last)
result = match(pattern)
```

## Programmable patterns

Programmable patterns use the same pattern operations as declarative patterns,
but call them through `ctx` inside the function body. The context provides
`ctx.value()`, `ctx.wait()`, `ctx.consume()`, `ctx.try_consume()`, `ctx.delay()`,
`ctx.capture()`, and `ctx.require()`. These operations have the same roles as
their declarative counterparts, while `ctx.value()` reads a scalar from a
waveform at the current sample.

Programmable patterns are useful when later steps depend on values read from
the waveform. The runtime invokes the body from each possible start cycle, so
the body can use ordinary Python conditionals and loops to choose the flow based
on values read during the match.

For this reason, precompute fixed `Waveform` expressions outside the body to
avoid processing the full waveform repeatedly. Use `ctx.value()` inside the
body when a decision depends on the current cycle.

```python
from wavekit.pattern import collect

cmd_fire = cmd_valid & cmd_ready
rsp_fire = rsp_valid & rsp_ready


def read_command(ctx):
    if not ctx.value(cmd_fire):
        return None

    opcode = int(ctx.value(cmd_op))
    length = int(ctx.value(cmd_len))
    data = []

    if opcode == 0:
        ctx.consume(rsp_fire, channel='response')
        for _ in range(length):
            ctx.consume(r_valid & r_ready, channel='read-data')
            data.append(int(ctx.value(r_data)))
        return {'opcode': 'read', 'data': data}

    if opcode == 1:
        for _ in range(length):
            ctx.consume(w_valid & w_ready, channel='write-data')
            data.append(int(ctx.value(w_data)))
        ctx.consume(rsp_fire, channel='response')
        return {'opcode': 'write', 'data': data}

    ctx.require(False, message=f'unknown opcode {opcode}')
    return None

commands = collect(read_command)
```

Use `ctx.try_consume()` for non-blocking arbitration or polling. For a linear
burst, `ctx.consume()` is usually clearer.

## Event consumption and channels

`wait()` observes an event without claiming it. `consume()` waits for an event
and claims it for a logical channel. The channel identifies the event stream used
for ownership and arbitration.

At most one match can claim an event on the same channel and cycle. Events on
different channels can be consumed independently.

## Results and failures

Pattern execution produces either match records or extracted Python values.

### Match records

`match()` accepts either a declarative `Pattern` or a programmable body and
returns `MatchRecords`.

For a programmable body, use:

- return `ctx.OK` to record a successful match;
- return `None` to skip the current start cycle without producing a record.

A `MatchRecords` object contains:

- `start`, `end`, and `duration`;
- `status`;
- named `captures`.

`end` is inclusive. Extract a matched waveform window with
`cycle_slice(start.clock, end.clock + 1)`.

Use:

- `filter_ok()` to keep successful rows;
- `filter_failed()` to keep failed rows;
- `filter_status(MatchStatus.Timeout)` to select a specific status.

### Collected values

`collect()` accepts a programmable body and returns a list containing each
non-`None` value returned by the body.

### Failures and timeouts

Pass `timeout=<cycles>` to `match()` or `collect()` to bound the duration of each
candidate match.

- `match()` records timeout and requirement failures in the result status;
- `collect()` raises `PatternError` when a timeout or requirement failure occurs.
