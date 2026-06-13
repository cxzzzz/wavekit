# Error Handling

> How errors are handled in this project.

---

## Overview

wavekit uses Python's built-in exceptions with descriptive messages. The library
does not define custom exception hierarchies — it relies on `ValueError`,
`TypeError`, and `RuntimeError` with clear error messages explaining what went
wrong and how to fix it.

---

## Error Types

| Exception | When Used | Example |
|-----------|-----------|---------|
| `ValueError` | Invalid input values, constraint violations | `width is None`, `signedness mismatch` |
| `TypeError` | Wrong argument types | `mask requires boolean numpy array` |
| `RuntimeError` | External dependency unavailable | FsdbReader when Verdi runtime missing |
| `PatternError` | Pattern definition/runtime errors (from `pattern/errors.py`) | Invalid pattern step configuration |

---

## Error Handling Patterns

### Input Validation at API Boundaries

All public methods validate inputs early and fail fast with clear messages:

```python
# src/wavekit/waveform.py:281
def __add__(self, other):
    if self.signed != other.signed:
        raise ValueError('signedness mismatch')

# src/wavekit/waveform.py:219-222
def mask(self, mask):
    if isinstance(mask, Waveform):
        if mask.width != 1:
            raise TypeError('mask requires waveform with width 1 or boolean dtype')
    elif not isinstance(mask, np.ndarray):
        raise TypeError('mask requires boolean numpy array')
```

### Mutually Exclusive Parameters

```python
# src/wavekit/readers/vcd/reader.py:91-93
if begin_time is not None and begin_cycle is not None:
    raise ValueError('begin_time and begin_cycle are mutually exclusive')
if end_time is not None and end_cycle is not None:
    raise ValueError('end_time and end_cycle are mutually exclusive')
```

### Graceful Degradation for Optional Dependencies

```python
# src/wavekit/__init__.py:53-59
class _FsdbReaderStub:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'FsdbReader requires the Verdi FSDB runtime (libNPI.so).\n\n'
            'Set WAVEKIT_NPI_LIB to the library path, set VERDI_HOME to the Verdi '
            'installation directory, or ensure libNPI.so is in LD_LIBRARY_PATH.'
        )
```

---

## Common Error Messages

### Waveform Operations

| Error | Cause | Fix |
|-------|-------|-----|
| `signedness mismatch` | Operating on signed + unsigned waveforms | Convert with `.as_signed()` or `.as_unsigned()` |
| `width is None` | Bit operation on width-unknown signal | Ensure signal was loaded with range info |
| `width mismatch: X and Y` | Bitwise op on different-width signals | Ensure both operands have same width |
| `width too large` | Operation on >64-bit signal | Use object dtype or reduce width |
| `mask requires waveform with width 1` | Using multi-bit waveform as mask | Use comparison operator: `wave == 1` |

### Pattern Matching

| Error | Cause | Fix |
|-------|-------|-----|
| `delay() requires n >= 0` | Negative delay value | Use non-negative delay |
| `Cannot specify both 'until' and 'when'` | Invalid loop configuration | Choose one: `until=` (do-while) or `when=` (while) |
| `Must specify either 'until' or 'when'` | Missing loop condition | Add `until=` or `when=` to loop() |
| `Waveform clock arrays are not aligned` | Pattern uses waveforms sampled on different clock axes | Load/construct all pattern waveforms from the same clock axis |

Declarative and programmable patterns validate waveform alignment lazily through
the shared `PatternRuntime` as `ctx.value()` / `ctx.cycle()` / `ctx.time()` or
compiled declarative steps observe waveforms. Both paths must reject different
clock arrays, not just different lengths. Do not add declarative-only eager
waveform collection back; branches that never execute should not validate their
unobserved waveforms.

### File Loading

| Error | Cause | Fix |
|-------|-------|-----|
| `path 'X' matched no signals` | Pattern doesn't match any signals | Check signal path or use `get_matched_signals()` first |
| `matched N signals but expected exactly one` | Multiple matches in single mode | Use exact path or `mode='zip'` |

---

## What NOT to Do

1. **Don't catch and suppress errors** — Let errors propagate to the caller
2. **Don't use bare `except:`** — Catch specific exception types
3. **Don't return error codes** — Use exceptions for error conditions
4. **Don't raise generic `Exception`** — Use specific types (`ValueError`, `TypeError`)
