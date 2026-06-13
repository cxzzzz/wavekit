# Consolidate Pattern Runtime Engine

## Goal

Keep only one Pattern execution engine while preserving declarative and programmable Pattern APIs. The result should make the architecture simpler, reduce duplicate semantics, organize files around clear responsibilities, and keep implementation maintainable and elegant.

## What I already know

* The previous task routed declarative `Pattern.match()` through `compile_declarative_pattern()` plus the unified runtime.
* `src/wavekit/pattern/engine.py` used to contain the old NFA-style declarative engine and helper functions.
* `src/wavekit/pattern/runtime.py` now owns the active cycle-major runtime used by programmable patterns and compiled declarative patterns.
* `src/wavekit/pattern/compiler.py` compiles declarative `Step` objects into an internal async program body.
* `src/wavekit/pattern/dsl.py` no longer performs declarative-only full waveform collection/alignment validation.
* `src/wavekit/pattern/steps.py` still contains clone/reset state support (`clone_steps`, `DelayStep.remaining`, `LoopStep.iteration_count`, `RepeatStep.times_remaining`) that is only needed by the old engine.
* User wants to discuss how to avoid retaining two engines, with simple architecture, reasonable file organization, and elegant implementation.

## Assumptions (temporary)

* Public APIs should remain stable for this task unless explicitly decided otherwise.
* `PatternEngine` should stop being a maintained implementation path.
* Behavior should stay covered by existing pattern tests plus targeted regressions if reorganization changes semantics.

## Open Questions

* None.

## Requirements (evolving)

* There must be only one production execution backend for Pattern matching.
* Declarative and programmable Pattern APIs should share the same runtime semantics.
* Delete the old declarative `PatternEngine` implementation instead of keeping a legacy/reference copy. (Done in earlier cleanup commit.)
* Move `PatternError` to `src/wavekit/pattern/errors.py`.
* Remove declarative-only full waveform collection/alignment validation.
* Rename `src/wavekit/pattern/program.py` to `src/wavekit/pattern/runtime.py` without keeping a compatibility shim. (Implemented.)
* Rename runtime-internal `Program*` concepts to clearer Pattern/runtime names:
  `ProgramRuntime` -> `PatternRuntime`, `ProgramContext` -> `PatternContext`,
  `ProgramInstance` -> `PatternInstance`, `ProgramOp` -> `RuntimeOp`,
  `ProgramCondition` -> `RuntimeCondition`, and `ProgramChannel` -> `RuntimeChannel`.
* Do not export `PatternContext` from `wavekit.pattern.__init__` in this task.
* Delete `src/wavekit/pattern/validation.py` after moving/removing its remaining responsibilities. (Implemented.)
* Declarative compilation may infer the scan axis from the first static `Waveform`, but runtime must be the only waveform alignment validation mechanism.
* Declarative patterns with only callable references and no explicit/static axis should continue to fail with the existing runtime axis-detection `PatternError`.
* Remove old-engine-only mutable step state and clone helpers from `steps.py`.
* Treat declarative `Step` objects as read-only AST nodes with no per-instance runtime state.
* Do not update README/README_ZH unless implementation unexpectedly changes public behavior.
* Record architecture notes in task docs/PRD rather than user-facing docs for this task.
* Keep internal boundaries clear: DSL/building, declarative compilation, runtime scheduling, result construction, and shared errors/helpers.
* Remove declarative `tick` compatibility now that same-cycle continuation is the established default.
* Users should express next-cycle continuation explicitly with `.delay(1)` instead of `tick=True`.
* Remove private compiler/runtime bridge methods `_wait_internal(...)` and `_delay_internal(...)`; they were temporary compatibility seams for passing `tick` and declarative blocking `require` into runtime ops.
* Public programmable and declarative APIs should both support blocking `require` on wait/consume/delay:
  * `await ctx.wait(cond, *, consume=False, channel=None, require=None)`
  * `await ctx.consume(cond, channel, *, require=None)`
  * `await ctx.delay(n, *, require=None)`
  * `Pattern().wait(cond, *, require=None, channel=None)`
  * `Pattern().consume(cond, channel, *, require=None)`
  * `Pattern().delay(n, *, require=None)`
* Declarative compilation should pass `require` through the public `PatternContext` methods above instead of using private bridge methods.
* Keep declarative blocking `require` semantics while removing the bridge methods:
  * `wait(require=...)`: check the wait condition first; if the wait succeeds, do not check `require` on the match cycle; only check `require` on blocked waiting cycles.
  * `delay(0, require=...)`: no-op and do not check `require`.
  * `delay(n > 0, require=...)`: check `require` on each blocked cycle before completion. If delay starts at cycle `t`, check cycles `t..t+n-1`, then resume on cycle `t+n` without checking `require` for that delay.

## Acceptance Criteria (evolving)

* [x] No public code path uses the old declarative `PatternEngine`.
* [x] `engine.py` is either removed or no longer contains an alternate runtime.
* [x] No `validation.py` module remains unless it owns generic behavior not already handled by runtime.
* [x] `PatternRuntime` and `PatternContext` live in `runtime.py`; internal imports refer to `runtime.py`.
* [x] No internal `ProgramRuntime` / `ProgramContext` / `ProgramInstance` / `ProgramOp` names remain.
* [x] Old-engine-only `clone_steps`, `clone()` methods, and mutable step runtime state are removed.
* [x] Existing declarative and programmable Pattern tests pass.
* [x] Lint, format, and type checks pass for affected files.
* [x] README/README_ZH remain unchanged unless public behavior changes.
* [x] No public `tick` parameter remains on declarative `wait()` or `consume()`.
* [x] No `tick` field remains in declarative step AST or runtime wait/consume ops.
* [x] No `_wait_internal(...)` or `_delay_internal(...)` methods remain.
* [x] Programmable `ctx.wait`, `ctx.consume`, and `ctx.delay` support `require` with the same blocking semantics as declarative steps.
* [x] Declarative compiler passes `require` through public `PatternContext` wait/consume/delay methods.
* [x] Existing `tick=True` tests are removed or rewritten to use explicit `.delay(1)`.
* [x] Same-cycle wait/capture behavior and explicit next-cycle `.delay(1)` behavior are both covered by focused tests.

## Definition of Done

* Tests added/updated where appropriate.
* Lint / typecheck green for affected files.
* Docs or notes updated if public behavior or import location changes.
* Migration risk considered for internal/private imports.

## Out of Scope (explicit)

* Replacing the async compiler with a bytecode/state-machine IR, unless we decide otherwise during design.
* README/README_ZH updates, unless necessary for changed public behavior.

## Follow-up TODOs

* Consider README/README_ZH updates after the Pattern API stabilizes around explicit `.delay(1)`.
* Consider declarative `Waveform` support for integer parameters such as `delay(n)` and `repeat(..., n=...)` in a separate task.

## Technical Notes

* Inspected `src/wavekit/pattern/dsl.py`.
* Inspected `src/wavekit/pattern/compiler.py`.
* Inspected `src/wavekit/pattern/runtime.py`.
* Inspected `src/wavekit/pattern/engine.py`.
* Inspected `src/wavekit/pattern/steps.py`.

## Decision (ADR-lite)

### Delete the old declarative engine

**Context**: Declarative `Pattern.match()` now routes through `compile_declarative_pattern()` and `PatternRuntime`, but `engine.py` still contained the old NFA-style `PatternEngine`. Keeping both implementations risks semantic drift and makes future fixes ambiguous.

**Decision**: Delete the old `PatternEngine` implementation rather than moving it to a legacy/reference module.

**Consequences**: There will be only one production runtime path. Historical reference remains available via git history. Any remaining shared helpers must move out of `engine.py` before deletion.

### Split errors out of engine.py and rely on runtime validation

**Context**: `engine.py` owned `PatternError` plus `_collect_waveforms()` / `_validate_waveforms()`, even though the old engine should be deleted. After consolidation, full declarative waveform collection duplicates runtime waveform-axis validation.

**Decision**: Add `errors.py` for `PatternError`. Do not keep declarative full waveform collection/alignment validation. Declarative compilation may infer a scan axis from the first static `Waveform`; `PatternRuntime.note_waveform()` validates every actually observed waveform's clock axis.

**Consequences**: Internal imports become explicit and `engine.py` can disappear entirely without turning another module into a generic dumping ground. Branches or captures that are never executed are not validated eagerly; this is intentional runtime semantics.

### Rename program.py to runtime.py

**Context**: `program.py` contained the unified runtime (`ProgramRuntime`, `ProgramContext`, and runtime ops). `runtime.py` is the clearer module name.

**Decision**: Rename `program.py` to `runtime.py` directly and do not keep a `program.py` compatibility shim.

**Consequences**: File organization matches actual responsibility. External users importing the private `wavekit.pattern.program` module will break, but this is not a documented public API.

### Rename Program* runtime concepts

**Context**: After renaming `program.py` to `runtime.py`, internal names like `ProgramRuntime` and `ProgramContext` no longer match the module responsibility. Bare names like `Runtime`, `Context`, and `Instance` are too generic once imported across `dsl.py`, `compiler.py`, tests, or type annotations.

**Decision**: Use Pattern-prefixed names for cross-module/runtime domain concepts and Runtime-prefixed names for scheduler internals: `PatternRuntime`, `PatternContext`, `PatternInstance`, `RuntimeOp`, `RuntimeCondition`, and `RuntimeChannel`. Keep concrete operation names `WaitOp`, `ConsumeOp`, and `DelayOp` unchanged.

**Consequences**: Naming stays explicit without preserving obsolete `Program*` aliases. External users importing private `Program*` names will break, but these names are not part of the documented public API.

### Keep PatternContext internal for now

**Context**: Programmable pattern users interact with `ctx`, and type annotations may eventually need a public context type. Exporting it now would expand the public API while the programmable interface is still settling.

**Decision**: Do not export `PatternContext` from `wavekit.pattern.__init__` in this task.

**Consequences**: Users can omit the annotation or import from the internal runtime module if they knowingly depend on internals. A future public export can be added once the API is stable.

### Preserve callable-only axis behavior

**Context**: Removing declarative full waveform collection means declarative axis inference should only use static `Waveform` references. Patterns made entirely of callables have no static waveform axis to infer.

**Decision**: Keep the existing behavior: callable-only declarative patterns without an explicit or inferable axis fail at runtime with the scan-axis `PatternError`.

**Consequences**: No new special support is introduced. Users who need callable-only scans must provide a waveform axis through an API that supports it.

### Remove old-engine Step runtime state

**Context**: The old `PatternEngine` cloned and mutated `Step` objects per instance using fields like `DelayStep.remaining`, `LoopStep.iteration_count`, and `RepeatStep.times_remaining`. The unified compiler treats steps as read-only declarative AST nodes.

**Decision**: Remove old-engine-only mutable step fields, `init_remaining()` helpers, `clone()` methods, and `clone_steps()`.

**Consequences**: `steps.py` becomes a pure declarative AST module. Any external use of clone helpers will break, but these were internal implementation details rather than documented public API.

### Keep user-facing docs unchanged

**Context**: This task is an internal architecture cleanup. The public Pattern API and documented behavior should remain unchanged.

**Decision**: Do not update README/README_ZH for this task. If notes are needed, keep them in PRD/task records or existing internal specs.

**Consequences**: Scope stays focused. User-facing docs avoid churn unrelated to public behavior.

### Remove tick compatibility and private bridge methods

**Context**: `tick=True` was kept only as a temporary declarative compatibility path after same-cycle `wait` / `consume` became the default. The long-term API model is explicit time movement with `.delay(1)`. The compiler currently relies on private `_wait_internal(...)` and `_delay_internal(...)` methods to pass compatibility-only fields such as `tick` plus declarative blocking `require` into runtime ops. That private bridge also made programmable and declarative `require` support unnecessarily asymmetric.

**Decision**: Remove the public declarative `tick` parameter, remove `WaitStep.tick` and `WaitOp.tick`, and delete `_wait_internal(...)` / `_delay_internal(...)`. Add public programmable `require` support to `ctx.wait`, `ctx.consume`, and `ctx.delay`, matching existing declarative blocking semantics. Declarative compilation should pass `require` through those public context methods rather than using private bridge methods. Next-cycle behavior must be represented by an explicit `DelayStep` / `.delay(1)` in user code or tests.

**Consequences**: Code using `Pattern().wait(..., tick=True)` or `Pattern().consume(..., tick=True)` will break and should migrate to `.wait(...).delay(1)` or `.consume(...).delay(1)`. Runtime scheduling becomes simpler because wait/consume success always resumes in the same cycle. Programmable users can express guarded blocking directly with `await ctx.wait(..., require=...)`, `await ctx.consume(..., require=...)`, and `await ctx.delay(..., require=...)`. Blocking `require` behavior must remain covered by tests while the bridge methods are removed.
