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
  * `await ctx.wait(cond, *, require=None)` observes a condition without consuming an event.
  * `await ctx.consume(cond, channel, *, require=None)`
  * `await ctx.delay(n, *, require=None)`
  * `Pattern().wait(cond, *, require=None)` observes a condition without consuming an event.
  * `Pattern().consume(cond, channel, *, require=None)`
  * `Pattern().delay(n, *, require=None)`
* Declarative compilation should pass `require` through the public `PatternContext` methods above instead of using private bridge methods.
* `wait` is observational and non-consuming in both declarative and programmable APIs. Exclusive/FIFO event ownership must be explicit through `consume(cond, channel)`. Channels may be `Channel` objects, hashable keys, or dynamic callables returning either.
* Keep declarative blocking `require` semantics while removing the bridge methods:
  * `wait(require=...)`: check the wait condition first; if the wait succeeds, do not check `require` on the match cycle; only check `require` on blocked waiting cycles.
  * `delay(0, require=...)`: no-op and do not check `require`.
  * `delay(n > 0, require=...)`: check `require` on each blocked cycle before completion. If delay starts at cycle `t`, check cycles `t..t+n-1`, then resume on cycle `t+n` without checking `require` for that delay.
* Decouple `PatternRuntime` from `Pattern` private fields. Runtime construction should receive explicit values (`program`, `axis`, `timeout_cycles`, `max_active`) rather than a `Pattern` object.
* Declarative `Pattern.match()` must not temporarily mutate `self._program` or `self._axis`; remove the current `try/finally` state restoration by constructing runtime inputs locally.
* Split `prepare_declarative_pattern()` into separate responsibilities:
  * `compile_declarative_pattern(steps)` for Step AST to async body compilation.
  * `infer_declarative_axis(steps)` for minimal first-static-waveform scan-axis hinting.
* The compiler should accept read-only `Step` lists rather than a full `Pattern` object.
* Move safe runtime/compiler/error imports in `dsl.py` to module top-level after the boundary cleanup, unless an actual import cycle appears in verification.
* Add a narrow `PatternContext.captures` accessor so compiler/runtime dynamic callables do not reach through `ctx._instance.captures`.
* Resolve dynamic consume channels at runtime blocking/commit time consistently. Avoid evaluating declarative dynamic channel callables only once when entering the step, and avoid resolving an impure/dynamic channel twice differently between ready and commit. Keep the existing `RuntimeOp.ready() -> bool` / `commit()` shape; a consume op may cache the resolved channel internally after `ready()` succeeds so `commit()` consumes that exact channel.
* Declarative AST nodes must mirror the public concepts: `WaitStep` observes and `ConsumeStep` owns. `WaitStep` must not carry a `channel` field; consuming behavior belongs only to `ConsumeStep`.
* Keep the first unguarded declarative `WaitStep` trigger optimization simple. Do not retain cached callable-result reuse solely to avoid double invocation side effects; callable side effects are out of scope for this cleanup.
* Systematize result success naming around `ok`: keep `MatchStatus.OK` / `ctx.OK` semantics, replace `MatchResult.valid` with `MatchResult.ok`, and replace `filter_valid()` with `filter_ok()`.
* Do not keep `valid` / `filter_ok()` compatibility aliases in this cleanup; delete the old result API directly to avoid inconsistent naming and confusion with hardware `valid` signals.
* Align runtime operation concepts with the public API and declarative AST split: `WaitOp` observes only and must not contain a `consume` flag or `channel`; `ConsumeOp` owns FIFO/exclusive matching and independently holds its channel plus ready-channel cache. Prefer clear boundaries over deduplicating every small piece of wait/consume logic.
* Keep condition validation lazy and centralized in `PatternRuntime.eval_condition()`; do not validate conditions eagerly when constructing `PatternContext` runtime ops.
* Cache waveform clock-axis validation by waveform object identity inside `PatternRuntime.note_waveform()` so each waveform's clock array is aligned once per runtime instead of on every cycle observation.
* Let dynamic channel callable exceptions propagate naturally from `PatternRuntime.resolve_channel()`; only invalid returned channel values should be converted to `PatternError`.

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
* [x] Plain declarative and programmable `wait` are observational/non-consuming; focused tests cover multiple candidates observing one event.
* [x] Explicit declarative and programmable `consume` provide FIFO/exclusive channel matching, including hashable and dynamic channel routing.
* [x] `PatternRuntime` does not accept a `Pattern` object or read `Pattern` private fields.
* [x] Declarative `Pattern.match()` constructs compiled program / axis inputs locally without mutating `self._program` or `self._axis`.
* [x] `prepare_declarative_pattern()` is removed in favor of separate compile and axis inference functions.
* [x] Compiler entry points accept `list[Step]` rather than a full `Pattern` object.
* [x] Dynamic consume channel evaluation is consistent between declarative and programmable paths and cannot resolve to different channels between ready and commit.
* [x] Declarative AST uses separate `WaitStep` and `ConsumeStep` nodes; no `WaitStep.channel` field remains.
* [x] First unguarded `WaitStep` trigger handling is simple and does not cache callable results for reuse by the compiled step body.
* [x] Result success APIs use `ok` / `filter_ok()` and no longer expose `valid` / `filter_valid()` aliases.
* [x] Runtime `WaitOp` and `ConsumeOp` are separate concepts with no `consume` flag or channel handling inside `WaitOp`.
* [x] Runtime condition type validation is centralized in `PatternRuntime.eval_condition()` with no separate eager `validate_condition()` path.
* [x] Waveform clock-axis alignment is cached by waveform object identity after the first validation in each runtime.
* [x] Dynamic channel callables are not wrapped by `resolve_channel()`; callable exceptions propagate naturally.

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
* Consider lightweight detection for programmable `RuntimeOp` objects created by `ctx.wait(...)` / `ctx.consume(...)` / `ctx.delay(...)` but never awaited, without flagging valid delayed-await or internal `try_consume()` usage.

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

### Make wait observational and consume explicit

**Context**: Earlier declarative `wait()` semantics used an implicit per-step FIFO channel, so multiple in-flight instances reaching the same wait step competed and only the oldest instance could advance on a matching cycle. Programmable `ctx.wait()` could not honestly provide the same implicit-channel behavior because a stable await-site identity is not available without fragile coroutine/frame introspection. Research into WAL and local Verdi transaction/protocol references also points to a split model: waveform/event queries are observational by default, while protocol pairing/ownership is modeled explicitly with transaction relations, keys, or analysis policy.

**Decision**: In both APIs, `wait(cond, require=...)` is observational and non-consuming. Exclusive/FIFO matching is spelled explicitly as `consume(cond, channel, require=...)`, where `channel` can be a `Channel`, a hashable key, or a dynamic callable returning either. Declarative `Pattern.wait()` should not own an implicit FIFO channel by default.

**Consequences**: This is a breaking semantic cleanup for declarative patterns that relied on implicit wait-step FIFO arbitration. Those patterns should migrate to `Pattern().consume(cond, channel)` with an explicit `Channel` or key. The API becomes easier to explain: `wait` observes, `consume` owns. Declarative and programmable behavior align, and future protocol/transaction analysis can build explicit relation/keyed matching on top of `consume` rather than hiding event ownership inside ordinary waits.

### Decouple runtime construction from Pattern builder state

**Context**: After declarative compilation moved onto the unified runtime, `Pattern.match()` temporarily mutates `self._program` and sometimes `self._axis`, constructs `PatternRuntime(self)`, then restores state in a `finally` block. `PatternRuntime.__init__` reads private fields from the `Pattern` builder (`_program`, `_axis`, `_timeout_cycles`, `_max_active`). `compiler.py` similarly accepts a full `Pattern` object only to read `_steps`, and `prepare_declarative_pattern()` mixes compilation with axis inference.

**Decision**: Treat `Pattern` as the public builder/config object and `PatternRuntime` as an execution engine configured by explicit values. `PatternRuntime` should be constructed with a program body, axis, timeout cycles, and max-active limit directly. Declarative `match()` should compile `self._steps`, infer an axis hint when bounded scanning needs one, and pass those local values to `PatternRuntime` without mutating `self`. Replace `prepare_declarative_pattern()` with `compile_declarative_pattern(steps)` and `infer_declarative_axis(steps)`.

**Consequences**: The runtime no longer depends on `Pattern` private fields, and declarative matching no longer needs `try/finally` state restoration. The compiler boundary becomes clearer: it translates read-only Step AST lists and does not know about the builder object. Safe imports in `dsl.py` can move to module top-level after this cleanup, making execution flow easier to read.

### Keep dynamic consume channel resolution runtime-owned

**Context**: Explicit `consume(cond, channel)` now owns FIFO/exclusive event matching. Channel values may be static `Channel` objects, hashable keys, or dynamic callables. Programmable runtime can evaluate dynamic channel callables using the current cycle index and captures, but declarative compilation must not pre-resolve a dynamic channel once at step entry if the callable depends on the current waiting cycle.

**Decision**: Runtime should own channel resolution for consume operations. The compiler should pass declarative channel expressions through in a form the runtime can evaluate at each blocking cycle. Keep the simple `RuntimeOp.ready() -> bool` / `commit()` protocol; when a consume operation becomes ready, it may store the resolved channel internally and `commit()` must consume that cached channel instead of resolving the callable a second time.

**Consequences**: Declarative and programmable dynamic channel timing stay aligned. Impure or capture-dependent channel callables cannot accidentally choose one channel for readiness and another for commit in the same cycle. This keeps `consume` semantics reliable for future keyed transaction/protocol matching.

### Split declarative WaitStep and ConsumeStep

**Context**: The public API now separates observational `wait()` from owning `consume()`, but the declarative AST still encoded both through `WaitStep.channel`. That internal shape obscured the new model and kept compiler logic coupled to a legacy "consuming wait" concept. The compiler also cached the first wait condition result to avoid invoking callable conditions twice in the same cycle, adding complexity for an out-of-scope side-effect concern.

**Decision**: Add a dedicated `ConsumeStep` AST node and remove `channel` from `WaitStep`. `Pattern.wait()` appends `WaitStep`; `Pattern.consume()` appends `ConsumeStep`. The compiler treats only an unguarded first `WaitStep` as the simple trigger shortcut and compiles it normally afterward; it does not cache callable readiness results for reuse. `ConsumeStep` is never used as a trigger shortcut because consume requires FIFO/ownership semantics.

**Consequences**: Public and internal concepts align: wait observes, consume owns. The AST becomes easier to inspect and extend, and the first-trigger path stays simple. Callable conditions on the first wait may be evaluated once for the trigger check and again by the compiled wait body on matching cycles; avoiding side effects in such callables remains the user's responsibility for now.

### Systematize MatchResult success naming around ok

**Context**: `MatchStatus.OK` and `ctx.OK` already name successful matches as OK, but `MatchResult.valid` and `filter_valid()` use a different word for the same result-success concept. `valid` is also easy to confuse with hardware valid signals, especially when pattern captures include protocol `valid` waveforms.

**Decision**: Use `ok` / `filter_ok()` for result-success queries. Keep the semantics of `MatchStatus.OK` and `ctx.OK`, but replace `MatchResult.valid` with `MatchResult.ok` and replace `filter_valid()` with `filter_ok()`. Do not keep `valid` / `filter_valid()` aliases in this task; remove the old API directly so the naming model is unambiguous.

**Consequences**: Code using `result.valid` or `result.filter_valid()` must migrate to `result.ok` or `result.filter_ok()`. This cleanup intentionally favors a consistent result API over compatibility aliases. Future work may add a more complete status-filtering family such as `failed`, `timed_out`, `require_failed`, `filter(status=...)`, or `filter_failed()`, but this round should stay focused on `ok` and `filter_ok()`.

### Split runtime WaitOp and ConsumeOp responsibilities

**Context**: The public API and declarative AST now distinguish observational `wait` from owning `consume`. Keeping a unified runtime op with a `consume` flag or channel field would preserve an internal version of the older mixed concept and make the runtime boundary less clear.

**Decision**: Make runtime operations mirror the public/API/AST model. `WaitOp` should be observational only and should not contain a `consume` flag or `channel`. `ConsumeOp` should be the only op that owns FIFO/exclusive event matching, holds the channel expression, and caches the channel resolved during `ready()` for use during `commit()`.

**Consequences**: The runtime may contain a small amount of duplicated wait/consume blocking logic, but the concept boundary is clearer and future changes are less likely to accidentally reintroduce implicit consumption into ordinary waits.
