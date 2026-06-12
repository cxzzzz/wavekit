# Consolidate Pattern Runtime Engine

## Goal

Keep only one Pattern execution engine while preserving declarative and programmable Pattern APIs. The result should make the architecture simpler, reduce duplicate semantics, organize files around clear responsibilities, and keep implementation maintainable and elegant.

## What I already know

* The previous task routed declarative `Pattern.match()` through `compile_declarative_pattern()` plus `ProgramRuntime`.
* `src/wavekit/pattern/engine.py` still contains the old NFA-style declarative engine and helper functions.
* `src/wavekit/pattern/program.py` now owns the active cycle-major runtime used by programmable patterns and compiled declarative patterns.
* `src/wavekit/pattern/compiler.py` compiles declarative `Step` objects into an internal async program body.
* `src/wavekit/pattern/dsl.py` currently imports `_collect_waveforms` / `_validate_waveforms` from `engine.py`, so `engine.py` cannot be deleted without moving those helpers.
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
* Delete the old declarative `PatternEngine` implementation instead of keeping a legacy/reference copy.
* Move `PatternError` to `src/wavekit/pattern/errors.py`.
* Move waveform collection/alignment validation helpers to `src/wavekit/pattern/validation.py`.
* Keep `src/wavekit/pattern/program.py` named as-is for this task.
* Remove old-engine-only mutable step state and clone helpers from `steps.py`.
* Treat declarative `Step` objects as read-only AST nodes with no per-instance runtime state.
* Do not update README/README_ZH unless implementation unexpectedly changes public behavior.
* Record architecture notes in task docs/PRD rather than user-facing docs for this task.
* Keep internal boundaries clear: DSL/building, declarative compilation, runtime scheduling, result construction, and shared errors/helpers.

## Acceptance Criteria (evolving)

* [ ] No public code path uses the old declarative `PatternEngine`.
* [ ] `engine.py` is either removed or no longer contains an alternate runtime.
* [ ] Shared helpers needed by the unified path live in appropriately named modules.
* [ ] Old-engine-only `clone_steps`, `clone()` methods, and mutable step runtime state are removed.
* [ ] Existing declarative and programmable Pattern tests pass.
* [ ] Lint, format, and type checks pass for affected files.
* [ ] README/README_ZH remain unchanged unless public behavior changes.

## Definition of Done

* Tests added/updated where appropriate.
* Lint / typecheck green for affected files.
* Docs or notes updated if public behavior or import location changes.
* Migration risk considered for internal/private imports.

## Out of Scope (explicit)

* Replacing the async compiler with a bytecode/state-machine IR, unless we decide otherwise during design.
* Removing public `tick` compatibility, unless explicitly chosen.
* Renaming `src/wavekit/pattern/program.py` to `runtime.py` in this task.
* README/README_ZH updates, unless necessary for changed public behavior.

## Follow-up TODOs

* Rename `src/wavekit/pattern/program.py` to `runtime.py` in a later focused cleanup.

## Technical Notes

* Inspected `src/wavekit/pattern/dsl.py`.
* Inspected `src/wavekit/pattern/compiler.py`.
* Inspected `src/wavekit/pattern/program.py`.
* Inspected `src/wavekit/pattern/engine.py`.
* Inspected `src/wavekit/pattern/steps.py`.

## Decision (ADR-lite)

### Delete the old declarative engine

**Context**: Declarative `Pattern.match()` now routes through `compile_declarative_pattern()` and `ProgramRuntime`, but `engine.py` still contains the old NFA-style `PatternEngine`. Keeping both implementations risks semantic drift and makes future fixes ambiguous.

**Decision**: Delete the old `PatternEngine` implementation rather than moving it to a legacy/reference module.

**Consequences**: There will be only one production runtime path. Historical reference remains available via git history. Any remaining shared helpers must move out of `engine.py` before deletion.

### Split errors and validation helpers out of engine.py

**Context**: `engine.py` currently owns `PatternError` plus `_collect_waveforms()` / `_validate_waveforms()`, even though the old engine should be deleted.

**Decision**: Add `errors.py` for `PatternError` and `validation.py` for waveform collection/alignment helpers.

**Consequences**: Internal imports become explicit and `engine.py` can disappear entirely without turning another module into a generic dumping ground.

### Defer program.py rename

**Context**: `program.py` now contains the unified runtime (`ProgramRuntime`, `ProgramContext`, and runtime ops). `runtime.py` would be a clearer name, but this task already touches engine removal and helper relocation.

**Decision**: Keep `program.py` unchanged in this task and record a follow-up TODO to rename it later.

**Consequences**: This task stays smaller and easier to review. File naming is not final, but the rename can be done separately without mixing semantic cleanup with move-only churn.

### Remove old-engine Step runtime state

**Context**: The old `PatternEngine` cloned and mutated `Step` objects per instance using fields like `DelayStep.remaining`, `LoopStep.iteration_count`, and `RepeatStep.times_remaining`. The unified compiler treats steps as read-only declarative AST nodes.

**Decision**: Remove old-engine-only mutable step fields, `init_remaining()` helpers, `clone()` methods, and `clone_steps()`.

**Consequences**: `steps.py` becomes a pure declarative AST module. Any external use of clone helpers will break, but these were internal implementation details rather than documented public API.

### Keep user-facing docs unchanged

**Context**: This task is an internal architecture cleanup. The public Pattern API and documented behavior should remain unchanged.

**Decision**: Do not update README/README_ZH for this task. If notes are needed, keep them in PRD/task records or existing internal specs.

**Consequences**: Scope stays focused. User-facing docs avoid churn unrelated to public behavior.
