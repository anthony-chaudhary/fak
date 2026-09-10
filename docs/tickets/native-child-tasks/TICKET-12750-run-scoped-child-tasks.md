<!-- fak-public-issue: anthony-chaudhary/fak#12750 -->
<!-- fak-agent-key: native-child-run-scoped-task-state -->
# Run-scoped native child tasks

Tracking: [#12750](https://github.com/anthony-chaudhary/fak/issues/12750). Related: #12740, #11414.

```routing
lane: agent
paths: ["internal/agent", "cmd/fak"]
expected_steps: 8
repo: fak
```

## Current state

The native CLI can execute child tasks, but task engine and gate lookup still use
the process-global armed state. The reusable child runner also lives in command
code. Concurrent served parents therefore lack an isolated task namespace and
an owned cancellation/join boundary.

## Core through-line

Parent `RunArm` -> local `TaskState` -> context-routed task gate and engine ->
reusable native child runner -> terminal result -> cancel and join owned effects.

## Gold-plating boundary

No gateway wiring, server-wide semaphore, serving policy, provider changes, or
new orchestration layer. Gateway admission is the next separate leaf.

## Done condition

- [x] `WithChildTaskRunner` creates fresh bounded state for every run.
- [x] Identical IDs and idempotency keys remain isolated between parents.
- [x] Parent cancellation and return cancel and join only owned children.
- [x] The CLI reuses the internal runner and retains current behavior.
- [x] Child catalogs remain depth one and preserve the governed tool floor.
- [x] Async status and wait results cannot replay stale or peer task state.
- [x] Concurrent scoped policies remain local to their parent kernels.

## Verifiable Witness

```bash
go test -race ./internal/agent ./cmd/fak -run 'Test(RunScopedChild|NativeChild)' -count=1
```

## Closure binding

The resolving commit cites #12750 with the `(fak agent)` leaf suffix.
