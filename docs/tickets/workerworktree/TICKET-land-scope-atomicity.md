<!-- fak-workerworktree-key: land-scope-atomicity-before-wip-normalization -->

# fix(workerworktree): admit selected scope before normalizing WIP tests

Process cause: harness-failure

## Parent context

Public issue-tagging landing commit `97d832f990` exposed this worker-land behavior; `75578b5a86` repaired the affected test file.

## Current state

`fak worktree worker land` can normalize the complete worker delta before it proves that every changed path belongs to the selected feature scope. A refused landing stripped `//go:build wip_mcp_search` from `internal/gateway/mcp_search_test.go`, even though that path was unrelated to the issue-tagging feature. A later broad-path landing carried the deletion into public `main` in `97d832f990`; repair commit `75578b5a86` restored the tag.

## Why this is next

A fail-closed landing refusal must preserve every out-of-scope byte. Mutating an unrelated test before admission can activate quarantined coverage and corrupt recoverable peer WIP.

## Working spine

Complete worker delta -> exact selected-path admission -> WIP normalization only inside admitted paths -> isolated commit or preserved reconciliation refusal.

## Core through-line

Move or guard WIP-fence normalization so it runs only after exact path admission and only for admitted files. Refusal must leave the worker tree byte-identical, including leading build constraints.

## Gold-plating boundary

Do not redesign isolated landing, leases, CAS, or build-tag policy. Do not change the semantics of an explicitly selected WIP test. Repair only ordering and scope of the existing normalization step.

## Routing

```routing
lane: workerworktree
paths:
  - internal/workerworktree/land.go
  - internal/workerworktree/land_test.go
expected_steps: 4
```

## Definition of done

- [ ] A refused landing with an unrelated `//go:build wip_mcp_search` file leaves that file byte-identical.
- [ ] WIP normalization cannot run before complete selected-path admission.
- [ ] Explicitly selected WIP-test normalization retains its existing behavior.
- [ ] Focused workerworktree tests pass on Windows.

## Done condition / witness

The focused workerworktree test exits zero and a regression fixture proves both out-of-scope preservation and explicitly selected WIP normalization.

## Witness

```text
go test ./internal/workerworktree -run 'Test.*Land.*(Scope|WIP|BuildTag)' -count=1
```

## Acceptance gate

The focused witness exits zero and a regression fixture proves both the out-of-scope preservation path and the explicitly selected normalization path.

## Closure binding

The resolving commit cites this ticket and includes the focused green witness. Do not close from a refusal receipt without byte-for-byte worker-tree read-back.

## Work estimate

Estimate: 3 points. Uncertainty: normalization may have more than one entry path.

## Overall completion contribution

Contribution: 3/100 points.

## Completion standard

integrated
