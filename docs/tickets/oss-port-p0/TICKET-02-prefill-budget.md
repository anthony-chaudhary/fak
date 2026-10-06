# fix(engine): bound native prefill by a resumable per-iteration token budget

## Parent context

Coordinates with parent #8395 (https://github.com/anthony-chaudhary/fak/issues/8395).

Port of the bounded, resumable prefill-chunk mechanism from mini-sglang at pinned
revision `9a91cfa` (MIT licensed). Ported by adapting the mechanism to this tree's
scheduler; the upstream file is not vendored.

<!-- fak-engine-key: oss-port-p0-prefill-budget -->
<!-- fak-cross-key: oss-port-p0-prefill-budget -->
<!-- fak-public-issue: anthony-chaudhary/fak#8395 -->

```routing
lane: modelengine/prefill
paths: ["internal/modelengine/nativesched_prefill.go", "docs/tickets/oss-port-p0/TICKET-02-prefill-budget.md"]
expected_steps: 3
public_issue: anthony-chaudhary/fak#8395
cross_key: oss-port-p0-prefill-budget
```

Process cause: verification-gap

## Scope class

single-package

## Current state

`internal/modelengine/nativesched_prefill.go` already carries the mechanism this
ticket ports, in three connected pieces:

- `SetQwenPrefillMaxTokensPerIteration` (line 289) stores a per-iteration prefill
  token ceiling on the scheduler, and refuses a positive ceiling below
  `nativeQwenPrefillMinChunkTokens` (16) rather than silently degrading to a
  token-loop prefill.
- `qwenPrefillChunkBudget` (line 306) returns the chunk size for one prefill step.
  It returns 0 — meaning no budget, synchronous admission — unless the admission is
  Q4_K, the session belongs to the scheduler's own model, the session cache is
  empty, and `nativeQwenResidentAppendEligible` (line 348) accepts the config.
- `advanceQwenPrefill` (line 481) advances a lane's `promptCursor` by at most that
  chunk, so an over-budget prefill is split and continued across scheduler
  iterations rather than run whole.

The gap is invocation and proof, not mechanism. A repository-wide search for
`SetQwenPrefillMaxTokensPerIteration` returns its definition plus ten call sites,
and every one of those call sites is inside
`internal/modelengine/nativesched_prefill_test.go`. No `cmd/`, `internal/`, or
`pkg/` production line sets the ceiling, so every production admission takes the
`budget < nativeQwenPrefillMinChunkTokens` early return and prefills whole.

The split-then-continue contract is therefore exercised only by tests that arm the
ceiling themselves. Nothing asserts that a prefill whose prompt exceeds the budget
is RESUMED to completion — that the cursor reaches the prompt end and the lane
transitions to decode — rather than silently truncated or restarted.

## For

Local serving clients whose prompts exceed one scheduler iteration's worth of
prefill work on the native model-engine path.

## Problem

A long prompt has no bounded, resumable prefill: it is admitted as one synchronous
unit, so a single request occupies the scheduler for its whole prefill and the
per-iteration latency bound the scheduler advertises does not hold.

## Today

The bounded-resumable mechanism is present in the tree and unreachable from
production. Every production prefill runs whole inside one scheduler iteration.

## Better

The scheduler arms its per-iteration prefill budget on the serving path, and a
prefill that exceeds the budget is split at the cursor and continued on later
iterations until the prompt is consumed and the lane enters decode.

## Why this is next

The mechanism is already written; what is missing is the production arming and the
resume witness. That makes this the cheapest unit in the cohort that closes a real
scheduling bound rather than a stylistic gap, and it is a prerequisite for any
later claim that a bounded prefill landed. It stays a single-file, single-package
leaf.

## Working spine

Serving request admitted -> scheduler arms its per-iteration prefill ceiling ->
prompt longer than the ceiling is consumed in chunks across iterations -> the
cursor reaches the prompt end -> the lane transitions to decode and the stream
opens.

## Core through-line

Arm the bounded prefill budget from the serving path and prove the resume: a
prefill larger than the budget completes, with no truncated prompt and no restart.

## Gold-plating boundary

No chunk-size autotuning or heuristic, no new configuration surface, no change to
the `nativeQwenPrefillMinChunkTokens` floor, no widening of the resident-Qwen
qualification gate in `nativeQwenResidentAppendEligible`, no disaggregated
prefill/decode split, no paged-KV rework, and no throughput or time-to-first-token
number claimed here.

## Verifiable Witness

```bash
go test ./internal/modelengine/ -run 'TestNativeSchedulerPrefillBudgetResumes' -count=1
```

`TestNativeSchedulerPrefillBudgetResumes` is added by this ticket; it does not
exist on the parent tree.

## Done condition / witness

Witness: `go test ./internal/modelengine/ -run 'TestNativeSchedulerPrefillBudgetResumes' -count=1`.

- [ ] [SW-VERIFIED] The new test FAILS on the parent tree and PASSES after the
  change. A test that passes before the change proves nothing.
- [ ] [SW-VERIFIED] At least one production line outside `*_test.go` arms
  `SetQwenPrefillMaxTokensPerIteration` on the serving admission path, and the test
  asserts that call path by name so deleting the arming turns the test red.
- [ ] [SW-VERIFIED] The test admits a prompt strictly longer than the armed budget
  and asserts the prefill RESUMES to completion: the whole prompt is consumed and
  the lane reaches decode, with no truncated prefix and no re-prefill of consumed
  tokens.
- [ ] [SW-VERIFIED] The refusal branch stays intact: a positive ceiling below
  `nativeQwenPrefillMinChunkTokens` is still refused rather than silently ignored.
- [ ] [HW-WITNESSED] Time-to-first-token on a live appliance under a long prompt.
  `unobserved` is recorded as a gap, never as a pass.

## Definition of done

Every checkbox in Done condition, plus a green focused witness recorded in the issue
or the resolving commit.

## Acceptance gate

The focused model-engine test is green, it fails against the parent tree, and the
routing fence and boundary checks are green.

## Closure binding

The resolving commit cites this repo-qualified public issue and carries the
`modelengine` leaf in its commit subject.

## Witness

The focused model-engine test above is the completion witness for the resume
contract. The latency read against a live appliance is a separate hardware
measurement and must cite its own receipt; a green software test is not that
measurement.

## Likely files

- `internal/modelengine/nativesched_prefill.go` — the budget, chunk sizing and
  resume loop this ticket arms and verifies.
- `internal/modelengine/nativesched_prefill_test.go` — gains the new resume test.
- the serving admission site that must arm the ceiling.
- `docs/tickets/oss-port-p0/TICKET-02-prefill-budget.md` — routing and acceptance
  contract.

## Lane

modelengine/prefill

## Blast radius and affected lanes

One package: `internal/modelengine`. Tokenisation, model weights, wire protocols
and non-serving callers remain outside the leaf. Arming the ceiling changes
prefill granularity only; the decode contract and the existing synchronous
admission fallback are unchanged.

## Quarantined fallback mechanism

The synchronous whole-prompt admission path continues unchanged while this leaf is
absent, and remains the fallback for every admission the qualification gate
rejects. If the resume cannot be completed without weakening the existing
preemption-protection invariant for partially-prefilled lanes, leave the ceiling
unarmed and keep this issue open rather than trading correctness for granularity.

- Centrality: Enabling (a bounded, resumable prefill budget is armed on the serving path)
- P1 Context: advanced — a named scheduler bound goes from unreachable to invoked.
- P2 Net value: advanced — long prompts stop monopolising one scheduler iteration.
- P3 Adaptation: preserved — the existing chunk loop and qualification gate are reused unchanged.
- P4 Operations: advanced — yields one bounded dispatchable unit with a resume witness.

## Expected steps

3

## Work estimate

Estimate: 3 points

## Overall completion contribution

Contribution: 3 points toward parent #8395.

## Completion standard

development

## Target operating envelope

- prefill tokens per scheduler iteration: not applicable (this leaf proves the resumable split, not a throughput figure on live hardware)

## Witnessed operating envelope

- prefill tokens per scheduler iteration: not applicable (no production line arms the ceiling yet, so no bounded prefill can be witnessed on the serving path)