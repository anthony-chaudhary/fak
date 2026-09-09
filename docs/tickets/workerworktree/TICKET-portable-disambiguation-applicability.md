# fix(workerworktree): gate concept oracle by repository contract

<!-- fak-workerworktree-key: portable-disambiguation-applicability -->
<!-- github-issue: 12457 -->

```routing
lane: workerworktree
paths:
  - internal/workerworktree/disambiguation.go
  - internal/workerworktree/disambiguation_test.go
  - internal/workerworktree/workerworktree_live_test.go
  - internal/workerworktree/workerworktree_test.go
  - docs/tickets/workerworktree/TICKET-portable-disambiguation-applicability.md
expected_steps: 6
priority: P1
dependencies: []
```

## Current state

`fak worktree worker land --root <selected-repository>` treats every changed Go or Markdown path as relevant to the concept-disambiguation oracle. The oracle then attempts to generate and compare `docs/concept-disambiguation-scorecard/README.md` and `INDEX.md` even when the selected repository and all three evaluated trees have never carried the analyzer contract. A portable land of an ordinary `cmd/*.go` change is therefore refused because generated artifacts that are not part of that repository do not exist.

## Why now

Managed worker isolation and landing are default-on and root-parameterized. The canonical landing primitive must distinguish an inapplicable repository-level invariant from a violated invariant without recognizing repository names or weakening the oracle where its contract exists.

## Parent ref

#11235

## Problem frame

- Centrality: Core
- P1: advanced - arbitrary selected repository roots can use the default governed land path.
- P2: advanced - applicability follows committed tree contents, not environment or repository identity.
- P3: preserved - a repository that carries or removes the analyzer contract still receives the complete before/worktree/post-apply oracle and fail-closed refusal.
- P4: preserved - existing typed reconciliation results remain authoritative when the applicable oracle fails.

## Working spine

Selected root -> candidate tree construction -> content-derived oracle applicability -> full three-tree invariant when applicable -> candidate validation -> CAS -> typed receipt.

## Core through-line

Run the concept-disambiguation oracle when any evaluated tree carries its canonical analyzer contract; skip it only when the selected root never carries that contract across HEAD, worker HEAD, and the post-apply tree.

## Gold-plating boundary

Do not add an environment bypass, repository-name allowlist, configurable policy language, second merge engine, or weaker scorecard semantics. Do not treat a missing analyzer after removal as inapplicable when an earlier evaluated tree carried it.

## Concrete repro witness

```powershell
fak worktree worker land --root <disposable-repository> --worktree <managed-worker> --base-sha <base> --paths cmd/demo/main.go --verify go-build
```

With a valid Go repository that has never tracked `tools/concept_disambiguation_scorecard.py`, landing currently refuses before CAS with a post-apply disambiguation failure because generated scorecard README/corpus files are absent.

## Exact file:line seams

- `internal/workerworktree/land.go:729-737` (`landIsolated`, unconditional path-based oracle admission)
- `internal/workerworktree/disambiguation.go:94-107` (`disambiguationRelevant`)
- `internal/workerworktree/disambiguation.go:206-218` (`verifyAppliedDisambiguation`)
- `internal/workerworktree/disambiguation.go:260-288` (`verifyWithinDeadline`)
- `internal/workerworktree/disambiguation_test.go` (oracle unit witnesses)
- `internal/workerworktree/workerworktree_live_test.go` (portable live landing witnesses)
- `internal/workerworktree/workerworktree_test.go:906-939` (existing fail-closed oracle witness)

## Blast radius and affected lanes

- Primary lane: `workerworktree`
- Affected: managed landing of source and documentation changes in generic repository roots.
- Preserved: public roots that carry the concept analyzer, analyzer removal/staleness detection, CAS/retry, recovery refs, verification, serving, model execution, and ABI.

## Quarantined fallback mechanism

Until fixed, the current lander refuses before CAS and preserves the worker. The implementation must retain that fail-closed behavior whenever any evaluated tree carries the analyzer contract. There is no shared-index or unverified fallback.

## Scoped acceptance criteria

- [ ] 1. [SW-VERIFIED] A disposable external repository with a changed `cmd/*.go` file and no analyzer contract completes managed verified landing.
- [ ] 2. [SW-VERIFIED] Applicability is derived generically from canonical contract paths in the evaluated Git trees, without repository names or environment bypasses.
- [ ] 3. [SW-VERIFIED] When HEAD, worker HEAD, or the post-apply tree carries the analyzer contract, the existing before/worktree/post-apply oracle runs in full.
- [ ] 4. [SW-VERIFIED] Removing the analyzer contract remains applicable and fails closed rather than being classified as not applicable.
- [ ] 5. [SW-VERIFIED] Existing staleness, semantic-validity, clarity, and coverage non-regression checks are unchanged for applicable repositories.
- [ ] 6. [SW-VERIFIED] Focused and full `internal/workerworktree` tests plus `go vet` pass.

## Definition of done

- [ ] Content-derived applicability gating is implemented before the expensive oracle.
- [ ] Never-applicable, contract-present, and contract-removal regression tests pass.
- [ ] Public scrub and exact-diff checks pass.

## Likely files

- `internal/workerworktree/disambiguation.go`
- `internal/workerworktree/disambiguation_test.go`
- `internal/workerworktree/workerworktree_live_test.go`
- `internal/workerworktree/workerworktree_test.go`
- `docs/tickets/workerworktree/TICKET-portable-disambiguation-applicability.md`

## Witness

`go test ./internal/workerworktree -run 'Test.*Disambiguation.*Applicab|TestLiveLand.*WithoutDisambiguationContract' -count=1` must exit 0.

### Verifiable Witness

```powershell
go test ./internal/workerworktree -run 'Test.*Disambiguation.*Applicab|TestLiveLand.*WithoutDisambiguationContract' -count=1
```

## Done condition

The canonical lander succeeds for a generic root that never carries the analyzer contract while retaining the full fail-closed oracle for contract-present, stale, and removal candidates.

## Done condition / witness

The focused witness, `go test ./internal/workerworktree -count=1`, and `go vet ./internal/workerworktree` all exit 0.

## Acceptance gate

`go test ./internal/workerworktree -count=1`

## Closure binding

The resolving signed-off commit cites the created GitHub issue number, carries `(fak workerworktree)`, and is independently landed, read back on `origin/main`, and pushed before closure.

## Expected steps

6

## Work estimate

Estimate: 2 points

## Overall completion contribution

Contribution: 2/10 points for portable default-on managed landing.

## Completion standard

production

## Target operating envelope

- applicable analyzer roots retaining the full fail-closed oracle: = 100 percent

## Witnessed operating envelope

- applicable analyzer roots retaining the full fail-closed oracle: = 100 percent

## Lane

workerworktree
