<!-- fak-leaseref-key: fenced-remote-release-resurrection -->
## Problem
For agents handing back exact-tree leases, successful `fak leaseref release` must not immediately restore the released record. The command currently calls ambient synchronization after ReleaseFenced deletes the local ref. The generated plan fetches before pushing: the remote record is reimported, and wildcard push cannot propagate deletion.

Source seams: cmd/fak/leaseref_release.go:80; cmd/fak/leaseref_sync_ambient.go:281; internal/loopdrive/leaseref_sync.go:75. Observed release returned ok=true for generation 1 while both local and remote refs retained the original OID. This blocks completed task handoff.

Lineage: closed #10849 introduced ambient lifecycle wiring. Open #11360 covers session-generation ABA; this issue specifically covers remote deletion convergence and must preserve its fencing invariants. Open #12410 concerns radixkv, not lease synchronization.

```routing
lane: leaseref
paths: ["cmd/fak/leaseref_release.go", "internal/leaseref/release.go", "internal/leaseref/release_test.go"]
expected_steps: 6
```

## Core through-line
Publish a narrowly targeted remote release with expected-object fencing, then delete locally with the same fence. Never fetch a just-deleted lease back as part of after-write publication. Existing public harness lease primitives own this repair; private factory workflows consume them.

## Gold-plating boundary
No broad sync protocol rewrite, force pruning, automatic deletion of peers, compute changes, or claim of distributed acquisition arbitration. Preserve explicit offline behavior and expose failed remote publication honestly.

## Witness
`go test ./internal/leaseref -run Release -count=1` plus a real-Git CLI release regression with a bare remote and two clones. Successful fenced release leaves the named local and remote refs absent after fetch; a remote replacement inserted between snapshot and deletion survives, and release refuses success. A different peer lease remains unchanged.

## Scoped Acceptance Criteria
- [ ] Real-Git regression fails before repair and passes afterward.
- [ ] Release cannot resurrect its own remote record.
- [ ] Stale remote OID or generation preserves replacement ownership.
- [ ] Unavailable remote never produces a misleading converged-release success.
- [ ] Focused tests and required validate gates pass before governed landing.

## Current state
Source diagnosis complete; repair and regression pending. Quarantined fallback: retain the lease and request fenced owner recovery; do not clear peer refs manually.

## Work estimate
3 points, bounded concurrency repair.

## Overall completion contribution
3 points toward a 100-point all-in-one governed local-agent baseline.

## Working spine
Governed agent task completion -> fenced lease release -> immediate safe handoff.

## Completion standard
development

## Done condition
The listed scoped acceptance criteria pass with real-Git regression evidence.


## In scope
Fenced remote deletion and direct CLI release integration with bounded tests.
## Out of scope
All other protocols and compute behavior.
## Parent context
Follow-on of #10849; coordinate with #11360.
## Why this is next
Existing completed work cannot hand back its lease.
## Acceptance gate
go test ./internal/leaseref -run Release -count=1 and required validate.
## Closure binding
Fixes this issue only after witnessed governed landing.
## Likely files
internal/leaseref/release.go
internal/leaseref/release_remote_test.go
cmd/fak/leaseref_release.go


Tracking: https://github.com/anthony-chaudhary/fak/issues/12877


## Compatibility and remaining protocol boundaries
The force option is explicitly local-only; remote force release is unsupported.
Remote deletion can precede a history-write or local-CAS failure; the error/refusal retains local state for fenced retry. The existing wildcard synchronization never included refs/fak/history, so cross-clone generation-history propagation remains #11360. This patch does not provide distributed acquisition arbitration or stop a third stale clone from later publishing its retained ref; it prevents the releasing CLI from fetching its just-deleted record back.

## Reproduction receipt
With ReleaseFencedRemote initially delegating to the unchanged local ReleaseFenced, TestReleaseFencedRemote/converges failed: released lease resurrected: exists=true. Replacement-boundary test also failed because release never attempted a fenced remote deletion. Final green witness pending.

## Verifiable Witness
go test ./internal/leaseref -run Release -count=1


## Verification receipt (2026-09-12)
- PASS: go test ./internal/leaseref -run Release -count=1 (56.963s), including five real-Git remote-release cases and existing local release tests.
- PASS: independent parent source review of final CAS ordering and refusal behavior.
- PASS: staged public leak audit; gofmt and git diff --check.
- PARTIAL: fak validate --mine the five scoped paths timed out after 4m0.064s during vet; all five overlays checked, zero skipped, test phase unrun. Landing remains gated pending a complete witness.
- CLI focused test suite pending.

## Final scoped evidence and landing blocker
- PASS: go test ./cmd/fak -run 'TestLeaseref(Release|AcquireReleaseRenewAmbientSync)' -count=1 (56.514s).
- PASS: 5-Gate staged boundary check, four Go files, zero violations.
- PASS: candidate executable build.
- RED: required fak validate --timeout 8m --mine five owned paths refused WSL capability preflight after 14.5s before workspace allocation; no fallback selected. Earlier attempt timed out during vet after four minutes. No validation gate was bypassed, no production lease was changed, and no trunk commit was made.
- Context authorship is COLLOCATED (assigned leaf/no recursive fanout), with independent parent review of the final concurrency changes.
