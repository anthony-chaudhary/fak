<!-- fak-gateway-key: corpus-resolver-no-module-temp-v1 -->

# test(gateway): isolate the no-module-root corpus resolver fixture

## Current state

`TestCalibrationDriftCorpusResolvesTheTrackedMirror/no_module_root_resolves_nothing`
assumes `t.TempDir()` is outside every Go module. Managed worker validation may
place temporary directories beneath the detached worktree, so
`resolveCorpusFrom` correctly walks upward to that worktree's `go.mod` and tracked
`docs/nightrun/gateway-usage.jsonl`. The test then fails at
`internal/gateway/calibration_corpus_drift_test.go:329` because its fixture did
not establish the condition it claims to test.

Go 1.25's `testing.T.TempDir` uses `GOTMPDIR` directly when it is set. The
failure reproduces on unchanged public `fak` source and is independent of
the native-context gateway changes that exposed it in a full-suite run. The
focused native-context gateway contracts remain green. Closed #5406 corrected
which corpus the guard selects; it did not make this no-module fixture
independent of the host's temporary-directory placement.

## Classification

- Portfolio tier: 1 — all-in-one gateway validation reliability.
- Centrality: Core.
- Work unit: S1 leaf.
- Priority: P2.

## Problem frame

- **Centrality:** Managed detached worktrees are the sanctioned landing route,
  and their full gateway suite must give the same verdict as a root checkout.
- P1: preserved - corpus data and serving behavior do not change.
- P2: advanced - remove an environment-dependent false red from the gateway
  validation gate.
- P3: preserved - keep the resolver's upward module-root search intact.
- P4: preserved - this is a filesystem fixture correction with no performance or
  physical-hardware claim.

## Parent context

Follow-up blocker discovered while validating #12663. It preserves the tracked
corpus behavior delivered by closed #5406.

## Why this is next

This is the sole full-gateway-suite limitation recorded during #12663 validation,
and it is directly triggered by the repository's required managed-worktree
environment.

## Core through-line

Explicit isolated filesystem fixture -> resolver starts outside a module -> no
module root found -> deterministic negative assertion in every checkout layout.

## Working spine

Gateway corpus resolver test -> synthetic filesystem root -> ancestor walk ->
managed-worktree full-suite witness.

## Gold-plating boundary

Do not change the tracked corpus location, gateway calibration constants,
percentile logic, production resolver behavior, or managed-worktree layout. Do
not weaken the assertion into a skip.

## Done condition

- [ ] The no-module-root case constructs an ancestor chain that cannot contain
      the checkout's `go.mod`.
- [ ] The live-only, tracked-only, and both-present synthetic module cases retain
      their #5406 behavior.
- [ ] The focused test passes both with the normal host temp directory and with
      `GOTMPDIR` rooted beneath the repository.
- [ ] The scoped gateway suite passes in a managed detached worktree.

## Definition of Done

- [ ] The test verdict is independent of `GOTMPDIR`/`TEMP`/`TMP` placement.
- [ ] Production corpus resolution is unchanged.
- [ ] No private host path appears in test output or committed fixtures.

## Acceptance gate

The exact focused command passes when the process temp root is inside the module,
all four resolver subtests still execute, and the scoped gateway suite is green
in a managed worktree.

## Closure binding

The DCO-signed resolving commit closes the created public issue and carries
`(fak agent)`.

## Witness

Run `go test ./internal/gateway -run '^TestCalibrationDriftCorpusResolvesTheTrackedMirror$' -count=1 -v`
with `GOTMPDIR` beneath the checkout. The negative subtest incorrectly resolves
the checkout's `docs/nightrun/gateway-usage.jsonl`.

### Verifiable Witness

```powershell
New-Item -ItemType Directory -Force .gotmp | Out-Null
$env:TEMP=(Resolve-Path .gotmp).Path
$env:TMP=$env:TEMP
$env:GOTMPDIR=$env:TEMP
$env:GOWORK='off'
go test ./internal/gateway -run '^TestCalibrationDriftCorpusResolvesTheTrackedMirror$' -count=1 -v
```

Current failure in the negative subtest:

```text
calibration_corpus_drift_test.go:329: resolved ".../docs/nightrun/gateway-usage.jsonl" outside any module root
```

## Likely files

- `internal/gateway/calibration_corpus_drift_test.go:268-332`
- `docs/tickets/gateway/TICKET-01-corpus-resolver-temp-root.md`

## Blast radius and affected lanes

- Affected: one gateway corpus-resolver test fixture and the full gateway
  validation witness.
- Unaffected: production corpus resolution, calibration values, native request
  admission, model catalogs, proxy traffic, and compute backends.
- Fallback: root checkouts with an external temp directory continue passing while
  the managed-worktree false red is tracked.

## Lane

gateway-tests

## Expected steps

2

```routing
lane: gateway-tests
paths: internal/gateway/calibration_corpus_drift_test.go, docs/tickets/gateway/TICKET-01-corpus-resolver-temp-root.md
expected_steps: 2
```

## Work estimate

Estimate: 1 point.

## Overall completion contribution

Contribution: 1/1 point.

## Completion standard

development

## Tracking and validation update

Tracked by [#12670](https://github.com/anthony-chaudhary/fak/issues/12670).
The full gateway suite passes when the task-specific `GOTMPDIR`, `TEMP`, and
`TMP` directories are outside the checkout. This isolates the fixture defect;
it does not fix the repository-local temporary-directory case.
