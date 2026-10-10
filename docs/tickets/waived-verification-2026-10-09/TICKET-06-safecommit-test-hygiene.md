# TICKET-06: safecommit real-git tests red after waived test-hygiene commits

## Current state

Covered commits: 365456b1e23 safecommit skipped-ref warnings; 1422aebebfd real-Git noncommit refs; 1ac2b1f42a7 canonical Go check declarations; 8c7b45c4dda, 230ad541ddb sessionctl; 94565f4dfb0 stepobs; 9705aa98a85 agent; cb2abedb667 model shape checks; eeaf26cd83d ctxmmu bulk clearing; 86d6dfdf87d cached scorecard owner lookups; b57f7aea858 pinned CUDA mirror images; 513e7f8a821 docs.

Green at `24033b4080f`: sessionctl, stepobs, ctxmmu, envconfiglint, tools.

Red in `internal/safecommit`:

- `TestPeerWIPAttributionBounded/real-git`: "fixture fast-import: exit status 1".
- `TestPeerWIPLiteralFilterPreservesRealGitOwnership`: context deadline exceeded.

Out of scope: `internal/agent` `TestReadEngineCompanionSymlinkLaunderingGuard` fails because this Windows host refuses symlink creation. That is a host-environment gap dating from 2026-10-02, not caused by this group.

## Working spine

1. Rerun `internal/safecommit` alone with `-count=3` to separate a load flake from a regression.
2. If it is a regression, bisect from 61d65787a5d and fix with fail-before/pass-after tests.
3. Record before/after numbers for the two perf commits.
4. Confirm the pinned CUDA image digests resolve.

## Witness

`go test -count=3 ./internal/safecommit/` exit 0, plus benchmark numbers.

## Done condition

- [ ] safecommit green, or root-caused with an issue filed.
- [ ] Perf deltas recorded.
