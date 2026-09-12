# Guard prelaunch system commit admission

Tracking: https://github.com/anthony-chaudhary/fak/issues/12876

## Problem and intended outcome

Direct guarded child launch currently starts the process before evaluating the existing system commit reserve. Evaluate the same reserve immediately before both child-start call sites, preserving runtime monitoring after admission. This core harness repair changes neither the threshold nor unsupported-platform behavior.

## Owned recovery paths

`cmd/fak/guard_child_resource.go`, `cmd/fak/guard_child_resource_supervision_test.go`, `cmd/fak/guard_child_supervision.go`, and `internal/procguard/commit*.go`.

## Witness

`go test ./internal/procguard ./cmd/fak -run 'Test(SystemCommitSnapshot|GuardPrelaunchSystemCommitHeadroom)' -count=1`, plus required isolated validation and the real CLI refusal witness recorded on #12876. Green focused tests alone do not establish successful landing.

## Recovery state, 2026-09-12

The preserved implementation was compared against current `origin/main`; its intended changes are still missing and the patch applies cleanly. Source worktrees remain unchanged. Recovery uses a scoped lease and a separate task worktree. Required isolated validation against `5b6a02396efa7d7f9d858e1c56d8c4a89b881660` passed formatting, build, and vet, then timed out in affected-package tests at 720013 ms. Smoke was not reached. No commit or push is claimed.

The first current-main attempt refused `WSL_CAPABILITY_PREFLIGHT_FAILED` before allocation. Direct WSL execution succeeds; a serialized retry follows format normalization. Previous worker reports on the issue also record unrelated full-package baseline failures and a stranded disambiguation cache lock. Those gates remain mandatory and are not bypassed.

The candidate is preserved in local WIP ref `refs/fak/wip/operational-recovery-12876-12877`; the initial recovery object is `975aff966c53df150deec92a8d8ab9fbcb8b2a92`. The validation receipt is `_scratch/operational-recovery/public-validation-retry.json` in the recovery task. Next check: complete affected-package tests and smoke under the required gate before attempting native landing. The timeout does not prove a guard test failed.

A subsequent focused diagnostic against newer observed main `2634b9edb4fc319f6b363b6a9f206ff311dafe90` refused at WSL capability preflight (15219 ms) before tests. Its claim is UNRUN. No repeated diagnostic or landing attempt followed.

## Done condition

Both launch paths refuse low headroom before process start, focused behavior and required package/landing gates pass, and the exact scoped commit is verified on remote main. Until then preserve the candidate and keep #12876 open.
