---
name: run-it-all-night
description: Plan and run one bounded overnight issue worker through fak's guarded dispatch path, with typed capacity admission, lane leases, explicit dry-run/live gates, and independent git/DOS/test reconciliation. Use when the operator asks to let a narrow repo task run unattended overnight.
allowed-tools: Bash, Read, Grep, Glob
metadata:
  opencode: claude-only   # capacity admission and lease discipline are load-bearing and not portable per-skill
---

Run one narrow, issue-bound task overnight without bypassing the kernel's account, lease,
or witness gates.

## 1. Resolve the workspace and issue

Start from the current repository root; never hard-code a checkout path. Confirm that the task
has one dedicated issue, a bounded done condition, likely files, and a focused acceptance
witness. If any of those are missing, repair the issue before launch.

```bash
git rev-parse --show-toplevel
gh issue view <issue> --json number,title,state,body,url
```

## 2. Capture capacity and ownership evidence

Run the authoritative dispatch preflight and honor every typed refusal, account park, retry-after,
session ceiling, and seat-depletion result. Never bypass a refusal with Claude dangerous-mode flags or an unguarded process launch.

Acquire the issue's exact file tree through `dos lease-lane` before any edit or live dispatch.
If another live holder owns the tree, stop this unit and choose non-overlapping work.

## 3. Dry-run before live launch

Render the issue-specific guarded command first. A dry-run must show the intended worker
identity, bounded fuel, file tree, account admission, and zero launches. For stale-work units,
use the issue-bound loop and keep issue creation separate from process launch:

```bash
fak stale-work loop --packet <packet.json> --issues <issues.json> --max-wave 1 --json
```

Inspect the plan. Do not add `--live-launch` until the issue is contract-valid, the typed
capacity preflight admits it, and the lane lease is held.

## 4. Launch one bounded worker

Enable exactly one live gate for exactly one worker. Prefer the issue-specific command rendered
by the dry-run (`fak dispatch tick ... --live`); for stale-work, use:

```bash
fak stale-work loop --packet <packet.json> --issues <issues.json> \
  --live-launch --max-wave 1 --json
```

A wider wave is allowed only after collision pricing proves disjoint trees and the account/seat
preflight admits the full width. Overnight duration is not permission for unbounded turns,
workers, retries, or spend.

## 5. Monitor and reconcile from independent witnesses

Monitor the dispatch receipt and run archive, but never accept the worker's narration as proof.
A ship requires all of:

- the expected scoped git diff and origin ancestry;
- the issue-specific test or read-back green;
- `dos commit-audit` returning `OK` / `diff-witnessed`; and
- independent issue state agreeing with the witnessed commit.

If the worker exits without a commit, posts a typed refusal, or makes no progress, record the
failure receipt on the issue. Do not relaunch until the cause is repaired and preflight is green.

## 6. Finish safely

Push promptly once the exact issue is green. Release the lane lease only after publication or a
recorded non-ship. Before launching another worker, repeat capacity preflight, dry-run, and the
full witness barrier.
