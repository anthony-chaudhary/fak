# Release blocked: scorecard control-pane grade-ratchet (2026-06-30)

**Status: `not yet` on the rolling release cut.** The release-process *improvement*
shipped; the *cut* is held by one HARD gate that is not safely fixable from a
Windows dev box.

## The gate

`release_decide` / `release_status` HOLD on `CI_BASE_RED`: every recent completed
main `ci.yml` run (8+ consecutive, e.g. runs 28473224802, 28474306222,
28474428994, 28474590135, 28474736011, 28475054510, 28475431833, 28475580496,
28475706635, 28476032824) is red. With the upstream cascade cleared (below), the
failing step is now:

```
scorecard control pane (folds every *-debt + trend; HARD raw+grade ratchet --check)
```

```
# on CI's Linux node (the gating measure):
RATCHET FAIL: regressed +118 vs @ac838e85 (was 944, now 1062);
grade-debt 43->46 (+3); worsened: seo, repo-hygiene, code-slop

# on a Windows dev box (NOT the gating measure — OS-divergent, do not pin from this):
RATCHET FAIL: regressed +294 vs @ac838e85 (was 944, now 1238);
grade-debt 43->50 (+7); worsened: seo, repo-hygiene, code-slop, milestone, tooling-quality
```

## Root cause (added 2026-06-30, second session): a shared-tree CLOBBER, not pure drift

A **correct Linux re-pin already existed today and was reverted by a worktree sync.**
Chronology of `tools/scorecard_baseline.json` (all 2026-06-30):

```
46be8990 @04:30  baseline -> a9bd997 / total_debt 959   (Linux measurement pin — GOOD)
 7b884e8d @05:09  baseline -> a9bd997 / 959              (still good)
35d34ebe @10:15  baseline -> ac838e85 / 944              (REVERTED by "chore(tools): sync shared worktree")
```

`git diff ac838e85 HEAD -- tools/scorecard_baseline.json` is empty, so the effective
floor is 410 commits behind HEAD. `35d34ebe` is the exact `fak-shared-tree-high-churn-commit`
failure class: a bulk worktree sync swept the baseline back to an older blob, undoing
`46be8990`'s good Linux pin. That clobber is why the ratchet reds *now*.

But the gap is **not** closed by restoring `a9bd997`/959 either: CI's current Linux
measure is 1062 > 959, so ~103 units of GENUINE new debt (orphaned integration/vendor
docs, accumulated slop) landed since that pin. A naive restore-or-pin-over would launder
real rot. The fix is **retire-then-pin on Linux**, and the re-pin commit must touch ONLY
`tools/scorecard_baseline.json` by explicit path so the next `sync shared worktree`
cannot re-clobber it (treat the baseline as a do-not-bulk-sync surface).

## Why this is not a Windows-fixable cut blocker

- The +294 is **peer-accumulated debt** across five scorecards **none of which this
  session touched** — it built up since today's 01:09 re-pin (`ac838e85
  chore(scorecard): re-pin the control-pane baseline after the raw-debt drop`).
- **Re-pinning** (`tools/scorecard_control_pane.py --pin`) would *bless* the
  regression as the new floor — the tool's own doc forbids pinning a regression,
  and it hides real rot from the ratchet.
- **Pinning from Windows is a documented trap**: scorecard debt is OS-dependent, so
  a Windows `--pin` mismatches CI's Linux measure and drifts the baseline stale
  before CI even measures it. The fleet's Linux `garden` cadence owns the re-pin.
- **Retiring +294 across five surfaces** (incl. code-slop / tooling-quality code
  measures, on a peer-churned, lock-storm trunk) is a multi-surface `score-2x`
  gardening pass, not a release-cut step.

## Next checkable step (for the fleet, on Linux)

1. On a Linux node (CI parity): `python tools/scorecard_control_pane.py --check`
   to reproduce the +294, then `--critical` per regressed scorecard.
2. Retire the genuine debt worst-first via the per-surface skills (`seo`,
   `repo-hygiene`, `slop-score`, `milestone`, `tooling-quality`) until the grade
   axis no longer slips (43→50 back toward 43).
3. Only after a *real* drop, re-pin from Linux (`--pin`) and let CI go green; then
   `release_decide` flips to `release` (minor: 0.36.0 → 0.37.0).

## What DID ship this session (the cleared cascade + the improvement)

| commit | what |
|---|---|
| `c99f5c02` | gofmt-align two peer struct literals — unblocked the `gofmt -l .` red |
| `30a4dd6b` | position 28 confusable tokens — disambiguation coverage 94.8% → 97.6% |
| `3c383f15` | re-ground intent-literal metric to live 982/1006 |
| `157ca7e9` | split `cmd/fak/guard.go` (1510 → 1229) into `guard_format.go` — cleared the steerability dispatch-god-file HARD floor |
| `68817f74` | **release-process improvement**: `ci_failure_diagnosis` now names the FAILED CI STEP (was a generic "fix CI") — collapses the exact multi-step hunt this session did into one line |

The architest tier row for `internal/promptaudit` (another push-gate blocker) was
landed by a peer (`fff49208`-era) before this session's duplicate.
