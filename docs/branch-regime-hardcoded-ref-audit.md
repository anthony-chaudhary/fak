---
title: "Branch regime hard-coded ref audit"
description: "Classification map for hard-coded main/master references during the dev/main branch-role migration."
---

# Branch Regime Hard-Coded Ref Audit

**Issue:** #1701.
**Status:** phase-1 audit. Workflow branch refs are freshness-checked by
`internal/workflowaudit`; non-workflow refs below are classified for replacement
or retention and still need a broader lint gate.

Do not use this report as permission for a blind `main` -> `dev` replacement.
The branch regime has two legitimate meanings for `main`: public/release front
door and legacy compatibility. Only development-source assumptions should move
to branch-role config.

## Regeneration Commands

```bash
rg -n "github\\.ref_name == 'main'|refs/heads/main|origin/main|\\bmain\\b|\\bmaster\\b" .github cmd internal tools docs README.md AGENTS.md CONTRIBUTING.md
fak workflow-audit --write-doc
go test ./internal/workflowaudit -count=1
```

## Workflow Refs

Workflow refs are already covered by
[`docs/ci/workflow-branch-audit.md`](ci/workflow-branch-audit.md). That generated
report classifies each `.github/workflows/*.yml` branch/tag reference as one of:

- `development`
- `release-front-door`
- `tag`
- `legacy`
- `unclassified`

The regression gate is `go test ./internal/workflowaudit -count=1`; a new
unclassified workflow development-path ref reds that package.

## Development-Source Assumptions To Replace

| Path family | Current hard-coded shape | Classification | Required move |
|---|---|---|---|
| `tools/extend_preflight.py` | `on-master`, `branch == "master"`, recovery text telling operators to `git switch master` | development-source assumption | Replace with the configured development branch or retire the stale preflight in favor of `fak commit` / hook checks. |
| `tools/fleet_control_pane.py` and tests | `DEFAULT_WORKTREE_MASTER_REF = "origin/master"`, `--master-ref`, `worktree_master_ref` | development-source assumption hidden behind legacy naming | Rename the config key or add a branch-role-derived default; keep compatibility parsing only as legacy input. |
| Release/status callers that still pass `main` as a default branch argument | literal fallback branch for release/front-door checks | mixed: some release-front-door, some development-source | Audit call by call; if the value means release branch or public front door, bind to `[branch_roles].release_branch` or `public_front_door`; if it means source, bind to `release_source`. |
| Dispatch, guard, or prompt text that says everyday work lands on `main` | current no-cutover operator law | development-source assumption, valid only before cutover | Keep until #1703 proceeds, then replace with configured development branch text in one coordinated prompt/hook update. |

## Intentional Main/Master Refs To Keep

| Path family | Classification | Why it stays hard-coded or explicitly named |
|---|---|---|
| Public install docs and release links | public front door | User-facing docs should stay anchored on `main` / tagged releases unless #1700 changes the public URL policy. |
| `.github/workflows/*` `github.ref_name == 'main'` release/feed gates | release-front-door | These jobs intentionally post or publish only from the public front door; see the workflow audit. |
| `.github/workflows/*` `master` arms | legacy | Compatibility arm captured in `internal/workflowaudit/allow.txt`; new legacy arms must be reviewed. |
| `docs/stable-releases/*`, historical proof docs, and imported tracker closeouts | historical archive | These quote old branch names and should not be rewritten as current policy. |
| `tools/bench_migrate*.py`, `tools/bench_node.README.md`, and test fixtures with `"branch": "master"` | fixture / imported benchmark metadata | These are data examples unless the tool uses the branch value to select current development source. |
| `internal/bench*` comments using "master goal" | not a git ref | Domain term for fan-out topology, not a branch name. |
| `tools/demo_robustness_scorecard.py` regex that flags `@main` / `@master` installs | public-link guard | The lint intentionally detects mutable install refs. |

## Missing Gate

`internal/workflowaudit` covers workflow refs only. #1701 remains open until a
non-workflow lint/test fails new unclassified development-source references such
as `origin/main`, `refs/heads/main`, or `branch == "master"` outside documented
public/front-door, fixture, historical, or compatibility locations.

## Next Replacement Order

1. Replace `tools/extend_preflight.py`'s `master` operator law with the
   branch-role vocabulary or retire it from the active preflight path.
2. Rename or role-bind `tools/fleet_control_pane.py`'s `worktree_master_ref`
   default.
3. Add a non-workflow hard-coded-ref lint that allows this report's intentional
   families and fails new unclassified development-source refs.
4. After #1703 records `proceed`, update agent/prompt text in one coordinated
   branch-role-aware change.
