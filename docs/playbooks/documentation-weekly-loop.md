# Weekly documentation reconciliation

- **Primary audience:** documentation maintainer
- **Lifecycle:** current
- **Generation:** current trunk process
- **Authority:** [documentation audience architecture](../project/DOCUMENTATION-AUDIENCE-ARCHITECTURE-2026-07-15.md)
- **Companion process:** [documentation cohort dispatch](documentation-cohort.md)
- **Last verified:** `fak hygiene`, `tools/docs_scorecard.py`, `tools/check_index_sync.py`, and GitHub issue CLI contracts on 2026-07-15

Use this playbook once per week to detect documentation-route drift on committed `main`, convert every actionable finding into a bounded issue, and select the next collision-safe cohort. The loop moves discovery before page assignment: workers receive known route ownership, lifecycle context, and witnesses instead of discovering architecture conflicts while editing.

## Choose the run mode

| Need | Mode | Default? | Authority |
|---|---|---:|---|
| Reconcile the documentation system | Clean-main weekly run | **Yes** | A clean checkout of committed `origin/main` |
| Check one route before committing | Changed-route check | No | The explicitly owned working-tree files |
| Investigate a historical or versioned set | Scoped audit | No | The named release, backend, or archived corpus |

A peer-dirty shared tree is not evidence of repository-wide drift. Run the weekly baseline in CI, a scheduled clean checkout, or another clean view of committed `main`. Use a dirty working tree only for a path-bounded check whose ownership is known.

## Weekly outputs

Create one dated run record outside the repository or in the configured run archive. Keep:

- the audited commit;
- hygiene and index-sync results;
- the scorecard JSON;
- the open-issue dedupe snapshot;
- findings classified as `fix`, `issue`, `accepted`, or `false-positive`;
- issue numbers created or reused;
- the selected next cohort and its owned paths.

The run is incomplete while a finding exists only in notes or a handoff.

## 1. Pin a clean authority

Start after syncing a clean checkout of `main`. Record the exact commit before measuring:

```powershell
git fetch origin main
git status --short --branch
git rev-parse origin/main
```

If the checkout is peer-dirty, move the scheduled run to a clean CI/fleet checkout; do not clean, stash, or reinterpret peers' files. For an interactive changed-route check, record the exact owned paths and label the result `working-tree`, not `main`.

## 2. Run cheap structural gates first

Structural failures are direct fixes, not scoring inputs:

```powershell
# Native Go gates: local links plus reciprocal INDEX.md/llms.txt reachability.
fak hygiene --gates INDEX_SYNC,BROKEN_LINK --json

# Independent reciprocal index witness used by the current CI fallback.
python tools/check_index_sync.py --audit-tree

# Generated answer-engine map must agree with its inputs.
python tools/gen_llms_full.py --check
```

Stop cohort selection when a front-door link is broken or an index is stale. File or repair that load-bearing route first, then rerun the gates. A network URL outage is recorded separately from a local route defect so transient network state does not rewrite documentation architecture.

## 3. Measure the maintained corpus

Run the scorecard from the same pinned checkout and save machine-readable evidence:

```powershell
$stamp = Get-Date -Format 'yyyy-MM-dd'
$score = Join-Path $env:TEMP "fak-docs-scorecard-$stamp.json"
python tools/docs_scorecard.py --scope reachable --json | Set-Content -Encoding utf8 $score
Get-Content -Raw $score | ConvertFrom-Json | Select-Object verdict,reason,next_action
```

Use `--scope core` for a fast changed-route preflight and `--scope all` only for an intentional full-corpus audit. On recurring automation, pass `--since <last-audited-commit>` first; an unchanged corpus can skip rescoring, while any changed corpus receives a full reachable scan.

The score is a prioritization signal, not permission to bulk-rewrite pages. Read `corpus.worst`, `coverage.defects`, and each affected page before creating work.

## 4. Inspect route drift

For each changed or flagged route, answer these questions against the audience architecture:

1. Who is the one primary audience and what job does the first screen support?
2. Is lifecycle explicit: `current`, `versioned`, `research`, or `archived`?
3. Is release, backend, mode, or maturity context stated where behavior varies?
4. Does the route lead with affirmative current behavior and an explicit default choice?
5. Are proof, scope, provenance, and support boundaries adjacent to the claim they qualify?
6. Do direct inbound navigation entries still describe the page accurately?
7. Does the route end in one checkable next action?
8. Is deeper contributor or historical rationale reachable without occupying the public path?

Classify each finding before acting:

| Class | Action |
|---|---|
| Broken front door, dangling local link, or generated-map drift | Fix or file as the next load-bearing issue. |
| One bounded route defect | Create or reuse one contract-ready issue. |
| Cross-route taxonomy or shared-navigation conflict | Serialize an architecture issue before page fanout. |
| Correct versioned or archived behavior | Record `accepted` with its lifecycle authority; do not modernize it silently. |
| Tool defect or false positive | File against the checker with the smallest reproducer. |

## 5. Dedupe and contract every issue

Capture open issues before filing so repeated weekly runs converge:

```powershell
$issues = Join-Path $env:TEMP "fak-docs-open-$stamp.json"
gh issue list --state open --limit 500 --json number,title,body,labels,url | Set-Content -Encoding utf8 $issues
```

Search title, owned path, and stable body marker. Reuse an existing issue when its owned route and done condition match. Otherwise create one issue through `fak issue create` with:

- reader job and primary audience;
- exact owned route, including direct navigation consumers;
- lifecycle and generation context;
- bounded change and observable done condition;
- local-link plus independent read-back witness;
- priority, generation, class, estimate, parent contribution, and completion standard required by project-work policy.

Refresh the snapshot after creating or reusing issues, then run strict review over the resulting issue set:

```powershell
gh issue list --state open --limit 500 --json number,title,body,labels,url | Set-Content -Encoding utf8 $issues
fak issue contract --from-issues $issues --strict-project-work --strict-witness --strict-scale --strict-born-routed --json
```

Repair held contracts before dispatch. The weekly run does not count an issue merely because it exists; it must be assignable without rediscovering scope.

## 6. Select the next cohort

Prioritize in this order:

1. broken public front doors and invalid commands;
2. mode or generation ambiguity that can send a reader down the wrong path;
3. missing authority, proof, or direct navigation reconciliation;
4. high-reach clarity and choice-model debt;
5. contributor, research, and archive cleanup.

Use [documentation cohort dispatch](documentation-cohort.md) to price path collisions, acquire leases, launch only disjoint routes, and independently witness landed effects. Shared files such as `README.md`, `START-HERE.md`, `llms.txt`, and `INDEX.md` are serial reconciliation points unless exact ownership proves otherwise.

## 7. Close the loop

Before ending the weekly run:

- rerun structural gates if any direct fix landed;
- record the audited commit and scorecard verdict;
- list every finding with an issue number or explicit accepted/false-positive rationale;
- record the next cohort's issue numbers and owned paths;
- verify no finding survives only as prose;
- schedule the next run from the newly audited commit.

A weekly run is complete when committed-main evidence has been measured, every finding has a durable disposition, and the next cohort is contract-reviewed and collision-priced. The next checkable step is to run the structural gates on a clean `origin/main` checkout and record the audited commit.

