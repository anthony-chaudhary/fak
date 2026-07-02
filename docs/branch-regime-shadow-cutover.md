---
title: "Branch regime shadow cutover checklist"
description: "Reversible shadow-mode checklist for proving the dev/main branch regime before changing global agent instructions."
---

# Branch Regime Shadow Cutover Checklist

**Issue:** #1703.
**Mode:** shadow or pilot only. `main` remains the accepted everyday agent commit
branch until this checklist records a proceed decision.

This runbook is the proof artifact for the reversible cutover. Fill it out in a
single issue comment or release note before any global prompt, hook, or dispatch
template tells agents to ship normal work to `dev`.

## Entry Criteria

- `dos.toml [branch_roles]` names one development branch, one release branch,
  one release source, and one public front door.
- `fak release status --json` reports the same branch-role tuple and no
  unexplained promotion blockers.
- `fak workflow-audit --write-doc` has a clean audit, with every workflow branch
  filter classified as development, release, tag, legacy, or public-front-door.
- `fak release ship --json` dry-runs from the configured release source and
  reports `source_branch`, `source_sha`, `target_branch`, `target_sha`, and
  `target_ancestry.ok=true`.
- The release source SHA has a green source CI witness or the run records a hold
  decision.
- Forge rulesets for the pilot `dev` branch are present and documented.
- A named worker cohort is selected. Everyone else remains on `main`.

## Pilot Cohort Mechanism

The cohort selector is the per-process env opt-in `FLEET_BRANCH_PILOT=1`
paired with the declared `dos.toml [branch_roles].pilot_development_branch`.
It breaks the chicken-and-egg this gate recorded (a pilot needs workers on
`dev`, but splitting `[branch_roles]` would flip every agent in the shared
tree at once):

- The shared `development_branch` never moves — non-opted workers, hooks, CI,
  and dispatch keep targeting it, so there is no split-brain.
- A worker launched with `FLEET_BRANCH_PILOT=1` resolves its development
  branch to the pilot value everywhere `branchrole.Load` is consumed: the
  `fak commit` OFF_TRUNK guard expects the pilot branch, the
  `reference-transaction` hook allowlists creating it, and
  `fak release status` reports `shadow_cutover.pilot.active=true`.
- Opting in without a declared pilot branch is a surfaced error and falls
  back to the shared development branch — a mislaunched pilot worker cannot
  silently invent a branch.
- The pilot value may never equal the release branch; config parsing refuses
  it.

`fak release status --json` shows the lever under `shadow_cutover.pilot`
(declared branch, opt-in env, whether this process is active).

## Execution Steps

1. Record the starting public-front-door SHA:

   ```bash
   git ls-remote origin refs/heads/main
   ```

2. Create or refresh `dev` from that exact SHA only if the branch does not
   already contain pilot work:

   ```bash
   git fetch origin main
   git push origin origin/main:refs/heads/dev
   ```

3. Capture branch-role status:

   ```bash
   fak release status --json --require-ci-green --limit-commits 50
   ```

4. Capture workflow role coverage:

   ```bash
   fak workflow-audit --write-doc
   go test ./internal/workflowaudit -count=1
   ```

5. Capture release-promotion dry run from the pilot source:

   ```bash
   fak release ship --json --source-branch dev --trunk main --base origin/dev
   ```

6. Move only the named pilot cohort to `dev` by launching each pilot worker
   with `FLEET_BRANCH_PILOT=1` (see Pilot Cohort Mechanism above). Record the
   exact prompt, dispatch packet, or account selector used.

7. Require every pilot worker to ship with the usual path-scoped commit witness.
   Record each commit SHA and `fak commit --preview` result.

8. Re-run release status and release-promotion dry run after the pilot commits.

9. Post a final #1703 decision: `proceed`, `hold`, or `backout`.

## Verification Commands

Use these commands as the minimum proof bundle:

```bash
git ls-remote origin refs/heads/main refs/heads/dev
fak release status --json --require-ci-green --limit-commits 50
fak workflow-audit --write-doc
go test ./internal/workflowaudit -count=1
fak release ship --json --source-branch dev --trunk main --base origin/dev
fak release prplan --base origin/main --head origin/dev --check
```

`fak release prplan` is the promotion-legibility artifact: it folds the
`main..dev` promotion range into PR units grouped by the `(fak <leaf>)`
ship-stamp — the "PRs managed in advance" bodies a human operator reads when
`dev` work ships to `main`. `--check` fails if the range holds unstamped
commits, so legibility is a gate, not a hope.

The proof bundle is incomplete unless it includes:

- the starting `main` SHA used to create or validate `dev`;
- the current `dev` SHA;
- the pilot worker cohort;
- at least one witnessed pilot commit to `dev`, or an explicit `hold` stating
  why the cohort did not run;
- the release-promotion dry-run JSON containing the exact `dev` `source_sha`;
- a final decision with links to command output or attached artifacts.

## Proceed Criteria

Proceed only when all of these are true:

- `dev` is protected by the intended development ruleset.
- Development CI is green on the exact `dev` SHA being considered.
- Release promotion dry-run can name and validate the exact `dev` source SHA.
- The pilot cohort shipped to `dev` without split-brain evidence.
- No public docs or install links point users at `dev`.
- The backout commands below still apply without force-pushing.

## Hold Criteria

Hold if any of these are true:

- branch-role status is missing, stale, or contradictory;
- workflow audit has unclassified development-path refs;
- release-promotion dry-run cannot prove source CI, target ancestry, or the
  promoted source range;
- any pilot worker commits ordinary work to `main` after being moved to `dev`;
- any non-pilot worker commits ordinary work to `dev`;
- forge protections differ from the intended rulesets;
- the proof bundle is unavailable or only verbal.

## Backout Steps

1. Stop pilot dispatches and prompt changes that target `dev`.
2. Put the pilot cohort back on the current configured development branch.
3. Leave `main` untouched as the public front door.
4. Do not delete, rewrite, or force-push `dev`.
5. Record the `dev` SHA and decide whether to promote, merge, or abandon its
   pilot commits with normal source-SHA evidence.
6. Re-run:

   ```bash
   fak release status --json --require-ci-green --limit-commits 50
   ```

7. Post a `backout` decision on #1703 with the command output and next blocker.

## Decision Record Template

```text
Decision: proceed | hold | backout
main start SHA:
dev SHA:
pilot cohort:
pilot commits:
release status artifact:
workflow audit artifact:
release promotion dry-run artifact:
blockers:
next action:
```
