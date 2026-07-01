---
title: "Branch regime ADR: dev integration and main public front door"
description: "Decision record and migration runbook for separating fak's hot development branch from the public release branch without split-brain agent behavior."
---

# Branch regime ADR - dev integration and main public front door

**Status:** proposed contract, not cut over.
**Issue:** #1694.
**Current law until cutover:** agents still commit to `main` under the existing
`OFF_TRUNK` guard. Do not update agent prompts, hooks, or CI to target `dev`
until the prerequisites below have landed.

## Decision

fak will move from one branch doing every job to two long-lived branch roles:

| Role term | Branch at cutover | Meaning |
|---|---|---|
| Development branch | `dev` | The hot shared integration branch where agents commit ordinary work by explicit path. |
| Release branch | `main` | The clean public branch used for releases, tags, install docs, and stable links. |
| Release source | `dev` | The git source a promotion reads from when preparing a release to `main`. |
| Public front door | `main` | The branch users see through GitHub default browsing, `go install ...@latest`, release notes, Pages links, and README install paths. |

The existing `OFF_TRUNK` token may remain, but after cutover it must mean "not
on the configured development branch" rather than "not on `main`." This keeps
the discipline and changes only the target.

## Invariants

- There is still one hot development trunk. `dev` is not permission to create
  per-feature branches or side worktrees.
- Everyday commits remain path-scoped, signed off, stamped, leak-checked, and
  guarded by stale-base checks.
- `main` remains active. It is not a dead branch; it is the public/release front
  door.
- Promotion from `dev` to `main` is an audited release operation with a source
  SHA witness, not a normal agent commit.
- Public docs and install commands must not point users at an unstable branch.
- Release tags must bind to `main` after the promotion witness, not to arbitrary
  local state.
- The branch-role vocabulary must have one authoritative config source before
  code, hooks, workflows, or prompts consume it.

## Non-goals

- Do not flip the GitHub default branch in this issue.
- Do not weaken `OFF_TRUNK`, path-scoped commit, DCO, claim tags, leak checks,
  or stale-base guards.
- Do not teach agents to commit to `dev` until `fak commit`, hooks, CI, release
  tooling, and prompts agree on the same branch roles.
- Do not make `main` a second shared development trunk.
- Do not treat `master` as a live role. It is only an external alias or legacy
  residual to audit.

## No-split-brain rule

At every point in the migration, there must be exactly one accepted everyday
agent commit branch.

Before cutover, that branch is `main`. After cutover, that branch is `dev`.
During the transition, any document, prompt, hook, or workflow that would make
some agents target `main` while others target `dev` is a blocker. If the system
cannot prove a single accepted development branch, keep the old `main` regime.

## Migration order

1. **#1694 - branch-role ADR.** Land this contract first. It is documentation
   only and does not change behavior.
2. **#1695 - role-aware commit and OFF_TRUNK guard.** Teach `fak commit`, hook
   text, and refusal details to read the configured development branch while
   preserving the guard.
3. **#1696 - forge rulesets.** Install branch protections/rulesets for hot `dev`
   and clean `main`; do not rely on local hooks alone.
4. **#1697 - CI branch targeting.** Retarget workflows so `dev` gets integration
   checks and `main` gets release/front-door checks.
5. **#1698 - release promotion.** Promote from `dev` to `main` with source-SHA
   witnesses and release evidence.
6. **#1699 - dispatch and agent prompts.** Only after the guard and CI agree,
   teach agents and dispatch packets that everyday work ships on `dev`.
7. **#1700 - public docs/install links.** Keep public docs, install commands,
   Pages, and release notes anchored on `main`.
8. **#1701 - hard-coded ref audit.** Replace remaining hard-coded `main`/`master`
   refs with role config where they are not public-front-door references. Use
   [`docs/branch-regime-hardcoded-ref-audit.md`](branch-regime-hardcoded-ref-audit.md)
   as the current classification map.
9. **#1702 - status visibility.** Surface `dev`/`main` drift, promotion blockers,
   and release-source freshness in status views.
10. **#1703 - reversible shadow cutover.** Run a shadow mode that reports what
    would happen under `dev` before switching agents. Cut over only after the
    report is clean. Use
    [`docs/branch-regime-shadow-cutover.md`](branch-regime-shadow-cutover.md)
    as the proof checklist for that run.

Global agent prompt changes are last-mile work: they belong after #1695, #1696,
#1697, and #1703 prove the target branch is enforceable and observable.

## Backout plan

The backout anchor is the current `main`-only regime.

If a cutover step causes inconsistent branch targets, failed release promotion,
or ambiguous `OFF_TRUNK` recovery text:

1. Stop prompt changes and dispatch launches that target `dev`.
2. Set the authoritative development branch back to `main`.
3. Keep public `main` docs/install links unchanged.
4. Re-run the status/drift report from #1702 to prove there is no unpromoted
   `dev` work required for a release.
5. Re-enable the next cutover attempt only after the failed rung has a witness.

No backout path may force-push `main` or erase `dev`. Reconciliation is a normal
merge/promotion decision with source-SHA evidence.

## Acceptance checklist

- Role terms in this document are the names code/config should use:
  `development branch`, `release branch`, `release source`, and
  `public front door`.
- Behavior remains unchanged until later tickets land.
- The migration order names every branch-regime follow-on ticket.
- The no-split-brain and backout rules define when agents may switch.
