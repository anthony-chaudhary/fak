---
title: "Worktree land merge-reconciliation contract — a rejected apply is a structured refusal, never a silent drop"
description: "land_worktree_diff (tools/worker_worktree.py, #1334/#3165) lands a worker's detached-worktree delta on the trunk via a plain `git apply` + pathspec commit. When the trunk moved inside the worker's diff region since its pinned base, the all-or-nothing apply rejects — and the old contract failed open with prose `ok:false`, silently dropping the worker's entire verified delta. This note compares three reconcile algorithms, picks serialized-apply-under-lane-lease-with-auto-rebase-on-reject, names the landing session as the deterministic conflict owner with an explicit STOP-and-replan path (COLLISION_RISK from the closed refusal vocabulary), and places the contract against the COMMIT_NOT_SELF_CONTAINED / TRUNK_WOULD_NOT_COMPILE compile-integrity rungs."
---

# Worktree land merge-reconciliation contract

Date: 2026-07-10. Fixes [#3207](https://github.com/anthony-chaudhary/fak/issues/3207).
Builds on the detached-worktree isolation primitive (#1334, epic #3165:
[`tools/worker_worktree.py`](../../tools/worker_worktree.py)) and should be read against
[`SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25`](SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md)
(the shared-trunk bet) and
[`SHARED-TRUNK-COMPILE-INTEGRITY-GAP-2026-07-06`](SHARED-TRUNK-COMPILE-INTEGRITY-GAP-2026-07-06.md)
(the compile-integrity rungs).

## 1. The hole

`land_worktree_diff` captures `git diff <base_sha>` inside the worker's detached
worktree and lands it on the trunk with `git apply --whitespace=nowarn` followed by
`git commit -s -- <paths>`, serialized by the lane lease the dispatcher holds. A plain
`git apply` is **all-or-nothing**: if the trunk moved since `base_sha` in any region the
patch touches — even context lines around an untouched hunk — the whole patch rejects and
nothing is applied. The old contract turned that reject into fail-open prose
(`{"ok": false, "reason": "git apply to trunk failed: …"}`), indistinguishable from any
other git hiccup. The caller shrugs, the worktree is reaped, and the worker's **entire
delta — which passed the in-worktree compile witness — silently evaporates.** That is the
same evaporation class the `base_sha` diff-base fix closed for in-worktree commits, one
seam later. A silent drop is the one outcome the repo's witness-not-self-report floor can
never audit: there is no artifact left to disbelieve.

## 2. Three candidate reconcile algorithms

**(a) Rebase-detached-worktree-onto-current-trunk-HEAD before every land.** Re-pin the
detached worktree to current trunk HEAD (replay the delta), re-capture the diff, then
apply — clean by construction, modulo the rebase-to-apply race (closed by the lease).
Cost: pays a replay on *every* land, though disjoint-lease admission makes a
moved-trunk-inside-the-region reject the rare case; and it still needs a conflict policy
for when the replay itself conflicts, so it does not remove the hard case — it only
prepays for it.

**(b) `git apply --3way` against the trunk.** One flag; auto-resolves pure context drift
via blob ancestry. Disqualified on the shared-trunk axis alone: on a genuine overlap it
leaves **conflict markers and a conflicted index in the SHARED trunk working tree** —
exactly the cross-worker WIP entanglement #1334 built the worktree to prevent. A
half-conflicted shared tree during a land wedges every peer whose lease touches those
files; the reconcile MUST materialize conflicts only in the worker's isolated tree.

**(c) Serialized-apply-under-lane-lease with auto-rebase-on-reject — CHOSEN.** Keep
today's plain apply as the fast path (the lease taxonomy makes rejects rare). On reject:
reconcile **in the isolated worktree** — re-pin it onto current trunk HEAD, replay the
delta there, re-run the compile witness there, re-capture the diff, retry the apply once
(the lane lease is held across the whole rebase+retry, so the window cannot reopen). If
the replay itself conflicts, that is a genuine overlapping-region edit — a *materialized*
lease collision — and the land STOPs with a structured refusal (§4).

Rationale: (c) strictly dominates (a) — it *is* (a), gated to the rare reject instead of
taxing every land — and beats (b) because conflicts only ever materialize inside the
worker's own tree, never as markers on the shared trunk. It also preserves the existing
atomicity invariant: on any reject, `git apply` has applied nothing and the trunk is clean.

## 3. The deterministic conflict owner

**The landing session owns the conflict.** On a reject it resolves in place: the
auto-rebase in its own worktree (mechanical, no judgment). On a genuine conflict it
**STOPs and replans** — emits the structured refusal, keeps the worktree intact (the
preserved work *is* the evidence; never reap on refusal), and routes to replan.

- **Never a human merge queue.** That is the field's per-agent-branch merge backlog the
  shared-trunk bet explicitly rejects
  ([`…ISOLATION…`](SHARED-TRUNK-VS-PER-AGENT-ISOLATION-2026-06-25.md) §3): re-introducing
  a person-owned serialization point per land would concede the whole trade.
- **Never a silent drop.** The refusal is loud, typed, and carries the git evidence.

## 4. The explicit STOP path

The contract change shipped with this note (`land_worktree_diff`, apply-reject arm):

```json
{"ok": false, "applied": false, "committed": false,
 "reason": "COLLISION_RISK",
 "detail": "git apply to trunk rejected (diff base <sha>): <git evidence>",
 "next_action": "replan: re-pin the worktree onto current trunk HEAD, re-verify, re-land; STOP on a genuine conflict"}
```

`COLLISION_RISK` is verified against the closed refusal vocabulary
(`dos man wedge COLLISION_RISK --explain` → `VALID reason`, `REFUSAL? yes`, category
OPERATOR_GATE, "route to replan"). The semantics fit: at land time the *risk* the
arbiter admits against has **materialized** — the trunk moved inside the worker's leased
region despite disjoint-lease admission, which means either a lease-taxonomy hole or an
unleased edit; both are replan-shaped, so the token routes the caller's loop correctly. A
finer-grained dedicated token (e.g. `LAND_PATCH_CONFLICT`) could later be declared in the
workspace's `dos.toml [reasons]` table; that is deliberately out of scope here (dos.toml
is owned elsewhere), and `COLLISION_RISK` is already emittable, verifiable, and refusable
today — the closed-vocabulary bar a new token would have to re-earn.

STOP invariants, all mechanically true on the reject path: nothing applied (apply
atomicity), nothing committed, trunk clean, worktree intact, refusal routed — the caller
replans instead of the work vanishing.

## 5. Relationship to the compile-integrity rungs

[`SHARED-TRUNK-COMPILE-INTEGRITY-GAP-2026-07-06`](SHARED-TRUNK-COMPILE-INTEGRITY-GAP-2026-07-06.md)
defines two fail-closed rungs: **`COMMIT_NOT_SELF_CONTAINED`** (commit seam, static
self-containment) and **`TRUNK_WOULD_NOT_COMPILE`** (push seam, affected-scoped isolated
build). The land seam sits between the worker's in-worktree verify witness and those
rungs, and a moved trunk adds the hazard textual reconciliation cannot see:

> A clean apply — or a clean auto-rebase — proves **textual** disjointness, not
> **semantic** compatibility. A peer's landed rename of a symbol the worker calls applies
> cleanly and breaks the build: the `runSuperloopDrive` shape, relocated to the land seam.

So the reconcile contract composes with the rungs rather than replacing them:

1. After any rebase-on-reject, the compile witness **re-runs in the re-pinned worktree**
   (the existing `verify=` / `_go_build_verify` seam) — rung-B semantics (isolated build
   against the *new* base) applied at land time, before anything touches the trunk.
2. The clean-apply-onto-a-moved-trunk case that never triggers the rebase is exactly what
   the push-seam `TRUNK_WOULD_NOT_COMPILE` rung backstops.
3. All three reasons — `COMMIT_NOT_SELF_CONTAINED`, `TRUNK_WOULD_NOT_COMPILE`,
   `COLLISION_RISK` — are the same move: a seam where the old behavior was silence (or
   prose) becomes a fail-closed, structured, replan-routable refusal from a closed
   vocabulary.

## 6. Non-goals

- No auto-resolution of genuine overlapping edits (no union merge, no ours/theirs) — a
  real conflict always STOPs to replan.
- This ticket ships the **contract** (the structured refusal on reject, landed in
  `land_worktree_diff`) and this design; the auto-rebase-on-reject arm itself is the
  follow-on wiring on top of the now-loud reject signal.
- Not a merge queue, not per-agent branches, not a human gate — the shared trunk stays.
