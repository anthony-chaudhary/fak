---
title: "#2404's no-progress verdict names two tokens that no closed vocabulary knows — and the acceptance spans two different namespaces"
description: "Triage for #2404 (structural no-progress detection as closed session-gate verdicts). Both proposed tokens (NO_PROGRESS, LIVELOCK_REFUSED_CALL) fail dos_check_reason as UNCLASSIFIED drift, and the acceptance asks one token to live in BOTH the ArmMetrics.StoppedBySession stop-reason family AND the DOS refusal vocabulary — which are separate closed sets that have never been unified. Implementation is blocked on an author/operator naming decision, not on engineering. Horizon classified gen/next by inheritance from program #2387."
---

# #2404 — the verdict token gap, and the two-vocabulary split under it

**Filed:** 2026-08-03, from a dispatch worker routed to resolve #2404.
**Status:** triage only. No detector was implemented. #2404 stays **open**.

## Why this note exists instead of a patch

#2404 asks for structural no-progress detection routed through the session gate
as *closed verdicts* `NO_PROGRESS` and `LIVELOCK_REFUSED_CALL`, "drawn from the
DOS-checkable refusal vocabulary", and its acceptance requires that
`dos_check_reason` validate the emitted token as known.

Neither token is in that vocabulary. Both are rejected:

```
dos_check_reason NO_PROGRESS          -> known=false, category=UNCLASSIFIED
dos_check_reason LIVELOCK_REFUSED_CALL -> known=false, category=UNCLASSIFIED
   "Do NOT emit it. Pick a real reason from dos_refuse_reasons, or have the
    workspace declare this one in dos.toml [reasons] first."
```

The live vocabulary is 82 tokens (`dos_refuse_reasons`, workspace `C:\work\fak`).
So the issue's acceptance criterion cannot be satisfied as written by any
implementation: the code it asks for would emit a token the gate it cites
refuses. This is a specification defect, not a missing engineering step.

## The deeper problem: these are two different closed sets

The acceptance asks one token to be simultaneously:

1. **an `ArmMetrics.StoppedBySession` value** — `internal/agent/loop.go:72-77`
   documents that family explicitly: *"a closed token: PAUSED, DRAINING,
   BUDGET_TURNS_EXHAUSTED, ..."*. This is the loop's own stop-reason namespace.
2. **a DOS refusal reason** — the `dos_check_reason` / `dos.toml [reasons]`
   governance vocabulary, where the nearest live tokens are `DOOM_LOOP` and
   `RELAY_NO_PROGRESS`.

These sets have never been unified, and the repo already demonstrates the split:
the livelock condition #2404 wants to route already has an in-repo token,
`LIVELOCK_FUSE` (`internal/gateway/adjudicate_proposed.go:35`), wired through
`annotateToolLivelock` (`adjudicate_proposed.go:130`) — and **that** token is
*also* UNCLASSIFIED to `dos_check_reason`. So the existing sibling of the work
#2404 proposes does not itself meet the bar #2404 sets.

Deciding which namespace owns termination-by-futility — or that the two should
be bridged — is an author/operator call with fleet-wide blast radius. It is not
a call a dispatch worker should make unilaterally, least of all on an issue whose
generation horizon was unclassified at dispatch (triage-only risk envelope).

## What is already built (so the next worker does not re-derive it)

The termination plumbing #2404 describes largely exists; only the *detector* and
the *token* are missing.

- `ArmMetrics.StoppedBySession` (`internal/agent/loop.go:77`) already makes
  "why did this arm stop" a queryable field rather than an inference.
- The turn-boundary stop seam is established and idiomatic. `internal/agent/loop.go:485-494`
  (mid-flight interrupt, #5158) is the exact shape a no-progress check should
  copy: a `(reason, stopped)` probe at a clean boundary that sets
  `m.StoppedBySession = reason` and returns.
- The session gate itself runs at that boundary: `cfg.gateTurn(ctx)` at
  `internal/agent/loop.go:499`, whose non-proceed verdict already ends the arm
  with a recorded reason.
- `Config.DecideSession` / `SessionVerdict` (`internal/gateway/gateway.go:1128-1144`)
  carry `Stop` and `Reason` fields already.
- `guardrsi.LivelockDetector` is real and wired, as the issue claims
  (`internal/gateway/adjudicate_proposed.go:253-276`).

**Two corrections to the issue body,** both minor but turn-wasting:
`repetitionLoopSteer` is *defined* in `internal/gateway/messages_resultnotes.go:291`,
not `messages.go` (it is only *called* from `messages.go:246`, which does confirm
the issue's claim that it is proxy-path-only); and the livelock reason the repo
actually stamps is `LIVELOCK_FUSE`, not `LIVELOCK_REFUSED_CALL`.

## The naming decision, stated as options

- **(a) Reuse `DOOM_LOOP`.** Already declared (`dos.toml:1128`), already
  DOS-checkable, and its summary is a near-exact semantic match: *"verified
  forward progress stays flat for K consecutive windows — a confirmed doom loop,
  distinct from a mere stall."* Costs nothing to adopt; changes #2404's stated
  token.
- **(b) Declare `NO_PROGRESS` in `dos.toml [reasons]`.** The sanctioned path the
  refusal itself names. `dos.toml` is governance config every sibling worker's
  kernel reads, so this is an operator edit.
- **(c) Keep the two namespaces separate** and treat `StoppedBySession` values as
  loop-local (as `LIVELOCK_FUSE` already is), dropping the DOS-checkable clause
  from the acceptance.

Note that `RELAY_NO_PROGRESS` is *not* a candidate: #2404 explicitly scopes it to
the relay-leg sibling (#1893/#1905), out of scope here.

## Generation classification

**Horizon: `gen/next`**, inherited from program **#2387**, which carries
`gen/next`. #2404 is a leaf of #2388 (untagged), which is part of #2387. Under
`docs/generation.md`, `gen/next` is "near-term foundation that should be runnable
by agents soon, but still needs a gate, dogfood run, schema, or default-exposure
proof" — which fits: the plumbing is built, the gate and the token schema are not.

Labels/milestone were **not** applied: GitHub issue writes are operator-gated on
this host. The intended repair is `gen/next` + `generation` + milestone
`Generation G1 - Next Gen`.

- **Promotion evidence** (what would move this toward `now`): a resolved token
  decision from the three options above, after which the detector is a small
  single-seam change against `loop.go:485`'s established pattern.
- **Demotion / retirement evidence:** if the fleet consolidates on `DOOM_LOOP`
  plus the existing `fak doomloop scan` floor (`internal/doomloop`) as the single
  stuck-detection surface, #2404 becomes a duplicate of that path and should be
  retired rather than built — its distinct value is only the *in-loop, gate-owned*
  termination, not the detection itself.
- **Invalidating assumption:** this note assumes `StoppedBySession` tokens are
  intended to stay a loop-local closed set. If the fleet's actual direction is to
  make every terminal verdict DOS-checkable, then option (c) is wrong, the real
  work is much larger than #2404 (it would also have to reclassify `LIVELOCK_FUSE`
  and the existing PAUSED/DRAINING family), and #2404 should be split.

## Suggested next step

Answer the naming question on #2404, then implement against `loop.go:485`'s
pattern. If option (a) is chosen the issue's acceptance text needs one edit
(`NO_PROGRESS` -> `DOOM_LOOP`) and the work is genuinely a small leaf.
