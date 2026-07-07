---
title: "trajctl dogfood — the witness report (2026-07-07)"
description: "The trajectory-control spine (epic #2533) exercised end-to-end on one real long-horizon session, with real ledger rows, a live regime-gate decision, and the honest miss list."
---

# trajctl dogfood — the witness report

**Issue:** #2541 (spine step 8 of epic #2533). **File the gaps, do not fix them.**

This is the witness step of the trajectory-control spine: the whole `declare → score
→ curve → (steer)` spin exercised on **one real long-horizon session**, with real
ledger rows and at least one regime-gate decision. It is deliberately honest about
what could *not* be exercised — that miss list is the point of a dogfood.

## The session under test

The real long-horizon session is the **trajectory-control epic build-out itself**
(#2533): steps 1–7 of the spine landed across many turns as a sequence of real commits
on trunk. That is a genuine multi-turn objective with witnessed progress, so it is the
right thing to score. The objective was declared with the seven spine steps as plan
phases, each phase bound to the **real commit** that landed it:

| phase | spine step | landing commit |
|---|---|---|
| phase-1 | objective score ledger spine (#2534) | `41e62c0b97dc` |
| phase-2 | declare/close/list objective CLI (#2535) | `0fdd64fa95b1` |
| phase-3 | scorer registry + W3 witnessed-commit progress (#2536) | `07f7f2f738ab` |
| phase-4 | W2 activity-divergence (stall) scorer (#2537) | `8f23d7f3bc18` |
| phase-5 | curve fold + closed signal vocabulary (#2538) | `4c7e852d2436` |
| phase-6 | score-at-turn-end runner + PreCompact twin (#2539) | `f649954e4d84` |
| phase-7 | re-anchor nudge — regime-gated steering (#2540) | *(none — #2540 is OPEN)* |

## How it was run

The turn ends were replayed through the **production sampler**
`trajctlhook.RunTurnEnd` with the **real git resolver** `GitEvidenceResolver(root)`
(which shells `git cat-file -e <sha>^{commit}` to confirm each SHA resolves before
crediting a phase). At turn *k* the evidence window carried exactly the phase→commit
bindings for phases 1..k — the honest historical replay of the epic as its steps
landed. Every row below was produced by the shipped fold; none is hand-authored.

> **Why a harness and not `fak trajctl`:** the CLI is parked (`var _ = cmdTrajctl`,
> no `case "trajctl"` in `main.go`) and the turn-end sampler has no live caller, so the
> spin had to be driven from a throwaway `go run` harness against the real Go API. That
> is itself the headline dogfood finding (gap G1/G2 below).

## Real ledger rows (`fak-trajctl/1`)

The declared objective row:

```json
{"schema":"fak-trajctl/1","kind":"objective","objective":{"id":"trajctl-epic-2533","statement":"trajectory control: score anything, steer by curve (epic #2533)","plan":[{"id":"phase-1","title":"objective score ledger spine (#2534)"}, … ,{"id":"phase-7","title":"re-anchor nudge — regime-gated steering (#2540)"}],"budget":{"turns":8},"status":"active"}}
```

The W3 witnessed-commit-progress score rows (first, mid, and the plateau — trimmed to
show the shape; every row is real output):

```json
{"schema":"fak-trajctl/1","kind":"score","score":{"objective_id":"trajctl-epic-2533","value":0.14285714285714285,"method":"witnessed-commit-progress","version":"1","witness":"W3","evidence":[{"kind":"commit","ref":"41e62c0b97dc","detail":"phase-1"}],"unix_millis":1751850000000,"session_id":"trajctl-dogfood-2026-07-07","run_id":"dogfood-1"}}
{"schema":"fak-trajctl/1","kind":"score","score":{"objective_id":"trajctl-epic-2533","value":0.8571428571428571,"method":"witnessed-commit-progress","version":"1","witness":"W3","evidence":[{"kind":"commit","ref":"41e62c0b97dc","detail":"phase-1"}, … ,{"kind":"commit","ref":"f649954e4d84","detail":"phase-6"}],"unix_millis":1751868000000,"session_id":"trajctl-dogfood-2026-07-07","run_id":"dogfood-1"}}
{"schema":"fak-trajctl/1","kind":"score","score":{"objective_id":"trajctl-epic-2533","value":0.8571428571428571,"method":"witnessed-commit-progress","version":"1","witness":"W3","evidence":[ … 6 commit refs … ],"unix_millis":1751871600000,"session_id":"trajctl-dogfood-2026-07-07","run_id":"dogfood-1"}}
```

The witnessed-progress curve, turn by turn:

```
turn 1: value=0.1429  evidence=1
turn 2: value=0.2857  evidence=2
turn 3: value=0.4286  evidence=3
turn 4: value=0.5714  evidence=4
turn 5: value=0.7143  evidence=5
turn 6: value=0.8571  evidence=6
turn 7: value=0.8571  evidence=6   ← phase-7 (nudge #2540) has no commit; curve holds at 6/7
```

## The curve fold + the regime-gate decision

```
== curve fold (schema: fak-trajctl-curve/1) ==
objective=trajctl-epic-2533 status=active signal=HEALTHY latest=0.8571 delta=+0.0000
detail: progress 0.86 (delta +0.00)
method=witnessed-commit-progress version=1 points=7
  curve: 0.143 0.286 0.429 0.571 0.714 0.857 0.857

== regime gate decision ==
decision=ABSTAIN reason=healthy-curve (signal=HEALTHY, no divergence)
```

**The regime-gate decision was a verified abstention.** The witnessed curve rose
monotonically 0.14 → 0.86 and then held steady with no competing stall/drift/overrun
evidence, so `classify()` returns `HEALTHY`. The nudge doctrine (#2540, #2533) is that a
HEALTHY curve is **never** nudged — mid-trajectory intervention degrades a high-success
session (arXiv:2602.03338) — so the gate withholds. This is the "at least one
regime-gate decision (nudge or **verified abstention**)" the issue requires, and it is
honest: the abstention is *forced by the evidence*, not chosen for convenience.

## What mis-scored / the blind spots the dogfood surfaced

- **A phase is credited on SHA-existence, not semantic match.** The W3 commit scorer
  credits a phase the moment *any* bound SHA resolves to a real commit object; it never
  checks that the commit actually *implements* that phase. A resolvable-but-wrong SHA
  (or a trailer pointing at an unrelated commit) would witness the phase just the same.
  The witness rung is "the commit exists," not "the commit does the work." Adjacent to
  the calibration (#2566) and anti-gaming (#2568) children.
- **The STALL (W2) path was never exercised.** The activity-divergence scorer only
  fires when session-audit rows are fed into the window; the dogfood fed none, so STALL
  was structurally unreachable and the HEALTHY verdict never had to out-rank it. The
  dogfood proves HEALTHY→abstain but does **not** prove DRIFT/STALL→nudge (the nudge
  actuator does not exist yet — #2540).
- **The curve plateau reads as HEALTHY even though the objective is unfinished.** At 6/7
  the spine is genuinely incomplete (the nudge is unlanded), yet a flat top curve with
  no divergence signal classifies as HEALTHY. "Steady near the top" and "steady but
  stuck below done" are the same signal without a completion/expectation reference. Not
  wrong per the pinned rules — but a real interpretive blind spot for a consumer.

## The honest miss list (what could NOT be exercised end-to-end)

1. **No live producer.** `trajctlhook.RunTurnEnd`/`RunCompaction` are called only by
   tests; nothing in a running session samples the curve at turn ends. → **filed as
   #3129.**
2. **No live source of phase→commit bindings.** Even a wired sampler would score W3=0
   forever, because nothing assembles `WindowInput.PhaseCommits` from commit trailers /
   a verify pass; the dogfood hand-built them from `git log`. → **filed as #3129.**
3. **The CLI is parked.** `fak trajctl` is unreachable from `main.go` dispatch. →
   already tracked by **#2765** (unpark + wire the regime-gated nudge).
4. **The nudge actuator is unshipped.** Only abstention is reachable; a real nudge on a
   degrading curve cannot be witnessed yet. → already tracked by **#2540** / **#2765**.

## Filed gap list (the witness links)

- **#3129** — the live turn-end producer: wire `RunTurnEnd` into a session + assemble
  `PhaseCommits` from commit trailers (new, filed by this dogfood).
- **#2540** — the re-anchor nudge (regime-gated steering) — the actuator this dogfood
  could only abstain in front of.
- **#2765** — unpark the `fak trajctl` CLI + wire the nudge onto the control bus.
- **#2566** — scorer calibration meter (score-vs-witnessed-outcome) — the frame for the
  SHA-existence blind spot.
- **#2568** — the anti-gaming fence (intent-literal rows, steering allowlist).

## Reproduce

Declare `trajctl-epic-2533` with the seven phases above, bind each phase to its listed
SHA, and replay turn ends through `trajctlhook.RunTurnEnd` with
`trajctlhook.GitEvidenceResolver(".")`; fold with `State.CurveReportFor`. The SHAs are
real trunk commits — `git cat-file -e <sha>^{commit}` confirms each. The package folds
are covered by `go test ./internal/trajctl/... ./internal/trajctlhook/...`.
