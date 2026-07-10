---
title: "Concept: anti-starvation dispatch aging — 'no ready unit waits"
description: "Proposes folding each ready unit's wait time into its effective dispatch weight, plus a hard starvation deadline, so no ready work waits forever."
---

# Concept: anti-starvation dispatch aging — "no ready unit waits forever"

*2026-07-11*

## The gap

The fleet's issue→worker dispatch order ranks **ready** work by base priority **absolutely**:

- `internal/dispatchtick.OrderLaneCandidates` sorts by priority weight (P0=1000, P1=400,
  P2=150, unlabeled=60) descending, with only a by-number recency **tiebreak**.
- `internal/dispatchorder.Plan` sorts kept units by declared `Priority` descending first,
  then a recency / `PreferOldest` / ID tiebreak.

In both, priority **leads**: a lighter unit never overtakes a heavier one — ties break by age,
but a lower weight never wins on age. So a ready unit that is perpetually out-weighted by a
steady drip of fresher, higher-priority arrivals is **never picked**. That is textbook
priority-scheduling **starvation**, and today nothing in the order prevents it. `dispatch
-conservation` can already *see* a symptom after the fact (its re-storm "churn" count flags
issues that burned worker-units while others got none), but no signal feeds *how long a unit has
been waiting* back into the pick decision.

This concept is that missing feedback term.

## The mechanism

`internal/dispatchaging` folds each ready unit's **wait time** into an **effective weight** and
adds a hard deadline:

```
effective_weight = base_weight + aging_boost
aging_boost      = BoostPerInterval * floor(wait / IntervalSeconds)   (capped at MaxBoostPoints)
standing         = starved  if wait >= StarvationSeconds              (force-served this tick)
                   aging    if aging_boost > 0
                   fresh    otherwise
```

Order: all **starved** units first (worst-starved — longest wait — first), then the rest by
descending effective weight; ties in both bands break by longer wait, then base weight, then ID.

`Standing` is the whole closed vocabulary — `fresh | aging | starved` — the fairness verdict a
human or a peer agent can read without re-deriving the arithmetic.

### Tuned defaults (`DefaultParams`)

Tuned against the priority-weight taxonomy: one unlabeled-tier's worth of weight (**+60**) accrues
every **10 min** waited, so an unlabeled unit climbs past a P2 in ~20 min, past a P1 in ~1 h, and
past a P0 in ~2.6 h — and the **6 h** hard deadline force-serves it regardless of the competition.
Soft boost is uncapped by default (`MaxBoostPoints = 0`).

## Two independent guarantees

1. **Soft aging (monotonic).** Effective weight rises with wait time. Because it is monotonic and
   uncapped by default, a long-waiting light unit *eventually* out-weighs any **fixed** heavier
   tier — so *permanent* starvation is impossible even with the hard deadline off.
2. **Hard starvation deadline.** A unit waiting ≥ `StarvationSeconds` is `starved` and admitted
   ahead of every non-starved unit this tick, whatever its base weight. This bounds the
   **worst-case** wait to a fixed number, independent of the boost arithmetic — the fail-closed
   rung an operator can reason about directly.

## Both lenses

Fairness control **and** throughput optimization are the same mechanism here: draining a unit that
would otherwise rot prevents the later re-storm churn `dispatch-conservation` charges as wasted
capacity, and removes the operator toil of hand-bumping a starving issue's priority label.

## Design decisions (why it looks like the rest of the tree)

- **Data, not code.** The clock is passed in as `Now` (the leaf never reads one) and every knob
  lives in a small declared `Params` struct with documented defaults — candidates for a `dos.toml`
  `[dispatch.aging]` table later, not hardcoded policy.
- **Evidence, not claims.** The standing is a pure function of the wait clock against each unit's
  `ReadySince` stamp; it trusts no worker's self-report.
- **Additive, no regression.** The zero-value `Params` disables both aging rungs, so `Fold` then
  orders by base weight → wait → ID — the *pre-aging* order. Proven by
  `TestAgingDisabledIsPreAgingOrder`.
- **Pure leaf / impure shell.** The fold is stdlib-only (`sort`) and total; gathering the ready
  candidates from the live backlog and acting on the order stays in `cmd/fak` — the same split
  `dispatchorder` (decision) and its wire use.

## Using it

`fak dispatch-aging` is a read-only diagnostic. It reads the candidate set as JSON (a bare array of
`{id, base_weight, ready_since}`, or an object under `candidates`/`order`) from `--in` or stdin, so
it composes with whatever lists the live backlog.

```
$ fak dispatch-aging --now 1000000 < ready.json
dispatch aging -- 5 ready: 1 starved, 1 aging, 3 fresh; oldest wait 7h
!! #0 104-starved  starved base=150 eff=2670 (+2520) waited=7h
   #1 101-fresh-p0 fresh   base=1000 eff=1000 (+0) waited=10s
 + #2 102-aged-default aging base=60 eff=780 (+720) waited=2h
   #3 103-fresh-p1 fresh   base=400 eff=400 (+0) waited=30s
   #4 105-unknown  fresh   base=60 eff=60 (+0) waited=0s
  pick: 104-starved
```

- `--json` emits the machine-readable `fleet-dispatch-aging/1` `Result`.
- `--interval-s`, `--boost`, `--max-boost`, `--starvation-s`, `--now` override the tuned defaults.
- `--fail-on-starved N` exits 1 when the starved count exceeds `N` — a CI / loop gate, mirroring
  `dispatch-conservation --fail-on-leak`.

## What ships in this spine

- `internal/dispatchaging/dispatchaging.go` — the pure fold + closed `Standing` vocabulary.
- `internal/dispatchaging/dispatchaging_test.go` — property tests (soft aging overtakes a fresh
  higher tier; starved beats everything; caps; edges) + the additive no-regression witness.
- `cmd/fak/dispatch_aging.go` — the read-only CLI diagnostic (`fak dispatch-aging`).
- Registration in `cmd/fak/main.go` and the `internal/devindex` verb/tier manifests.

## Follow-on backlog (the fan-out)

The spine is a **decision leaf plus a diagnostic** — it computes and reports the fair order but does
not yet *steer* the live picker. The follow-ons, smallest-first:

1. **Feed the term into the live order.** Have `dispatchtick.OrderLaneCandidates` /
   `dispatchorder.Plan` consume `dispatchaging.Fold` (or its effective weight) so the *actual*
   dispatch pick honors aging — behind a default-off flag first, then default-on once observed.
   This is the "computed & reported but not yet **enforced**" gap, the same shape called out in
   `tier-account-routing.md` and `generation-super-loop-budgets.md`.
2. **`dos.toml [dispatch.aging]` config.** Promote `Params` to declared workspace config (interval,
   boost, max-boost, starvation) with the tuned defaults, so operators tune fairness without a
   rebuild — mirroring how other knobs graduate from constants to `dos.toml`.
3. **`ReadySince` provenance.** Wire the real "became dispatchable" timestamp (issue-eligible time,
   not created-at) from the backlog source into `Candidate.ReadySince`; document the fallback
   (unknown ⇒ waits 0, never starves) and which source is authoritative.
4. **Starvation observability.** Surface `starved_count` / `oldest_wait_seconds` in
   `fak dispatch progress` / a fleet pane, and consider a `dispatch-aging --fail-on-starved 0` CI
   gate in the fleet loop so a starving unit pages before it rots a full window.
5. **Interaction with `attemptbudget` / cooldown.** A unit that is `COOLING_DOWN` should not accrue
   starvation pressure while it is *ineligible*; define whether the wait clock pauses during
   cooldown, and reconcile with `dispatch-conservation`'s churn accounting.

Each item is a candidate issue; none is required for the spine to stand on its own.
