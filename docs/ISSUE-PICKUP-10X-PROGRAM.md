---
title: "fak issue-pickup 10x program — from ungated creation to born-pickup-ready"
description: "The trajectory ledger for making fak's default GitHub issue surface 10x easier to pick up and 10x less wasteful: the baseline, the target, and the ordered ladder of changes that shift the issue-hygiene scorecard's checks left into the creation path — with proof."
---

# The issue-pickup 10x program — make every issue born pickup-ready

fak files GitHub issues from a fan of independent producers — `fak issue create`,
the dogfood bridge, `fak complain`, the maturity route, idea-scout, the signal
bots — that all converge on `gh issue create`. Creation is deliberately fast and
structurally **ungated**: the issue contract (`internal/issuecontract`), the
near-duplicate index (`internal/issuededup`), and the class/priority taggers
(`tools/issue_lane_router.py`, `tools/issue_triage.py`) are all *post-hoc audit*
surfaces, not wired into the create path. The result is a backlog where an unknown
fraction of issues are **not pickup-ready**, and every one of those is wasted
pickup attention:

- No `class:*` label → invisible to the dev-leaves / infra / front-door views the
  dispatch loop selects from.
- No `priority/P?` label → ranks at the default weight, so it cannot be ordered
  for pickup.
- A body missing the worker-ready sections → a fresh-context agent picks it up,
  cannot tell what "done" means, and bounces.
- A silent near-duplicate of work already filed → two agents collide, or one does
  work that was already done.

This program is the durable ledger for driving that waste to zero: the baseline,
the target, and what each pass actually moved to make the next pickup *one
obvious, ready choice*, with the evidence that proves it.

> The living per-run snapshot is `fak score issue-hygiene` (against the live
> backlog); this file is the hand-kept **trajectory** — baseline, target, and the
> ordered ladder, with the witness each rung must show. Sibling of
> [`CODE-2X-PROGRAM.md`](CODE-2X-PROGRAM.md), one surface up: it grades the
> *backlog we create*, not the *code we ship*.

## The measure — `fak score issue-hygiene`

`fak score issue-hygiene` (Go: `internal/issuehygiene`, shell:
`cmd/fak/issuehygienescore.go`) reads the live open backlog via `gh issue list`
and folds it into a composite score (0–100) plus the headline metric,
**`issue_hygiene_debt`**: the count of concrete, re-derivable HARD defects on the
**dispatchable** set — the open issues that are not an epic, not blocked on a
human, and not carrying a triage/provenance label (the same exclusions the
ready-leaves pickup view applies). Same backlog snapshot + same reference clock →
same number, because the core is a pure function of `(issues, nowUnix)`; the only
impure part, the `gh` fetch, lives in the cmd shell.

| KPI | HARD/SOFT | a unit of issue-hygiene debt |
|---|---|---|
| `class_coverage` | HARD | a dispatchable issue with no `class:*` label |
| `priority_coverage` | HARD | a dispatchable issue with no `priority/P?` label |
| `contract_completeness` | HARD | a dispatchable issue whose body is missing a worker-ready section (mirrors `issuecontract.missingRequiredIssueSections`) |
| `dedupe_integrity` | HARD | a dispatchable issue that near-duplicates a lower-numbered open issue (via `internal/issuededup`) |
| `leaf_shape` | HARD | an epic with no linked children (nothing to dispatch under it) |
| `kind_area_coverage` | SOFT | a dispatchable issue missing a kind/area label (advisory — anti-gaming) |
| `triage_backlog` | SOFT | an open issue waiting in the triage inbox (advisory) |
| `staleness` | SOFT | a dispatchable, unassigned issue untouched > 60d (advisory) |

**Why three KPIs are SOFT.** They score (a triage-heavy or stale backlog grades
lower) but never gate. The idea-scout inbox is *supposed* to hold un-triaged
items awaiting a human; kind/area labels are nice-to-have, not pickup-blocking;
and staleness is a lagging signal, not a creation defect. Making them HARD would
reward triage-inbox purges and cosmetic labelling over real pickup-readiness — the
scorecard-family anti-gaming rule.

**The single headline.** `pickup_ready_pct` — the share of dispatchable issues
that clear *every* HARD axis (tagged, contract-complete, unique). This is the
number the 10x target lives on.

## The 10x target (honest framing)

On any real backlog the composite score sits high and cannot double — it caps at
100. So the meaningful "10×" lives on the **waste axis**: a *wasted pickup* is an
attempt on an issue that is not pickup-ready (untagged, so never surfaced;
contract-incomplete, so the agent bounces; or a duplicate, so the work collides or
was already done). Drive `issue_hygiene_debt` on newly-created issues toward zero
and `pickup_ready_pct` toward 100, so wasted pickups fall by ≥10×. The lever is
**shift-left**: move each HARD check out of the post-hoc audit and into the
creation path, so an issue cannot be *born* un-pickup-ready.

## The ladder — ordered by leverage

Each rung is an epic child with a concrete mechanism and the witness it must show.
The ordering is deliberate: measure first, then close the creation seams the
measurement exposes, highest-leverage first.

1. **Baseline the live backlog** *(shipped: this program + `fak score
   issue-hygiene`)*. Run `fak score issue-hygiene --json` against the live backlog,
   record `issue_hygiene_debt` and `pickup_ready_pct` in the Trajectory table
   below. **Witness:** a committed baseline row.

2. **Creation-time contract gate.** Wire the scorecard's three per-issue HARD axes
   (`class`, `priority`, `contract`) into `fak issue create` as a pre-create check
   that *repairs or rejects* — reusing `issuecontract.ReviewIssueDraft` and
   `issuecontract.BuildTemplateRepairPlan`, which already exist as audit surfaces.
   An issue cannot be created without a class, a priority, and the worker-ready
   sections. **Witness:** `issue_hygiene_debt` computed over the *newly-created*
   cohort (by `createdAt`) trends to 0.

3. **Auto-class + auto-priority at create.** Run `tools/issue_lane_router.py`'s
   label inference in `--apply` mode *at create time* (not as a nightly backfill),
   so every issue is born with a `class:*` and a defaulted `priority/P?`.
   **Witness:** `class_coverage` and `priority_coverage` hold at 100 on the new
   cohort without a backfill pass.

4. **Dedup-at-create.** Promote `internal/issuededup` from a post-hoc census to a
   create-time gate: check the draft against the open backlog and block (or warn +
   link) on a ≥0.80 near-duplicate before the issue is filed. **Witness:**
   `dedupe_integrity` holds at 100 as issues are created.

5. **Per-issue pickup score → dispatch ranking.** Expose the scorecard's per-issue
   verdict so the dispatch loop ranks pickup-ready issues first and routes the rest
   to repair instead of to an agent. **Witness:** the dispatch tick only hands
   agents issues that clear the HARD axes; bounced pickups fall.

6. **Scheduled auto-triage sweep.** A periodic `fak score issue-hygiene` run opens
   label-backfill / body-repair PRs (or `gh` label writes behind `--live`) for the
   residual, so the backlog self-heals toward pickup-ready. **Witness:**
   `triage_backlog` and `staleness` trend down over successive runs.

7. **Fold into the RSI control pane.** Land the card in the static scorecard
   control pane via a *committed feature snapshot* (so CI stays deterministic and
   offline — no live `gh` in the control-pane path), the way the other live-signal
   cards are folded. **Witness:** the card appears in the roster with a
   reproducible number. *(Deferred deliberately: folding a live-backlog card into
   the static pane before the shift-left rungs land would re-pin a moving
   baseline.)*

## Trajectory

| Date | issue_hygiene_debt | pickup_ready_pct | pass | witness |
|---|---|---|---|---|
| _pending first live run_ | — | — | baseline | `fak score issue-hygiene --json` |

Record each pass by explicit path and witness it with `dos commit-audit HEAD`
(must print `[diff-witnessed]`), so this ledger records pickup-readiness that was
*proven*, not asserted.

## Appendix — fileable epic

The epic body below is ready for `gh issue create` (title + labels + the ladder as
a task list). It is intentionally **not auto-filed** — creating a public issue is
an outward-facing action; file it when you want the tracker to carry it.

```text
Title: epic(dispatch): make every GitHub issue born pickup-ready (10x less wasted pickup)

Labels: epic, class:dev, priority/P1, dispatch

## Current state
Issue creation fans through `gh issue create` ungated. The issue contract, the
near-dup index, and the class/priority taggers are all post-hoc audits, so an
unknown share of the backlog is not pickup-ready (untagged, contract-incomplete,
or duplicated) — each a wasted pickup. `fak score issue-hygiene` now measures this
as `issue_hygiene_debt`; this epic drives it to zero at the source.

## Scope
Shift each HARD issue-hygiene check left, out of the post-hoc audit and into the
creation path, highest-leverage first.

## Done condition
`pickup_ready_pct` on the newly-created cohort holds at ~100 with no backfill
pass; wasted pickups fall ≥10x.

## Children
- [ ] Creation-time contract gate in `fak issue create` (reuse issuecontract.ReviewIssueDraft)
- [ ] Auto-class + auto-priority at create (issue_lane_router --apply at create time)
- [ ] Dedup-at-create gate (promote internal/issuededup to a write-time check)
- [ ] Per-issue pickup score → dispatch ranking
- [ ] Scheduled auto-triage sweep for the residual
- [ ] Fold the card into the static RSI control pane via a committed snapshot

## Witness
`fak score issue-hygiene --json` — issue_hygiene_debt over the newly-created
cohort trends to 0; `dos commit-audit HEAD` is `[diff-witnessed]` on each pass.
```
