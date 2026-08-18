# Top next GitHub work — 2026-08-18

**Verdict:** protect the release path first, then build the missing cost join and one managed-GPU deployment spine in parallel; use those artifacts to run the comparative value-chain proof. The new Graft-derived retrieval issue is valuable, but it follows the release/value-chain critical path.

This is a point-in-time execution plan, not a replacement for GitHub. Issue state, acceptance criteria, and milestones remain authoritative in GitHub; this note records why these issues are the next coherent sequence.

## Value frame

- **For:** maintainers choosing what the next agents should ship from a 500-issue open backlog.
- **Problem:** label-only ranking mixes old speculative P0s, release blockers, recent research, and prerequisites for net-true product evidence. It does not expose the dependency chain that converts work into an operator outcome.
- **Today:** `fak console issues` reports 500 open issues, including 19 P0, 127 P1, 146 orphan P0/P1, and 259 without priority. The labels are useful triage inputs but not a sufficient execution order.
- **Better because:** the sequence below binds recent studies to open issue contracts and orders them by prerequisite leverage, user-visible outcome, and strength of the eventual witness.
- **Witness:** each row names the source evidence, the GitHub issue, and the artifact that retires it; the wave boundaries prevent downstream benchmarks from running before their billing and deployment joins exist.

## Problem frame

- **Centrality:** Stewardship. This plan decides which Core and Enabling work gets scarce trunk and fleet capacity next.
- **P1 managed context:** recent research, live issue state, release constraints, and value-chain dependencies are joined in one bounded handoff.
- **P2 net-true efficiency:** cost work precedes cost claims; comparative proof cannot silently substitute token estimates for billing evidence.
- **P3 bounded adaptation:** only seven issues enter the execution set, with explicit serial and parallel boundaries.
- **P4 integrated operations:** the sequence spans safe release, provider cost, managed deployment, live outcome proof, retrieval, and maturity rather than optimizing one isolated package.

## Ranked execution set

| Rank | Issue | Why now | Completion witness |
|---:|---|---|---|
| 1 | [#7325 — bind every release cutter to one auditable lease](https://github.com/anthony-chaudhary/fak/issues/7325) | A concurrent cutter can publish from the wrong HEAD. Every later result is harder to trust if release ownership is ambiguous. This is a hot-tree change and runs alone. | The issue's two-cutter test proves exactly one winner, loser-side no-op, and one lease-linked release receipt. |
| 2 | [#7324 — prevent fast-gate supersession starvation](https://github.com/anthony-chaudhary/fak/issues/7324) | Superseded release checks can starve the last valid fast-gate result. It follows #7325 so release scheduling is repaired behind a trustworthy cutter boundary. | A concurrency witness proves a newer run owns publication and older cancellations cannot starve it. |
| 3 | [#6651 — authoritative session-keyed provider cost ledger](https://github.com/anthony-chaudhary/fak/issues/6651) | `fak value-chain audit` deliberately leaves cost absent without authoritative billing coverage. This is the missing denominator join, not optional observability polish. | Two-provider/two-root tests prove attribution, retries, absent cost, and ambiguous joins never cross-fold; reconciliation uses provider billing export provenance. |
| 4 | [#7346 — one auditable Lambda managed-GPU deployment spine](https://github.com/anthony-chaudhary/fak/issues/7346) | The Lambda strategic-fit study found a concrete external gap: fak has serving and guard planes but no single managed-GPU provision-to-teardown path. The issue was filed from that evidence rather than leaving the recommendation in prose. | A fixture-backed lifecycle selfcheck plus one scrubbed live Lambda provision → endpoint → fak request → teardown receipt. |
| 5 | [#7192 — prove cost per safe outcome on the latest harness](https://github.com/anthony-chaudhary/fak/issues/7192) | This converts #6651 and #7346 into the observable user claim. Running it earlier would either lack authoritative cost or benchmark a hand-built deployment that is not the product path. | A paired raw-vs-fak value-chain packet reports task success, unsafe side effects, turns, tokens, authoritative cost coverage, and explicit unknowns. |
| 6 | [#7338 — bounded Go structural retrieval through CLI and guarded MCP](https://github.com/anthony-chaudhary/fak/issues/7338) | The Graft study identified a cheap, high-leverage borrow from infrastructure that already exists in `internal/codegraph`. It improves agent context acquisition without importing Graft's Python server or database stack. | Deterministic symbol/caller/callee/skeleton queries, stale-index behavior, guarded MCP parity, and a measured comparison against grep/full-file reads. |
| 7 | [#7313 — admit historical-session exploration to the maturity ladder](https://github.com/anthony-chaudhary/fak/issues/7313) | Recent session-history work is already shipped and dogfooded, but its maturity/default-on gap is invisible. This follows the critical path and should absorb the current maturity/session WIP rather than create another competing issue. | `fak maturity --json` reports source-bound `sessionmine` evidence and promotes only from a valid runtime proof. |

## Wave plan

### Wave 0 — serialize release trust

Run **#7325, then #7324**, with no parallel release/workflow writer. Both own hot release surfaces and their value is prerequisite trust, not feature breadth.

### Wave 1 — parallel enabling spines

After Wave 0 is green, run **#6651**, **#7346**, and **#7338** in parallel only if DOS pricing confirms disjoint file trees and lane leases. The expected seams are fleet/value-chain, deployment/integration, and codegraph/MCP respectively; arbitration, not this note, decides actual concurrency.

### Wave 2 — close the value loop

Run **#7192** only after #6651 supplies an authoritative cost join and #7346 supplies a repeatable managed deployment receipt. If Lambda capacity is temporarily unavailable, preserve the fixture witness and dispatch the live run through the sanctioned compute path; do not replace the live witness with an estimate.

### Wave 3 — promote what is already working

Fold the existing maturity/session edits into **#7313**, validate only their explicit paths, and land as one issue/one commit. Do not open a second maturity ticket for the same capability.

## Why other apparent top issues are not in this batch

- The deterministic issue projection currently contains 19 P0 and 127 P1 rows, but 146 of those high-priority rows are orphaned and 259 open issues lack priority. A priority label without a recent dependency or outcome witness is not enough to preempt this chain.
- Broad platform epics such as immutable cross-machine deployment remain larger than the one-provider spine. #7346 deliberately proves one integrated path before another platform matrix or scheduler.
- Provider cache-hint work (#7308–#7311) is valuable but does not unlock the immediate cost-per-safe-outcome claim as directly as #6651 → #7192.
- Graft follow-ons beyond #7338 wait for the minimal retrieval spine and measured value. The study explicitly rejects importing its Python/FastAPI/SQLite architecture wholesale.

## Evidence used

- `fak console issues --state open --json`, captured 2026-08-18: 500 open; 19 P0; 127 P1; 146 orphan P0/P1; 259 missing priority.
- [`docs/end-to-end-value-chain.md`](../end-to-end-value-chain.md) and [`docs/value-chain-audit.md`](../value-chain-audit.md): organization-defined outcomes, paired comparisons, shared setup charging, and absent-cost honesty.
- [`docs/notes/LAMBDA-FAK-STRATEGIC-FIT-2026-08-18.md`](LAMBDA-FAK-STRATEGIC-FIT-2026-08-18.md): one provider-neutral managed-GPU adapter as the first external spine; no premature multi-cloud scheduler.
- [`docs/research/graft-study-2026-08-18.md`](../research/graft-study-2026-08-18.md): borrow bounded structural retrieval through existing Go codegraph seams; reject the external Python/database stack.
- [`docs/notes/WIP-CHECKPOINT-LIFECYCLE-2026-08-18.md`](WIP-CHECKPOINT-LIFECYCLE-2026-08-18.md): current session durability semantics and the boundary between checkpoint protection and normal landing.
- Live GitHub issue bodies and labels read on 2026-08-18. #6651 and #7192 were assigned `priority/P1` and the cache-value milestone; #7346 was filed with `priority/P1` in Generation G1.

## Replan triggers

Re-run this plan after any of these events:

1. #7324 or #7325 closes or produces a new release-path blocker.
2. #6651 changes its ledger schema or #7192 changes its required cost join.
3. #7346's live provider witness disproves Lambda API/capacity assumptions.
4. DOS pricing finds file-tree collisions between the proposed Wave 1 issues.
5. A newer study supplies stronger external evidence that changes the value-chain critical path.

Nothing else discovered in this pass needs a new issue: the Lambda gap is #7346, the Graft gap is #7338, and the remaining recommendations map to existing issues above.