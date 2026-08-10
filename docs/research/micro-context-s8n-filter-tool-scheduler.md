# S8n: filter/tool micro-window scheduler with witnessed sufficiency

Date: 2026-08-10  
Issue: #6167  
Inputs: S8m tool fold (`bce85d6330`)

## Verdict

The controlled matrix supports a narrower general claim: bounded records make **task-shaped stopping,
queue-slot release, and selective stage admission** executable and auditable. They do not make a model
controller universally best. At identical fixture quality, the deterministic planner wins the existence
query; adaptive routing wins top-k and exhaustive work under the declared calibrated stage model; and
selective hedging improves the two expensive adaptive slices while universal hedging spends extra work.
These are controlled scheduler milliseconds, not live provider latency, tokens, dollars, or billing.

## Workload and authority

The frozen workload joins:

- 32 S8m records: 21 majority `read_only`, 11 majority `current_state`, preserving the source fold digest;
- 16 declared exact-filter controls, split 8 tune / 8 test, because S8m contains no `none` labels.

Six arms share 48 records, four workers, a 900 ms per-attempt deadline, and one quality oracle:

1. tuned run-all;
2. tuned fixed cascade;
3. deterministic planner;
4. adaptive three-seam micro-window scheduler;
5. adaptive scheduler with selective read-only hedging;
6. adaptive scheduler with universal hedging.

The stage catalog is closed: exact filter, control window, semantic filter window, repository read, and
live read tool. No effect/write stage exists. A selector changes neither capabilities nor arguments.
Receipts distinguish unopened, completed, timed-out, sufficiency-cancelled, and hedge-loser work.

## Task-specific stop contracts

- **Existence:** stop after a witnessed positive; otherwise inspect all records.
- **Top-k:** stop only when the witnessed kth score is at least every unopened record's structural upper
  bound. Merely collecting k candidates is insufficient.
- **Exhaustive:** never cancel primary work; all records must complete. Hedge losers may be cancelled.

Tests compare every early fold with the all-record oracle. Timeout receipts explicitly say
`deadline_released_slot`; sufficiency cancellations say `witnessed_sufficiency_slot_released`.

## Results

All 18 task/policy cells meet the controlled quality floor (`quality=1.0`). Values are deterministic
calibrated milliseconds over five identical replay trials.

| Task | Policy | Wall ms | Work ms | Opened | Cancelled | Hedges |
|---|---|---:|---:|---:|---:|---:|
| existence | run-all | 481.62 | 1,926.48 | 9 | 3 | 0 |
| existence | fixed cascade | 2.69 | 10.78 | 8 | 3 | 0 |
| existence | **planner** | **1.17** | **4.68** | 4 | 3 | 0 |
| existence | adaptive | 5.23 | 20.90 | 4 | 3 | 0 |
| existence | adaptive selective hedge | 6.34 | 25.38 | 4 | 3 | 0 |
| existence | adaptive universal hedge | 5.40 | 21.62 | 4 | 3 | 0 |
| top-k | run-all | 3,391.81 | 13,567.24 | 48 | 3 | 0 |
| top-k | fixed cascade | 1,426.21 | 5,628.62 | 48 | 1 | 0 |
| top-k | planner | 528.54 | 2,114.15 | 11 | 3 | 0 |
| top-k | adaptive | 323.24 | 1,292.95 | 12 | 3 | 0 |
| top-k | **adaptive selective hedge** | **267.25** | **1,069.02** | 12 | 3 | 0 |
| top-k | adaptive universal hedge | 288.08 | 1,152.33 | 14 | 3 | 0 |
| exhaustive | run-all | 3,528.44 | 13,735.35 | 48 | 0 | 0 |
| exhaustive | fixed cascade | 1,543.98 | 5,746.39 | 48 | 0 | 0 |
| exhaustive | planner | 2,329.48 | 8,884.92 | 48 | 0 | 0 |
| exhaustive | adaptive | 1,359.88 | 4,917.08 | 48 | 0 | 0 |
| exhaustive | **adaptive selective hedge** | **1,218.14** | **4,669.38** | 48 | 0 | 0 |
| exhaustive | adaptive universal hedge | 1,363.17 | 5,095.95 | 50 | 2 | 2 |

Interpretation:

- Existence has a structural answer and ordering signal, so a deterministic planner dominates the
  controller. This is an intended static-policy win, not a failure to hide.
- Top-k repays adaptive stage admission because it avoids irrelevant expensive stages and has a sound
  upper-bound stop. Selective hedging reduces both observed wall and total work in this deterministic
  schedule because faster winners release slots before expensive queued primaries open.
- Exhaustive work cannot use early stopping. Adaptive still avoids semantically inapplicable stages;
  selective hedging improves the modeled tail, while universal hedging opens two duplicates and loses.
- Run-all is a correctness reference but not a tuned economic winner for this heterogeneous catalog.

## Artifact and exact rerun

Artifact: `experiments/microcontext/s8n-filter-tool-scheduler-2026-08-10.json`.

```powershell
go run ./cmd/microcontextdemo `
  -filter-tool-scheduler-fold experiments/microcontext/s8m-semantic-tool-fold-2026-08-10.json `
  -filter-tool-scheduler-output experiments/microcontext/s8n-filter-tool-scheduler-2026-08-10.json `
  -filter-tool-scheduler-trials 5

go run ./cmd/microcontextdemo `
  -verify-filter-tool-scheduler experiments/microcontext/s8n-filter-tool-scheduler-2026-08-10.json
```

The verifier requires all 18 cells, quality 1.0, positive work/wall metrics, receipt digests, no
exhaustive primary cancellation, source/gold provenance, and a complete claim boundary.

## Steelman and falsification boundary

- **Planner:** wins when query structure exposes the right stage and stop rule, as existence does here.
  Adding a model controller would be overhead.
- **Fixed cascade:** remains easier to test and likely dominates under stable stage distributions without
  strong per-record heterogeneity.
- **Run-all:** is operationally simple and can be rational when catalogs are tiny, stages cheap, or audit
  costs dominate compute.
- **Live endpoint:** may reverse every calibrated ranking through network variance, provider batching,
  prefix caching, rate limits, cancellation billing, or model quality. S8n does not answer that question.
- **Gold:** S8m is majority-labeled and low-unanimity, with zero naturally adjudicated `none` records.
  Exact controls are explicit fixtures, not disguised adjudicator output.
- **Hedging:** selective's controlled win depends on the tail predictor and free-slot rule. A poor
  predictor or billed cancellation can erase it; universal hedging already loses here.

Therefore S8n validates the scheduling mechanism and a static/adaptive decision boundary. It does not
close #6033 or #6111; the quality-qualified live comparison remains `not-yet`.
