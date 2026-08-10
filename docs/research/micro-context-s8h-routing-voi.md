# Micro-context S8h: executable value-of-information routing

**Status:** controlled executable experiment; not live provider evidence<br>
**Issues:** #6105, parent #6033; live follow-up #6110; semantic corpus #6124<br>
**Artifact:** `experiments/microcontext/s8h-routing-voi-2026-08-10.json`

## Question

Should one globally pre-filter a large input, send every unresolved record to a model, or let each bounded micro-context select and cancel its own filter/model/tool stages?

The important extension is that a micro-context is not merely a smaller prompt. It is an admission and scheduling boundary. Given one record—or one value inside a record—it can:

1. resolve the record with a deterministic filter;
2. ask a bounded selector whether further information can change the answer;
3. dispatch an expensive semantic filter/model, a read-only tool, or a bounded combination;
4. cancel unopened stages as soon as the quality contract is settled;
5. return a typed fact or abstention for a provenance-preserving fold.

## Experiment

The `-routing-voi-output` mode executes four policies over five workload mixtures. Each mixture uses 48 deterministic trials of 500 records (24,000 decisions), with a frozen seed and common utility function:

```text
utility = 100*correct - 100*wrong - 15*abstain - latency_ms - 8*cost_units
```

The record classes deliberately require different evidence:

- `exact`: candidate-visible structure is sufficient;
- `semantic`: model interpretation is strongest;
- `fresh`: current external state makes the read-only tool strongest;
- `boundary`: either model or tool may resolve the case, so routing errors matter.

Policies:

- **always-model** — exact filter, then model for every unresolved record;
- **always-filter-tool** — exact filter, then expensive filter plus tool for every unresolved record;
- **adaptive** — exact filter, selector, then only the eligible model/tool/filter stage;
- **oracle** — hindsight chooses the highest-utility admissible route per record; it is a regret lower bound, not deployable.

Stage latency and cost are deterministic calibrated units. Outcomes vary reproducibly by record and policy, which allows quality/cost/latency/regret accounting without pretending fixture values are provider billing.

## Results

| Mixture | Winner | Always-model utility | Filter+tool utility | Adaptive utility | Oracle utility | Adaptive regret |
|---|---:|---:|---:|---:|---:|---:|
| Exact-heavy (98% exact) | filter+tool | 98.74 | **99.11** | 98.47 | 99.74 | 1.27 |
| Semantic-heavy | adaptive | 71.58 | 44.59 | **79.83** | 90.23 | 10.40 |
| Fresh-tool-heavy | adaptive | 14.55 | 85.08 | **87.15** | 91.97 | 4.82 |
| Mixed | adaptive | 51.91 | 71.26 | **85.49** | 92.55 | 7.07 |
| Boundary-heavy | adaptive | 41.01 | 68.97 | **81.03** | 90.98 | 9.95 |

The exact-heavy crossover matters: adaptive routing is **not universally best**. A selector has a price. When almost everything is solved structurally, a simpler static path wins by avoiding selector overhead. As heterogeneity rises, adaptive selection repays that tax by avoiding systematically wrong or irrelevant expensive stages.

The artifact also records stage-call and cancellation counts. Cancellation is a first-class result: it measures expensive stages that were never opened, rather than inferring savings from a smaller final prompt.

## What this says about filters and tools

### Filters are routable work

“Filtering” is not one mandatory preprocessing pass. Filters can form a staged decision graph:

```text
exact predicate
  ├─ settled → typed fact
  └─ unresolved → selector
       ├─ semantic filter/model
       ├─ current-state tool
       ├─ bounded combination
       └─ abstain/cancel
```

A cheap selector can choose the filter family, but only where workload heterogeneity repays its own latency and token cost. Stable homogeneous workloads should compile to a static route.

### Tool calls are evidence acquisition, not context decoration

A read-only tool should run only when its possible observation can change the answer. The micro-window limits authority and output amplification, while the fold accepts only independently read-back observations. This prevents two common failures:

- calling tools for records already settled by visible data;
- treating a timeout or cancelled call as negative evidence.

Effectful tools remain a separate stage: selection may propose an effect, but approval, idempotency, execution, read-back, and receipt folding stay outside model authority.

## Steelmanned interpretations

- **Global-preprocessing advocate:** one compiled SQL/search pipeline is simpler and cheaper for stable exact-heavy distributions. The crossover confirms this rather than defining adaptive routing as the winner by construction.
- **Long-context advocate:** a capable long-context model can amortize cross-record synthesis and avoid selector mistakes. This experiment does not test tasks whose answer genuinely depends on global interactions.
- **Retrieval advocate:** a shared index can dominate per-record dispatch when semantic neighborhoods are reusable. A production router should admit retrieval as another stage, not force everything into isolated windows.
- **Micro-context advocate:** heterogeneous records need different evidence. Per-window admission bounds failures, enables cancellation, and makes cache keys and provenance local.
- **Security advocate:** selection must not grant authority. The stage catalog, budgets, tool modes, and receipt checks remain deterministic and default-deny.
- **Benchmark skeptic:** calibrated units demonstrate routing mechanics and crossover, not dollar savings or model superiority. That claim boundary is explicit and verifier-enforced.

## Reproduce

```powershell
go run ./cmd/microcontextdemo `
  -routing-voi-output experiments/microcontext/s8h-routing-voi-2026-08-10.json `
  -routing-voi-seed 6105 `
  -routing-voi-trials 48 `
  -routing-voi-records 500

go run ./cmd/microcontextdemo `
  -verify-routing-voi experiments/microcontext/s8h-routing-voi-2026-08-10.json

go test ./cmd/microcontextdemo -run 'TestRoutingVOI|TestVerifyRoutingVOI|TestEveryVerifyFlag' -count=1
```

The verifier requires all five mixtures, all four policies, sane accounting, both an adaptive-winning and a static-winning region, non-negative regret, and an explicit three-part claim boundary.

## Remaining evidence boundary

This result makes #6105 executable and falsifiable, but it does not close #6033. #6124 must provide independently adjudicated semantic residuals and #6110 must replace calibrated stage units with live endpoint tokens, dollars, TTFT/tail, retries, cache/batch provenance, and tool costs. The sanctioned endpoint was reachable during this run; because S8f has zero eligible semantic calls under the frozen #6109 tuner, issuing model calls against it would be benchmark theater rather than valid #6110 evidence.
