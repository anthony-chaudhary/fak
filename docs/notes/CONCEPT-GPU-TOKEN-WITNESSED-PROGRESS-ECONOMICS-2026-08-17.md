# From $/GPU to $/token to $/witnessed progress

**Date:** 2026-08-17  
**Status:** CONCEPT / measurement contract, not a shipped savings claim  
**Centrality:** Core — this names the economic output of the agent kernel rather than another mechanism metric.

## Verdict

The economic ladder should end at **cost per independently witnessed unit of useful progress**.
For heterogeneous agent work, the comparison form should be **witnessed human-equivalent minutes per dollar** (WHEM/$, higher is better), with **$/witnessed outcome** retained as the simpler within-cohort view.

That makes fak the **trajectory-yield layer**:

```text
$/GPU-hour  ->  $/token  ->  $/witnessed progress
 infrastructure   model       agent-system outcome
```

`$/GPU-hour` asks how cheaply compute is rented. `$/token` asks how efficiently a model endpoint turns compute into model output. fak asks how efficiently the whole agent trajectory turns paid model work, tools, context, retries, and operator attention into an accepted result that an independent witness can verify.

The short answer to “what is fak here?” is therefore:

> **fak is witnessed-progress goodput: useful, independently verified progress per dollar and per unit time.**

It is not a new token denomination and not merely “cheaper tokens.” fak can improve the final ratio even when it uses more tokens, if those tokens prevent retries, preserve the right context, choose a better model, safely execute tools, or get a larger unit of work across an independent acceptance gate.

## Feynman-simple value frame

- **For:** teams operating long-running coding and tool-using agents.
- **Problem:** GPU and token economics stop before the thing the buyer wanted is delivered.
- **Today:** a cheap token or a high cache-hit ratio can still fund loops, unsafe calls, rejected patches, and operator cleanup.
- **Better because:** fak controls the trajectory seam where context reuse, routing, tool policy, adaptation, and witnessed stopping can turn model activity into accepted progress.
- **Witness:** paired runs report all-in cost, elapsed time, predeclared work size, independent acceptance, and rework/operator burden; the resulting WHEM/$ must beat the next-best system on the same cohort.

## Why the denominator must move

### 1. $/GPU-hour is an input price

This is useful for capacity procurement and self-hosted serving, but says nothing about utilization, batching, model quality, or whether any task completed. A cheap idle accelerator remains waste.

### 2. $/token is a production price

This absorbs part of the serving stack: utilization and throughput become a billable model-work unit. Provider price sheets and tracing systems naturally stop here because token counts are observable and attributable. But tokens are still activity, not value:

- input, cache-read, cache-write, and output tokens have different prices;
- models tokenize differently;
- one system may spend extra tokens to avoid a failed or unsafe action;
- repeated, rejected, or irrelevant output remains billable;
- local tools, verification, waiting, and operator recovery are outside the token count.

Thus “tokens per dollar” can improve while delivered work gets worse.

### 3. $/task is closer, but heterogeneous tasks break it

A one-line edit and a four-hour investigation each count as one task. Pass rate alone also hides the cost of attempts and the size of accepted work. This repository already chose a size-aware denominator in [the agentic trajectory denominator note](AGENTIC-TRAJECTORY-DENOMINATOR-2026-08-06.md): predeclared human-equivalent minutes, with completion gated by an independent witness.

### 4. $/witnessed progress is the economic output

Count progress only when the requested acceptance criterion passes independently. Size it using the fixed, predeclared human baseline; do not let the producing agent award itself points. Failed attempts keep their costs and contribute zero delivered progress.

This is the same move FinOps unit economics makes from technical units such as VM, GB, or token toward business units such as transaction, customer, or case resolved. For agent work, the business unit is not universally “a token” or even “a task”; it is a witnessed unit of the operator's declared objective.

## Measurement contract

For a comparable cohort, define:

- `C_run`: all attributable run cost: provider bill or GPU occupancy, non-model service cost, and priced operator intervention. Report components separately as well as the total.
- `H_i`: the human-equivalent minutes assigned to work item `i` **before** treatment is revealed.
- `A_i`: independent acceptance, `1` only when the predeclared witness passes, otherwise `0`.
- `R_i`: independently estimated human-equivalent rework or cleanup minutes caused by the run.
- `W`: net witnessed human-equivalent minutes delivered.

```text
W = max(0, sum(H_i * A_i) - sum(R_i))

primary efficiency: WHEM/$ = W / C_run             (higher is better)
reciprocal cost:     $/WHEM = C_run / W             (lower is better)
cohort shorthand:    $/witnessed outcome = C_run / sum(A_i)
time goodput:        WHEM/hour = W / elapsed_hours   (higher is better)
```

If `W = 0`, `$/WHEM` is undefined/infinite, not zero. Report cost and zero delivered progress. Never average away a failed run by dropping it.

When human-minute sizing is unavailable, use a predeclared local unit such as independently resolved issues, accepted support cases, or policy-compliant completed transactions. Label it local and do not compare it across unlike cohorts. This follows the net-true standard: a narrower observed quantity is better than a broad invented conversion.

### Factorization: where fak acts

For self-hosted inference, the reciprocal view makes the ladder explicit:

```text
WHEM/$ = (GPU-hours/$)
       * (billable-token-equivalents/GPU-hour)
       * (WHEM/billable-token-equivalent)
```

For an API endpoint, the first two factors collapse into provider `token-equivalents/$`.

The last factor is **trajectory yield**. It includes task success, useful context selection, retry/rework avoidance, safe tool completion, and independent acceptance. That is fak's primary layer. fak can also improve the middle factor economically through prefix reuse, cache preservation, compaction, and model routing, but those are mechanisms whose value must survive the final trajectory-yield gate.

Use provider-price-weighted **billable token equivalents**, not raw token totals, when input/output/cache classes differ. Keep the underlying token classes visible so a favorable conversion rate cannot hide a workload-shape change.

## Worked example

Suppose two paired systems attempt three work items, each sized at 30 human minutes before the run.

| System | All-in cost | Accepted items | Rework | Net WHEM | WHEM/$ | $/WHEM |
|---|---:|---:|---:|---:|---:|---:|
| baseline | $5.00 | 2 | 10 min | 50 | 10.0 | $0.100 |
| fak treatment | $6.00 | 3 | 5 min | 85 | 14.17 | $0.071 |

The treatment spends 20% more dollars and may use more tokens, yet delivers about 42% more witnessed progress per dollar. A token-only dashboard could call it a regression; the outcome metric correctly calls it a gain. Conversely, if a cache optimization lowers spend to $4.00 but only one item passes and cleanup is 10 minutes, it delivers 20 WHEM, or 5 WHEM/$: cheaper activity, worse economics.

This is why fak must not optimize `$/token` as the terminal objective.

## Guardrails against gaming

1. **Predeclare the objective, cohort, work-size estimate, acceptance gate, and cost boundary.** Post-hoc task sizing lets the treatment grade itself.
2. **Use an independent witness.** A model's “done” message, commit subject, cache hit, or emitted artifact is not acceptance.
3. **Keep quality and safety as gates, not purchasable offsets.** A cheap unsafe outcome does not become valuable by multiplying it by a soft score.
4. **Charge failures, retries, verification, and cleanup.** Failed work contributes zero progress but retains its full cost and elapsed time.
5. **Report dollars and time side by side.** Reserved GPU economics can make marginal dollars look free while wall-clock capacity remains scarce.
6. **Separate observed, estimated, and derived fields.** Provider invoices are observed; GPU-hour allocations and human-minute baselines may be estimated; WHEM/$ is derived.
7. **Compare against the real next-best alternative.** The baseline is the same agent stack without the fak treatment, not “do nothing,” unless doing nothing is genuinely the operator's alternative.
8. **Publish the distribution.** Median WHEM/$ alone can hide catastrophic tails; include acceptance rate, zero-progress runs, p50/p95 elapsed time, and intervention burden.

## What existing fak metrics mean in this ladder

| Existing metric | Layer | What it can prove | What it cannot prove alone |
|---|---|---|---|
| GPU occupancy, throughput, TTFT/ITL | infrastructure/serving | capacity and latency behavior | accepted agent work |
| provider bill, token-equivalent spend | model economics | attributable model cost | useful outcome |
| cache-hit or reuse rate | mechanism | work was reused | net savings or completion |
| avoided prompt work / cost per long turn | mechanism economics | less repeated model work under a declared price model | end-to-end task value |
| pass rate / resolved issues | outcome | independently accepted count in one cohort | cross-cohort work size or all-in cost |
| human-equivalent minutes completed | sized outcome | comparable delivered work under the declared baseline | economic efficiency without cost |
| **WHEM/$ and WHEM/hour** | **agent-system goodput** | **net-true economic and temporal delivery** | broad business value outside the declared objective |

The project should keep the lower-level rows because they diagnose *why* the top-line moved. It should not promote a mechanism gain to a product-economics claim until the paired outcome row also moves.

## Problem checklist

- **P1 managed context:** count context savings only if the same or more witnessed progress survives; otherwise compaction simply made a cheaper failure.
- **P2 net-true efficiency:** include all run costs, retries, verification, and cleanup; compare paired WHEM/$ and WHEM/hour.
- **P3 bounded adaptation:** require the acceptance and safety gates to remain fixed across treatment and baseline, so adaptation cannot redefine success.
- **P4 integrated operations:** derive the row from trajectory, billing/occupancy, witness, and operator-intervention evidence that can be joined and audited.

## Research anchors (observed 2026-08-17)

These sources establish the adjacent layers; none by itself establishes a fak gain.

- [MLCommons, MLPerf Inference: Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/) frames serving around throughput and latency scenarios. This is the GPU-to-serving layer.
- [LangSmith cost tracking](https://docs.langchain.com/langsmith/cost-tracking) automatically derives LLM costs from token counts or accepts directly supplied costs. This is the token-to-dollar observability layer.
- [SWE-bench](https://www.swebench.com/) reports percent of real GitHub issue instances resolved and exposes resolved-versus-cost views. This moves from activity to independently tested task outcomes, but instances remain heterogeneous.
- [METR, “Measuring AI Ability to Complete Long Software Tasks”](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) characterizes capability by the human time required for tasks completed at a fixed success probability. This supports size-aware outcome units rather than flat task counts.
- [FinOps Foundation, Unit Economics](https://www.finops.org/framework/capabilities/unit-economics/) distinguishes technical units such as cost per token from outcome units such as cost per transaction or case resolved, and recommends outcome proxies when direct revenue attribution is unavailable.
- Internal constraints: [net-true value](../standards/net-true-value.md), [agentic trajectory denominator](AGENTIC-TRAJECTORY-DENOMINATOR-2026-08-06.md), and [end-to-end value chain](../end-to-end-value-chain.md).

## Decision

Adopt this vocabulary in research and benchmark design:

- **North-star economic row:** WHEM/$ (and reciprocal $/WHEM).
- **North-star temporal row:** WHEM/hour.
- **Within-cohort operator shorthand:** $/witnessed outcome.
- **Name for fak's layer:** trajectory yield or witnessed-progress goodput.
- **Diagnostic rows:** $/GPU-hour, token-equivalents/$, cache/reuse, acceptance rate, rework, operator intervention, and latency.

Do **not** claim that fak currently improves WHEM/$ from this note. The concept becomes a shipped claim only after a paired baseline/treatment run joins observed cost, predeclared work size, independent acceptance, elapsed time, and rework under the same cohort.
