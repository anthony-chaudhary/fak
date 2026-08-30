# Cross-layer sweeps: seven response surfaces, one evidence contract

> **Study ID:** `study_5eef7ef15ae699456b2b0f7245691745f39e28ecf1f76af577901944a972afd4`  
> **Observed:** 2026-08-29  
> **External source:** [`mlcommons/inference@cfb0df14b21a3898521891f021a1c6aadec2ab2c`](https://github.com/mlcommons/inference/tree/cfb0df14b21a3898521891f021a1c6aadec2ab2c), Apache-2.0. The pin was authored and committed on 2026-08-26; repository metadata observed a 2026-08-29T21:11:11Z push.  
> **FAK spine:** [#10189](https://github.com/anthony-chaudhary/fak/issues/10189) is open. This note maps evidence; it does not claim that the shared contract has shipped.

## Verdict

FAK already has response surfaces from host memory through whole-agent fleets. Their experiment runners and mathematics should remain local to each layer. The missing common seam is smaller. It covers comparable identity and point provenance. It also types missing observations, censored boundaries, and finding lineage.

That seam is the purpose of #10189. It must not become a universal scheduler or smoothing library. Each layer retains its own definition of "knee." Host-memory plateaus, serving SLA knees, cache budget thresholds, and routing-policy crossovers answer different questions.

## Seven-layer map

`PRESENT` means the runner and its stated fold exist. `PARTIAL` means a useful surface exists but live evidence, identity binding, or claim consumption is still incomplete.

| Layer | Varied axis | Frozen identity and constraints | Finding | Receipt or witness | Status and gap |
|---|---|---|---|---|---|
| Kernel / host | Geometric worker count `1..threads` | Working-set bytes, trial count, target duration, machine class, traffic accounting, and runtime budget | Raw maximum, conservative adjacent-point sustainable roof, and the first stable-suffix saturation knee | [`internal/modelperfobs/bandwidth_roofline.go:35`](../../internal/modelperfobs/bandwidth_roofline.go#L35), fold at [line 548](../../internal/modelperfobs/bandwidth_roofline.go#L548); [#9519 live receipt](../_witnesses/issue-9519-host-memory-saturation/README.md) | PRESENT, live. The receipt preserves all four points and five trials per point. Apple unified-memory, multi-NUMA local/remote, and production-pattern matrices remain separate gaps #9427, #9619, and #9147. |
| Native model | Prompt length `{128,512,2048,4096,8192}` | Exact served model, endpoint, backend, node, FAK commit, Go version, max output, stream mode, and the full-MLA-on-sm80 scope | Per-length TTFT and prefill tok/s; a failed length stays `FAIL` with null metrics while later points continue | [`internal/glm52prefillsweep/glm52prefillsweep.go:43`](../../internal/glm52prefillsweep/glm52prefillsweep.go#L43), lineage at [line 137](../../internal/glm52prefillsweep/glm52prefillsweep.go#L137), live loop at [line 820](../../internal/glm52prefillsweep/glm52prefillsweep.go#L820) | PARTIAL. A live, provenance-bearing runner exists, but this inventory found no committed live sweep report. The historical GLM-5.2 surface is not current Qwen3.8 evidence and does not establish a native-model peak or knee. |
| Serving | Request concurrency | Ordered workload digest, exact model, engine and engine-receipt digest, batch capacity and source, machine, and p99 budgets | Maximum measured token throughput and highest-throughput valid point satisfying every configured SLA | [`internal/webbench/serving_sweep.go:20`](../../internal/webbench/serving_sweep.go#L20), honesty rules at [line 200](../../internal/webbench/serving_sweep.go#L200), validation at [line 289](../../internal/webbench/serving_sweep.go#L289); closed [#10028](https://github.com/anthony-chaudhary/fak/issues/10028) | PRESENT, analyzer and CLI. Invalid, over-capacity, sparse, or failed points do not become zero. Claim text still does not consume these receipts; [#10079](https://github.com/anthony-chaudhary/fak/issues/10079) owns that gap. |
| Cache / context | Finite cached-token budget, plus one unbounded pass; the KV-depth campaign separately varies prefix depth, churn, concurrency, and pressure | Exact ordered trace and logical timestamps for budget replay; the depth campaign requires model/runtime revisions, tokenizer, order, reset, and pressure arm | Reuse-versus-budget curve, infinite-cache ceiling, smallest 99%-of-ceiling budget; deepest reliable prefix and cliff remain the live-depth question | [`internal/cachesweep/cachesweep.go:1`](../../internal/cachesweep/cachesweep.go#L1), fold at [line 99](../../internal/cachesweep/cachesweep.go#L99); [#3952](https://github.com/anthony-chaudhary/fak/issues/3952), [#8424](https://github.com/anthony-chaudhary/fak/issues/8424) | PARTIAL. Budget replay is deterministic and modeled through the real radix tree. The live depth/cliff campaign has fixtures and a contract, but its cache-evidence availability bounds what can be called measured. |
| Routing / policy | Policy `{always-model, always-filter-tool, adaptive, oracle}` across five workload mixtures | Seed, trials, records per trial, mixture weights, stage calibration, and one declared utility | Policy winner, adaptive regret versus oracle, quality, latency, cost units, abstentions, stage calls, and cancellations | [`cmd/microcontextdemo/routing_voi.go:13`](../../cmd/microcontextdemo/routing_voi.go#L13), experiment loop at [line 203](../../cmd/microcontextdemo/routing_voi.go#L203); [`s8h` report](../research/micro-context-s8h-routing-voi.md) and [artifact](../../experiments/microcontext/s8h-routing-voi-2026-08-10.json) | PARTIAL, controlled. The surface locates crossovers in calibrated units. It is not live provider latency, cost, or general model quality; #6110 and #6124 own those evidence gaps. |
| Agent workload | Shared-prefix tokens × sub-turns × sub-agent count | Versioned profile and cost model, root seed, trial count, goal-pool geometry, invalidation mode, and one orchestration topology | Cross-agent dedup uplift, prefix tokens avoided, coordination overhead, and modeled prompt-cache economics | [`internal/turnbench/fanout.go:377`](../../internal/turnbench/fanout.go#L377), product sweep at [line 398](../../internal/turnbench/fanout.go#L398); [`FANOUT-BENCH-RESULTS`](../benchmarks/FANOUT-BENCH-RESULTS.md) | PRESENT, mixed authority. Kernel events and exact prefix geometry are measured; dollar and parallel-speed projections remain modeled. A shared contract must preserve that split point by point. |
| Fleet / operations | Turns per agent × fleet agent count | Versioned fleet profile and cost model, root seed, trials, invalidation granularity, and serial cell execution against process-global world state | Shared-versus-isolated cache uplift, invalidation crossover, calls removed, and turn-tax surface | [`internal/turnbench/fleet.go:375`](../../internal/turnbench/fleet.go#L375), serial product sweep at [line 399](../../internal/turnbench/fleet.go#L399); [`FLEET-SWEEP-RESULTS`](../benchmarks/FLEET-SWEEP-RESULTS.md) | PRESENT, kernel-measured. The 2,500-cell surface measures kernel events rather than model serving or accepted-task throughput. Those denominators must remain separate. |

The map also covers the ablation-arm planner in [#2831](https://github.com/anthony-chaudhary/fak/issues/2831). Its shared workload hash and arm provenance are useful companion evidence, but the planner searches feature combinations rather than defining another layer-wide response contract.

## “Sweep” also names cleanup and inventory work

Repository vocabulary contains commands named `sweep` that do not sample a response surface:

- [`fak sweep`](../../cmd/fak/sweep.go#L19) inventories a dirty shared tree and groups work by lane. It can commit one owned unit.

- [`fak dispatch sweep`](../../internal/dispatchsweep/dispatchsweep.go#L1) repeats guarded dispatch ticks until capacity, queue, or another typed stop condition ends the campaign.

- [`SweepScratch`](../../internal/treedoctor/sweepscratch.go#L27) previews or reaps declared disposable scratch.

- [`SweepGuard`](../../internal/wipattr/sweep.go#L5) classifies broad-staging hazards by ownership.

- [`internal/sweepconfig`](../../internal/sweepconfig/sweepconfig.go) loads profiles; it does not run or certify an experiment.

These are cleanup, reconciliation, inventory, or bounded campaign loops. They should not emit an optimization peak or be wired through #10189 merely because their names contain `sweep`.

## What MLCommons contributes

The pinned LoadGen source carries a useful worldview: traffic generation and logging are reusable, while the benchmark owns model and dataset semantics and a separate checker decides submission validity.

- [`loadgen/README.md:5-84`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/loadgen/README.md#L5-L84) separates a reusable load generator from model and dataset adapters. It records issued and completed traffic. Rule-specific verification stays outside the generator.

- [`loadgen/test_settings.h:190-255`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/loadgen/test_settings.h#L190-L255) makes duration, query count, RNG seeds, and repeated-versus-unique sample controls explicit.

- [`loadgen/loadgen.cc:250-315`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/loadgen/loadgen.cc#L250-L315) derives queries from those frozen settings. [`loadgen.cc:622-651`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/loadgen/loadgen.cc#L622-L651) freezes sample-set generation behind a separate seed.

- [`loadgen/loadgen.cc:739-839`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/loadgen/loadgen.cc#L739-L839) widens a bound, then binary-searches within a bracket. [`loadgen.cc:939-1002`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/loadgen/loadgen.cc#L939-L1002) records the important limitation: Server monotonicity prevents automatic discovery of a valid lower bound.

- [`loadgen/results.cc:483-518`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/loadgen/results.cc#L483-L518) keeps minimum duration, sample/query count, latency constraints, and early stopping as separate validity inputs.

- The external [`performance_check.py:76-101`](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/tools/submission/submission_checker/checks/performance_check.py#L76-L101) registers independent artifact checks; [lines 191-280](https://github.com/mlcommons/inference/blob/cfb0df14b21a3898521891f021a1c6aadec2ab2c/tools/submission/submission_checker/checks/performance_check.py#L191-L280) rechecks sample counts, seeds, and latency constraints from the logged result.

FAK should borrow the separation of generation, immutable settings, raw logs, and independent validity. It should not copy LoadGen's C++, submission rules, scenario vocabulary, or binary-search algorithm into every domain. No source bytes were copied for this study.

## Negative knowledge

- A terminal best sampled point is not automatically an interior optimum. It is censored unless the admissible range is proven closed.

- Missing, failed, and identity-invalid points are not numeric zero. They can widen a bracket or make a finding non-identifiable.

- Multiple threshold crossings do not collapse into one invented knee.

- MLCommons FindPeakPerformance applies only to its Server scenario and relies on a valid user lower bound. It is precedent for explicit validity, not proof that arbitrary FAK surfaces are monotone.

- A shared certificate cannot upgrade evidence authority. Modeled cost and controlled calibration retain their labels. Exact geometry, host-copy throughput, and kernel counters do not become live end-to-end serving evidence.

- Closed domain issues show that their local runners shipped. They do not prove #10189's common contract or #10079's claim gate has shipped.

## Coverage, exclusions, and refresh

This study inspected all seven rows, their primary runners, and checked-in artifacts. It also inspected the cleanup names most likely to be conflated. Issue review covered #10189 plus companions #10028/#10079/#9519/#3952/#8424/#2831.

The external read covered the pinned LoadGen overview and settings. It also covered query and sample generation. Peak search, result validity, the current performance checker, and repository metadata completed the read.

The following were excluded:

- exhaustive history and forge review for either repository;

- every FAK benchmark or matrix;

- external MLCommons policies, accuracy checkers, power rules, and reference models;

- live reruns and raw private fleet logs; and

- any claim that these are the only useful response surfaces.

The map is representative and decision-oriented. It is not a complete file census.

Refresh on any of these events:

- #10189 or #10079 lands;

- a mapped schema changes or a third consumer needs a wire contract;

- a live artifact closes a `PARTIAL` row;

- MLCommons changes LoadGen peak-search or validity semantics;

- the monitored pin moves; or

- a new surface exposes a finding class the four proposed folds cannot represent.
