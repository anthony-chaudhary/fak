---
title: "Research route: hypotheses, evidence, and promotion"
description: "Index of fak research notes and captured stage witnesses, with the lifecycle rule that keeps a hypothesis out of the maintained authority set."
---

# Research: hypotheses, evidence, and promotion

**Primary audience:** research readers evaluating an idea's provenance, evidence, and maturity before using it to guide implementation.
**Lifecycle:** research; conclusions here remain hypotheses or scoped observations until a maintained authority adopts them.
**Generation:** usually `gen/second-next` or `gen/future`, with some research performed to validate `gen/next` work; the route itself is `gen/now` documentation.
**Authority:** current behavior comes from maintained guides, contracts, tests, and shipped implementation linked by the [documentation index](../index.md).
**Support:** use the [contributor route](../../CONTRIBUTING.md) for implementation help and the [operator guides](../operator/) for supported operation.

**Next action:** before applying a research claim, find its provenance, maturity, and promotion witness; if any is missing, keep the claim in research and verify it independently.

## Active focus

- [Related-system inventory maps](inventory/) — pinned local-checkout maps generated with `fak study-inventory` before deep `study-repo` borrowing, so broad source coverage has a concrete denominator.

- [Structured session intent](../notes/structured-session-intent-2026-08-18.md) — own-prompt inventory plus recent scheduler/hook research, with a validated minimum/target/maximum, trigger, recurrence, and lifecycle-hook declaration spine.

- [Micro-context fabrics for 100–10,000 parallel agents](micro-context-fabrics.md) — split one cached agent base into many bounded logical contexts; includes the runnable 10k synthetic spine and controlled-kernel/API-only research ladder.
- [Micro-context operators for general large-input LLM work](micro-context-large-input-operators.md) — partition record/field/group inputs, adaptively select filters or tools, emit typed facts, fold with provenance, and stop only under an answer-safe contract.
- [Micro-window routing across filters and tool calls](micro-context-filter-tool-routing.md) — typed value-of-information routing over deterministic filters, semantic windows, read tools, widen/stop/escalate, with kernel-owned authority and budgets (#6105/#6106).
- [Micro-context S8 large-input operator witness](micro-context-s8-large-input-operator.md) - fixture-backed 1,000-record partition, deterministic prefilter, semantic map, exact reuse/invalidation, bounded fold, and oracle proof.
- [Micro-context S8a adaptive filter-selector witness](micro-context-s8a-filter-selector.md) — allowlisted per-record routing across exact exclusion, semantic filtering, group widening, and escalation with confusion/cost/replay telemetry.
- [Micro-context S8b read-only tool-enrichment witness](micro-context-s8b-tool-enrichment.md) — typed allowlisted reads with cross-record dedupe, quotas, timeout/retry, cancellation, restart cache, recursive output bounds, and independently read-back citations.
- [Micro-context S8c provenance-fold witness](micro-context-s8c-provenance-fold.md) — bounded typed reducers, content-addressed trees, minority/uncertainty retention, source-resolved claims, and path-local invalidation.
- [Micro-context S8d tuned-baseline falsification spine](micro-context-s8d-falsification-spine.md) — fixture-modeled five-pipeline decision boundaries that expose both micro-context wins and losses while leaving live net-true evidence open.
- [Micro-context S8e witnessed effect receipts](micro-context-s8e-effect-receipts.md) — capability/idempotency/resource-bound effects with denial, conflict, partial failure, cancellation ambiguity, restart replay, breaker, and independent read-back states.
- [Micro-context S8f non-fixture corpus and grader](micro-context-s8f-corpus-grader.md) — frozen 1,000 real-issue snapshot, train/tune/test isolation, separately hashed answers, leakage checks, and a strict held-out grader (#6108).
- [Micro-context S8g tuned baselines and exact frontier](micro-context-s8g-tuned-baselines.md) — tune/held-out protocol, zero-semantic-residual falsification, adaptive zero-call result, and the boundary on model/performance claims (#6109).
- [Micro-context S8h executable value-of-information routing](micro-context-s8h-routing-voi.md) — controlled filter/model/tool policy crossover, cancellation accounting, oracle regret, and explicit live-evidence boundary (#6105).
- [Micro-context S8i independently adjudicated semantic residual](micro-context-s8i-semantic-residual.md) — blinded tune/test packet, two-model live adjudication, exact-agreement/abstention fold, and blind semantic grader (#6124).
- [Micro-context S8j live semantic matrix](micro-context-s8j-live-semantic-matrix.md) — same-endpoint retrieval/long/chunk/micro execution with observed tokens, cache, TTFT/tail, retries, strict quality, and a no-winner falsification (#6110).
- [Micro-context S8k strengthened live baselines](micro-context-s8k-strong-live-baselines.md) — leave-one-out top-k retrieval, concurrent chunks, tune-only abstention calibration, tail outlier evidence, and continued no-winner result (#6151).
- [Micro-context S8l live cancellation and partial-fold policies](micro-context-s8l-tail-policies.md) — wait-all/deadline/early-stop/hedge receipts, typed partial folds, live latency-quality tradeoffs, and cancellation billing limits (#6160).
- [Micro-context S1 real-endpoint witness](micro-context-s1-real-endpoint.md) — 100/100 contexts through four bounded workers, with TTFT/usage telemetry and a retained 16-worker overload finding.
- [Micro-context S2 shared-prefix A/B](micro-context-s2-prefix-ab.md) — no cache benefit observed on the first real endpoint; scoped concurrency improved aggregate work while worsening TTFT.
## Read research by maturity

Research material makes options and uncertainty inspectable. It can identify a useful mechanism, report a measurement, or define a future architecture without claiming that fak currently supports it.

| Maturity | What the material establishes | What you may do next |
|---|---|---|
| **Hypothesis** | A falsifiable idea, assumption set, or proposed mechanism. | Reproduce or design the named experiment; do not present it as product behavior. |
| **Observed** | A result with a named source, environment, date, and measurement method. | Re-run it in the target environment before generalizing beyond its stated scope. |
| **Validated option** | Evidence has retired specified technical uncertainty, but product integration or support gates remain. | Follow the promotion trigger and linked implementation issue. |
| **Promoted** | A maintained contract, guide, test, or shipped implementation has adopted the conclusion. | Leave this route and use that maintained authority for current behavior. |

A paper, benchmark, model output, or committed study can be valuable evidence without being support evidence. Support begins only where a maintained product route states its scope and names a reproducible witness.

## Provenance contract

A research claim should make these fields discoverable:

1. **Question or hypothesis** — the proposition being tested.
2. **Source and date** — paper, repository revision, dataset, model, interview, or local observation.
3. **Method and environment** — baseline, hardware, software version, workload, mode, and constraints.
4. **Result and uncertainty** — what was observed, its scope, and what remains unknown.
5. **Maturity and generation** — hypothesis, observed, validated option, or promoted; plus the applicable generation stream.
6. **Promotion witness** — the evidence and maintained destination required before the claim becomes implementation guidance.

For performance or efficiency claims, apply the [net-true-value standard](../standards/net-true-value.md): compare against the real tuned alternative, include added costs, state scope and provenance, and provide a reproducible witness.

## Promotion, demotion, and retirement

The [Generation Contract](../generation.md#promotion-verbs) governs generation changes. Promote research closer to `gen/now` only when its named blocker is retired by evidence. Demote it when an assumption fails or a witness regresses. Retire it when the option is superseded, rejected with evidence, completed, or no longer has an owner and witness path.

Promotion does not rewrite history. Put durable instructions in the maintained contract, guide, test, or implementation route, then link the research record for rationale. Superseded research remains available through the [dated-notes route](../notes/); retired material belongs in the [archive](../archive/) with its replacement or retirement decision.

## Mode and discovery

Research often depends on a specific backend, model, release, hardware tier, dataset, or offline/live mode. Apply a result only to the mode and generation it names. A future-generation label expresses horizon rather than support, and priority expresses importance rather than maturity.

The repository's studies and dated investigations currently live under [`docs/notes/`](../notes/); the curated [Notes & research index](../../INDEX.md#notes--research-docsnotes) is the human route. [`docs/sota/`](../sota/) tracks state-of-the-art comparisons. Use [`llms.txt`](../../llms.txt) for machine-oriented discovery.

## Quantization and runtime evaluations

- [CubicQuant bounded evaluation](quantization/cubicquant.md) — bounded scalar-reconstruction study for the CubicQuant format against tuned uniform and non-uniform baselines.
- [Heterogeneity-aware microscaling](quantization/heterogeneous-microscaling.md) — bounded interoperability evaluation for mixed-shape microscaling groups.
- [Output-aware INT2 KV-cache rotation](quantization/int2-kv-rotation.md) — bounded evaluation of low-bit KV rotation under output-aware reconstruction.
- [LightRot bounded low-bit evaluation](quantization/lightrot.md) — bounded reproduction surface for the LightRot quantization proposal.
- [QEvict recoverable quantized KV eviction](quantization/qevict.md) — bounded evaluation of quantized eviction with recovery semantics.
- [Recurrent Residual Quantization evaluation](quantization/recurrent-residual.md) — bounded reconstruction study for recurrent residual quantization.
- [ReQuant fixed-grid refinement evaluation](quantization/requant.md) — bounded evaluation of refinement over a fixed quantization grid.
- [Qwen3.8 Metal OSS hot-path study](qwen38-metal-oss-hotpath-study.md) — captured hot-path observations for the open-source Metal route serving Qwen3.8.
- [Frame loss between master agents and subagents](relativity-frame-loss.md) — study of context and control loss between coordinating and delegated agents.
- [TTL-upgrade refusal corpus study](../notes/ttl-upgrade-refusal-corpus-2026-08-09.md) — dated refusal-corpus study for time-to-live upgrade behavior.

- [Micro-context cache-value Track-1 fold](micro-context-cachevalue-track1.md) — controlled S2b reuse enters the witnessed P&L while synthetic and provider-dollar evidence remain fenced out.
- [Micro-context S2b controlled in-kernel prefix-cache A/B](micro-context-s2b-kernel-cache-ab.md) — fresh-process arms reconcile response usage with RadixAttention counters and observe a fixture-scoped 2.16x shared-base service gain.
- [Micro-context S3 hibernation/restart](micro-context-s3-hibernation-restart.md)
- [Micro-context S4a lightweight descriptor](micro-context-s4-lightweight-descriptor.md)
- [Micro-context S4b compatibility scheduler](micro-context-s4-compatibility-scheduler.md)
- [Micro-context S4c effect safety](micro-context-s4-effect-safety.md)
- [Micro-context S4d: bounded multi-turn continuation](micro-context-s4d-multi-turn-descriptor.md) — exact 1,000×3 turn accounting with continuation-token and byte-verified mid-task restore.
- [Micro-context S4e real compatibility-batch execution](micro-context-s4e-compat-batch-execution.md) — planner batches execute through the in-kernel batch seam; the first mixed-length CPU fixture is an honest 0.539x negative result.
- [Micro-context S5a controlled-kernel 1,000-context ramp](micro-context-s5a-controlled-kernel-1k.md)
- [Micro-context S6 API-only adapter](micro-context-s6-api-only.md)
- [Micro-context health scorecard](micro-context-health-scorecard.md) — deterministic witness fold grades the controlled 1k CUDA ledger A/100 and names the missing second-run drift baseline.
- [Micro-context outcome counters](micro-context-outcome-counters.md) — the existing quality ledger exposes reconciled success/error/refusal totals; controlled 1k CUDA readout is 1000/0/0.
- [Micro-context quality and observability ledger](micro-context-quality-ledger.md)
- [Micro-context S7 mixed-tenant fairness](micro-context-s7-fairness.md)
- [S8m: three-adjudicator tool-routing gold stabilization](micro-context-s8m-tool-gold.md)
- [S8n: filter/tool micro-window scheduler](micro-context-s8n-filter-tool-scheduler.md)
- [S8o: live quality-qualified filter/tool scheduler](micro-context-s8o-live-filter-tool-scheduler.md)

- [S8p live scheduler disagreement audit](micro-context-s8p-disagreement-audit.md) — blinded error atlas finds disputed gold and no stable pre-answer admission signal; records `not-yet`.

## Frontier infrastructure and workload expectations

- Human overview and field assumptions: [`frontier-infrastructure/README.md`](frontier-infrastructure/README.md)
- Machine-readable dated evidence ledger: [`frontier-infrastructure/index.json`](frontier-infrastructure/index.json)
- Benchmark assumption registry (batching, clusters, heavy tails/Zipf, user and agent workloads): [`frontier-infrastructure/workload-assumptions.md`](frontier-infrastructure/workload-assumptions.md)
- Offline structural validation commands are embedded in the corpus README.

This corpus separates production measurements, official statements, vendor claims,
reported estimates, analysis, and rumors. Its explicit coverage gaps are part of the
evidence contract; the index must not imply that the open web is finite or fully observed.

## Named coding-workload patterns

- [`coding-workload-vocabulary.md`](coding-workload-vocabulary.md) — cited proposal separating workload shape, orchestration topology, verification strategy, and failure mode; names reusable patterns/subpatterns and rejects common conflations.
- [`coding-workload-vocabulary.json`](coding-workload-vocabulary.json) — deterministic machine companion with source provenance, inclusion/exclusion boundaries, aliases, and stable candidate IDs.

Use `fak workpattern list|source|trajectory|report` to consume the canonical seed catalog and evidence miners. The report is bounded to explicit detectors and scrubbed/local inputs; it does not claim universal taxonomy consensus or infer private-chat intent.

- [S8q/S8r true pre-answer tool admission](micro-context-s8qr-true-tool-admission.md) — paired model-distinct consensus gold; two-stage admission matches quality while opening 50% fewer reads on the scoped envelope.

- [S8s/S8t natural multi-tool decision surface](micro-context-s8st-natural-multitool-surface.md) — five evidence classes show fixed/adaptive/parallel crossover by tool cost, with quality gated first.

- [`tensor-build-local-study-2026-08-15.md`](../notes/tensor-build-local-study-2026-08-15.md) — deep, snapshot-pinned study of local TensorBuild: typed engine identity, evidence tiers, artifact liveness, agent/human control parity, and work-cost attribution; dedupes current fak coverage and files #6874-#6876.

- [`CONCEPT-STUDY-TENSOR-BUILD-2026-08-29.md`](../notes/CONCEPT-STUDY-TENSOR-BUILD-2026-08-29.md) — current whole-tree, snapshot-pinned TensorBuild recheck: 26 evidence/measurement/native-runtime candidates, exact FAK witnesses, explicit TensorRT ablations, and 13 bounded leaves (#10268-#10271, #10278-#10286).

## Architecture and kernel studies

- [AI-Ops storage qualification study (2026-08-29)](ai-ops-storage-qualification-study-2026-08-29.md) — artifact-first SSD qualification architecture, license gate, and trace-to-storage-envelope borrow filed as #10267.
- [Agent-serving composition architecture (2026-08-26)](agent-serving-composition-architecture-study-2026-08-26.md) — evidence-backed composition study and follow-on map.
- [NVIDIA KVTC study](nvidia-kvtc-study.md) — upstream KV-transfer mechanism study and fak relevance map.
- [Incumbent inference architecture bottlenecks (2026-08-28)](incumbent-inference-architecture-bottlenecks-2026-08-28.md) — vLLM/SGLang/Dynamo/MAX bottleneck study (#9894): the binding constraint is the interaction cross-product among scheduling, reusable state, specialization, and compilation lifecycle, not one missing kernel.
- [metrics-service study (2026-08-29)](metrics-service-study-2026-08-29.md) — snapshot-pinned study of an external Go observability runtime (#10287): deadline-aligned collection, normalized validated snapshots, and concurrent sink fan-out, with the bounded borrow/disposition map.
- [ActaClad/plumbline study (2026-08-31)](plumbline-study-2026-08-31.md) — Python static-analyzer study (#10452): borrow four bounded mechanisms (stable evidence paths, negative fixtures, a new-findings gate, export into existing surfaces); do not adopt a second analyzer framework.
- [Composable system prompt algebra](composable-system-prompt-algebra-2026-09.md) — algebra and transformation rules for composable agent system prompts.
- [Qwen3.8 and GLM-5.3 deep subagent inventory](qwen38_glm53_deep_subagent_inventory.md) — inventory and capability analysis of deep subagent hierarchies.
- [Qwen3.8 and GLM-5.3 Flash inventory](qwen38_glm53_flash_inventory.md) — feature and compatibility inventory for fast-tier models.

