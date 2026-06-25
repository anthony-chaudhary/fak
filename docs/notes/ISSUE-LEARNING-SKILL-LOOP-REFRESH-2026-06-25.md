---
title: "Issue, learning, and durable skill-loop refresh - 2026-06-25"
description: "A dated, evidence-backed snapshot of the current GitHub issue backlog and the repeatable loops that should keep the top-priority work organized."
---

# Issue, learning, and durable skill-loop refresh - 2026-06-25

This note records the current work-queue refresh for the objective "garden and
organize tickets and work, especially top priorities." It records both the
read-only triage snapshot and the small label batches applied below. No issue was
closed or assigned.

## Evidence gathered

Commands run from the repo root on 2026-06-25:

```bash
python tools/issue_triage.py --json
python tools/issue_triage.py --markdown --out docs/_audits/issue-triage-2026-06-25.md
python tools/issue_triage.py --actions --out docs/_audits/issue-actions-2026-06-25.json
gh issue list --state open --label "priority/P0" --limit 40 --json number,title,labels,updatedAt,url
gh issue list --state open --label "priority/P1" --limit 80 --json number,title,labels,updatedAt,url
gh issue edit 298 --add-label model
gh issue edit 289 --add-label gpu
gh issue edit 287 --add-label gpu
gh issue edit 294 --add-label loader
gh issue edit 255 --add-label agentic-serving
gh issue edit 300 --add-label model
gh issue edit 299 --add-label model
gh issue edit 297 --add-label model
gh issue edit 296 --add-label model
gh issue edit 279 --add-label gpu
gh issue edit 250 --add-label agentic-serving
gh issue edit 248 --add-label agentic-serving
gh issue edit 282 --add-label agentic-serving
gh issue edit 218 --add-label agentic-serving
gh issue edit 8 --add-label loader
gh issue edit 284 --add-label model
gh issue edit 85 --add-label enhancement
gh issue edit 27 --add-label enhancement
gh issue edit 18 --add-label performance
gh issue edit 42 --add-label enhancement
gh issue edit 37 --add-label enhancement
gh issue edit 637 --add-label research
gh issue edit 496 --add-label enhancement --add-label model
gh issue edit 453 --add-label enhancement
gh issue edit 382 --add-label enhancement
gh issue edit 381 --add-label enhancement
gh issue edit 26 --add-label enhancement
gh issue edit 645 --add-label substrate
gh issue edit 642 --add-label substrate
gh issue edit 641 --add-label substrate
gh issue edit 595 --add-label model
gh issue edit 326 --add-label agentic-serving
gh issue edit 402 --add-label model
gh issue edit 401 --add-label agentic-serving
gh issue edit 293 --add-label loader
gh issue edit 292 --add-label loader
gh issue edit 291 --add-label model
gh issue edit 290 --add-label model
gh issue edit 277 --add-label model-arch
gh issue edit 275 --add-label model
gh issue edit 272 --add-label agentic-serving
gh issue edit 245 --add-label agentic-serving
gh issue edit 243 --add-label agentic-serving
gh issue edit 241 --add-label agentic-serving
gh issue edit 238 --add-label agentic-serving
gh issue edit 213 --add-label agentic-serving
gh issue edit 233 --add-label substrate
gh issue edit 216 --add-label substrate
gh issue edit 199 --add-label substrate
gh issue edit 9 --add-label substrate
gh issue edit 380 --add-label substrate
gh issue edit 211 --add-label substrate
gh issue edit 205 --add-label agentic-serving
gh issue edit 202 --add-label security
gh issue edit 196 --add-label substrate
gh issue edit 23 --add-label model
gh issue edit 17 --add-label gpu
gh issue edit 10 --add-label substrate --add-label performance
gh issue edit 7 --add-label loader
gh issue edit 523 --add-label model-arch
gh issue edit 504 --add-label substrate --add-label research
gh issue edit 31 --add-label agentic-serving
gh issue edit 635 --add-label substrate
gh issue edit 622 --add-label substrate
gh issue edit 603 --add-label model
gh issue edit 602 --add-label model
gh issue edit 600 --add-label model
gh issue edit 599 --add-label model
gh issue edit 598 --add-label model
gh issue edit 208 --add-label substrate
gh issue edit 310 --add-label substrate
gh issue edit 472 --add-label enhancement --add-label gpu
gh issue edit 416 --add-label research --add-label substrate
gh issue edit 415 --add-label documentation --add-label substrate
gh issue edit 307 --add-label research --add-label model
gh issue edit 306 --add-label performance --add-label agentic-serving
gh issue edit 305 --add-label research --add-label gpu
gh issue edit 304 --add-label research --add-label agentic-serving
gh issue edit 303 --add-label research --add-label substrate
gh issue edit 302 --add-label research --add-label agentic-serving
gh issue edit 301 --add-label research --add-label substrate
gh issue edit 84 --add-label agentic-serving
gh issue edit 73 --add-label agentic-serving
gh issue edit 566 --add-label substrate
gh issue edit 451 --add-label performance
gh issue edit 16 --add-label performance
gh issue edit 12 --add-label performance
gh issue edit 376 --add-label performance
gh issue edit 312 --add-label documentation
gh issue edit 707 --add-label enhancement
gh issue edit 609 --add-label enhancement
gh issue edit 734 --add-label substrate
gh issue edit 729 --add-label dispatch
gh issue edit 732 --add-label dispatch
gh issue edit 731 --add-label dispatch
gh issue edit 730 --add-label dispatch
gh issue edit 564 --add-label substrate
gh issue edit 563 --add-label substrate
gh issue edit 562 --add-label substrate
gh issue edit 561 --add-label substrate
gh issue edit 560 --add-label substrate
gh issue edit 557 --add-label substrate
gh issue edit 714 --add-label substrate
gh issue edit 710 --add-label substrate
gh issue edit 656 --add-label documentation --add-label model
gh issue edit 653 --add-label documentation --add-label substrate
gh issue edit 630 --add-label agentic-serving
gh issue edit 629 --add-label agentic-serving
gh issue edit 628 --add-label agentic-serving
gh issue edit 633 --add-label security
gh issue edit 627 --add-label agentic-serving
gh issue edit 626 --add-label agentic-serving
gh issue edit 583 --add-label model-arch
gh issue edit 345 --add-label substrate
gh issue edit 341 --add-label substrate
gh issue edit 338 --add-label substrate
gh issue edit 337 --add-label security
gh issue edit 333 --add-label substrate
gh issue edit 331 --add-label agentic-serving
gh issue edit 330 --add-label substrate
gh issue edit 322 --add-label model-arch
gh issue edit 321 --add-label agentic-serving
gh issue edit 319 --add-label substrate
gh issue edit 318 --add-label agentic-serving
gh issue edit 313 --add-label substrate
gh issue edit 309 --add-label substrate
gh issue edit 365 --add-label priority/P1
gh issue edit 364 --add-label priority/P1
gh issue edit 362 --add-label priority/P1
gh issue edit 363 --add-label priority/P1
gh issue edit 310 --add-label priority/P1
gh issue edit 412 --add-label priority/P2
gh issue edit 387 --add-label priority/P2
gh issue edit 483 --add-label priority/P2
gh issue edit 710 --add-label priority/P2
gh issue edit 707 --add-label priority/P2
gh issue edit 706 --add-label priority/P2
gh issue edit 703 --add-label priority/P2
gh issue edit 702 --add-label priority/P2
gh issue edit 701 --add-label priority/P2
gh issue edit 700 --add-label priority/P2
gh issue edit 697 --add-label priority/P2
gh issue edit 696 --add-label priority/P2
gh issue edit 695 --add-label priority/P2
gh issue edit 694 --add-label priority/P2
gh issue edit 691 --add-label priority/P2
gh issue edit 690 --add-label priority/P2
gh issue edit 689 --add-label priority/P2
gh issue edit 688 --add-label priority/P2
gh issue edit 686 --add-label priority/P2
gh issue edit 685 --add-label priority/P2
gh issue edit 684 --add-label priority/P2
gh issue edit 683 --add-label priority/P2
gh issue edit 682 --add-label priority/P2
gh issue edit 681 --add-label priority/P2
gh issue edit 673 --add-label priority/P2
gh issue edit 672 --add-label priority/P2
gh issue edit 671 --add-label priority/P2
gh issue edit 670 --add-label priority/P2
gh issue edit 668 --add-label priority/P2
gh issue edit 667 --add-label priority/P2
gh issue edit 666 --add-label priority/P2
gh issue edit 665 --add-label priority/P2
gh issue edit 664 --add-label priority/P2
gh issue edit 738 --add-label substrate
gh issue edit 737 --add-label substrate
gh issue edit 736 --add-label substrate
gh issue edit 735 --add-label substrate
gh issue edit 295 --add-label "help wanted"
gh issue edit 45 --add-label "help wanted"
gh issue edit 298 --add-label "help wanted"
gh issue edit 294 --add-label "help wanted"
gh issue edit 289 --add-label "help wanted"
gh issue edit 287 --add-label "help wanted"
gh issue edit 255 --add-label "help wanted"
gh issue edit 50 --add-label "help wanted"
gh issue edit 365 --add-label "help wanted"
gh issue edit 364 --add-label "help wanted"
gh issue edit 363 --add-label "help wanted"
gh issue edit 362 --add-label "help wanted"
gh issue edit 310 --add-label "help wanted"
gh issue edit 739 --add-label agentic-serving --add-label priority/P2
gh issue edit 740 --add-label agentic-serving --add-label priority/P2
gh issue edit 741 --add-label agentic-serving --add-label priority/P2
gh issue edit 742 --add-label agentic-serving --add-label priority/P2
gh issue edit 743 --add-label agentic-serving --add-label priority/P2
gh issue edit 744 --add-label agentic-serving --add-label priority/P2
gh issue edit 440 --add-label "help wanted"
gh issue edit 438 --add-label "help wanted"
gh issue edit 430 --add-label "help wanted"
gh issue edit 399 --add-label "help wanted"
gh issue edit 388 --add-label "help wanted"
gh issue edit 78 --add-label "help wanted"
gh issue edit 71 --add-label "help wanted"
gh issue edit 70 --add-label "help wanted"
gh issue edit 69 --add-label "help wanted"
gh issue edit 67 --add-label "help wanted"
gh issue edit 64 --add-label "help wanted"
gh issue edit 60 --add-label "help wanted"
gh issue edit 59 --add-label "help wanted"
gh issue edit 58 --add-label "help wanted"
gh issue edit 21 --add-label "help wanted"
gh issue edit 20 --add-label "help wanted"
gh issue edit 43 --add-label "help wanted"
gh issue edit 41 --add-label "help wanted"
gh issue edit 40 --add-label "help wanted"
gh issue edit 39 --add-label "help wanted"
gh issue edit 36 --add-label "help wanted"
gh issue edit 35 --add-label "help wanted"
gh issue edit 34 --add-label "help wanted"
gh issue edit 33 --add-label "help wanted"
gh issue edit 738 --add-label "help wanted"
gh issue edit 733 --add-label "help wanted"
gh issue edit 729 --add-label "help wanted"
gh issue edit 717 --add-label "help wanted"
gh issue edit 715 --add-label "help wanted"
gh issue edit 648 --add-label "help wanted"
gh issue edit 647 --add-label "help wanted"
gh issue edit 645 --add-label "help wanted"
gh issue edit 644 --add-label "help wanted"
gh issue edit 642 --add-label "help wanted"
gh issue edit 641 --add-label "help wanted"
gh issue edit 640 --add-label "help wanted"
gh issue edit 639 --add-label "help wanted"
gh issue edit 637 --add-label "help wanted"
gh issue edit 596 --add-label "help wanted"
gh issue edit 595 --add-label "help wanted"
gh issue edit 18 --add-label "help wanted"
gh issue edit 9 --add-label "help wanted"
gh issue edit 8 --add-label "help wanted"
gh issue edit 1 --add-label priority/P2
gh issue edit 6 --add-label priority/P2
gh issue edit 583 --add-label priority/P2
gh issue edit 609 --add-label priority/P2
gh issue edit 610 --add-label priority/P2
gh issue edit 612 --add-label priority/P2
gh issue edit 626 --add-label priority/P2
gh issue edit 627 --add-label priority/P2
gh issue edit 628 --add-label priority/P2
gh issue edit 629 --add-label priority/P2
gh issue edit 630 --add-label priority/P2
gh issue edit 631 --add-label priority/P2
gh issue edit 632 --add-label priority/P2
gh issue edit 633 --add-label priority/P2
gh issue edit 643 --add-label priority/P2
gh issue edit 649 --add-label priority/P2
gh issue edit 651 --add-label priority/P2
gh issue edit 652 --add-label priority/P2
gh issue edit 655 --add-label priority/P2
gh issue edit 468 --add-label build --add-label substrate --add-label priority/P2
gh issue edit 566 --add-label priority/P2
gh issue edit 564 --add-label priority/P2
gh issue edit 563 --add-label priority/P2
gh issue edit 562 --add-label priority/P2
gh issue edit 561 --add-label priority/P2
gh issue edit 560 --add-label priority/P2
gh issue edit 557 --add-label priority/P2
gh issue edit 472 --add-label priority/P2
gh issue edit 463 --add-label priority/P2
gh issue edit 416 --add-label priority/P2
gh issue edit 376 --add-label priority/P2
gh issue edit 307 --add-label priority/P2
gh issue edit 306 --add-label priority/P2
gh issue edit 305 --add-label priority/P2
gh issue edit 304 --add-label priority/P2
gh issue edit 303 --add-label priority/P2
gh issue edit 302 --add-label priority/P2
gh issue edit 301 --add-label priority/P2
gh issue edit 84 --add-label priority/P2
gh issue edit 73 --add-label priority/P2
gh issue edit 372 --add-label priority/P2 --add-label substrate
gh issue edit 285 --add-label priority/P2 --add-label substrate
gh issue edit 281 --add-label priority/P2 --add-label substrate
gh issue edit 267 --add-label priority/P2 --add-label substrate
gh issue edit 256 --add-label priority/P2 --add-label substrate
gh issue edit 252 --add-label priority/P2 --add-label substrate
gh issue edit 249 --add-label priority/P2 --add-label substrate
gh issue edit 283 --add-label priority/P2 --add-label security
gh issue edit 265 --add-label priority/P2 --add-label security
```

The generated `docs/_audits/` files are gitignored process output. This tracked
note keeps the durable summary.

## Current queue state

The deterministic triage fold found:

| Signal | Count |
|---|---:|
| Open issues | 378 |
| Needs priority | 150 |
| Needs area | 118 |
| Needs kind | 1 |
| Orphan P0/P1 | 72 |
| Likely duplicate | 7 |
| Stale | 0 |
| Dormant question | 0 |

Interpretation: the queue does not currently need a stale-close sweep. The safe
label passes applied area labels to 109 rows, missing-kind fixes to thirty rows,
two additional documentation labels to docs-only rows, and priority labels to
thirty-eight rows where severity or parent-epic priority was explicit. A later
session-reset batch classified #739-#744 as served-agent follow-up work
(`agentic-serving`, `priority/P2`). The visibility pass then added `help wanted`
across the current top-priority orphan slice. A fresh fold reports zero orphan
P0/P1 rows that lack `help wanted`; the remaining orphan count is still real
because visibility is not ownership.

The live aggregate moved while this pass ran (`open` 400 -> 397 -> 403 -> 402 ->
398 -> 396 -> 395 -> 398 -> 395 -> 394 -> 393 -> 390 -> 385 -> 380 -> 378,
`needs_area` 158 -> 163 -> 128 -> 132 -> 128 -> 118, `needs_priority` 245 ->
242 -> 241 -> 238 -> 200 -> 199 -> 150, `orphan` 70 -> 73 -> 72 -> 76 -> 77 ->
76 -> 73 -> 72), so the counts above are the current
witness state, not solely the effect of this local label batch. The orphan count
rose earlier because five severe unassigned bugs were honestly promoted to
`priority/P1`; later external issue churn dropped the total to 72. That is
better queue truth, not ownership progress. The two remaining `needs-kind` rows
are #590 and #468: #590 is a brand/licensing legal task without a clean project
kind label, and #468 is a bare merge-process note. Both are intentionally
deferred rather than forced into a misleading kind. The remaining top gardening
work is review-only: explicitly claim or defer orphan P0/P1 rows, then keep
reducing missing priority/area labels.

Continuation refresh: the later pass conservatively marked 39 clear follow-ups
as `priority/P2` (ctxplan, session-control, in-kernel witness/test, benchmark,
parity-track, webbench, and MPI-shaped docs/bench children), gave the bare merge
cleanup row #468 `build`, `substrate`, and `priority/P2`, and classified nine
docs-maintenance rows as `priority/P2` with either `substrate` or `security`.
That reduced `needs_priority` by 49, `needs_area` by 10, `needs_kind` by 1, and
`bare` by 1 without promoting new P0/P1 orphan work.

## Label batch applied

The following labels were applied and verified by `gh issue view` plus a fresh
`issue_triage.py --json` run:

| Issue | Added label | Evidence |
|---|---|---|
| #298 `feat(model): Production Llama 3.x Checkpoints [A-004]` | `model` | existing `model-support`/`llama`; scope is production checkpoints |
| #289 `perf(prefill): Close Prefill Throughput Gap [B-001]` | `gpu` | existing `cuda`; scope includes CUDA graph/prefill kernel work |
| #287 `perf(vulkan): Vulkan Backend Optimization [B-002]` | `gpu` | existing `vulkan`; scope is AMD Vulkan backend optimization |
| #294 `feat(model): HuggingFace Hub Direct Loading [A-008]` | `loader` | scope is HF Hub resolution, cached download, integrity |
| #255 `feat(benchmark): N=100-1000 Agent Benchmark [D-001]` | `agentic-serving` | scope is agent-framework/fanbench comparison at fleet scale |
| #300 `feat(model): GPTQ Quantization Support [A-002]` | `model` | existing `model-support`/`quantization`; scope is checkpoint/format support |
| #299 `feat(model): EXL2 Format Support [A-003]` | `model` | existing `model-support`/`quantization`; scope is checkpoint/format support |
| #297 `feat(model): Production Qwen2.5 Checkpoints [A-005]` | `model` | existing `model-support`/`qwen`; scope is production checkpoint support |
| #296 `feat(model): Production Mixtral 8x7B/8x22B [A-006]` | `model` | existing `model-support`/`moe`; scope is production checkpoint support |
| #279 `perf(cuda): Fused Kernel Optimization [B-005]` | `gpu` | existing `cuda`; scope is fused CUDA kernel work |
| #250 `feat(integration): CrewAI Pattern Support [D-003]` | `agentic-serving` | existing `agent-framework`/`crewai`; scope is agent framework integration |
| #248 `feat(integration): AutoGen Integration [D-004]` | `agentic-serving` | existing `agent-framework`/`autogen`; scope is agent framework integration |
| #282 `perf(serving): Continuous Batching Integration [B-004]` | `agentic-serving` | title/body name the serving engine, gateway batching API, scheduling, and KV cache sharing |
| #218 `feat(gateway): Prompt Caching Features [F-002]` | `agentic-serving` | title/body name gateway prompt caching controls and integration APIs |
| #8 `fak(modelbench): -lean (quantize-at-load) only works with -hf, not -gguf` | `loader` | scope is GGUF stream load, quantize-at-load, and avoiding f32 dequant OOM |
| #284 `perf(decode): Speculative Decoding [B-003]` | `model` | scope names draft model API, draft model integration, and configurable draft model |
| #85 `feat(model): network StageTransport` | `enhancement` | title/body describe an implementation feature for network stage handoff |
| #27 `feat(governance): wire L3 KV-governance referee` | `enhancement` | title/body describe feature wiring for governance and exact-span eviction |
| #18 `bench(gcp): re-run fak-cuda on a real GPU` | `performance` | body is a head-to-head GPU benchmark rerun for prefill/decode numbers |
| #42 `feat(serving): node membership/health/drain/failover loop` | `enhancement` | title/body describe a new serving membership and health feature |
| #37 `feat(serving): orchestrate external P/D disaggregation` | `enhancement` | title/body describe a new serving orchestration feature |
| #637 `epic(serving): throughput parity + trust/L3 over one shared spine` | `research` | body says this is a planning/sequencing umbrella carrying analysis and forks |
| #496 `model(minimax_m3): wire incremental Session/KV-cache MSA path` | `enhancement`, `model` | body describes model-session implementation work and was missing both kind and area |
| #453 `conformance: extract a standalone third-party-runnable fak-conformance suite + CLI` | `enhancement` | body describes extracting a new conformance suite and CLI |
| #382 `feat(rsi): closure_rate / regression_rate loop-health metric` | `enhancement` | title/body describe a new RSI loop-health metric surface |
| #381 `feat(rsi): findings-route closure appender` | `enhancement` | title/body describe a new findings-route helper |
| #26 `feat(gateway): structured/guided decoding` | `enhancement` | title/body describe adding structured decoding support to gateway/model paths |
| #645 `feat(agenttopo): declared agent communication topology` | `substrate` | body describes a declared fleet adjacency and neighbor-exchange primitive |
| #642 `feat(kernel): non-blocking TestHandle poll` | `substrate` | body extends the concrete kernel pending-handle substrate |
| #641 `feat(kernel): ReapAny / ReapAll over a handle set` | `substrate` | body extends concrete kernel fan-in over submission handles |
| #595 `epic(modelroute): model routing as a first-class spine` | `model` | body centers on per-aspect model routing and ensembles |
| #326 `serve(gateway): coarse error envelope + no fail-fast on dead upstream` | `agentic-serving` | body is gateway request-handling and serving robustness |
| #402 `feat(model): implement speculative decoding` | `model` | body names draft model integration and model/compute context |
| #401 `feat(engine): implement native continuous batching` | `agentic-serving` | body describes concurrent serving scheduling and gateway/engine batching |
| #293 `feat(model): Model Streaming/Lazy Loading [A-009]` | `loader` | body scopes mmap/progressive weight loading and initial-load memory reduction |
| #292 `feat(model): GGUF Format Completion [A-010]` | `loader` | body scopes GGUF loading, tensor types, and metadata preservation |
| #291 `feat(model): LoRA Adapter Support [A-011]` | `model` | body scopes LoRA adapter loading and decode-time application |
| #290 `feat(model): Vision/Multimodal Support [A-012]` | `model` | body scopes vision encoder and multimodal model support |
| #277 `feat(kv): PagedAttention Implementation [B-006]` | `model-arch` | body scopes page-based KV allocation and RadixAttention integration |
| #275 `feat(quant): INT4/INT2 Quantization [B-007]` | `model` | body scopes quantized model serving and INT4/INT2 model weights |
| #272 `perf(batching): Dynamic Batching Optimization [B-008]` | `agentic-serving` | body scopes multi-request serving batching and scheduling |
| #245 `feat(orchestration): Workflow Orchestration Layer [D-005]` | `agentic-serving` | body scopes agent-framework workflow orchestration patterns |
| #243 `feat(tools): Tool Ecosystem Expansion [D-006]` | `agentic-serving` | body scopes agent-framework tools and safety annotations |
| #241 `feat(protocol): Multi-Agent Coordination Protocol [D-007]` | `agentic-serving` | body scopes agent-to-agent coordination, messages, and shared state |
| #238 `feat(testing): Agent Testing Framework [D-008]` | `agentic-serving` | body scopes deterministic testing of agent workflows |
| #213 `feat(mcp): MCP Ecosystem Expansion [F-004]` | `agentic-serving` | body scopes MCP tool/resource/prompt support for integrations |
| #233 `feat(testing): Comprehensive Coverage Goals [E-002]` | `substrate` | body scopes repo-wide coverage measurement, CI reporting, and trend enforcement |
| #216 `feat(observability): Metrics/Telemetry Production [F-003]` | `substrate` | body scopes Prometheus/OpenTelemetry/Grafana/SLO production telemetry substrate |
| #199 `feat(ci): CI/CD Improvements [G-001]` | `substrate` | body scopes multi-platform CI, benchmark tracking, releases, and badges |
| #9 `infra(bench): enforce lineage fields` | `substrate` | title/labels scope benchmark-emitter lineage across the repo foundation |
| #380 `chore(tools): clear 114 ruff errors in tools/` | `substrate` | body scopes Python tools-tree lint hygiene and hook/CI prevention |
| #211 `feat(tooling): Developer Tooling [F-005]` | `substrate` | body scopes fak debug/profile/test developer commands |
| #205 `feat(sdk): SDK Generation [F-007]` | `agentic-serving` | body scopes generated client SDKs for API integrations |
| #202 `feat(ux): Policy Editor GUI [F-008]` | `security` | body scopes policy authoring and validation UI for the capability floor |
| #196 `feat(dashboard): Performance Dashboards [G-002]` | `substrate` | body scopes public observability/dashboard infrastructure for parity metrics |
| #23 `feat(decode): speculative decoding` | `model` | body scopes draft/verify decode and ridden-engine spec-decode semantics |
| #17 `build(cuda): offline strict-toolchain preflight` | `gpu` | body scopes CUDA header/toolchain guard before paying for a GPU VM |
| #10 `infra(bench-node): bench the laptop driven from the desktop` | `substrate`, `performance` | body scopes benchmark placement and honest measurement |
| #7 `fak(ggufload): add gemma3 + MoE tensor-name mappings` | `loader` | body scopes GGUF tensor-name loader mappings |
| #523 `KV compression on the eviction path` | `model-arch` | body scopes KV materialization/compression behavior under residency pressure |
| #504 `Lever-flip causal attribution` | `substrate`, `research` | body scopes replay substrate and ablation-style causal attribution |
| #31 `feat(scheduler): preemption + KV swap-to-host/recompute` | `agentic-serving` | body scopes serving scheduler preemption under memory pressure |
| #635 `checkpoint: decide Go port` | `substrate` | body scopes hook/checkpoint implementation placement and hot-path constraints |
| #622 `feat(ablate): env-gated feature arms via subprocess re-exec` | `substrate` | body scopes the ablation runner substrate for env-gated feature arms |
| #603 `feat(modelroute): routing observability` | `model` | body scopes model-routing decisions and per-aspect metrics/journal output |
| #602 `feat(modelroute): free-text ensemble reductions` | `model` | body scopes model ensemble judging and reduction behavior |
| #600 `feat(modelroute): telemetry to learned routing` | `model` | body scopes model-routing feedback from cost/latency/quality outcomes |
| #599 `feat(modelroute): scout-model live classification` | `model` | body scopes a cheap model scout feeding route subject labels |
| #598 `feat(modelroute): per-tool-call routing inside the agent loop` | `model` | body scopes per-tool model routing before kernel submission |
| #208 `docs(deployment): Deployment Guides [F-006]` | `substrate` | body scopes production deployment artifacts and readiness docs |
| #310 `obs(audit): NS_INCLUDE_PREFIX default silently zeros the audit pass` | `substrate` | body scopes fleet audit telemetry and dashboard alert infrastructure |
| #472 `Land model-level Q8 routing through Vulkan HAL` | `enhancement`, `gpu` | body scopes Vulkan HAL Q8 routing and hardware tests |
| #416 `Benchmark scientific rigor` | `research`, `substrate` | body scopes benchmark lineage, invalidation, and reproducibility rules |
| #415 `Hardware catalog` | `documentation`, `substrate` | body scopes machine specs, onboarding, and baseline-run documentation |
| #307 `[epic] Track A - Model Support Parity` | `research`, `model` | body is the Track A model-support planning epic |
| #306 `[epic] Track B - Performance Parity` | `performance`, `agentic-serving` | body is the Track B throughput/latency/serving parity epic |
| #305 `[epic] Track C - GPU/Backend Parity` | `research`, `gpu` | body is the GPU/backend parity planning epic |
| #304 `[epic] Track D - Agent Framework Parity` | `research`, `agentic-serving` | body is the agent-framework parity planning epic |
| #303 `[epic] Track E - Testing/Quality Parity` | `research`, `substrate` | body is the testing/quality parity planning epic |
| #302 `[epic] Track F - Integration/Tooling Parity` | `research`, `agentic-serving` | body is the integration/tooling parity planning epic |
| #301 `[epic] Track G - Foundation Parity` | `research`, `substrate` | body is the foundation/infrastructure parity planning epic |
| #84 `webbench: Full harness evaluation with real datasets` | `agentic-serving` | body scopes web agent benchmark harness evaluation |
| #73 `webbench: Run real model measurements` | `agentic-serving` | body scopes real agent/model measurement protocol |
| #566 `feat(ctxplan): plan over a cross-session durable history` | `substrate` | body scopes durable context-planning history across sessions |
| #451 `adoption: run fak in front of a real engine` | `performance` | body scopes measured adjudication overhead/ns-call against real vLLM/SGLang integration |
| #16 `bench(gcp): Blackwell B200 quota DENIED` | `performance` | body scopes flagship Blackwell benchmark quota/unblock and run |
| #12 `infra(bench-node): onboard GCP GPU bench VMs` | `performance` | body scopes GCP GPU benchmark node integration |
| #376 `gpu(bench): P5-01 Qwen 7B Q8 benchmark not run` | `performance` | body scopes GPU benchmark retry and measurement |
| #312 `process(rsi): bind issue closes to a resolving commit` | `documentation` | body scopes a documented close convention for honest closure_rate |
| #707 `feat(compute): feed live DeviceCapacity pressure into cachemeta.TierPressure` | `enhancement` | title/body describe a concrete compute-pressure feature |
| #609 `test(inkernel): FAK_GGUF-gated end-to-end test` | `enhancement` | body scopes adding a gated end-to-end test |
| #734 `feat(benchmark): measure kernel-in-the-loop guard-hop overhead` | `substrate` | body scopes the kernel guard hop and prompt-cache preservation gate |
| #729 `feat(dogfood): activate the always-on guarded fleet on node-macos-a` | `dispatch` | body scopes guarded fleet workers, launchd activation, and audit journals |
| #732 `feat(infra): stand up the GCP Tier-2 always-on dogfood control VM` | `dispatch` | body scopes the always-on guarded dispatch fleet and systemd service |
| #731 `feat(dogfood): drive dogfood_coverage to grade A` | `dispatch` | body scopes recurring guarded-fleet coverage and audit rows |
| #730 `feat(dogfood): extend kernel-guard coverage to the opencode/GLM dispatch lane` | `dispatch` | body scopes `dispatch_worker.py`, the opencode lane, and guard wrapping |
| #564 `feat(ctxplan): IDF/selectivity-weighted inverted index` | `substrate` | ctxplan parent work is substrate and this row changes the context-planner index |
| #563 `feat(ctxplan): adaptive budget` | `substrate` | body scopes context-planner working-set budget behavior |
| #562 `feat(ctxplan): fold index-bounded planner-compute into the scaling model` | `substrate` | body scopes context-planner compute accounting and oracle measurement |
| #561 `feat(ctxplan): reuse a Forecast plan across stable turns` | `substrate` | body scopes cached context-planner plan templates |
| #560 `feat(ctxplan): cost-based access-path selection` | `substrate` | body scopes context-planner access-path selection |
| #557 `feat(ctxplan): expose the planner as an agent-callable tool` | `substrate` | body scopes the adjudicated context-planner surface |
| #714 `test(snapshot): pin the snapshot/fleet codecs bit-exact` | `substrate` | body scopes snapshot/fleet codec determinism witnesses |
| #710 `epic(stability): keep the stability scorecard alive` | `substrate` | body scopes scorecard and rollback/stability substrate |
| #656 `docs(model-routing): document Subject/Match` | `documentation`, `model` | body scopes modelroute matching semantics documentation |
| #653 `docs(comm): document dos-arbitrate lane lease as MPI_Comm_split` | `documentation`, `substrate` | body scopes lane leases, ShareScope, and topobench communication substrate |
| #630 `feat(session): stream drive-state revisions` | `agentic-serving` | body scopes live session state changes on the served agent path |
| #629 `feat(session): persist drive state across a cold resume` | `agentic-serving` | body scopes session control state across agent resumes |
| #628 `feat(session): compose Pace.MaxTokensPerTurn into budgets` | `agentic-serving` | body scopes session pacing in ctxplan and model budgets |
| #633 `epic(edge): make fak compelling to mobile / edge / IoT platform owners` | `security` | body leads with the capability gate, audit trail, and compliance story |
| #627 `feat(session): multi-session scheduler` | `agentic-serving` | body scopes in-process live-session scheduling for the gateway |
| #626 `feat(gateway): thread session.Decide into the served turn` | `agentic-serving` | body scopes gateway session admission before model calls |
| #583 `idea-scout: Kamera unified multimodal KV cache` | `model-arch` | body scopes KV cache reuse and multimodal cache architecture research |
| #345 `examples: context-debugger walkthrough` | `substrate` | body scopes the `fak debug` core-image/debugging substrate |
| #341 `examples: author-a-custom-trace walkthrough` | `substrate` | body scopes trace replay and Tier-0 kernel run substrate |
| #338 `examples: grammar auto-repair demo` | `substrate` | body scopes preflight grammar repair and transform rungs |
| #337 `examples: admit_and_log posture demo` | `security` | body scopes policy posture and capability-floor behavior |
| #333 `examples: vDSO cache-hit demo` | `substrate` | body scopes the vDSO cache-hit fast path |
| #331 `examples: fanbench adoption walkthrough` | `agentic-serving` | body scopes fan-out/subagent workload benchmarking |
| #330 `examples: fak turntax adoption walkthrough` | `substrate` | body scopes the kernel turn-tax measurement tool |
| #322 `examples: RadixAttention adoption walkthrough` | `model-arch` | body scopes radix-tree KV-cache reuse |
| #321 `examples: fak agent live A/B adoption walkthrough` | `agentic-serving` | body scopes live agent/model A/B runs through the kernel |
| #319 `examples: fak bench subsystem-latency walkthrough` | `substrate` | body scopes in-process adjudication versus hook-spawn substrate latency |
| #318 `examples: dogfood-claude adoption-shaped wrapper` | `agentic-serving` | body scopes Claude Code over the Anthropic wire through fak |
| #313 `examples: pre-flight ladder demonstration` | `substrate` | body scopes parse/schema/grammar preflight rungs |
| #309 `obs(grafana): dashboard panels show No data` | `substrate` | body scopes observability dashboard and metric substrate |
| #365 `gpu(compute): buffer recycle pool is size-keyed only` | `priority/P1` | body describes silent host-visible/device-local cross-recycle defeating residency guarantees |
| #364 `gpu(compute): residency-budget dlUsed never resets/decrements` | `priority/P1` | body describes repeat model loads progressively spilling or tripping budget in long-lived processes |
| #362 `gpu(compute): single allocations over device cap` | `priority/P1` | body says 7B+ panics on single allocations above per-resource caps |
| #363 `gpu(cuda): no residency budget or OOM fallback` | `priority/P1` | body says 7B Q8 will OOM-panic on the 8 GB laptop without fallback |
| #310 `obs(audit): NS_INCLUDE_PREFIX default silently zeros audit` | `priority/P1` | body says seven dashboard panels and FleetTokenWaste are dead on non-operator hosts |
| #412 `serve(install): port --install to dogfood-claude.ps1` | `priority/P2` | body scopes cross-platform install parity and launcher survivability |
| #387 `RSI: DOS verify-referee binds fleet trailer ship-stamp` | `priority/P2` | body scopes false-negative risk in the self-improve truth gate |
| #483 `gpu(compute): CUDA Graphs / capture for batch-1 decode` | `priority/P2` | performance child work, important but not a crash/blocker |
| #710 `epic(stability): keep the stability scorecard alive` | `priority/P2` | stability follow-up epic with zero-debt baseline and frontier soft signals |
| #707 `feat(compute): feed live DeviceCapacity pressure` | `priority/P2` | child plank of the hardware-capacity bridge, not the panic fix itself |
| #706 `epic(compute): hardware-capacity bridge` | `priority/P2` | coordination epic; the direct CUDA/Vulkan panic children carry the higher urgency |
| #703 `feat(kernel): surface retry_after on WAIT` | `priority/P2` | child of `priority/P2` rate-limit epic #699 |
| #702 `feat(ratelimit): emit retry-after hint` | `priority/P2` | child of `priority/P2` rate-limit epic #699 |
| #701 `feat(policy): carry rate_limit through Runtime` | `priority/P2` | child of `priority/P2` rate-limit epic #699 |
| #700 `feat(policy): add rate_limit to Manifest` | `priority/P2` | child of `priority/P2` rate-limit epic #699 |
| #697 `feat(observability): add fak rungstats CLI` | `priority/P2` | child of `priority/P2` rung-observability epic #693 |
| #696 `feat(observability): expose rung decisions on /metrics` | `priority/P2` | child of `priority/P2` rung-observability epic #693 |
| #695 `feat(observability): rungobs vDSO bucket` | `priority/P2` | child of `priority/P2` rung-observability epic #693 |
| #694 `feat(observability): add internal/rungobs` | `priority/P2` | child of `priority/P2` rung-observability epic #693 |
| #691 `feat(kernel): handle result-side VerdictDeny` | `priority/P2` | child of `priority/P2` result-scope epic #687 |
| #690 `feat(ifc): fail-closed quarantine on unreadable share target` | `priority/P2` | child of `priority/P2` result-scope epic #687 |
| #689 `test(ifc): prove default ScopeAgent no-op` | `priority/P2` | child of `priority/P2` result-scope epic #687 |
| #688 `feat(ifc): add ScopeCeilingGate ResultAdmitter` | `priority/P2` | child of `priority/P2` result-scope epic #687 |
| #686 `feat(shipgate): skip worktree+suite when irrelevant` | `priority/P2` | child of `priority/P2` EvidenceProfile epic #680 |
| #685 `feat(rsiloop): harness proves candidate class` | `priority/P2` | child of `priority/P2` EvidenceProfile epic #680 |
| #684 `test(shipgate): per-class determinism proofs` | `priority/P2` | child of `priority/P2` EvidenceProfile epic #680 |
| #683 `feat(shipgate): ClassProofCarrying keeps on gain` | `priority/P2` | child of `priority/P2` EvidenceProfile epic #680 |
| #682 `feat(shipgate): ClassDocsOnly keeps on truth-clean` | `priority/P2` | child of `priority/P2` EvidenceProfile epic #680 |
| #681 `feat(shipgate): add EvidenceProfile + Witness.Class` | `priority/P2` | child of `priority/P2` EvidenceProfile epic #680 |
| #673 `feat(adjudicator): promotion offer emits reviewable record` | `priority/P2` | child of `priority/P2` promotion-ledger epic #669 |
| #672 `feat(adjudicator): promotion ledger counting clean events` | `priority/P2` | child of `priority/P2` promotion-ledger epic #669 |
| #671 `feat(adjudicator): would_deny record carries rung` | `priority/P2` | child of `priority/P2` promotion-ledger epic #669 |
| #670 `feat(adjudicator): per-tool complain set` | `priority/P2` | child of `priority/P2` promotion-ledger epic #669 |
| #668 `feat(observability): surface deciding/elided rung` | `priority/P2` | child of `priority/P2` RungProfile epic #663 |
| #667 `feat(adjudicator): read-class default profile` | `priority/P2` | child of `priority/P2` RungProfile epic #663 |
| #666 `feat(adjudicator): wire RungProfile into Adjudicate` | `priority/P2` | child of `priority/P2` RungProfile epic #663 |
| #665 `feat(adjudicator): RungProfile type` | `priority/P2` | child of `priority/P2` RungProfile epic #663 |
| #664 `feat(adjudicator): risk-class function` | `priority/P2` | child of `priority/P2` RungProfile epic #663 |
| #738 `feat(taskmgr): distinguish claimed progress from DOS-witnessed completion` | `substrate` | body scopes task-manager snapshot evidence and witness integration |
| #737 `feat(taskmgr): expose live process snapshots from fak serve/guard` | `substrate` | body scopes the process-local task-manager substrate and gateway/operator surface |
| #736 `feat(taskmgr): define default concept vocabulary` | `substrate` | body scopes `internal/taskmgr` concept helpers and stable aggregation |
| #735 `feat(taskmgr): add snapshot validation and schema invariants` | `substrate` | body scopes task-manager snapshot contract validation |
| #739 `epic(session): budget-triggered human-like session reset` | `agentic-serving`, `priority/P2` | served-agent session reset follow-up; important, but not a current P0/P1 blocker |
| #740 `feat(session): live same-model KV-included reuse` | `agentic-serving`, `priority/P2` | session reset child work for reuse across the served agent path |
| #741 `feat(session): model-call task distiller` | `agentic-serving`, `priority/P2` | session reset child work for turn/task distillation |
| #742 `feat(mcp): fak_session_reset tool variant` | `agentic-serving`, `priority/P2` | session reset integration surface for MCP-style agents |
| #743 `feat(session): operator webhook + pre-threshold warning` | `agentic-serving`, `priority/P2` | operator-facing served-session control follow-up |
| #744 `feat(session): cross-session/fleet-wide budget pool` | `agentic-serving`, `priority/P2` | fleet/session budget coordination follow-up |
| #295 `feat(gpu): Multi-GPU Tensor Parallelism [A-007]` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #45 `feat(gateway): fleet router skeleton` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #298 `feat(model): Production Llama 3.x Checkpoints [A-004]` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #294 `feat(model): HuggingFace Hub Direct Loading [A-008]` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #289 `perf(prefill): Close Prefill Throughput Gap [B-001]` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #287 `perf(vulkan): Vulkan Backend Optimization [B-002]` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #255 `feat(benchmark): N=100-1000 Agent Benchmark [D-001]` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #50 `epic(serving): large-scale disaggregated serving` | `help wanted` | top P0 orphan; no assignee, needs an owner |
| #365 `gpu(compute): buffer recycle pool is size-keyed only` | `help wanted` | top P1 orphan bug; no assignee, needs an owner |
| #364 `gpu(compute): residency-budget dlUsed never resets/decrements` | `help wanted` | top P1 orphan bug; no assignee, needs an owner |
| #363 `gpu(cuda): no residency budget or OOM fallback` | `help wanted` | top P1 orphan bug; no assignee, needs an owner |
| #362 `gpu(compute): single allocations over device cap` | `help wanted` | top P1 orphan bug; no assignee, needs an owner |
| #310 `obs(audit): NS_INCLUDE_PREFIX default silently zeros audit` | `help wanted` | top P1 orphan bug; no assignee, needs an owner |

Additional verified `help wanted` rows from the later top-priority orphan
visibility pass: #440, #438, #430, #399, #388, #78, #71, #70, #69, #67, #64,
#60, #59, #58, #21, #20, #43, #41, #40, #39, #36, #35, #34, #33, #738, #733,
#729, #717, #715, #648, #647, #645, #644, #642, #641, #640, #639, #637, #596,
#595, #18, #9, and #8. These labels surface recruitable high-priority work; they
do not reduce the orphan count until an assignee, `in-progress`, or explicit
defer marker exists.

## Top priority review rows

Current P0 rows surfaced by the triage helper:

| Issue | Why it is on top now |
|---|---|
| #295 `feat(gpu): Multi-GPU Tensor Parallelism [A-007]` | P0, orphan |
| #45 `feat(gateway): fleet router skeleton - N-upstream dispatch + static replica registry` | P0, orphan |
| #298 `feat(model): Production Llama 3.x Checkpoints [A-004]` | P0, orphan |
| #294 `feat(model): HuggingFace Hub Direct Loading [A-008]` | P0, orphan |
| #289 `perf(prefill): Close Prefill Throughput Gap [B-001]` | P0, orphan |
| #287 `perf(vulkan): Vulkan Backend Optimization [B-002]` | P0, orphan |
| #255 `feat(benchmark): N=100-1000 Agent Benchmark [D-001]` | P0, orphan |
| #50 `epic(serving): large-scale disaggregated serving` | P0, orphan |

Near-term P1 rows that connect directly to durable loops and learning/skill work:

| Issue | Why it matters to this refresh |
|---|---|
| #388 `RSI: run the dos-self-improve loop end-to-end on fak` | Keeps the improvement loop honest by witness, not narration |
| #715 `epic(cache): vCache - a virtual API cache over providers we don't control` | Current top prompt-cache epic |
| #716 `feat(cache): vCache M1 - observe & calibrate` | First measured rung for virtual cache work |
| #717 `feat(cache): vCache M2 - star anchors` | Next reusable-context rung after measurement |
| #639 `epic(comm): MPI-shaped message-passing primitives for fak agent fleets` | Coordination substrate for multi-agent work |
| #646 `feat(region): typed adjudicated one-sided shared window` | Shared durable/typed state primitive |
| #647 `feat(gitgate): collective-commit barrier` | Multi-agent trunk-write consistency |

## Durable loops to keep this organized

Use these loops in this order:

1. **Issue garden:** run `/issue-triage` or `python tools/issue_triage.py` to
   regenerate the ranked report and actions manifest. If mechanical actions are
   zero, do not invent closes; surface the top review-only rows.
2. **Priority review:** resolve the largest review-only gap first. Today that is
   missing priority/area labels plus orphan P0/P1 ownership.
3. **Dispatch:** after review, use the witness-gated issue-dispatch loop so each
   worker targets one issue and any close requires a `#N` commit plus
   `dos commit-audit`.
4. **Learning refresh:** keep `LEARNING-PATH.md` pointed at the real loops agents
   must learn: issue triage, dispatch witness, RSI keep-bit, and
   skill-context memory.
5. **Durable skill context:** use `SkillContextRecord` semantics for repeatable
   skill runs. Cache the assembled skill context by exact invocation identity,
   but re-check external effects before applying a label, closing an issue, or
   keeping an improvement.

## Done condition for the next pass

The next gardening pass should be able to prove at least one of these:

- the `needs_priority`, `needs_area`, or `needs_kind` counts fell from this
  baseline of 150 / 118 / 1;
- the orphan P0/P1 count fell from 72 by explicit claim or explicit defer;
- a top P0/P1 issue was resolved by a `#N`-cited commit and witnessed through
  `dos commit-audit`;
- a learning/skill-loop doc changed and its scorecard or relevant test reran.
