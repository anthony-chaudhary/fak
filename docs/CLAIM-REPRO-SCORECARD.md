---
title: "fak claim-reproducibility scorecard — are claims falsifiable from a clean clone?"
description: " fak's deterministic claim-reproducibility scorecard: validates that every witness in CLAIMS.md and BENCHMARK-AUTHORITY.md resolves to a real artifact, test, or command path."
---

# Claim-reproducibility scorecard

<!-- claim-repro-scorecard: 2026-06-28 · process: tools/claim_repro_scorecard.py -->

This scorecard validates that every witness handle in ``CLAIMS.md`` (``[SHIPPED]``/``[SIMULATED]``/``[STUB]`` claims) and every artifact path or ``Reproduce:`` command in ``BENCHMARK-AUTHORITY.md`` is **resolvable by an outsider from a clean clone**. An un-falsifiable claim — a ``Witness: TestFooBar`` that names a non-existent test, or a ``Reproduce: go run ./cmd/gone`` pointing at a deleted binary — is the worst failure mode for a skeptical reader, because it looks checkable and isn't.

> Regenerate: ``python tools/claim_repro_scorecard.py --markdown --stamp DATE > docs/CLAIM-REPRO-SCORECARD.md``

## Headline

| Metric | Value |
|---|---|
| **Un-falsifiable claims (total HARD defects)** | **79** |
| Composite score | 0.0/100 (grade F) |
| Advisory (soft) signals | 0 |

## Per-KPI

Two KPIs, each 0–100. ``debt`` = units of HARD un-falsifiable claims in that KPI.

| KPI | Score | Debt | Detail |
|---|---:|:--:|---|
| ``claims`` | 0 | 44 | 159 claims, 44 un-falsifiable |
| ``benchmarks`` | 0 | 35 | 151 benchmarks, 35 un-falsifiable |

## Un-falsifiable claim work-list

### ``claims`` — 44 defect(s), score 0
- un-falsifiable claim: - [SHIPPED] The whole module passes the Go data-race detector with **zero data races**, enforced by the `race-detector`  — missing package path: ...
- un-falsifiable claim: - [SHIPPED] `recall`: a finished session persists as a durable **core image** — `manifest.json` (the page table: roles + — missing artifact: manifest.json, missing artifact: cas.json, missing artifact: RECALL-RESULTS.md
- un-falsifiable claim: - [SHIPPED] `cdb`: the **context debugger** — `IngestSession` turns a REAL Claude Code transcript into a core image (one — missing artifact: CDB-RESULTS.md
- un-falsifiable claim: - [SHIPPED] ECC-style metadata integrity for recall cells (#783/#785, epic #782): the CAS already self-verifies a page's — test pattern ''Syndrome|ClassifyFault|FaultClass|Verify'' not found in package internal/recall
- un-falsifiable claim: - [SHIPPED] Served-session long-context reset budget: `session.Budget.ContextTokensLeft` is debited from provider-normal — test pattern '"TestContextBudget|TestSessionContextBudget|TestTraceForUsesConfiguredDefaultTrace|TestSetDefaultTraceIDAdvancesOmittedCallerTrace|TestBudgetExhaustedCallbackReceivesServedTranscript|TestSessionCLIContextBudget|TestDebitSessionHookDebitsContextBudget|TestResetOnBudgetContinuesTransparently|TestResetServedSessionOnBudgetRecontinuesWithCarryover|TestMCPSessionResetDebitsAndRearmsContinuation|TestGuardBudgetRestarterRecontinuesAndEmitsSeed|TestBuildGuardChildIncludesRestartEnv|TestGuardRestartSeedFileAndEnv"' not found in package internal/session
- un-falsifiable claim: - [SHIPPED] `sessionimage`: a portable, versioned, **model-agnostic** SESSION image — composes the drive (`session.json` — missing artifact: session.json, missing artifact: manifest.json, missing artifact: cas.json, missing artifact: index.json
- un-falsifiable claim: - [SHIPPED] The RESUME-CACHE decision is a first-class, deterministic, observable verb — the priced answer to "I am resu — test pattern '"TestHeadline250kColdResume|TestWarmResume|TestUnknownIdle|TestBreakEvenMatchesExplainer|TestResumePlan"' not found in package internal/resume
- un-falsifiable claim: - [SHIPPED] The shared task contract has executable docs and fixtures: `tools/shared_task_contract.py` validates JSON ex — missing artifact: python tools/shared_task_contract.py validate-doc docs/shared-task-record-contract.md
- un-falsifiable claim: - [SHIPPED] Rung-1 default-expire promotion gate: `recall.Page` carries a `Durability` field, and a `PromotionMode` gate — missing artifact: manifest.json, missing artifact: cas.json
- un-falsifiable claim: - [SHIPPED] A pure-Go SmolLM2-135M forward pass (134.5M params / 272 tensors) runs in-process with the **KV cache as a k — missing artifact: IN-KERNEL-MODEL-RESULTS.md
- un-falsifiable claim: - [SHIPPED] Multi-node compute is runnable, not just loopback-tested: `fak cluster` runs a real cross-NODE collective (A — test pattern ''DistComm|Pipeline|TP'' not found in package internal/model
- un-falsifiable claim: - [SHIPPED] Parity lane: parallel matmul across output rows + batched prefill GEMM + an 8-accumulator `fdot`, each outpu — missing artifact: MODEL-BASELINE-RESULTS.md
- un-falsifiable claim: - [SHIPPED] KV-quarantine bridge: a ctxmmu `Quarantine` verdict on poison bytes mechanically **evicts that result's K/V  — missing artifact: KV-QUARANTINE-BRIDGE-RESULTS.md
- un-falsifiable claim: - [SHIPPED] Planned view MEASURED over the heaviest REAL session transcripts (issue #559, `cmd/ctxplanbench`): the empir — missing cmd dir: cmd/ctxplanbench -selfcheck, missing cmd dir: cmd/ctxplanbench -heaviest 5
- un-falsifiable claim: - [SHIPPED] Planner per-turn COMPUTE flatten MEASURED over the heaviest REAL session transcripts (issues #558/#559, `cmd — missing cmd dir: cmd/ctxplanbench -selfcheck
- un-falsifiable claim: - [SHIPPED] Provable-deletion certificate (`internal/deletioncert`, demo `cmd/deletioncert`): a `DeletionCertificate` bi — missing cmd dir: cmd/deletioncert -selfcheck
- un-falsifiable claim: - [SHIPPED] Poly-model serving core (`internal/polymodel`, foundation leaf, stdlib-only): the deterministic **"host many — missing cmd dir: cmd/polymodelbench -selfcheck
- un-falsifiable claim: - [SHIPPED] RadixAttention parity vs SGLang: SGLang's KV-cache radix attention (radix tree of token sequences + runtime  — missing artifact: experiments/radixattention/*.json, missing artifact: RADIXATTENTION-RESULTS.md
- un-falsifiable claim: - [SHIPPED] `fak guard -- <agent>` is the one-command adopter front door that collapses the dogfood path (a shell launch — missing artifact: settings.json
- un-falsifiable claim: - [SHIPPED] **Serving-latency observability — percentile-capable TTFT / TPOT / end-to-end histograms on `/metrics`.** Th — test pattern 'TestInferenceLatencyHistograms' not found in package internal/gateway, test function not found: TestInferenceLatencyHistograms
- un-falsifiable claim: - [SHIPPED] **Oversized tool_result elision — the bounded-loss `req.Raw` byte-splice sibling of compaction (`--elide-res — test pattern ''TestElide|TestObjectValueSpan'' not found in package internal/agent
- un-falsifiable claim: - [SHIPPED] `fak turntax` replays a class-labeled trace through the real kernel and prices the extra error-code MODEL tu — missing artifact: TURN-TAX-RESULTS.md
- un-falsifiable claim: - [SHIPPED] `fak ablate` generalizes the 2-arm `fak bench` (vDSO on/off) into an N-ARM feature sweep: it replays ONE fro — missing cmd dir: cmd/fak ablate --sweep vdso
- un-falsifiable claim: - [SHIPPED] **`tools/cross_agent_ablate.py` ran the first live cross-agent (Regime B) ablation (epic #607 rung 3, #623)  — missing artifact: RESULT.txt, missing artifact: python tools/cross_agent_ablate.py report --reps experiments/ablate/cross-agent-pong-opus.json
- un-falsifiable claim: - [SHIPPED] `fanbench` sweeps the one-master-goal → N-subagent fan-out (the orchestrator-worker / lead-subagent topology — missing artifact: FANOUT-BENCH-RESULTS.md
- un-falsifiable claim: - [SIMULATED] The token-multiplier, prefix-cache `tax_clawed_back` (~62% at the N≈256 plateau on the default cost model) — missing artifact: FANOUT-BENCH-RESULTS.md
- un-falsifiable claim: - [SHIPPED] `fanrun` (`cmd/fanrun`, engine `internal/bench/fanrun.go`) is the MEASURED live capstone to `fanbench`'s mod — missing artifact: FANOUT-BENCH-RESULTS.md
- un-falsifiable claim: - [SIMULATED] The fan-out **task-quality** litmus (`tools/fanout_taskquality.py`, artifact `experiments/fanout/taskquali — missing artifact: LIVE-RESULTS.md, missing artifact: fanbench-research.csv, missing artifact: FANOUT-BENCH-RESULTS.md
- un-falsifiable claim: - [SHIPPED] `longctxbench` / `turnbench.RunLongContextLadder` compute the EXACT, contention-free work floor for the >100 — missing artifact: ULTRA-LONG-CONTEXT-RESULTS.md
- un-falsifiable claim: - [SHIPPED] **Capacity pressure sweep — report→policy→execute loop for KV pressure.** `internal/engine.RunCapacityPressu — test pattern '"Test(CapacityPressureSweep|PlanPlacementForDevice)"' not found in package internal/engine
- un-falsifiable claim: - [SHIPPED] **Classed capacity fit + GGUF serve load-time refusal (Plank 1/5 of the capacity bridge).** `compute.MemoryP — test pattern '"Test(DeviceMemoryInfo|HostMemoryInfo|HostSystemMemoryProbe|CapacityProbe|DeviceCapacity|HostCapacity|FitsOnDevice|FitsOnHost|FreeUnknown|FitVerdict|Refuse|MemoryPlan|EstimateKVStore|EstimateHALTransient)"' not found in package internal/compute, test pattern '"Test(Estimate|Fit)"' not found in package internal/ggufload, test pattern '"Test(GatewayLoadProfileCarriesServeMemoryPlanAndCapacity|FitServeGGUFOnDevice|ServeGGUFMemoryPlan|ServeGGUFCPUOffload)"' not found in package cmd/fak, test pattern '"TestModelLoadMetricsSuppressedUntilSet"' not found in package internal/gateway
- un-falsifiable claim: - [SHIPPED] **Classed runtime device OOM recovery, request-time fit refusal, and visibility on the served in-kernel path — test pattern '"Test(HALHostUploadsCarryMemoryClass|BackendKernelRuntimeUploadsCarryActivationClass|PagedKernelPageInUsesOffloadClass)"' not found in package internal/model, test pattern '"Test(RecoverDevicePanic|PrepareDeviceOOMRetry|InKernelRequest|InKernelOOMRetryStats|InKernelRequestPressureTrim)"' not found in package internal/agent, test pattern '"Test(InKernelOOMMetricsAndDebugVars|InKernelOOMRetryMetricsAndDebugVars|InKernelPressureTrimMetricsAndDebugVars|UpstreamErrorStatus)"' not found in package internal/gateway
- un-falsifiable claim: - [SHIPPED] **Request-time in-kernel capacity precheck.** Before a served device-HAL `Complete` runs prefill/decode, `ag — test pattern '"Test(InKernelRequest|InKernelRequestPressureTrim)"' not found in package internal/agent, test pattern '"Test(RequestMemoryMetricsAndDebugVars|RequestMemoryAggregateMetricsAndDebugVars|InKernelPressureTrimMetricsAndDebugVars|InKernelOOMMetricsAndDebugVars|UpstreamErrorStatus_InKernelCapacity)"' not found in package internal/gateway
- un-falsifiable claim: - [SHIPPED] **Resident KV-prefix memory visibility.** The in-kernel planner now implements an optional `agent.KVMemoryRe — test pattern '"Test(KVMemoryMetrics|MetricsExposesKVPrefixReuse)"' not found in package internal/gateway, test pattern '"TestInKernelKVMemoryStats"' not found in package internal/agent, test pattern '"TestHostSystemMemoryProbeIsSaneWhenAvailable"' not found in package internal/compute
- un-falsifiable claim: - [SHIPPED] **Memory-classed engine cache-event visibility for DDR cache tiers.** `engine.CacheEventMetrics` now project — test pattern '"Test(CacheTierMemoryClassProjection|CacheEventMetricsExposedAsPrometheus)"' not found in package internal/engine
- un-falsifiable claim: - [SHIPPED] **AMD GPU backend (Vulkan compute).** A `//go:build vulkan` `compute.Backend` (`internal/compute/vulkan*` +  — missing artifact: VULKAN-AMD-RESULTS.md
- un-falsifiable claim: - [SHIPPED] **llama.cpp parity — numerical YES, throughput NO.** The Vulkan f32 path is correct but ~37× slower than lla — missing artifact: VULKAN-AMD-RESULTS.md
- un-falsifiable claim: - [SHIPPED] The live seam is now exercised end-to-end: `fak agent` drove this kernel with a real OpenAI-compatible model — missing artifact: experiments/agent-live/*.json, missing artifact: LIVE-RESULTS.md
- un-falsifiable claim: - [SHIPPED] Issue-dispatch **closed loop** (`tools/issue_*`, `tools/dispatch_*`): the witness-gated GitHub-issue backlog — missing artifact: .dispatch-runs/dispatch-status.md
- un-falsifiable claim: - [SHIPPED] Harvest-corpus **advisory adjudication model** (`internal/advmodel`, #580): the consumer of the `internal/ha — missing artifact: testdata/adjudicator.json
- un-falsifiable claim: - [SHIPPED] The vCache **chains & recall decision engine** (`internal/vcachechain`, #719) — the M4 milestone: the prefix — test pattern '"TestProveRecall|TestPlanRecall|TestRebuild|TestBreakEven|TestPlaceBreakpoints|TestTopologicalReplay|TestPrefixDAG|TestRunVCacheProveRecall"' not found in package internal/vcachechain
- un-falsifiable claim: - [SHIPPED] The vCache **Governor decision engine** (`internal/vcachegov`, #720) — the steady-state policy that turns th — test pattern '"TestRunVCache|TestProve"' not found in package internal/vcachegov
- un-falsifiable claim: - [SHIPPED] The vCache **readiness scorecard** (`internal/vcachescore`, the `fak vcache score` verb; #789 dogfood, #791  — test pattern '"TestRunVCacheScore|TestEconomicsBlock|TestTelemetryOverrides|TestDefaultScore"' not found in package internal/vcachescore, missing artifact: go run ./cmd/fak vcache score --telemetry experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl --json --out experiments/agent-live/vcache-score-codex-telemetry-2026-06-25.json
- un-falsifiable claim: - [SHIPPED] The vCache **per-sub-concept observability lens** (`internal/vcacheobserve`, the `fak vcache observe` verb)  — test pattern '"Observe"' not found in package internal/vcacheobserve

### ``benchmarks`` — 35 defect(s), score 0
- un-falsifiable benchmark: | **fak CPU Q8 single-stream vs llama.cpp CPU (M3 Pro)** — CANONICAL | **decode 0.55–0.73× (fak 38.1; llama 68.7 @−t6 →  — missing artifact: model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json
- un-falsifiable benchmark: | **RadixAttention live speedup (model ladder)** | **4.58× → 6.95×** | SmolLM2-135M → Qwen2.5-1.5B Q8 | Full re-prefill  — missing artifact: radixbench-*-agents-fresh-20260619.json
- un-falsifiable benchmark: | **README headline: 50-turn × 5-agent reuse win** | **60.3× vs naive · 4.1× vs tuned** | Qwen2.5-1.5B Q8, T=50 A=5 P=20 — missing artifact: headline-qwen-50x5.json
- un-falsifiable benchmark: | **Fleet 5-agent × 200-turn 7B in <10 min — on a Metal forward (MEASURED, M3 Pro)** | **8.2 min (llama.cpp Metal forwar — missing artifact: session/macbook-m3pro-7b-batched-{bench,ctx}.log, missing artifact: fleet-5x200-7b-projection-20260622.json, missing artifact: FLEET-5X200-7B-10MIN-RESULTS.md
- un-falsifiable benchmark: | **Session value-add (high-T ladder)** | **24.9× → 139.3×** | SmolLM2-135M Q8, T=64 → T=512 | Naive stateless | `92896a — missing artifact: highT-smollm2-135m-*-fresh-20260619.json
- un-falsifiable benchmark: | Session value-add (1.5B "realistic model") | 7.2× → 10.0× | Qwen2.5-1.5B Q8, T=8 → T=16 | Naive stateless | `92896a4`  — missing artifact: smoke-qwen2.5-1.5b-T8-16-fresh-20260619.json
- un-falsifiable benchmark: | Session value-add (SmolLM2 P=512, re-measured) | **5.3–7.4×** | SmolLM2-135M Q8 | Naive stateless | `885ae8a` | `bench — missing artifact: benchmark-run-opencode-20260619/sessionbench-smollm2-135m-q8-authority.json
- un-falsifiable benchmark: | Qwen2.5-7B fak decode | 8.7 tok/s | Qwen2.5-7B Q8 | llama.cpp Metal 17.6 tok/s | `34c74f4` | `model-ladder/modelbench- — missing artifact: model-ladder/modelbench-qwen25-7b-q8.json
- un-falsifiable benchmark: | Qwen2.5-7B fak/llama.cpp ratio | 0.50× decode / 0.083× prefill | Qwen2.5-7B Q8 | llama.cpp Metal | `34c74f4` | `QWEN25 — missing artifact: QWEN25-7B-RESULTS.md
- un-falsifiable benchmark: | Qwen2.5-7B greedy parity | ✅ full 7-token match | Qwen2.5-7B Q8 | llama.cpp ("2+2 is 4.") | `34c74f4` | `QWEN25-7B-RES — missing artifact: QWEN25-7B-RESULTS.md
- un-falsifiable benchmark: | Qwen3.5-0.8B hybrid-GDN runs in fak | ✅ coherent ("pong") | Qwen3.5-0.8B f32 | instruction-following | `6a376b8` | `QW — missing artifact: QWEN35-0.8B-RESULTS.md
- un-falsifiable benchmark: | Qwen3.6-27B fak/llama.cpp ratio | 0.12× decode / 0.01× prefill | Qwen3.6-27B q4_k_m | llama.cpp Metal | `1698eff` | `m — missing artifact: model-ladder/qwen36-perf-gate-m3-20260619.md
- un-falsifiable benchmark: | Qwen3.6-27B fak **Metal Q4_K** decode / prefill | **1.2 / 0.6 tok/s** (0.16× / 0.012× of SOTA) | Qwen3.6-27B q4_k_m, M — missing artifact: runs/by-machine/node-macos-a/20260626T055239Z-q4k-metal-decode-27b/score.json
- un-falsifiable benchmark: | Qwen3.6-27B token parity | 2-token match (drift @3) | Qwen3.6-27B q4_k_m | llama.cpp oracle | `d03be46` | `model-ladde — missing artifact: model-ladder/qwen36-resident-q4k-parity-20260619.json
- un-falsifiable benchmark: | Qwen3.6-27B surface smoke | 4/4 surfaces PASS | Qwen3.6-27B (served) | agent/gateway/mcp/dogfood | `8a0f5bc` | `model- — missing artifact: model-ladder/qwen36-surfaces-dogfood-opencode-20260619.json
- un-falsifiable benchmark: | **Qwen3.6-27B 8-GPU SGLang serving — fak-gateway vs raw-SGLang (model-ladder Rung 4 headline, [#921](https://github.co — missing artifact: COMPARE.md, missing artifact: fak-gateway.json, missing artifact: raw-sglang.json, missing artifact: surface-smoke.json
- un-falsifiable benchmark: | **Qwen3.6-27B cold standup on 8-GPU datacenter server — raw-SGLang serving baseline (Rung-4 fresh-standup witness, [#9 — missing artifact: STANDUP.md, missing artifact: samples.json
- un-falsifiable benchmark: | Synthetic model live ratio | 1.64× | 64h/4L wiring | Full re-prefill | `a200c3d` | `radixbench-synthetic.json` | — missing artifact: radixbench-synthetic.json
- un-falsifiable benchmark: | **GPU Q8 decode (Vulkan, RX 7600)** | **24.6 tok/s · 1.49× vs GPU f32** | SmolLM2-135M Q8 | Same forward, f32 weights  — missing artifact: q8gpu-smollm2-135m-{gpu-q8,gpu-f32}-20260619.json
- un-falsifiable benchmark: | **GPU/CPU Q8 decode crossover** | **CPU lead 7.2× (135M) → 1.16× (1.5B)** | SmolLM2-135M → Qwen2.5-1.5B Q8 | CPU Q8 (l — missing artifact: crossover-qwen2.5-1.5b-{gpu,cpu}-q8-20260619.json
- un-falsifiable benchmark: | **GPU decode parity (reusable CUDA graph, RTX 4070)** — README headline | **~120 tok/s (119–120, f32) · parity with ll — missing artifact: LLAMACPP-HEADTOHEAD-RESULTS.md
- un-falsifiable benchmark: | **Pure-kernel admission latency (M3 Pro)** | **1.8–14 µs** scan · 3.3–15.8 µs Admit · 29–87 µs chain | ctxmmu / normga — missing artifact: MAC-M3PRO-KERNEL-BENCH-2026-06-20.md
- un-falsifiable benchmark: | **Ultra-long-context work floor (>100k tokens, EXACT/contention-free)** | **single ~10× · 5-agent fleet ~40×+ vs naive — missing artifact: session/ultra-long-context-floor-20260622.json, missing artifact: ULTRA-LONG-CONTEXT-RESULTS.md
- un-falsifiable benchmark: | **Decode vs prefill worker-count scaling (x86_64 32-core, within-run ratio)** | **decode all-cores-default penalty 2.5 — missing artifact: WORKER-SCALING-DESKTOP-X86-20260624.md
- un-falsifiable benchmark: | **Self-ablation feature sweep — vDSO on/off (deterministic, Regime A of epic #607)** | **vdso_hits 0→7 · engine_calls  — missing artifact: ABLATE-RESULTS.md
- un-falsifiable benchmark: | **Cross-agent ablation — bare `claude` vs `fak guard -- claude` (Regime B of epic #607, [#623](https://github.com/anth — missing artifact: ABLATE-RESULTS.md
- un-falsifiable benchmark: | **ToolSandbox/tau3 policy-state adapter smoke ([SIMULATED] local fixture)** | **raw safe pass^1 1/2 (0.500) -> fak saf — missing artifact: go run ./cmd/toolsandboxbench -suite testdata/toolsandbox/policy_state_smoke.json -out experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json -md experiments/agent-live/toolsandbox-policy-state-smoke-20260625.md
- un-falsifiable benchmark: | **Local-model coding witness — `fak guard --gguf` (PENDING — path assembled, awaiting real run)** | **PENDING — run co — missing artifact: LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md
- un-falsifiable benchmark: | SmolLM2-135M (30L) | **4.58×** | 7.50× | 62.1% → 86.7% (100% of optimal) | `radixbench-smollm2-135m-q8-agents-fresh-20 — missing artifact: radixbench-smollm2-135m-q8-agents-fresh-20260619.json
- un-falsifiable benchmark: | SmolLM2-360M (32L) | **5.40×** | 7.50× | 62.1% → 86.7% | `radixbench-smollm2-360m-q8-agents-fresh-20260619.json` | — missing artifact: radixbench-smollm2-360m-q8-agents-fresh-20260619.json
- un-falsifiable benchmark: | Qwen2.5-0.5B (24L) | **6.20×** | 7.50× | 62.1% → 86.7% | `radixbench-qwen2.5-0.5b-q8-agents-fresh-20260619.json` | — missing artifact: radixbench-qwen2.5-0.5b-q8-agents-fresh-20260619.json
- un-falsifiable benchmark: | Qwen2.5-1.5B (28L) | **6.95×** | 7.50× | 62.1% → 86.7% | `radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json` | — missing artifact: radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json
- un-falsifiable benchmark: | **gpu-q8** | **24.6** | 15.6 → 24.8 | `q8gpu-smollm2-135m-gpu-q8-20260619.json` | — missing artifact: q8gpu-smollm2-135m-gpu-q8-20260619.json
- un-falsifiable benchmark: | gpu-f32 | 16.5 | 12.6 → 18.7 | `q8gpu-smollm2-135m-gpu-f32-20260619.json` | — missing artifact: q8gpu-smollm2-135m-gpu-f32-20260619.json
- un-falsifiable benchmark: | cpu-q8 | 176.9 | 969 → 1519 | `q8gpu-smollm2-135m-cpu-q8-20260619.json` | — missing artifact: q8gpu-smollm2-135m-cpu-q8-20260619.json

