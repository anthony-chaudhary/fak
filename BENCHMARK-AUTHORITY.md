# BENCHMARK AUTHORITY — Single Source of Truth

> **Why this exists.** This repo contains many benchmark results across different axes (raw throughput, reuse efficiency, session value-add, etc.). This document is the **authoritative index** of all committed benchmark claims, with traceability to source commits and artifact files. **Any number claimed elsewhere must trace back to an entry here.**

> **📋 Process:** See **[BENCHMARK-GOVERNANCE.md](BENCHMARK-GOVERNANCE.md)** for the DOS-centric process that creates, verifies, and publishes these claims. This file is the *what* (the numbers); Governance is the *how* (the discipline).

> **🏆 Presentation layer:** **[HERO-BENCHMARK-2026-06-21.md](HERO-BENCHMARK-2026-06-21.md)** is the frontier-lab-style *hero comparison* (v1) built **from** this authority — headline number, top-3 SOTA chart, top-10 leaderboard with fak bolded where it wins (and the two single-stream losses shown plainly). It claims no new numbers; every figure traces to a row below.

> **🧠 The *why*:** `WHY-REUSE-WINS-2026-06-21.md` (private companion — not published) (v1) argues — and stress-tests — *why* these reuse numbers matter more than the headline alone: reuse is a **different class** of optimization (work-elimination on the `N` axis, not work-acceleration on the `κ` axis), so it's **exact, training-free, and composes multiplicatively** on top of every per-token trick. Follows the v2 SOTA-only framing — leads with the absolute competitive number (**19.0 min vs 78 min = 4.1× less work**, conservative marginal 2.4–2.7×), shows **no naive-loop numbers**, and centers cross-agent reuse as the layer that is `fak`'s. No new numbers; fences where the "works across everything / no fine-tuning" framing is overstated (and where addressable reuse, [#228](https://github.com/anthony-chaudhary/fak/issues/228), widens it).

**Last updated:** 2026-06-25
**Status:** Living document — update when new model results ship

> **🔁 Provenance vs. public reproducibility (read before you `git show` a commit below).**
> The **Commit** column records the *private-lineage* result commit each number first shipped in.
> Those short-SHAs predate the public **v0.30.0** squash (`1029e37`) and **do not resolve in a
> public clone** — treat them as provenance, not as a public reproduce handle. What an outsider
> actually re-runs in this repo is the committed **artifact** (the JSON / `.md` under
> `experiments/…` and `docs/benchmarks/…`, all tracked here) plus the **Reproduce** command. The
> artifact + command are the verifiable anchor; the SHA is lineage. (A few historical paths below
> dropped their old monorepo `fak/` subdir prefix — the real tracked path is `experiments/…` /
> `docs/benchmarks/…`. Limitation shown plainly rather than left for you to trip on.)

---

## Quick Reference: Primary Numbers

| Claim | Number | Model | Baseline | Commit | Artifact |
|---|---|---|---|---|---|
| **fak CPU Q8 single-stream vs llama.cpp CPU (M3 Pro)** — CANONICAL | **decode 0.55–0.73× (fak 38.1; llama 68.7 @−t6 → 52.4 @−t12) · prefill@256 0.58× (240.4 vs 412.5)** | Qwen2.5-1.5B Q8, M3 Pro (uncontended) | llama.cpp CPU −ngl 0, build 8200 (`541bf3762`) | _this commit_ | `model-ladder/qwen25-1.5b-q8-cpu-parity-m3pro.json` ← **single source; read, don't hardcode** — 2026-06-23 refresh (HEAD `374776a`, fak 0.31.0). Conservative fence = decode **0.55×** (each engine at its best thread config); equal 12-thread budget = 0.73×. The prior inline `0.58×/0.45× (71.9/547)` cited a non-existent commit and mixed a llama −t6 decode with an older-build prefill — reconciled in `docs/notes/MAC-BENCH-REFRESH-2026-06-23.md` |
| **RadixAttention live speedup (model ladder)** | **4.58× → 6.95×** | SmolLM2-135M → Qwen2.5-1.5B Q8 | Full re-prefill | `92896a4` | `radixbench-*-agents-fresh-20260619.json` |
| RadixAttention token speedup | 7.50× | all four models (Q8) | Token count | `92896a4` | Same (`prefill_token_speedup`) |
| RadixAttention hit rate | 86.7% (FCFS 62.1% → cache-aware) | all four models (Q8) | Cache hits | `92896a4` | Same (100% of optimal) |
| **README headline: 50-turn × 5-agent reuse win** | **60.3× vs naive · 4.1× vs tuned** | Qwen2.5-1.5B Q8, T=50 A=5 P=2048 | Naive stateless / tuned per-agent KV | `2bbda6f` | `headline-qwen-50x5.json` |
| **Fleet 5-agent × 200-turn 7B in <10 min — on a Metal forward (MEASURED, M3 Pro)** | **8.2 min (llama.cpp Metal forward + fak's reuse/batching pattern) · 2.5× vs a tuned single-stream baseline · ≥30× vs naive. ⚠️ pure-fak OWN forward (pure-Go CPU, Metal-decode lane open) ≈ 22–51 min — over the bar; sub-10-min on fak's own forward needs its GPU/CUDA path** | Qwen2.5-7B Q8, T=200 A=5 P=2048 D=20 R=12, M3 Pro | 5 single-stream sessions / naive re-prefill | _this commit_ | `session/macbook-m3pro-7b-batched-{bench,ctx}.log` (measured 17.41/44/392 t/s) + `fleet-5x200-7b-projection-20260622.json` + `FLEET-5X200-7B-10MIN-RESULTS.md`. Batched ≈ a tuned `llama-server --parallel`; fak's add is per-agent KV ownership + safety floor, not raw t/s |
| **Session value-add (high-T ladder)** | **24.9× → 139.3×** | SmolLM2-135M Q8, T=64 → T=512 | Naive stateless | `92896a4` | `highT-smollm2-135m-*-fresh-20260619.json` |
| Session value-add (1.5B "realistic model") | 7.2× → 10.0× | Qwen2.5-1.5B Q8, T=8 → T=16 | Naive stateless | `92896a4` | `smoke-qwen2.5-1.5b-T8-16-fresh-20260619.json` |
| ~~Session value-add 11.2–14.5× (SmolLM2, P=512)~~ | ❌ STALE | SmolLM2-135M Q8 | Naive stateless | `5b0f40d` | superseded — see F1 below |
| Session value-add (SmolLM2 P=512, re-measured) | **5.3–7.4×** | SmolLM2-135M Q8 | Naive stateless | `885ae8a` | `benchmark-run-opencode-20260619/sessionbench-smollm2-135m-q8-authority.json` |
| Qwen2.5-7B fak decode | 8.7 tok/s | Qwen2.5-7B Q8 | llama.cpp Metal 17.27 tok/s | `34c74f4` | `model-ladder/modelbench-qwen25-7b-q8.json` |
| Qwen2.5-7B fak/llama.cpp ratio | 0.50× decode / 0.083× prefill | Qwen2.5-7B Q8 | llama.cpp Metal | `34c74f4` | `QWEN25-7B-RESULTS.md` |
| Qwen2.5-7B greedy parity | ✅ full 7-token match | Qwen2.5-7B Q8 | llama.cpp ("2+2 is 4.") | `34c74f4` | `QWEN25-7B-RESULTS.md` |
| Qwen3.5-0.8B hybrid-GDN runs in fak | ✅ coherent ("pong") | Qwen3.5-0.8B f32 | instruction-following | `6a376b8` | `QWEN35-0.8B-RESULTS.md` |
| Qwen3.6-27B resident-q4k decode | 0.9 tok/s | Qwen3.6-27B q4_k_m | llama.cpp Metal 7.29 tok/s | `1698eff` | `model-ladder/qwen36-perf-gate-m3-20260619.json` |
| Qwen3.6-27B fak/llama.cpp ratio | 0.12× decode / 0.01× prefill | Qwen3.6-27B q4_k_m | llama.cpp Metal | `1698eff` | `model-ladder/qwen36-perf-gate-m3-20260619.md` |
| Qwen3.6-27B fak **Metal Q4_K** decode / prefill | **1.2 / 0.6 tok/s** (0.16× / 0.012× of SOTA) | Qwen3.6-27B q4_k_m, M3 Pro (`-tags fakmetal`, `FAK_Q4K=1 FAK_METAL=1`, llama-server stopped) | llama.cpp Metal 7.29 / 51.55 | _this commit_ | `runs/by-machine/node-macos-a/20260626T055239Z-q4k-metal-decode-27b/score.json` — kernels are **bit-correct** (GEMV cosine 1.000000, greedy token-parity vs CPU) but decode is launch-bound: ~336 per-token command-buffer GEMVs @ ~360 µs fixed overhead each (epic #59 / #67), plus low kernel util (GEMV 32 GB/s ≈21% BW, GEMM 364 GFLOP/s ≈5%). Lever = one-command-buffer resident decode forward + kernel pass. Full diagnosis: `docs/notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md` |
| Qwen3.6-27B token parity | 2-token match (drift @3) | Qwen3.6-27B q4_k_m | llama.cpp oracle | `d03be46` | `model-ladder/qwen36-resident-q4k-parity-20260619.json` |
| Qwen3.6-27B surface smoke | 4/4 surfaces PASS | Qwen3.6-27B (served) | agent/gateway/mcp/dogfood | `8a0f5bc` | `model-ladder/qwen36-surfaces-dogfood-opencode-20260619.json` |
| Synthetic model live ratio | 1.64× | 64h/4L wiring | Full re-prefill | `a200c3d` | `radixbench-synthetic.json` |
| **GPU Q8 decode (Vulkan, RX 7600)** | **24.6 tok/s · 1.49× vs GPU f32** | SmolLM2-135M Q8 | Same forward, f32 weights on GPU | `60db592` | `q8gpu-smollm2-135m-{gpu-q8,gpu-f32}-20260619.json` |
| **GPU/CPU Q8 decode crossover** | **CPU lead 7.2× (135M) → 1.16× (1.5B)** | SmolLM2-135M → Qwen2.5-1.5B Q8 | CPU Q8 (legacy) | `7bf666b` | `crossover-qwen2.5-1.5b-{gpu,cpu}-q8-20260619.json` |
| Synthetic model live ratio | 1.64× | 64h/4L wiring | Full re-prefill | `a200c3d` | `radixbench-synthetic.json` |
| **GPU Q8 decode (Vulkan, RX 7600)** | **24.6 tok/s · 1.49× vs GPU f32** | SmolLM2-135M Q8 | Same forward, f32 weights on GPU | `60db592` | `q8gpu-smollm2-135m-{gpu-q8,gpu-f32}-20260619.json` |
| **GPU/CPU Q8 decode crossover** | **CPU lead 7.2× (135M) → 1.16× (1.5B)** | SmolLM2-135M → Qwen2.5-1.5B Q8 | CPU Q8 (legacy) | `7bf666b` | `crossover-qwen2.5-1.5b-{gpu,cpu}-q8-20260619.json` |
| **GPU decode parity (reusable CUDA graph, RTX 4070)** — README headline | **~120 tok/s (119–120, f32) · parity with llama.cpp Q8_0** | SmolLM2-135M, RTX 4070 Laptop sm_89 / WSL2 (gated `FAK_CUDA_GRAPH=1`) | llama.cpp Q8_0 (120 ± 15, `-ngl 99`) | `1029e37` | `GPU.md` §3b + `LLAMACPP-HEADTOHEAD-RESULTS.md` (on-box bench/test witness; reproduce: `FAK_CUDA_GRAPH=1 go run -tags cuda ./cmd/modelbench -dir internal/model/.cache/smollm2-135m -backend cuda`) |
| **Pure-kernel decide latency (M3 Pro)** | **362 ns** allow · 560–605 ns w/ ArgPredicates | syscall/adjudicator Decide | per-call decision | `bcad56e` | `experiments/mac-m3pro-kernel-20260620/kernel-latency-mac-m3pro-20260620.json` |
| **Pure-kernel admission latency (M3 Pro)** | **1.8–14 µs** scan · 3.3–15.8 µs Admit · 29–87 µs chain | ctxmmu / normgate+ctxmmu | per-result admission (cited "~1,300 ns" = cheapest scan layer only) | `bcad56e` | same (`MAC-M3PRO-KERNEL-BENCH-2026-06-20.md`) |
| **Syscall boundary tax (M3 Pro, refreshed)** | **~2,849×** in-process vs spawned `fak hook` | in-process adjudication | process-per-decide baseline (n=100) | `bcad56e` | `report.json` + `experiments/mac-m3pro-kernel-20260620/report.json` |
| **Causal invalidation-on-external-write** | **PASS · max\|Δ\|=0** (1 evicted, sibling warm, re-admit refused) | vDSO `Revoke` + cachemeta external-invalidation | blunt world-flush / stale serve | `0fc39aa` | `experiments/causal-invalidation-20260620/causalbench-witness-20260620.json` |
| **Ultra-long-context work floor (>100k tokens, EXACT/contention-free)** | **single ~10× · 5-agent fleet ~40×+ vs naive (4.3× vs tuned)** | Qwen2.5-7B geometry, P=100k T=10 C=1/5 D=200 R=500 (arithmetic, no model) | Naive re-prefill (A/C ref) / warm per-agent KV (B/C) | _this commit_ | `session/ultra-long-context-floor-20260622.json` + `ULTRA-LONG-CONTEXT-RESULTS.md`. WORK floor (token = sessionbench `prefillTokens`; FLOP = O(L²)-aware), not a wall-clock; anchor token A/C 62.0× reproduces the committed 50×5 token floor; live wall-clock anchor at >100k is separately gated |
| **README front-page webbench hero — WebVoyager fleet prefill (MODELED geometry, no model)** | **8-worker A/C 9.7× vs naive floor · B/C 1.10× vs tuned per-agent KV · A/B turn-tax 8.8× (worker-independent)** | WebVoyager 643-task set, turns derived per-task from difficulty (`geometry_sources: difficulty=643`), BasePrefix=3000 / Action=150 / DOMState=2000 | A: naive re-prefill-every-turn (A/C) · B: tuned per-agent KV (B/C) | _this commit_ | `experiments/webbench/webvoyager-geometry-20260625.json` (emit with `fak webbench describe --dataset testdata/webbench/webvoyager-converted.jsonl --workers 1,2,4,8 --out …`). PREFILL-TOKEN WORK FLOOR from a deterministic geometry model — closed-form integer formula in `internal/webbench/geometry.go::ComputeArms`, **no model, no wall-clock, not "measured"**. README leads with the B/C value-stack number (1.10×); the 9.7× is explicitly the vs-naive-floor figure |

| **Decode vs prefill worker-count scaling (x86_64 32-core, within-run ratio)** | **decode all-cores-default penalty 2.5× (1.5B) → 2.1× (3B) → 1.14× (7B); decode peaks ≤8–16w, prefill scales to all cores** | Qwen2.5-1.5B/3B/7B Q8, x86_64 32-core agent-host (contended) | best worker count vs 32w default — same box, same run | _this commit_ | `experiments/session/worker-scaling-desktop-x86-20260624.json` + `WORKER-SCALING-DESKTOP-X86-20260624.md`. WITHIN-RUN ratio only; absolute tok/s is contended agent-host, NOT comparable to the uncontended M3 Pro rows. CPU-threading analogue of the GPU launch-bound small-model artifact |
| **Self-ablation feature sweep — vDSO on/off (deterministic, Regime A of epic #607)** | **vdso_hits 0→7 · engine_calls 12→5 · tokens 937→417 (−520)** | tau2-airline-smoke frozen trace (12 calls), mock engine, no model | all-off baseline (vDSO off) | _this commit_ | `experiments/ablate/tau2-smoke-vdso-ablation.json` + `ABLATE-RESULTS.md`. Counter fields (workload_hash/vdso_hits/engine_calls/tokens/denies/quarantines) reproduce byte-identical (kernel event counters on a frozen trace); only p50_ns/wall_seconds/buckets are single-box. Rung 1 sweeps the one runtime knob only; env-gated features + cross-agent (Regime B) arms are separate rungs |
| **Cross-agent ablation — bare `claude` vs `fak guard -- claude` (Regime B of epic #607, [#623](https://github.com/anthony-chaudhary/fak/issues/623))** | **K=5/arm, both 5/5 success · output 0.98× · turns 1.00× · total-ingested 1.56× (−28 986 tok, kernel overhead) · +fak: 5 ALLOW / 0 deny** | `pong` 1-tool-call task w/ deterministic check, `claude-opus-4-8`, same OAuth acct, single Windows host | `claude_code` (bare `claude -p`) baseline | _this commit_ | `experiments/ablate/cross-agent-pong-opus.json` + `ABLATE-RESULTS.md`. Regime B is DISTRIBUTIONAL (mean ± CI95 over K≥5; the `WorkloadHash` guard does NOT apply); success-gated, model-named, tokens decomposed never summed. ONE tiny tool-light task on ONE host ⇒ deny/repair/quarantine counters an honest zero, cache-split is cold-prefix illustrative not a fleet SLA. Tool: `tools/cross_agent_ablate.py` (17 hermetic tests) |
| **AgentDojo structural safety floor (local, model-free)** | **full-stack ASR 0/38 (0.000) vs detection-only 29/38 (0.763) · benign controls 2/2 · gate PASS** | deterministic AgentDojo-style red-team, no model | detection-only lexical gates | _this commit_ | `experiments/agent-live/agentdojo-fak-fullstack-20260625.json` (reproduce: `go run ./cmd/agentdojoredteam -json`; corpus `sha256:ddc5b9ae08df0b37224a290fae212525228d2930e77afecb7bfc868b06ca1060`). LOCAL structural floor only — not an official external AgentDojo leaderboard result or raw-model arm |
| **ToolSandbox/tau3 policy-state adapter smoke ([SIMULATED] local fixture)** | **raw safe pass^1 1/2 (0.500) -> fak safe pass^1 2/2 (1.000); fak denied 1 policy/minefield call; `result_claim_allowed=false`** | `offline-trace`, 2 ToolSandbox-shaped tasks, no live model | raw trace replay without fak mediation | `c92bb2c` | `experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json` + `.md` (reproduce: `go run ./cmd/toolsandboxbench -suite testdata/toolsandbox/policy_state_smoke.json -out experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json -md experiments/agent-live/toolsandbox-policy-state-smoke-20260625.md`). Adapter smoke only - not an official Apple ToolSandbox or tau3 leaderboard result |

> **The model-ladder thesis.** Live wall-clock ratio climbs toward the deterministic
> 7.50× token-speedup ceiling as per-token compute grows (135M 4.58× → 360M 5.40× →
> 0.5B 6.20× → 1.5B 6.95×). This confirms that the residual gap below 7.50× is
> clone/memcpy overhead that becomes negligible on larger models — not an
> architectural limit. The deterministic metrics (token speedup, hit rate) are
> hardware-independent and reproduce the committed JSON exactly; only the live
> wall-clocks are single-box (within-run ratios authoritative per
> [BENCHMARK-GOVERNANCE.md](BENCHMARK-GOVERNANCE.md) regime rules).

---

## AgentDojo Structural Safety Floor (2026-06-25)

**Date:** 2026-06-25
**Commit:** _this commit_
**File:** `experiments/agent-live/agentdojo-fak-fullstack-20260625.json`
**Reproduce:** `go run ./cmd/agentdojoredteam -json`

### What this measures

This is Packet A from `docs/notes/AGENTIC-BENCHMARK-RUN-PACKETS-2026-06-25.md`: the
local, deterministic AgentDojo-style structural safety floor. It compares the same
38-attack corpus against two configurations:

- **detection-only:** content detectors only (`normgate` + `ctxmmu`);
- **full-stack:** the shipped detector stack plus IFC provenance taint and sink-gate.

It is model-free and preserves the benchmark fence: this row does **not** claim an
official external AgentDojo leaderboard score, and it does not replace a future raw
model-vs-fak external harness arm.

### Results

| Metric | Artifact field | Value |
|---|---|---:|
| Task / attack count | `task_count` | 38 |
| Detection-only attack successes | `asr_detection_succeeded` / `asr_detection` | 29 / 0.763 |
| Full-stack attack successes | `asr_fullstack_succeeded` / `asr_fullstack` | **0 / 0.000** |
| Harvest corpus rows / catches | `corpus_rows` / `corpus_catches` | 38 / 38 |
| Closed catch reasons | `catch_reasons` | `MALFORMED=9`, `TRUST_VIOLATION=29` |
| Benign controls completed | `benign_completed` / `benign_completion_rate` | 2 / 1.000 |
| Gate verdict | `gate` | **PASS** |

Corpus identity: `sha256:ddc5b9ae08df0b37224a290fae212525228d2930e77afecb7bfc868b06ca1060`.
The artifact also records the reproduce command, the attack ids, policy mode
(`detection-only-vs-full-stack-ifc`), and source revision metadata.

### Honesty fences

- This is a **local structural safety floor**, not a claim that fak beats the
  official AgentDojo benchmark or any model leaderboard.
- The detection-only arm is an internal lexical-gate baseline, not a raw frontier
  model arm.
- A safety win counts here because the runner now reports benign full-stack controls
  alongside ASR; broader task utility still requires the external AgentDojo-compatible
  adapter described in issue #868/#869.

### Verification

- `go test ./cmd/agentdojoredteam ./internal/agentdojo` -> PASS.
- `go run ./cmd/agentdojoredteam -json` -> exit 0 and writes `gate=PASS`.
- JSON parse/read-back confirmed the fields in the table above.

---

## ToolSandbox/tau3 Adapter Smoke and Agentic Authority Shape (2026-06-25)

**Claim class:** `[SIMULATED]` benchmark fixture; the adapter code path is shipped,
but the tasks are a local ToolSandbox/tau3-shaped smoke, not external harness rows.
**Result commit:** `c92bb2c`
**Files:** `experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json`,
`experiments/agent-live/toolsandbox-policy-state-smoke-20260625.md`
**Reproduce:**

```powershell
go run ./cmd/toolsandboxbench `
  -suite testdata/toolsandbox/policy_state_smoke.json `
  -out experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json `
  -md experiments/agent-live/toolsandbox-policy-state-smoke-20260625.md
```

### What this measures

This is Packet E from `docs/notes/AGENTIC-BENCHMARK-RUN-PACKETS-2026-06-25.md`.
It is the first raw-vs-fak agentic authority entry shape for issue #876: every
quoted number below names the artifact, reproduce command, task ids, model/trace
configuration, utility metric, safety metric, and limitation.

The raw arm replays the same tool trace without fak mediation, then adjudicates
the calls after the fact to count policy breaches and minefield hits. The fak arm
adjudicates before execution and only completes a milestone after an `ALLOW` or
`TRANSFORM` verdict. `pass^1` means all benchmark milestones completed; `safe
pass^1` means milestone completion with zero policy breaches and zero minefield
hits.

### Configuration

| Field | Value |
|---|---|
| Benchmark | `toolsandbox-shaped-smoke` |
| Model | `offline-trace` |
| Task ids | `retail-refund-policy-minefield`, `banking-address-update-benign` |
| Task count | 2 |
| Raw/fak parity guard | `same_task_ids=true`, `same_trace=true` |
| External grader | none; local fixture only |
| Evidence class | `SIMULATED_LOCAL_FIXTURE` |
| Result claim allowed | `false` |

### Results

| Metric | Artifact field | Raw | fak |
|---|---|---:|---:|
| pass^1 | `summary.*.pass_1` | 2/2 (1.000) | 2/2 (1.000) |
| safe pass^1 | `summary.*.safe_pass_1` | 1/2 (0.500) | 2/2 (1.000) |
| policy breaches | `summary.*.policy_breaches` | 1 | 0 |
| minefield hits | `summary.*.minefield_hits` | 1 | 0 |
| denied calls | `summary.*.denied_calls` | 0 | 1 |
| argument repairs | `summary.*.argument_repairs` | 0 | 0 |

Derived deltas: `safe_success_delta=1`, `policy_breach_delta=1`,
`minefield_hit_delta=1`.

### Promotion Gate

- A raw-vs-fak agentic authority row must include: artifact path, reproduce
  command, model/runner configuration, task ids, raw/fak parity guard, utility
  metric, safety/evidence metric, and limitations.
- A local fixture row may be cited only as `[SIMULATED]` adapter evidence. It must
  not be promoted into a leaderboard, README headline, or external benchmark claim.
- The ToolSandbox smoke artifact now carries this as data:
  `evidence_class=SIMULATED_LOCAL_FIXTURE`, `official_harness.available=false`,
  promotion requirements, and `result_claim_allowed=false`.
- An official tau3, ToolSandbox, AgentDojo, SWE-bench, Terminal-Bench, or browser
  benchmark row needs benchmark-native tasks and grader output attached or linked,
  with raw and fak arms sharing the same task ids, model, budget, and retry policy.
- If a number is not in this file with those fields, treat it as a run note, not a
  quotable benchmark claim.

### Verification

- `go run ./cmd/toolsandboxbench ...` regenerated the JSON and Markdown witnesses.
- JSON read-back confirmed `schema=fak.toolsandbox-adapter-report.v1`,
  `task_count=2`, raw safe successes `1`, fak safe successes `2`, fak denied
  calls `1`, `evidence_class=SIMULATED_LOCAL_FIXTURE`, and
  `result_claim_allowed=false`.
- `go test ./internal/toolsandbox ./cmd/toolsandboxbench` -> PASS.

---

## Pure-kernel latency — Apple M3 Pro (2026-06-20)

**Date:** 2026-06-20
**Commit:** `bcad56e`
**Files:** `experiments/mac-m3pro-kernel-20260620/kernel-latency-mac-m3pro-20260620.json` *(the anchor)*, `MAC-M3PRO-KERNEL-BENCH-2026-06-20.md` *(narrative companion — not published in the public repo)*
**Machine:** Mac15,7 — Apple M3 Pro, 12 core, arm64, darwin, go1.26.0. Medians of count=8 trials on an idle box.

### What this adds (and why)

The Authority's model-bench rows left the **pure-kernel latency stack** uncommitted: the
syscall bench (`report.json`) was the one pure-kernel artifact and was explicitly "narrow",
and a "~1,300 ns" admission figure cited in `DISAGGREGATED-AGENT-MEMORY.md` and
`MEMORY-LAYERS-EXPLAINER.md` had no committed artifact. This pass witnesses the stack via
`go test -bench` (the most reproducible form) so every cited number traces to a committed
artifact + reproduction command. Full decomposition and honest fences in the results doc.

### Results

| Layer | p50 ns/op | B/op | allocs/op | verdict |
|---|---:|---:|---:|---|
| **Decide** (canonical allow) | **362** | 256 | 5 | ALLOW |
| Decide w/ ArgPredicates (0→2000 unrelated) | 560 → 605 | 600 | 14 | — |
| **ScreenBytes** scan — secret (regex) | **1,812** | 0 | 0 | caught |
| ScreenBytes scan — benign (full) | 4,482 | 128 | 2 | allow |
| ScreenBytes scan — injection (nested) | 14,062 | 417 | 2 | caught |
| **Admit** (full gate, +page-out) — secret | 3,337 | 2,022 | 26 | QUARANTINE |
| Admit — injection | 15,799 | 2,662 | 28 | QUARANTINE |
| **AdmitChain** (normgate+ctxmmu) — benign | 29,171 | 1,662 | 25 | ALLOW |
| AdmitChain — injection | 87,056 | 4,854 | 38 | QUARANTINE |

Plus the refreshed syscall A/B: in-process p50 **2,427 ns** vs spawned `fak hook` p50
**6.913 ms** (n=100) → **~2,849×** boundary tax, `gate_primary=pass`.

### The honest finding on the cited figure

The "~1,300 ns" is the **narrow reading** — the cheapest `ScreenBytes` path (secret regex)
measures ~1.8 µs here, same order, **not fabricated**. But it names only one layer: the
general scan is 4.5–14 µs, the full `Admit` (with the page-out side-effect) is 3.3–15.8 µs,
and the full normgate+ctxmmu chain the kernel `Reap` runs is 29–87 µs. The single cited
number undersells the composed path by ~an order of magnitude on the worst payload; the
decomposition above replaces it. (Governance rule #4: the old figure is marked, not
silently removed.)

### Verification

- New `internal/ctxmmu/bench_test.go` compiles + vets clean; `go test ./internal/ctxmmu
  ./internal/adjudicator` → PASS (existing ctxmmu tests unaffected by the normgate
  registration the chain bench enables).
- `dos_commit_audit bcad56e` → **ABSTAIN** (the subject is a witness/documentation claim, not
  a falsifiable code claim; the diff is nonetheless real — it lists `bench_test.go` + the
  committed JSON artifacts). The load-bearing witness is `dos verify fak benchmark` below.
- `dos verify fak benchmark` → **SHIPPED** (`bcad56e`, rung `trailer` — the `(fak benchmark)`
  stamp binds the commit as a unit of benchmark work, confirmed by evidence, not self-report).

---

## Causal invalidation-on-external-write (2026-06-20) — the CPU-only strategic witness

**Date:** 2026-06-20
**Commit:** `0fc39aa`
**File:** `experiments/causal-invalidation-20260620/causalbench-witness-20260620.json`
**Reproduce:** `go run ./cmd/causalbench -selfcheck` (zero files, exits non-zero on any violation)

### What this witnesses (and why it is the cheapest strategic proof)

This is matrix row 6 of `PLAN-cloud-neocloud-rightsizing-2026-06-20.md` — the one
genuinely **net-new** strategic concept with **no hardware dependency** ($0, CPU-only,
unblocked). It proves the property `STRATEGIC-TIMING-2026-06-20.md` ranks #3: an external
write makes a cached read stale, and the system **itself** discovers *which* cached reads
depended on the now-stale world-state and evicts exactly those, byte-exact, refusing
re-admission. It is the causal sibling of the provable-deletion witness (`cmd/deletioncert`,
row 5): deletion evicts a span an operator *chose*; this evicts the reads an external write
*caused* to go stale — the MESI-invalidate analogue in the integrity direction.

The kernel mechanism was already shipped (`cachemeta.PlanExternalInvalidations`,
`vdso.Revoke`, `internal/gateway/coherence.go` wiring it onto live `fak serve` traffic).
What was missing — and what this adds — is a single self-checking end-to-end witness that
binds the whole chain, the artifact this row anchors. It runs on the **real process-global
`vdso.Default`** (the same `Lookup`/`Emit`/`Revoke` path live traffic uses), no model and
no weights, because the property is structural over cache identity + the witness ledger,
not numeric.

### The chain it proves (every row an asserted invariant; the demo exits non-zero otherwise)

| Invariant | Artifact field | Value |
|---|---|---|
| Two reads admitted under two external witnesses serve byte-exact from cache | `w1_hit_before_write` / `w2_hit_before_write` | true / true |
| Cached bytes equal a fresh engine call (a hit *is* a fresh call) | `w1_served_byte_exact` | true |
| External write refutes one witness → **exactly** the dependent read evicted | `w1_evicted_by_write` | **1** (targeted, not a flush) |
| The sibling under the unrefuted witness stays byte-identical across the write | `w2_byte_identical_across` | true (**max\|Δ\|=0**) |
| The refuted read now misses → goes to the engine, fresh (no stale serve) | `w1_miss_after_write` | true |
| A re-fill under the refuted witness does **not** repopulate (CAS can't resurrect it) | `w1_readmission_refused` | true |
| Refuting an unrelated witness evicts **0** local entries | `unrelated_witness_evicts` | 0 |
| The refutation is broadcast on the coherence bus (cross-agent propagation) | `coherence_broadcast_fired` | true |
| The integrity clock advances on refutation | `trust_epoch_advanced` | true |

### Honesty fences

- **This is a containment/coherence witness, not a throughput number.** It proves the
  causal-eviction *property* holds byte-exact on the real kernel path; it says nothing
  about tok/s or scale. Pool-scale behaviour under many concurrent agents is row 15 of the
  right-sizing plan and remains `[DEFERRED]` / projected.
- **Structural, not numeric.** Like `cmd/deletioncert`, it uses inline payloads and the
  witness ledger, so `max|Δ|=0` is an exact byte comparison of served payloads, not an
  approximate tolerance. No model is loaded; the claim is about cache identity, which is
  hardware-independent and reproduces the committed JSON exactly.
- **The witness is not keyed into the tier-2 key yet** (per `revoke.go`'s own honesty
  note): two agents reading under *different* witnesses still share by `(tool,args,worldVer)`.
  This witnesses the revocation axis (C4 causal-consumer eviction), which is the
  load-bearing half; witness-keying is the natural follow-on.

### Verification

- `go run ./cmd/causalbench -selfcheck` → exit 0 (all 12 guarded invariants hold — the
  9-row table above is the headline subset); `main_test.go` guards the same chain via the
  portable `go test ./cmd/causalbench/` → `ok cmd/causalbench` (on Windows: `.\fak\test.ps1`).
- `dos_commit_audit 0fc39aa` binds the result commit (diff-witnessed: the demo, its test,
  and the committed JSON artifact).
- The number is a correctness verdict (PASS / `max|Δ|=0`), not a wall-clock — hardware-
  independent and re-derivable from the artifact alone.

---

## README Headline — 50-turn × 5-agent Qwen2.5-1.5B (the number on the front page)

**Date:** 2026-06-19
**Commit:** `2bbda6f`
**File:** `experiments/session/headline-qwen-50x5.json`
**Chart:** `experiments/session/chart1-headline-walltime.svg`

This is the number a first-time visitor sees in README §1: *"On a realistic 50-turn ×
5-agent run (Apple M3 Pro, Qwen2.5-1.5B), fak did in ~19 minutes what the naive loop
needs an estimated ~19 hours."* Every figure in that sentence traces here.

### Shape & arms

`T=50 agents=5 prefix=2048 decode=32 result=64`, Qwen2.5-1.5B-Instruct Q8 (lean
quantize-at-load). Three arms over the **same Q8 forward pass**: A naive-stateless, B
per-agent-KV tuned, C fak fused (prefix prefilled once, cloned into the agents, batched
decode).

### Results (from the artifact)

| Metric | Value | Artifact field |
|---|---|---|
| **Reuse win vs naive** | **60.3×** | `net_value_add_vs_naive=60.346` |
| **Reuse win vs tuned warm-cache** | **4.1×** | `net_value_add_vs_tuned=4.125` |
| Arm A naive total | ~19.1 h | `arm_A_naive_stateless.total_ms=68,726,015` |
| Arm C fak total | ~19.0 min | `arm_C_fak_fused.total_ms=1,138,871` |
| Exact prefill-token ratio A/C | 62.0× | `prefill_tokens.a_over_c=62.05` |
| Turn-tax A/B | 14.6× | `turn_tax_A_over_B=14.63` |

### Honesty fences (matching the README's own)

- The **60.3×** is **vs the naive loop** (re-send the whole growing context every turn).
  Vs a *tuned* warm-cache stack the honest gain is **4.1×** — a few-fold, as stated.
- Arm A's ~19h is **modeled** from `prefillCost(L)` sampled at six lengths
  (`prefill_model.lens/ms` in the artifact), not run live (it would take ~19h).
  The model is **validated within ~0.4%**: `live_validate.anchored_computed_over_live
  = 1.0039` (a reduced live shape confirms the projection). The README's "within ~1%"
  is conservative against this.
- Arms B and C run **fully live** (attention growth captured); arm A's decode is set
  byte-identical to arm B's live decode. Disclosed in the artifact `methodology` field.

### Verification

- `dos_commit_audit 2bbda6f` binds the result commit.
- Bit-identity gates green (`TestBatchedDecodeMatchesSerial`,
  `TestBatchFromPrefixMatchesIndependentPrefill`): the three arms emit identical tokens,
  so the win is reuse, not a numerics shortcut.

### F1 — tombstone note (2026-06-19, Governance rule #4)

The old SmolLM2 session cells **11.2×/14.5×** (P=512, T=8/16, A=4) do **not reproduce** on the
current kernel: re-measured as **5.3×/7.4×**. Root cause: commits `70a2cab` (Q8 prefill softcap),
`6e5fda3` (SEAM-0 decode fold), `eb9a2e5` (q8 scratch reuse) between the `5b0f40d` measurement and
HEAD made the Q8 prefill ~2× faster, so computed arm A (naive re-prefill) got cheaper and A/C
shrank. The fak arm C still matches (12.0s/28.2s old vs 11.2s/24.4s re-measured). The old number
shrank **because fak got faster at the prefill arm A depends on**, not from any regression. Full
diagnosis: `benchmark-run-opencode-20260619/BENCHMARK-RUN-OPENCODE-20260619.md` finding F1.

---

## RadixAttention Results (SmolLM2-135M Q8)

**Date:** 2026-06-18
**Commit:** `a200c3d`
**File:** `experiments/radixattention/radixbench-smollm2-135m-q8.json`

### What This Measures

Compares **baseline** (full re-prefill per request) vs **radix** (automatic prefix-cached KV reuse using the same algorithm as SGLang's RadixAttention paper).

### Workload: Agents

- **Shape:** 5 agents × 6 turns = 30 requests
- **System prefix:** 128 tokens (shared across all agents)
- **Per-turn step:** 24 tokens
- **Model:** SmolLM2-135M Q8_0 (30 layers, real checkpoint)

### Results

| Metric | Baseline | Radix | Speedup/Improvement |
|---|---|---|---|---|
| **Wall-clock** | 240,994 ms (~241s) | 49,452 ms (~49s) | **4.87×** |
| **Tokens computed** | 6,360 | 848 | **7.50×** fewer |
| **Cache hit rate** | 0% | 86.7% | Matches SGLang band (50-99%) |

### Verification

- `dos_commit_audit a200c3d` → **OK** (diff-witnessed)
- Committed JSON artifact exists and is readable
- Token counts are exact integers (hardware-independent)

### Why Wall-Clock (4.87×) < Token Speedup (7.50×)

On the synthetic 64-hidden/4-layer wiring model, the memcpy cost of cloning cached KV masks the compute savings (1.64× live ratio). On SmolLM2-135M, per-token compute dominates memcpy, so live speedup approaches the theoretical token figure (4.87×).

**Both results are committed and real** — they document different regimes.

> **Superseded as the headline by the 2026-06-19 model ladder below.** The single
> 135M point (`a200c3d`, contended 4.87×) remains a real committed measurement; the
> fresh ladder re-runs it at 4.58× (reps3, lightly contended) and extends it across
> three more models. Cite the ladder for the release; this row stays as provenance.

---

## RadixAttention Model Ladder (2026-06-19) — climbs to the token-speedup ceiling

**Date:** 2026-06-19
**Commit:** `92896a4`
**Files:** `experiments/radixattention/radixbench-{smollm2-135m,smollm2-360m,qwen2.5-0.5b,qwen2.5-1.5b}-q8-agents-fresh-20260619.json`

### What This Adds

The same RadixAttention `agents` workload (5 agents × 6 turns, 128-token shared
system prefix, 24-token per-turn step) run across four real Q8 checkpoints. The
**deterministic** metrics (token speedup, hit rate, FCFS→cache-aware recovery) are
byte-identical across all four (model-independent); the **live wall-clock** ratio is
the one that moves, and it climbs monotonically toward the 7.50× token ceiling as the
model grows.

### Results — `agents` workload

| Model | Live wall-clock | Token speedup | Hit rate (FCFS → cache-aware) | Artifact (`live_prefill_speedup`) |
|---|---|---|---|---|
| SmolLM2-135M (30L) | **4.58×** | 7.50× | 62.1% → 86.7% (100% of optimal) | `radixbench-smollm2-135m-q8-agents-fresh-20260619.json` |
| SmolLM2-360M (32L) | **5.40×** | 7.50× | 62.1% → 86.7% | `radixbench-smollm2-360m-q8-agents-fresh-20260619.json` |
| Qwen2.5-0.5B (24L) | **6.20×** | 7.50× | 62.1% → 86.7% | `radixbench-qwen2.5-0.5b-q8-agents-fresh-20260619.json` |
| Qwen2.5-1.5B (28L) | **6.95×** | 7.50× | 62.1% → 86.7% | `radixbench-qwen2.5-1.5b-q8-agents-fresh-20260619.json` |

Deterministic hit rates reproduce committed `a200c3d` exactly: few-shot 88.2%,
multi-turn-chat 79.5%, tree-of-thought 77.2%, agents 86.7%. Policy-eviction witness
green on every run.

### Verification

- Each row's `live_prefill_speedup` read directly from its committed JSON (verified
  2026-06-19: 4.581 / 5.40 / 6.20 / 6.951 → rounded above).
- `internal/radixkv` split-reuse == recompute (max|Δ|=0) → **PASS** (numerics are
  reuse, not a shortcut).
- Token counts (`prefill_token_speedup=7.5`, `radix_computed_tokens=848`) are exact
  integers, hardware-independent.
- **Cross-platform reproduction (2026-06-19):** the 135M `agents` deterministic fields
  reproduce **bit-for-bit on Windows x86_64** (hit 86.7%, token 7.50×, reused 5512,
  computed 848) vs the Mac M3 arm64 committed artifact; the live ratio moves (2.60× on
  x86 vs 4.58× on Mac) exactly as the small-model clone-overhead thesis predicts. See
  [`experiments/radixattention/CROSS-PLATFORM-REPRO-20260619.md`](experiments/radixattention/CROSS-PLATFORM-REPRO-20260619.md).

---

## Session Value-Stack High-T Ladder (2026-06-19) — the O(T²)→O(T) contrast

**Date:** 2026-06-19
**Commit:** `92896a4`
**Files:** `experiments/session/highT-smollm2-135m-{64-128-256,512}-fresh-20260619.json`

### What This Adds

The session value-stack (A=naive-stateless, B=per-agent-KV tuned, C=fak fused) pushed
to high turn counts on SmolLM2-135M (P=512, D=4, R=8, C=2) to expose the naive arm's
O(T²) re-prefill signature against fak's near-linear curve.

### Results

| T | A naive | B tuned | C fak | **A/C vs naive** | turn-tax A/B | exact prefill-tok A/C |
|---|---|---|---|---|---|---|
| 64  | 268.1s | 14.3s | 10.8s | **24.9×** | 18.7× | 74.9× |
| 128 | 908.8s | 30.7s | 23.0s | **39.5×** | 29.6× | 128.2× |
| 256 | 3982.1s | 74.8s | 54.4s | **73.2×** | 53.2× | 227.7× |
| 512 | 20424.4s (~5.7h) | 211.5s | 146.6s | **139.3×** | 96.6× | 421.7× |

The naive arm A explodes ~4× per T-doubling (268→909→3982s) — the O(T²) re-prefill
signature — while B and C stay near-linear.

### Honest methodology (carried from the artifact)

Arms **B and C run end-to-end LIVE** (attention growth captured). Arm **A's prefill
is modeled** from `prefillCost(L)` measured at sampled lengths (the O(L²)
prefill-attention captured, summed over the exact per-turn contexts), because running
A fully live at T=512 would take ~5.7h per cell; arm A's decode is set byte-identical
to arm B's live decode. The `validate` shape runs arm A fully live to confirm the
model. This is disclosed in each JSON's `methodology` field.

### Verification

- T=512 cell read from artifact: `net_value_add_vs_naive=139.278`,
  `turn_tax_A_over_B=96.564`, `prefill_tokens.a_over_c=421.716`.
- Bit-identity gates green: `TestBatchedDecodeMatchesSerial`,
  `TestBatchFromPrefixMatchesIndependentPrefill` (arms produce identical tokens).

---

## GPU Q8 Throughput — Vulkan on the AMD RX 7600 (2026-06-19)

**Date:** 2026-06-19
**Commit:** `60db592` (path unblocked by `84c2e6c`)
**Files:** `experiments/gpu/q8gpu-smollm2-135m-{gpu-q8,gpu-f32,cpu-q8}-20260619.json`
**Doc:** `experiments/gpu/VULKAN-Q8-RX7600-20260619.md`

### What This Adds

The first committed Q8-on-GPU throughput numbers from the `modelbench` harness. Until
`84c2e6c`, `modelbench -backend vulkan -quant` hard-refused ("compute HAL sessions are
f32-only today") even though the Q8 weight-upload + device-GEMM path was fully wired in
`internal/model/hal.go`. Three arms over the **same SmolLM2-135M Q8 forward pass** on the
real RX 7600 (Vulkan 1.4.349, native Windows), 64 decode steps / 3 reps.

### Results

| arm | decode tok/s | prefill P=16 → 512 | artifact |
|---|---:|---:|---|
| **gpu-q8** | **24.6** | 15.6 → 24.8 | `q8gpu-smollm2-135m-gpu-q8-20260619.json` |
| gpu-f32 | 16.5 | 12.6 → 18.7 | `q8gpu-smollm2-135m-gpu-f32-20260619.json` |
| cpu-q8 | 176.9 | 969 → 1519 | `q8gpu-smollm2-135m-cpu-q8-20260619.json` |

### Two honest findings

1. **Q8 weight-narrowing buys ~1.49× decode on the GPU** (24.6 vs 16.5 tok/s) and ~25–30%
   on prefill at every length — same forward, same device, only the weight dtype changes.
   The decode path is memory-bound, so cutting weight traffic ~4× directly raises throughput.
2. **On 135M the CPU wins outright** — cpu-q8 decode 176.9 tok/s is **7.2×** the GPU, and CPU
   prefill (batched GEMM) is **40–75×** the GPU's single-token-looped device prefill. The GPU
   path is **launch-bound** (~330 device ops/token × a fixed dispatch tax that dwarfs 135M's
   per-op compute), the same regime the CUDA/RTX-4070 lane documents — now confirmed on a
   second vendor. The device path is the architecture that scales to models too big for CPU
   residency, **not** a win at 135M. Lever: batched device prefill + capture-replay graph
   (`Async`/`GraphCompile` both `false` in the RX 7600 caps today).

### Verification

- Correctness gated on the real GPU: `TestHALVulkanQ8ForwardMatchesComputeQ8` →
  **prefill cosine = 1.0, step cosine = 1.0**; `TestHALVulkanForwardMatchesNative` →
  argmax-exact, cosine 1.0. The throughput win is reuse + narrower traffic, not a numerics
  shortcut.
- Each row read directly from its committed JSON (`decode.tok_per_sec`, `prefill[].tok_per_sec`).
- `precision`/`backend.selected`/`backend.tier` fields in each artifact make the provenance
  self-describing (e.g. gpu-q8: `precision=Q8_0`, `selected=vulkan`, `tier=discrete:AMD Radeon RX 7600`).

---

## GPU/CPU Q8 Crossover — the device path catches the CPU as the model grows (2026-06-19)

**Date:** 2026-06-19
**Commit:** `7bf666b` (unblocked by the `8c74fd9` q8_matmul input-tiling fix)
**Files:** `experiments/gpu/crossover-qwen2.5-1.5b-{gpu,cpu}-q8-20260619.json`
**Doc:** `experiments/gpu/CROSSOVER-1P5B-RX7600-20260619.md`

### What This Adds

The 135M GPU result above showed the device path **launch-bound** — 7.2× behind the CPU. The
obvious question: does that gap close on a bigger model, where the per-token GEMM is large
enough to amortize the fixed ~330-op/token dispatch tax? Measured on Qwen2.5-1.5B Q8 (the
`q8_matmul` shader's old inDim≤2048 cap, which the 1.5B FFN's inDim=8960 exceeded, was lifted
in `8c74fd9` — verified bit-correct by `TestVulkanQ8MatMulWideInput`, cosine ≥ 0.9999).

### Results — Q8 decode tok/s, GPU (Vulkan RX 7600) vs CPU (pure-Go legacy)

| model | CPU Q8 decode | GPU Q8 decode | **CPU / GPU ratio** |
|---|---:|---:|---:|
| SmolLM2-135M | 176.9 | 24.6 | **7.2×** |
| Qwen2.5-1.5B | 18.4 | 15.9 | **1.16×** |

The CPU's lead collapses **7.2× → 1.16×** as per-token compute grows ~11×. This is direct
evidence for the device-path thesis: the GPU wins as the model grows (one more size step, 3B+,
likely flips it to a GPU win). The launch-bound regime is a small-model artifact, not a
ceiling.

### Honest fences

- **Decode only.** Prefill still favors the CPU heavily (the device prefill loops single tokens
  — HAL prefill isn't batched — so it runs at decode speed; the CPU batches its prefill GEMM).
  Batched device prefill is the standing next lever; it does not affect the decode crossover.
- A transient large-prefill-shape VRAM-allocation panic exists on the 1.5B (the pool's
  drain-and-retry usually absorbs it; a smaller `-prefill-sizes` avoids it). The decode number
  is stable across reps.

### Verification

- Each ratio read from the committed JSON `decode.tok_per_sec` (GPU 15.900, CPU 18.428 → 1.16×;
  135M GPU 24.620, CPU 176.898 → 7.19×).
- Q8 device GEMM bit-close to the CPU Q8 reference at the 1.5B FFN width
  (`TestVulkanQ8MatMulWideInput`, in=8960, cosine ≥ 0.9999); HAL forward gate argmax-exact.

---

## Session Value-Stack Results (SmolLM2-135M Q8)

**File:** `docs/benchmarks/SESSION-VALUE-STACK-RESULTS.md`

### What This Measures

Compares three arms running the **same Q8 forward pass**:
- **A — naive-stateless**: Re-prefills entire context every turn (common local pattern)
- **B — per-agent-KV**: Prompt-cache/persistent KV per agent, no cross-agent sharing
- **C — fak fused**: Prefix prefilled once + cloned into C agents, batched decode

### Results

| Turns | Agents | Prefix | Naive (A) | Tuned (B) | fak (C) | A/C | B/C |
|---|---|---|---|---|---|---|---|
| 8 | 4 | 512 | 135.1s | 32.4s | 12.0s | **11.2×** | 2.70× |
| 16 | 4 | 512 | 409.4s | 67.9s | 28.2s | **14.5×** | 2.41× |

### Key Point

The **11.2–14.5×** value-add is **vs naive stateless serving**, not vs SGLang or any other tuned baseline. This is the "common local pattern" comparison.

> **Low-T anchor for the high-T ladder above.** This T=8/16 authority-shape result
> (C=4, D=24) is the conservative point; the 2026-06-19 high-T ladder pushes the same
> A-vs-C comparison to T=512 → 139.3× by isolating T-scaling with a smaller per-turn
> step. Both are vs the same naive-stateless baseline; they differ only in shape.

---

## Baseline Comparisons: What Each Number Means

| Number | Compares Against | Regime |
|---|---|---|
| 4.58× → 6.95× | Full re-prefill per request | RadixAttention live ladder (135M → 1.5B), climbing to the 7.50× ceiling |
| 7.50× | Token count reduction | Theoretical compute saved (deterministic, model-independent) |
| 86.7% | SGLang's published 50-99% band | Cache hit rate (FCFS 62.1% → cache-aware, 100% of optimal) |
| 11.2–14.5× (T=8/16) → 139.3× (T=512) | Naive stateless (no KV persistence) | Session value-add, O(T²)→O(T) as T grows |
| 2.4–2.7× | Tuned single-tenant (per-agent KV) | Marginal value over warm cache |

---

## Cross-Index

### SGLang RadixAttention Paper
- **Source:** Lianmin Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," arXiv:2312.07104; NeurIPS 2024
- **fak replication:** `docs/benchmarks/RADIXATTENTION-RESULTS.md`
- **Claim:** fak achieves 86.7% hit rate (inside SGLang's 50-99% band)

### SmolLM2-135M Reference
- **Role:** In-kernel bit-exact anchor for GPU/CPU equivalence gates
- **Proof:** `IN-KERNEL-MODEL-DESIGN.md` R0–R14 *(narrative companion — not published in the public repo; the bit-exact equivalence ships as tests, e.g. `TestHALVulkanForwardMatchesNative`)*
- **Status:** Proven bit-for-bit vs HF oracle

---

## Reproduce

```bash
# RadixAttention benchmark
go run ./cmd/radixbench \
  -dir internal/model/.cache/smollm2-135m \
  -quant \
  -out experiments/radixattention/radixbench-smollm2-135m-q8.json

# Session value-stack
go run ./cmd/sessionbench \
  -turns 8,16,32 -agents 4 -prefix 512 -decode 24 -result 48 \
  -out experiments/session/smoke-smollm2.json
```

---

## Tombstoned/Outdated Claims

The following claims have been superseded or should not be used:

| Old Claim | Status | Replacement |
|---|---|---|
| "~13× speedup, P=512,T=5,C=5" | ❌ Not found in committed evidence | Use 4.87× (RadixAttention) or 11.2–14.5× (value-stack) |
| "SmolLM2-135M achieves ~370s → ~30s" | ❌ No committed artifact for this exact config | See committed results above |
| Any uncommitted/transient benchmark numbers | ❌ Must ship via commit + JSON | See authority table |

---

## Next Model Results Template

When benchmarking a new model, add an entry following this structure:

```markdown
### Model-Name Results

**Date:** YYYY-MM-DD
**Commit:** <hash>
**File:** `path/to/artifact.json`

| Metric | Baseline | Optimized | Speedup |
|---|---|---|---|
| Wall-clock | XXX ms | YYY ms | **Z.Z×** |
```

---

## DOS Verification Discipline

Every claim in this document is backed by:
1. **Committed artifact** (JSON in repo)
2. **Git commit** with `dos_commit_audit` verification
3. **Reproducible command** in "Reproduce" section

No claim exists without a traceable source.
