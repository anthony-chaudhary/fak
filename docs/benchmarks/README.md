---
title: "docs/benchmarks — directory index"
description: "Navigation index for every benchmark sheet: fleet/cache value, context safety, in-kernel model parity, GPU platforms, serving head-to-heads, and the external benchmark suites."
---

# Benchmark sheets — directory index

Navigation only, not authority. Numbers are only authoritative in
[`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) (the governed authority
ledger), and every claim carries a tag in [`../../CLAIMS.md`](../../CLAIMS.md). Charts live
in [`../../BENCHMARK-GALLERY.md`](../../BENCHMARK-GALLERY.md). Each sheet below keeps its
own provenance labels (MEASURED / MODELED / WITNESSED / PENDING / GATED); this page just
tells you where to look.

## Fleet, cache & session value

| File | Kind | One-liner | Headline |
| --- | --- | --- | --- |
| [ABLATE-RESULTS.md](ABLATE-RESULTS.md) | results | The deterministic, $0, no-model half of the self-ablation benchmark harness (epic #607). | vDSO fast path serves 7 of 12 calls from cache, cutting 520 tokens (937 → 417) on the frozen tau2-airline-smoke trace |
| [CDB-RESULTS.md](CDB-RESULTS.md) | results | How the fak context debugger attaches to a finished session as a core image and faults in only the working set. | A real ~350k-token-class session decomposes into an 18 KB page table over a 1.2 MB swap device; a follow-up demand-pages 1.8–6.2% of resident bytes (measured, witnessed by go tests + committed cdb-report.json) |
| [FANOUT-BENCH-RESULTS.md](FANOUT-BENCH-RESULTS.md) | results | Prices one master goal decomposed into N sub-agents swept from 1 to 1024, measuring cross-agent dedup and shared-prefix KV reuse. | MEASURED: 1,024 real agents complete one goal end-to-end in 364 ms on a laptop with no GPU (headline 72.8× parallel speedup is a MODELED projection) |
| [FLEET-5X200-7B-10MIN-RESULTS.md](FLEET-5X200-7B-10MIN-RESULTS.md) | results | Measured on an Apple M3 Pro (Qwen2.5-7B Q8, llama.cpp Metal): the 5-agent × 200-turn fleet lands at ~8.2 min. | MEASURED: batched + shared prefix fleet = ~8.2 min on M3 Pro — under the 10-minute bar (naive re-prefill ≥ 4.0 hours); host Metal forward, not the pure fak kernel |
| [FLEET-SWEEP-RESULTS.md](FLEET-SWEEP-RESULTS.md) | results | Measures the turns-by-agents turn-tax surface across a 1-50 grid, quantifying the cross-agent cache uplift. | At the T=50×A=50 corner the read-only fleet deletes 2344 of 2500 calls (94%), cross-agent uplift +370 — measured tier-2 kernel events, not modeled |
| [GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md](GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md) | pending | PENDING — Results not yet collected; describes the result packet shape for the live GLM-5.2 cache-value run. | Cache value (reused tokens): PENDING; offline WITNESSED-derived prefill-elimination floor A/C 17.9× → 23.4× (workers 1→16) |
| [GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK.md](GLM52-FAK-KERNEL-CACHE-VALUE-RUNBOOK.md) | runbook | End-to-end path to observe fak's OWN in-kernel KV-prefix cache value on a real solved ticket via a GLM-5.2 gateway | Status: the observation seam is SHIPPED and tested; the live GLM-5.2 number is the box residual — nothing here invents a tok/s or a reuse figure |
| [GUARD-HOP-OVERHEAD-PENDING.md](GUARD-HOP-OVERHEAD-PENDING.md) | pending | Harness + row structure landed; the live MEASURED wall-clock is PENDING (hardware-gated). | PROJECTED per-turn 2.90–19.42 µs (8 calls/turn); per-session 0.14–0.97 ms (50 turns) |
| [MEMORY-DREAM-CLEANUP-RESULTS.md](MEMORY-DREAM-CLEANUP-RESULTS.md) | results | An offline pass that re-screens, repairs, and prunes a finished session's core image against today's gate. | |
| [SELF-TAX-TREND.md](SELF-TAX-TREND.md) | trend | Living trend companion to the self-tax row in BENCHMARK-AUTHORITY.md; a dated, net-true-labeled series. | ~0.55 ns/op · 0 allocs · FLAT N=1→1000 drivers (WITNESSED, adjudication read-path floor) |
| [SESSION-VALUE-STACK-RESULTS.md](SESSION-VALUE-STACK-RESULTS.md) | results | Measures the fused agent kernel's net work-reuse value-add (60.3x vs naive, 4.1x vs tuned) on a 50-turn session on M3 Pro. | 60.3× net value-add vs naive (A/C), 4.1× vs tuned single-tenant (B/C) — MEASURED for the live arms; arm A computed-from-measurement (validated live) |
| [ULTRA-LONG-CONTEXT-RESULTS.md](ULTRA-LONG-CONTEXT-RESULTS.md) | results | Reread-elimination win at the >100k-token regime, proven as an exact contention-free work floor | ~10x vs naive on a single >100k session, ~40x+ on a 5-agent fleet each >100k (exact contention-free work floor, not a wall-clock) |

## Context safety & quarantine

| File | Kind | One-liner | Headline |
| --- | --- | --- | --- |
| [KV-QUARANTINE-BRIDGE-RESULTS.md](KV-QUARANTINE-BRIDGE-RESULTS.md) | results | Context-MMU quarantine verdict on a tool result's bytes also evicts its K/V span from the attention cache. | max\|Δ\| evict-vs-never = 0.000e+00 — the evicted cache is bit-identical to one that never saw the poison (go test witnessed; wiring witness uses a synthetic model) |
| [LIVE-RESULTS.md](LIVE-RESULTS.md) | results | A live multi-turn A/B where a real model drives the same airline task twice, counting turns, tokens, and quarantines. | The injection reached the baseline's context in 100% of runs (5/5) and the kernel quarantined it in 100% of runs (5/5). |
| [RECALL-RESULTS.md](RECALL-RESULTS.md) | results | Proves a tool result quarantined by the context-MMU cannot be paged into a new context across the process boundary. | Core image: 4 pages (2 benign, 2 sealed), 442 bytes CAS — the quarantine survived the persist→reload boundary; 5/5 skeptic claims CONFIRMED |
| [TOOL-RESULT-TREE-KV-RESULTS.md](TOOL-RESULT-TREE-KV-RESULTS.md) | results | Pins the precise claim that fak alone removes a tool result from a sequence middle bit-identically to never-saw. | fak's middle-span removal is bit-identical to never-saw: TestKVQuarantineEqualsNeverSaw matches HF token-for-token, max\|Δ\|=0 (sibling-isolation 0.000e+00; fixture-gated HF rung; not yet wired into the live fak agent loop) |
| [TURN-TAX-RESULTS.md](TURN-TAX-RESULTS.md) | results | Measures fak's deterministic safety floor on a live kernel trace plus the extra error-recovery turns its one-shot syscall path eliminates when self-hosted. | Safety floor (the moat): 1 injection quarantined, 1 destructive op denied (baseline 1/1 vs fak 0/0) on the 14-call turntax-airline slice — real kernel counters, engine-agnostic; efficiency upside −9 turns is a cache-favorable self-host-only slice (real-world ~0.7% rate → 0.33 turns/session) |

## In-kernel model correctness & parity

| File | Kind | One-liner | Headline |
| --- | --- | --- | --- |
| [FAK-NATIVE-QWEN35-RESULTS.md](FAK-NATIVE-QWEN35-RESULTS.md) | results | fak's own in-kernel forward pass runs the Qwen3.5/3.6 hybrid Gated-DeltaNet end-to-end, 0.8B to 27B GGUF on M3. | Clean headline: decode 1.2 tok/s (64 tok / 54.12 s), prefill 0.6 tok/s (29 tok / 48.27 s) vs the llama.cpp-Metal bar 7.29 / 51.55 and the 3× goal 2.7 (MEASURED on Apple M3 Pro) |
| [IN-KERNEL-MODEL-RESULTS.md](IN-KERNEL-MODEL-RESULTS.md) | results | A kernel-owned in-process CPU forward pass for SmolLM2-135M, proven rung-by-rung bit-for-bit against HuggingFace. | Every rung proven against HuggingFace transformers: argmax exact 3/3 prompts, greedy generation token-for-token identical, KV quarantine and prefix reuse max\|Δ\| = 0.000e+00 |
| [M3-LLAMACPP-RESULTS.md](M3-LLAMACPP-RESULTS.md) | results | Adds an arm64 NEON Q8 kernel that flips int8 to 1.9x faster than f32, runs Qwen2.5-1.5B, and measures the gap to llama.cpp | fak now lands in the same order of magnitude as llama.cpp's CPU path (decode ~2.2x behind, prefill ~6.5x behind) — not at parity |
| [MODEL-BASELINE-RESULTS.md](MODEL-BASELINE-RESULTS.md) | results | Measures the in-kernel forward pass against HF and llama.cpp on CPU, then closes it to decode parity. | Prefill per-token slope (raw compute): llama.cpp Q8_0 0.337 ms/tok vs fak Q8 0.346 ms/tok = 1.03x — parity (measured 2026-06-17 under live 6-session fleet load) |
| [MODEL-BATCHING-RESULTS.md](MODEL-BATCHING-RESULTS.md) | results | Measures in-kernel multi-user batched decode over per-user KV caches, scaling throughput while bit-identical to serial Step. | Aggregate decode throughput: 2916 tok/s at B=960 = 151.9× the unbatched f32-serial baseline and 41.3× the real single-stream Q8 decode (measured, native, under live fleet load) |
| [QWEN25-7B-RESULTS.md](QWEN25-7B-RESULTS.md) | results | fak's first 7B dense run on M3 Pro Q8, reporting the honest throughput gap vs llama.cpp Metal plus full greedy parity. | 8.7 tok/s decode (115.2 ms/tok) — fak is 0.083× prefill / 0.50× decode vs llama.cpp Metal |
| [QWEN35-0.8B-RESULTS.md](QWEN35-0.8B-RESULTS.md) | results | End-to-end in-chat f32 hybrid-GDN path for Qwen3.5-0.8B, proving the architecture works on a tiny model before scaling up. | One Qwen3.5-0.8B f32 modelbench number, and it is in the Authority (#113): pinned cold load time 1203ms (M3 Pro) |
| [QWEN36-LOAD-PROFILE-440.md](QWEN36-LOAD-PROFILE-440.md) | results | Phase-attributed load profile of the Qwen3.6-27B GGUF->Q8 path, with a same-host arena-reuse before/after. | Witnesses the #440 page-churn fix at -4.0% peak RSS / -4.0% page faults (measured 2026-06-26) |
| [QWEN36-PARITY-RESULTS.md](QWEN36-PARITY-RESULTS.md) | results | The witnessed llama.cpp Metal reference for Qwen3.6-27B on M3 Pro, the speed bar fak's own engine targets. | Metal, full offload: prefill 51.55 tok/s, decode 7.29 tok/s (witnessed; fak decode progression 0.1 → 0.9 → 1.2 tok/s, not yet speed-parity) |
| [QWEN36-PARITY-ROLLUP-2026-06-28.md](QWEN36-PARITY-ROLLUP-2026-06-28.md) | rollup | One authoritative reconciliation of the Qwen3.6-27B-vs-llama.cpp parity status: PROVEN vs `not yet`, with gated repros | Correctness parity PROVEN at architecture level but REFUTED at 27B scale (token-3 argmax flip); speed parity `not yet` — 1.2 tok/s vs the 7.29 tok/s llama.cpp-Metal bar (recorded prior Mac witnesses, none re-measured here) |
| [ROPE-SCALING-RESULTS.md](ROPE-SCALING-RESULTS.md) | results | NTK-by-parts RoPE rescale letting an 8K-trained model attend at 128K, HF-faithful and byte-identical when unset. | llama3 inv_freq rescale — 8K→128K, proven against HF's own reference; device lanes NOT yet wired (known gap) |
| [SLIDING-WINDOW-RESULTS.md](SLIDING-WINDOW-RESULTS.md) | results | Reports fak's per-layer sliding-window attention bounding cost from O(N) to O(W), shipped as a read-time mask. | Measured: 1,000,000 tokens decoded with the KV cache never exceeding 256 positions (window 128), vs the ~67 GB a full 1M-token cache would need for SmolLM2-135M |

## GPU & hardware platform results

| File | Kind | One-liner | Headline |
| --- | --- | --- | --- |
| [GCP-H100-RESULTS.md](GCP-H100-RESULTS.md) | results | First end-to-end run of fak's own CUDA engine on a live GCP H100, head-to-head vs llama.cpp CUDA, with an honest gap verdict | MEASURED: fak-cuda decodes Qwen2.5-3B at 96.3 tok/s (f32) vs llama.cpp's 361.6 tok/s (Q8_0) on an H100 — ~3.75x behind on decode, verified on-device (-require-non-reference) |
| [GCP-L4-RESULTS.md](GCP-L4-RESULTS.md) | results | The completed GCP L4 head-to-head for Qwen2.5-3B Q8_0: llama.cpp CUDA, fak-cpu, and fak-cuda on one NVIDIA L4. | fak-cuda measured on a real L4: backend selected "cuda", tier sm_89, 16.6 prefill / 18.7 decode tok/s (f32) vs llama.cpp CUDA Q8_0 at 6,638.3 / 70.8 — the gap is not a win claim |
| [GPU-QWEN-RESULTS.md](GPU-QWEN-RESULTS.md) | results | fak-CUDA decode parity with llama.cpp f16 on Qwen2.5-1.5B and a Q8_0 path reaching Qwen2.5-3B on an 8 GB RTX 4070, argmax-exact. | Equal-precision decode parity, Qwen2.5-1.5B-Instruct: fak-CUDA f16 ~36.6 tok/s vs llama.cpp F16 34.3 (fak ~1.07x), witnessed greedy argmax-exact vs the f32 reference |
| [H100-KERNEL-5X-ROADMAP.md](H100-KERNEL-5X-ROADMAP.md) | plan | Evidence-backed decomposition of the measured fak-on-H100 gap into ranked, code-anchored levers with expected multipliers. | Every speedup number below is a projection gated on a measured Hopper run — decode 3.75× behind, prefill ~380× behind (MEASURED baseline; levers GPU-gated "not yet") |
| [QWEN36-AMD-VULKAN-RESULTS.md](QWEN36-AMD-VULKAN-RESULTS.md) | results | Witnessed run of Qwen3.6-27B Q4_K_M on an AMD/Vulkan Windows desktop, proving the model serves all three fak surfaces. | Result: 3/3 surfaces passed (witnessed 2026-06-19 on the AMD/Vulkan desktop node) |
| [VULKAN-AMD-RESULTS.md](VULKAN-AMD-RESULTS.md) | results | Witnesses fak's Vulkan compute backend reaching numerical parity on a real AMD RX 7600 while still ~60x slower than llama.cpp. | Numerical parity on real AMD silicon (argmax-exact greedy decode, prefill-logit cosine = 1.0); throughput NOT at parity — best 2.5 tok/s, ~58-60x slower than llama.cpp CPU (all MEASURED, nothing SIMULATED) |

## Engine & serving head-to-heads

| File | Kind | One-liner | Headline |
| --- | --- | --- | --- |
| [GPU-SERVER-GLM52-VLLM-AGENTIC-BENCHMARKS.md](GPU-SERVER-GLM52-VLLM-AGENTIC-BENCHMARKS.md) | pending | A pending-measurement plan for GLM-5.2 on vLLM through fak's gateway: agentic benchmarks + 20-task SWE-bench slice. | Status: pending measurement. This document carries commands and gates, not results. |
| [LLAMACPP-HEADTOHEAD-RESULTS.md](LLAMACPP-HEADTOHEAD-RESULTS.md) | results | Measures fak against llama.cpp on CPU across decode, prefill, batched throughput, and shared-prefix, with honest verdicts. | single-stream decode is parity (1.12×), batched is parity/slight fak lead (2916 vs ~2816 tok/s); cross-agent shared-prefix is OPEN (preliminary, settings-dependent) |
| [QWEN36-27B-GPU-SERVER-RESULTS.md](QWEN36-27B-GPU-SERVER-RESULTS.md) | results | fak serving and coding-agent surfaces on Qwen3.6-27B with an 8-GPU SGLang backend, reporting gateway tax across a 1-to-128 concurrency sweep. | All three fak surfaces PASS on the 27B; peak 64-concurrency 1085.6 tok/s fak-gateway vs 1451.6 raw SGLang (0.75×), converging to ~3% tax at conc 128 (0.97×) |
| [RADIXATTENTION-RESULTS.md](RADIXATTENTION-RESULTS.md) | results | Head-to-head of fak's KV-cache reuse against SGLang's RadixAttention on cache hit rate, with a 4.58x live speedup. | 4.58× live wall-clock speedup on SmolLM2-135M Q8 (agents workload); 77–88% cache hit rate inside SGLang's verified 50–99% band (MEASURED) |
| [VLLM-HEADTOHEAD-RESULTS.md](VLLM-HEADTOHEAD-RESULTS.md) | pending | A gateway adjudication-tax witness and vLLM engine bench; pending-measurement — every vLLM cell is a placeholder. | Status: pending-measurement (GATED scaffold). No vLLM GPU run has landed yet — every numeric cell in the vLLM tables is a placeholder (TBD); the only real vLLM-comparison numbers are the measured SGLang sibling's (0.75× at peak, ~3% tax at saturation). |

## External benchmark suites & contracts

SWE-bench, LiveCodeBench, Terminal-Bench, FrontierSWE, and the local coding witness — plus
the contract map that governs which of these may ever carry a public result claim.

| File | Kind | One-liner | Headline |
| --- | --- | --- | --- |
| [BENCHMARK-CONTRACT-MAP.md](BENCHMARK-CONTRACT-MAP.md) | contract | The single map binding each mediated fak eval to its official public benchmark, oracle, artifacts, and caveats. | Every row with result_claim_allowed=false is a contract or local fixture, not a public-leaderboard result; local smokes quotable only as [SIMULATED] |
| [FRONTIERSWE-ENV-ADAPTER.md](FRONTIERSWE-ENV-ADAPTER.md) | adapter | How fak stands up a co-resident fak serve gateway inside a FrontierSWE task sandbox without claiming a benchmark result. | |
| [FRONTIERSWE-RESULTS.md](FRONTIERSWE-RESULTS.md) | pending | The authority page for fak's FrontierSWE time-to-solution (TTS) claim. Created empty and gated: no number yet. | GATED, no number yet — no wall-clock or turn-count TTS number is recorded until the official grader has produced both arms' reward.json and score-parity holds |
| [FRONTIERSWE-SCORING-PARITY.md](FRONTIERSWE-SCORING-PARITY.md) | contract | How internal/frontierswe's Go scorer maps field-for-field onto the published leaderboard oracle score_from_reward.py | |
| [FRONTIERSWE-TTS-RUNBOOK.md](FRONTIERSWE-TTS-RUNBOOK.md) | runbook | The exact end-to-end recipe for a raw-vs-fak FrontierSWE time-to-solution (TTS) comparison. | No TTS number is claimed until the official grader has run and score-parity holds (GATED) |
| [LIVECODEBENCH-EPIC.md](LIVECODEBENCH-EPIC.md) | plan | Repo-side index mirroring the GitHub epic anchor for first-class LiveCodeBench support (#2085). | this index does not carry a score; LiveCodeBench results stay `pending run` |
| [LIVECODEBENCH-RESULTS.md](LIVECODEBENCH-RESULTS.md) | pending | Status: pending run. Authority placeholder recording the fields a real LiveCodeBench result must carry; no pass rate. | pass@1: pending run |
| [LIVECODEBENCH-RUNBOOK.md](LIVECODEBENCH-RUNBOOK.md) | runbook | The command path and evidence contract for running LiveCodeBench through fak without turning a dry run into a score claim. | Status: runbook assembled; native fak adapter pending child issues — all score cells remain `pending run` |
| [LIVECODEBENCH-SUBMISSION-PACKET.md](LIVECODEBENCH-SUBMISSION-PACKET.md) | packet | Assembly index for a future LiveCodeBench leaderboard submission; makes no LiveCodeBench result claim. | Status: BLOCKED_PRECREDENTIAL - no result claim, no authority row yet. |
| [LOCAL-MODEL-CODING-WITNESS-2026-06-27.md](LOCAL-MODEL-CODING-WITNESS-2026-06-27.md) | results | Measured results of a small local model behind fak guard on a minimal coding task, with an honest local-vs-frontier A/B. | MEASURED — the capability axis was run live: both 3B and 7B Qwen2.5-Coder rungs produced the correct fix (1-fail → 2-pass); governance witnessed via replay-trace (8 adjudicated verdicts, 3 dangerous calls denied) |
| [LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md](LOCAL-MODEL-CODING-WITNESS-RUNBOOK.md) | runbook | End-to-end command sequence to measure a small local model behind fak guard on a minimal coding task, with A/B. | Status: the path is ASSEMBLED; the measured numbers are PENDING a real run. |
| [SWEBENCH-PURE-KERNEL-RUNBOOK.md](SWEBENCH-PURE-KERNEL-RUNBOOK.md) | runbook | Exact end-to-end command sequence to resolve SWE-bench Verified with fak's native CUDA engine (no SGLang in the path) | Resolve-rate on the pure kernel: pending GPU run — the path is ASSEMBLED in code and the harness is witnessed |
| [SWEBENCH-RESULTS.md](SWEBENCH-RESULTS.md) | results | A fak-native SWE-bench Verified benchmark directly comparable to the bench tool on cost, cache-reuse, turns, and adjudication | THEORETICAL (MODELED) value-stack floor A/C 17.9×→23.4× (workers 1→16); MEASURED in-process adjudication p50 ~2.4 µs vs ~5.8 ms spawn (~2 400×); resolve-rate not MEASURED/VERIFIED (GPU server-gated) |
| [SWEBENCH-VERIFIED-GPU-SERVER-RESOLVE-COMPARE.md](SWEBENCH-VERIFIED-GPU-SERVER-RESOLVE-COMPARE.md) | results | A real coding-agent SWE-bench Verified run driving Qwen3.6-27B through the fak gateway and raw SGLang, official harness. | The same model resolves the instance through raw SGLang (1/1) but is blocked by fak's capability/trust floor (0/1) — overall completion is decided by the floor, not the model. |
| [TERMINAL-BENCH-2.1-FAILURE-TAXONOMY.md](TERMINAL-BENCH-2.1-FAILURE-TAXONOMY.md) | taxonomy | The general agent behavior that classifies a failed Terminal-Bench task and decides a legal recovery. | Status: engine shipped, pending live-run wiring — no Terminal-Bench number claim; result_claim_allowed stays false |
| [TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md](TERMINAL-BENCH-2.1-SUBMISSION-PACKET.md) | packet | Assembly index tying every checked-in campaign artifact to the promotion gate; makes no Terminal-Bench result claim. | Status: BLOCKED_PRECREDENTIAL — no result claim, no authority row yet; result_claim_allowed=false across every artifact |
