---
title: "fak industry scorecard — agent"
description: "The agent dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# agent — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Cache-aware routing (`request-routing`)

### ○ Cache-aware request routing across replicas (route to the worker that already holds the prefix) — fak: **no-claim**

*Why it matters:* At fleet scale a naive load balancer scatters requests so the same prefix is recomputed on every replica. Locality-aware routing turns single-node prefix caching into a fleet-wide property, the difference between a cache that works at 1 GPU and one that works at 100.

- **SOTA bar:** NVIDIA Dynamo's KV-cache-aware Smart Router reports TTFT and average-request-latency speedups by routing to replicas already holding matching KV (measured on 100K real DeepSeek-R1 requests, 4K/800 in/out, 8x R1-Distill-Llama-70B on 2x HGX-H100); SGLang's cache-aware router and the Meta/vLLM router pioneered the pattern. Combined with disaggregation Dynamo cites up to 30x more requests served (DeepSeek-R1 671B on GB200 NVL72). Exact tokens/GPU/s not disclosed (projected).
- **Leading systems:** NVIDIA Dynamo (Smart Router), SGLang router, vLLM production router
- **Source:** [https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) (2025-03-18)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure if it ever scales beyond one engine. fak is a single in-process kernel: it owns one KV cache and has a cross-agent coherence BUS (vdso.Revoke broadcast, causal invalidation witnessed) but NO multi-replica router that steers a request to the worker already holding its prefix (the Dynamo Smart Router / SGLang cache-aware router pattern). fak's cache-aware scheduling is intra-kernel (FCFS->DFS recovers 62.1%->86.7%), not cross-replica routing. No fak number exists; honest verdict is no-claim, not parity.
- **Trace:** No row in BENCHMARK-AUTHORITY.md; CLAIMS.md has coherence_broadcast (causal-invalidation) but no replica-routing claim

## Agent / fleet serving (`agent-fleet`)

### ▲ Cross-agent fleet serving time (N agents × T turns): work eliminated by shared-prefix fusion — fak: **lead**

*Why it matters:* fak's product category. When a fleet of agents shares a large system/context prefix, fusing that prefix once across all agents eliminates re-prefill work that even a tuned per-agent cache repeats. This is the axis fak is built to win.

- **SOTA bar:** A tuned warm per-agent KV cache (the SGLang / vLLM APC / OpenAI prompt-caching discipline) is the honest floor a serious operator already runs; fak measures the work eliminated ON TOP of it.
- **Leading systems:** SGLang RadixAttention, vLLM Automatic Prefix Caching, OpenAI prompt caching
- **Source:** [https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) (2024-12)
- **fak:** lead — 19 min (shipped)
- **fak note:** Both arms run live on ONE shared kernel held constant, so the 4.1× isolates cross-agent prefix fusion — work ELIMINATED, not a faster kernel. A production vLLM/SGLang would shrink both arms' absolute minutes but is expected to leave the reuse ratio ~intact (κ·N: uniform per-token speedup scales both arms).
- **Trace:** 2bbda6f · experiments/session/headline-qwen-50x5.json · BENCHMARK-AUTHORITY.md

### ▲ Marginal value of cross-agent prefix fusion ON TOP of an already-warm per-agent cache — fak: **lead**

*Why it matters:* The honest few-fold: not the headline-vs-naive number, but the gain over the tuned warm-cache a real operator runs. This is the conservative claim a skeptic should be shown.

- **SOTA bar:** An already-hot per-agent KV cache (1×). fak's cross-agent fusion adds a conservative 2.4–2.7× on top, measured on its own kernel held constant.
- **Leading systems:** Tuned warm per-agent KV cache
- **Source:** [https://www.lmsys.org/blog/2024-01-17-sglang/](https://www.lmsys.org/blog/2024-01-17-sglang/) (2024-01)
- **fak:** lead — 2.4 × (shipped)
- **fak note:** Cross-agent prefix fusion ON TOP of an already-warm per-agent cache — the honest marginal few-fold (2.4–2.7×, conservative end shown), NOT the 60× headline vs naive. FENCE: the 'tuned' baseline is fak's OWN warm-KV arm B (bit-exact, kernel held constant), NOT a live SGLang/vLLM process — a real competing engine also fuses a shared prefix once, so the marginal-vs-a-live-tuned-process number is unmeasured (see row 'marginal-vs-tuned-process').
- **Trace:** 92896a4 · SESSION-VALUE-STACK-RESULTS.md · BENCHMARK-AUTHORITY.md

### ○ Cross-agent reuse marginal value vs a LIVE tuned shared-prefix engine (head-to-head) — fak: **no-claim**

*Why it matters:* A live SGLang/vLLM ALSO fuses a shared prefix once. The marginal win vs a real competing process — not vs fak's own warm-KV arm — is the number a buyer comparing engines actually wants, and fak has not measured it.

- **SOTA bar:** SGLang/vLLM fuse a shared prefix once across requests; on the WebVoyager-style workload the marginal-vs-tuned win is only ~1.0–1.10×.
- **Leading systems:** SGLang RadixAttention, vLLM Automatic Prefix Caching
- **Source:** [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104) (2024-01)
- **fak:** no-claim — no number (projected)
- **fak note:** Every committed fleet multiplier (4.1×, 60.3×, 139×) is measured vs NAIVE re-prefill or fak's OWN kernel held constant in warm-KV mode — NOT a head-to-head against a live SGLang/vLLM process, which ALSO fuses a shared prefix once. The honest marginal-vs-a-real-tuned-process number is UNMEASURED; the repo's WebVoyager baseline-stratification record puts the marginal-vs-tuned win at only ~1.0–1.10× on that workload. Naming this gap stops the 4.1× from being read as a vs-live-competitor result.
- **Trace:** HERO-BENCHMARK-2026-06-21.md · docs/webbench-real-measurements-summary.md

## Model routing (`model-routing`)

### ▲ Model routing granularity: per-aspect + first-class ensemble routing vs whole-request single-model selection — fak: **lead**

*Why it matters:* Every mainstream 'LLM router' (RouteLLM, Martian, NotDiamond, Unify, OpenRouter, Portkey, LiteLLM Router) answers one question — which SINGLE model should serve this WHOLE request — and the only shipped model ensemble is a single fixed recipe (OpenRouter Fusion). Routing a sub-request aspect (one tool call, one reasoning step) to its own model, or declaring a configurable per-aspect ensemble with a reduction, is a granularity layer no surveyed product exposes; it is the axis that decides whether routing is a per-request pick or a first-class, in-loop decision.

- **SOTA bar:** Surveyed 2025-2026 routers/gateways (RouteLLM, Martian, NotDiamond, Unify, OpenRouter, Portkey, LiteLLM Router) all route the WHOLE request to ONE model; OpenRouter Fusion is the only shipped model ensemble and it is a fixed parallel-synthesize recipe, not a configurable per-aspect reduction. None routes a sub-request aspect to its own model. (Aurelio Semantic-Router routes to an intent, not a model; vLLM/SGLang routers balance replicas of ONE model for KV locality — a different layer.)
- **Leading systems:** RouteLLM (LMSYS), Martian, NotDiamond, Unify.ai, OpenRouter (+ Fusion), Portkey, LiteLLM Router
- **Source:** [https://github.com/anthony-chaudhary/fak/blob/main/docs/model-routing.md](https://github.com/anthony-chaudhary/fak/blob/main/docs/model-routing.md) (2026-06)
- **fak:** lead — no number (shipped)
- **fak note:** To our knowledge / among surveyed 2025-2026 routers and gateways (RouteLLM, Martian, NotDiamond, Unify, OpenRouter, Portkey, LiteLLM Router), every one routes the WHOLE request to ONE model and the only shipped model ensemble is OpenRouter Fusion — a fixed parallel-synthesize recipe, not a configurable reduction. fak routes at the ASPECT level (request | tool_call | query | step), so one request can send different aspects to different models, with first-class ENSEMBLES folded by a configurable reduction (first | vote | best_of | all_reduce | concat), expressed as one deterministic, verifiable policy. The decision spine (Route + Combine) and the offline benchmark (fak routebench) are SHIPPED (commit 2298421, witnessed by go test), so this is a CATEGORICAL capability lead on routing granularity and ensemble expressiveness — NOT a benchmarked speed/quality win. Any '10x' is a target to be measured, never an inferred or borrowed number; the live multi-model dispatch that executes a decision on real engines is STUB, so the end-to-end competitive latency/cost/quality delta is unmeasured.
- **Trace:** 2298421 · docs/model-routing.md (survey + status); BENCHMARK-AUTHORITY.md (fak routebench)

