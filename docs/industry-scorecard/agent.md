---
title: "fak industry scorecard — agent"
description: "The agent dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# agent — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Cache-aware routing (`request-routing`)

### ≈ Cache-aware request routing across replicas (route to the worker that already holds the prefix) — fak: **parity**

*Why it matters:* At fleet scale a naive load balancer scatters requests so the same prefix is recomputed on every replica. Locality-aware routing turns single-node prefix caching into a fleet-wide property, the difference between a cache that works at 1 GPU and one that works at 100.

- **SOTA bar:** Cross-region KV-locality-aware routing (GORGO-proxy) makes network latency an explicit routing term and cuts median TTFT ~2.5x vs least-load / prefix-trie baselines (224ms vs 568ms) while preserving prefix-cache locality across regions.
- **Leading systems:** GORGO / GORGO-proxy (cross-region, arXiv 2602.11688), SkyWalker (locality-aware cross-region LB, EuroSys 2026, arXiv 2505.24095)
- **Source:** [https://arxiv.org/html/2602.11688v1](https://arxiv.org/html/2602.11688v1) (2026-02)
- **fak:** parity — 93.97 % cross-replica prefix-KV cache-hit (shipped)
- **fak note:** fak now ships the multi-replica router this row asked for: gateway.FleetCacheRouter steers a request to the instance already holding its prefix KV, homing a cold prefix on the least-loaded instance (the Dynamo Smart Router / SGLang cache-aware router pattern), composed from the on-instance residency signal + the cross-agent coherence bus (vdso.Revoke). WITNESSED number: 93.97% cross-instance prefix-KV hit rate on a Zipf-skewed shared-prefix agent fleet (4 instances x 8 prefixes vs 28 families), a +10.4% lift over the cache-blind round-robin that gateway.ReplicaRouter.pick does today (85.13%). Sits inside the published cross-replica cache-hit band (50-99%) and clears Baseten-on-Dynamo's 0.89 -> parity, the fleet analogue of the on-instance 86.7%. FENCE: this is the routing-DECISION hit rate from a deterministic host-free simulation of the placement policy (no GPUs) -> apples_to_apples=false (same metric class as Baseten's live cross-replica hit, different apparatus). The downstream wall-clock win those hits produce on real GPUs (Baseten -50% TTFT / +62% output tok/s; GORGO 2.5x median TTFT) is host-gated and explicitly NOT claimed by this number.
- **Trace:** experiments/kv-fleet-routing/kv-fleet-routing-hitrate-20260627.json · internal/gateway/kv_fleet_routing.go (MeasureKVAwareFleetRouting) witnessed by TestKVAwareFleetRoutingHitRate; artifact kv-fleet-routing-hitrate-20260627.json (verify kv_aware_locality.hit_rate=0.9396551724137931). The cross-instance analogue of the on-instance row 'cache-aware-scheduling-routing' (FCFS 62.1% -> 86.7%).

## Agent / fleet serving (`agent-fleet`)

### ▲ Cross-agent fleet serving time (N agents Ã— T turns): work eliminated by shared-prefix fusion — fak: **lead**

*Why it matters:* fak's product category. When a fleet of agents shares a large system/context prefix, fusing that prefix once across all agents eliminates re-prefill work that even a tuned per-agent cache repeats. This is the axis fak is built to win.

- **SOTA bar:** The competing floor remains tuned warm per-agent KV reuse (vLLM Automatic Prefix Caching / SGLang RadixAttention / OpenAI prompt caching); fak's own measured 50x5 fleet-serving result (19 min vs 78 min warm-KV baseline, 4.1x from cross-agent prefix fusion) is the live number on this dim and no external SOTA has displaced the warm-KV-reuse baseline class.
- **Leading systems:** vLLM Automatic Prefix Caching, SGLang RadixAttention, OpenAI prompt caching (warm-KV reuse floor)
- **Source:** [https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) (2026-01)
- **fak:** lead — 19 min (shipped)
- **fak note:** Both arms run live on ONE shared kernel held constant, so the 4.1× isolates cross-agent prefix fusion — work ELIMINATED, not a faster kernel. A production vLLM/SGLang would shrink both arms' absolute minutes but is expected to leave the reuse ratio ~intact (κ·N: uniform per-token speedup scales both arms).
- **Trace:** 2bbda6f · experiments/session/headline-qwen-50x5.json · BENCHMARK-AUTHORITY.md

### ▲ Marginal value of cross-agent prefix fusion ON TOP of an already-warm per-agent cache — fak: **lead**

*Why it matters:* The honest few-fold: not the headline-vs-naive number, but the gain over the tuned warm-cache a real operator runs. This is the conservative claim a skeptic should be shown.

- **SOTA bar:** The conservative SOTA point is fak's own already-warm per-agent KV arm (baseline 1.0x); cross-agent prefix fusion on top of it yields a measured ~2.4-2.7x marginal gain. No external benchmark redefines the already-warm-cache baseline for this dim.
- **Leading systems:** Tuned warm per-agent KV cache, already hot (vLLM APC / SGLang RadixAttention class)
- **Source:** [https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) (2026-01)
- **fak:** lead — 2.4 × (shipped)
- **fak note:** Cross-agent prefix fusion ON TOP of an already-warm per-agent cache — the honest marginal few-fold (2.4–2.7×, conservative end shown), NOT the 60× headline vs naive. FENCE: the 'tuned' baseline is fak's OWN warm-KV arm B (bit-exact, kernel held constant), NOT a live SGLang/vLLM process — a real competing engine also fuses a shared prefix once, so the marginal-vs-a-live-tuned-process number is unmeasured (see row 'marginal-vs-tuned-process').
- **Trace:** 92896a4 · SESSION-VALUE-STACK-RESULTS.md · BENCHMARK-AUTHORITY.md

### ○ Cross-agent reuse marginal value vs a LIVE tuned shared-prefix engine (head-to-head) — fak: **no-claim**

*Why it matters:* A live SGLang/vLLM ALSO fuses a shared prefix once. The marginal win vs a real competing process â€” not vs fak's own warm-KV arm â€” is the number a buyer comparing engines actually wants, and fak has not measured it.

- **SOTA bar:** Head-to-head marginal value of cross-agent reuse vs a LIVE tuned shared-prefix engine (vLLM APC / SGLang RadixAttention, which also fuses a shared prefix once) remains UNMEASURED; fak's own WebVoyager stratification puts the marginal-vs-tuned win at only ~1.0-1.10x on that workload.
- **Leading systems:** SGLang RadixAttention (live process), vLLM Automatic Prefix Caching (live process)
- **Source:** [https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) (2026-01)
- **fak:** no-claim — no number (projected)
- **fak note:** Every committed fleet multiplier (4.1×, 60.3×, 139×) is measured vs NAIVE re-prefill or fak's OWN kernel held constant in warm-KV mode — NOT a head-to-head against a live SGLang/vLLM process, which ALSO fuses a shared prefix once. The honest marginal-vs-a-real-tuned-process number is UNMEASURED; the repo's WebVoyager baseline-stratification record puts the marginal-vs-tuned win at only ~1.0–1.10× on that workload. Naming this gap stops the 4.1× from being read as a vs-live-competitor result.
- **Trace:** HERO-BENCHMARK-2026-06-21.md · docs/webbench-real-measurements-summary.md

## Model routing (`model-routing`)

### ▲ Model routing granularity: per-aspect + first-class ensemble routing vs whole-request single-model selection — fak: **lead**

*Why it matters:* Every mainstream 'LLM router' (RouteLLM, Martian, NotDiamond, Unify, OpenRouter, Portkey, LiteLLM Router) answers one question â€” which SINGLE model should serve this WHOLE request â€” and the only shipped model ensemble is a single fixed recipe (OpenRouter Fusion). Routing a sub-request aspect (one tool call, one reasoning step) to its own model, or declaring a configurable per-aspect ensemble with a reduction, is a granularity layer no surveyed product exposes; it is the axis that decides whether routing is a per-request pick or a first-class, in-loop decision.

- **SOTA bar:** Surveyed 2025-2026 routers/gateways (RouteLLM, Martian, NotDiamond, Unify, OpenRouter, Portkey, LiteLLM Router) all route the WHOLE request to ONE model; OpenRouter Fusion is the only shipped model ensemble and it is a fixed parallel-synthesize recipe, not a configurable per-aspect reduction. None routes a sub-request aspect to its own model. (Aurelio Semantic-Router routes to an intent, not a model; vLLM/SGLang routers balance replicas of ONE model for KV locality â€” a different layer.)
- **Leading systems:** RouteLLM (LMSYS), Martian, NotDiamond, Unify.ai, OpenRouter (+ Fusion), Portkey, LiteLLM Router
- **Source:** [https://github.com/anthony-chaudhary/fak/blob/main/docs/model-routing.md](https://github.com/anthony-chaudhary/fak/blob/main/docs/model-routing.md) (2026-06)
- **fak:** lead — no number (shipped)
- **fak note:** To our knowledge / among surveyed 2025-2026 routers and gateways (RouteLLM, Martian, NotDiamond, Unify, OpenRouter, Portkey, LiteLLM Router), every one routes the WHOLE request to ONE model and the only shipped model ensemble is OpenRouter Fusion — a fixed parallel-synthesize recipe, not a configurable reduction. fak routes at the ASPECT level (request | tool_call | query | step), so one request can send different aspects to different models, with first-class ENSEMBLES folded by a configurable reduction (first | vote | best_of | all_reduce | concat), expressed as one deterministic, verifiable policy. The decision spine (Route + Combine) and the offline benchmark (fak routebench) are SHIPPED (commit 2298421, witnessed by go test), so this is a CATEGORICAL capability lead on routing granularity and ensemble expressiveness — NOT a benchmarked speed/quality win. Any '10x' is a target to be measured, never an inferred or borrowed number; the live multi-model dispatch that executes a decision on real engines is STUB, so the end-to-end competitive latency/cost/quality delta is unmeasured.
- **Trace:** 2298421 · docs/model-routing.md (survey + status); BENCHMARK-AUTHORITY.md (fak routebench)

## Client-side context compaction (`client-compaction`)

### ○ Long-session history compaction that preserves the provider prompt-cache prefix (drop-and-splice vs summarize-and-resend) — fak: **no-claim**

*Why it matters:* An agent re-sends its whole transcript every turn; the provider prompt cache discounts the unchanged prefix ONLY while it stays byte-for-byte identical. Almost every built-in compaction (Aider, LangChain ConversationSummaryMemory, Codex CLI, Copilot CLI, Anthropic API context-editing) summarizes or clears old turns and re-sends a REWRITTEN prompt â€” which by the providers' own docs breaks that exact-match prefix and re-bills it at full price on the compacting turn. Whether a tool can shrink a long conversation WITHOUT busting the cache is the axis that decides if compaction saves money on the turn it fires or costs it.

- **SOTA bar:** The field splits two ways. Summarizers (Aider, LangChain ConversationSummaryMemory, Codex CLI, Copilot CLI) and Anthropic's API context-editing REWRITE/clear content and break the prefix cache (OpenAI docs: 'when you drop, summarize or compact earlier turns ... you'll break the cache'; Anthropic context-editing docs: 'Invalidates cached prompt prefixes when content is cleared'). The cache-preserving sub-field is small: LangChain trim_messages (pure sliding-window drop, no rewrite) and Copilot's DELIBERATE cache-boundary reset reason about it; no surveyed tool ships a PROVEN byte-identity splice on by default. Context-editing pairs clearing with a memory tool for recovery (a different axis: recoverability), reporting up to ~84% token reduction on an Anthropic agent eval.
- **Leading systems:** Anthropic API context-editing (clear_tool_uses_20250919), OpenAI Codex CLI (summary compaction), GitHub Copilot CLI (cache-boundary compaction), Aider (ChatSummary), LangChain (trim_messages / ConversationSummaryMemory)
- **Source:** [https://platform.claude.com/docs/en/docs/build-with-claude/context-editing](https://platform.claude.com/docs/en/docs/build-with-claude/context-editing) (2026-01)
- **fak:** no-claim — no number (in-flight)
- **fak note:** HONEST NO-CLAIM, the audit's single biggest open risk. fak proves the shipped prefix is byte-identical, which makes a cache hit ELIGIBLE; whether the provider's lookup CASCADES to the head breakpoint after the middle is dropped (vs missing at the shifted recent breakpoint and re-billing the dropped middle as fresh input) is unverified by direct telemetry. If the cascade fails, the claimed shed savings collapse toward zero on the compacting turn. No fak_value is asserted until witnessed shed is correlated against observed cache_read on real Anthropic traffic (off vs on). The byte-identity row above is a mechanism lead; THIS is the cost realization, deliberately kept separate so the first is never read as the second.
- **Trace:** The instrument exists but is unmeasured on real traffic: /metrics emits fak_gateway_compaction_shed_tokens_total (WITNESSED, 'what fak SENT') next to fak_gateway_compaction_cache_read_tokens_total (OBSERVED, provider cache_read, 'attribute nothing to fak from it alone') — internal/gateway/metrics.go writeCompactionMetrics. The dogfood (compact-100k-session-dogfood-2026-06-25.json) runs vs a MOCK upstream, so it proves byte-identity, not provider reuse. Settling this is one credentialed Anthropic session scraped, not new code. Epic #745.

## Long-horizon time-to-solution (`time-to-solution`)

### ○ Long-horizon agent time-to-solution: wall-clock time for an agent (or agent fleet) to actually FINISH a multi-hour engineering task — fak: **no-claim**

*Why it matters:* The serving field competes on tokens/s, goodput, and latency - the throughput of a single forward pass. None of that measures the thing an agent operator actually pays for: how long until the WORK is done. FrontierSWE exists precisely because frontier models barely make progress on real ultra-long-horizon engineering tasks even given 20 hours each, so nobody is yet competing on FINISHING the same work faster. That is the dimension fak value stack (cross-agent prefix fusion, cache-value, disinterested-referee orchestration) is built to win, so it belongs on the scorecard as a first-class axis with FrontierSWE as the SOTA bar.

- **SOTA bar:** FrontierSWE (Proximal Labs, published 2026-04-16) is the field ultra-long-horizon coding benchmark: 17 real tasks (5 implementation, 9 performance-engineering, 3 ML-research) with a 20-hour-per-task budget, graded on a zero-to-one scale rather than binary pass/fail. It is UNSATURATED - the top model (Claude Fable 5, avg rank 2.35 / 90 percent dominance) still fails to solve almost all tasks, and agents burn many hours (e.g. Opus 4.6 ~6.6-13.8h/task) making little progress. There is no published time-to-solution winner: the SOTA bar is that long-horizon completion is an OPEN problem the field measures but nobody yet finishes fast.
- **Leading systems:** FrontierSWE (Proximal Labs) - the ultra-long-horizon coding benchmark, Claude Fable 5 / Opus 4.6 + agent scaffold (top FrontierSWE ranks), GPT-5.4 + agent scaffold
- **Source:** [https://www.proximal.ai/blog/frontierswe](https://www.proximal.ai/blog/frontierswe) (2026-04)
- **fak:** no-claim — no number (projected)
- **fak note:** HONEST NAMED GAP, not a win. The competitive thesis - that fak cross-agent prefix fusion + cache-value + disinterested-referee orchestration FINISH long-horizon work faster - is projected/gated, NOT yet witnessed. FrontierSWE is the SOTA bar precisely because the field has no time-to-solution winner: models barely progress in 20h. fak has run NO FrontierSWE task end-to-end, so there is no fak time-to-solution number; this row exists to set the bar the eventual witnessed measurement (epic #1706 C14/C15) gets scored against. Any speedup here is a target to be measured, never inferred or borrowed.
- **Trace:** epic #1706 (FrontierSWE first-class support - time-to-solution); C4 (TTS model) -> C14/C15 close this row with a witnessed number. Related on-axis: [[fleet-serving-50x5]] (cross-agent fleet serving) and [[marginal-vs-tuned-process]] (the honest marginal-vs-live-engine gap).

