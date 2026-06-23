---
title: "fak industry scorecard — decoding"
description: "The decoding dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# decoding — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Decoding acceleration (`speculative-decoding`)

### ○ Speculative decoding throughput effect across batch sizes (does it survive at high concurrency?) — fak: **no-claim**

*Why it matters:* Speculative decoding accelerates single-stream decode but consumes extra compute per step, so at the high batch sizes a busy server runs it can REDUCE throughput. The differentiating question for a serving operator is not the single-request speedup but whether the method still nets a throughput gain at production batch sizes; only newer drafters do.

- **SOTA bar:** On a single H100 (SGLang), EAGLE-3 gives 1.81x throughput at batch 2 and still 1.38x at batch 64, whereas EAGLE-2 drops to 0.93x at batch 24 (a net loss); acceptance rates approach ~80% with ~1.74-2.4 accepted tokens/step.
- **Leading systems:** EAGLE-3, EAGLE-2 (regresses at batch), vLLM / SGLang / TensorRT-LLM speculative backends
- **Source:** [https://docs.sglang.ai/backend/speculative_decoding.html](https://docs.sglang.ai/backend/speculative_decoding.html) (2025)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for fak's shipped engine: fak borrows the speculative-decoding PATTERN for batched adjudication (verify a set of decisions in one pass) but does not ship a draft/target speculative-decoding generation backend, so there is no EAGLE-style acceptance-rate or batch-scaling throughput number. No fak value exists; named as out-of-scope rather than a measured gap.
- **Trace:** No speculative-decoding throughput-vs-batch number in BENCHMARK-AUTHORITY.md. CLAIMS.md L28 references speculative-decoding only as inverted PRIOR ART for batch ADJUDICATION (set-shape decision equals serial in one pass), not as a generation-speed spec-decode backend; no acceptance-rate or batch-scaling tok/s is measured.

### ○ Speculative decoding effect on ITL (acceptance-rate-bound) — fak: **no-claim**

*Why it matters:* Speculative decoding (EAGLE-3, Medusa, MTP) verifies multiple draft tokens per forward pass, cutting ITL/TPOT 2-4x WITHOUT touching TTFT - the cheapest way to hit aggressive per-user token-rate SLOs at low batch size. The realized speedup is bound by draft acceptance rate, which is workload-dependent, so acceptance rate is itself a first-class latency dimension.

- **SOTA bar:** EAGLE-3 reports ~0.80-0.88 acceptance on coding/instruction tasks giving ~3-4x decode token-rate (ITL) gains on H100/H200, and outperforms EAGLE-2/Medusa-2 by 15-25% tokens/s at batch size 1. Speedup primarily reduces ITL, not TTFT.
- **Leading systems:** EAGLE-3, Medusa-2, TensorRT-LLM / vLLM speculative decoding
- **Source:** [https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/) (2026-01)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP (mechanism built, number unmeasured). fak has the correctness machinery for speculative decoding (Medusa/EAGLE-2/SpecInfer-style tree verify, bit-exact via KVCache.Evict) and the closed-form EffectiveTokensPerVerify model, but no measured acceptance rate or ITL speedup - only the arithmetic, pending the bench harness (#535). EAGLE-3 reports ~0.80-0.88 acceptance / ~3-4x decode-rate on H100/H200. fak claims no number here; no-claim until the lane runs on a GPU.
- **Trace:** CLAIMS.md: polymodel AcceptGreedy/AcceptTree + single-pass batched/tree verify (internal/spec, #533) are SHIPPED as the accept DECISION + bit-exact rollback, proven token-identical to plain greedy - but explicitly 'no GPU, so no tokens/sec', off mainline (NOT in defconfig, FAK_POLYMODEL default off). No acceptance-rate or ITL-speedup number is committed.

### ○ Speculative-decoding wall-clock speedup (single-stream / low-concurrency) — fak: **no-claim**

*Why it matters:* The headline reason operators turn on speculative decoding: how much faster each request finishes at interactive (batch~1) load. It directly sets the latency a chat/agent user feels, and is the number buyers compare across engines and draft methods.

- **SOTA bar:** EAGLE-3 reports up to 4.79x speedup on LLaMA-3.3-70B with no quality loss and is the de-facto industrial standard supported in vLLM and SGLang; vLLM's P-EAGLE reaches ~4-5x over standard decoding (20-30% over EAGLE-3 alone) on coding benchmarks
- **Leading systems:** EAGLE-3 (SafeAILab), vLLM, SGLang, P-EAGLE (vLLM)
- **Source:** [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) (2025-09)
- **fak:** no-claim — no number (projected)
- **fak note:** REAL GAP fak should measure. fak has shipped speculative-decode MECHANISM (greedy + tree verify, single-pass batched), proven token-identical to greedy — but ZERO measured wall-clock speedup: no GPU path, no real draft model, synthetic CPU only, off-mainline by construction. EAGLE-3's up-to-4.79x is a measured wall-clock on LLaMA-3.3-70B; fak has no comparable number, so no verdict is honest. The honest bench is gated on #535.
- **Trace:** CLAIMS.md In-kernel model: internal/polymodel AcceptGreedy/AcceptTree + internal/spec SpeculativeGreedy/SpeculativeTree + model.VerifyForward (rungs #533); honest fence states it runs on the CPU synthetic PreNorm regime, NO GPU, so NO tokens/sec — the speedup is only the closed-form EffectiveTokensPerVerify arithmetic; a measured number needs bench harness #535. Off mainline (FAK_POLYMODEL default off).

### ○ Draft acceptance length / acceptance rate (mean accepted tokens per verify step) — fak: **no-claim**

*Why it matters:* Acceptance length tau is the upstream driver of every spec-decode speedup; it is the engine-independent quality metric of a draft method and the thing that determines whether the verify-step overhead pays off. Operators tune draft depth/tree against measured tau on their own traffic.

- **SOTA bar:** EAGLE-class drafts reach ~0.6-0.8 per-token acceptance (tau ~3-5+ tokens/step with tree decoding); DeepSeek-V3 native MTP shows 85-90% acceptance of the 2nd token (~1.8x TPS); Medusa heads are lower, typically 0.55-0.70
- **Leading systems:** EAGLE-3, DeepSeek-V3 MTP, Medusa
- **Source:** [https://arxiv.org/pdf/2412.19437](https://arxiv.org/pdf/2412.19437) (2024-12)
- **fak:** no-claim — no number (projected)
- **fak note:** REAL GAP fak should measure. fak proves the accept/rollback ACCOUNTING is correct (KEEP+EVICT conservation, lossless), but reports no empirical tau / acceptance rate from a real draft+target pair. EAGLE-class ~0.6-0.8 per-token and DeepSeek-V3 MTP ~85-90% 2nd-token are measured on real models; fak's only number is a parameterized model, so it cannot claim an acceptance rate. Competitor value shown is DeepSeek-V3 MTP 2nd-token acceptance.
- **Trace:** CLAIMS.md: polymodel.EffectiveTokensPerVerify is a geometric-series speedup MODEL (closed-form arithmetic), not an empirical acceptance rate; AcceptGreedy/AcceptTree KEEP/EVICT counts are proven conserved against the bit-exact KVCache.Evict rollback but are exercised on synthetic/adversarial drafts (cmd/polymodelbench forces a rollback every round), not measured on a real draft model against a real target.

### ○ Breadth of speculative methods supported (draft-model, EAGLE/-2/-3, Medusa, MTP, lookahead, n-gram/prompt-lookup) — fak: **no-claim**

*Why it matters:* No single draft method wins on every model/workload, so an engine's menu of methods is a portability and future-proofing axis. Buyers want one engine that covers self-speculative (MTP/Medusa), feature-level (EAGLE), retrieval (n-gram), and external-draft so they can pick per model without re-platforming.

- **SOTA bar:** TensorRT-LLM supports draft-target, Medusa, ReDrafter, Lookahead, and EAGLE(-1/-2, EAGLE-3 via two-model/disaggregated); vLLM supports EAGLE/EAGLE-3, Medusa, MTP, n-gram/prompt-lookup; SGLang supports EAGLE/EAGLE3 and standard draft
- **Leading systems:** TensorRT-LLM, vLLM, SGLang
- **Source:** [https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) (2025-12)
- **fak:** no-claim — no number (projected)
- **fak note:** REAL GAP. fak's tree-verify mechanism is genuinely EAGLE-2/Medusa/SpecInfer-shaped and lossless, but as an unwired CPU-synthetic accounting layer it supports no method end-to-end on the live serving path. TensorRT-LLM/vLLM ship 4-6 production methods (draft-target, Medusa, EAGLE/-3, MTP, lookahead, n-gram). fak claims no breadth; naming this prevents the shipped accept-core from being read as method parity.
- **Trace:** CLAIMS.md In-kernel model: internal/polymodel ships AcceptGreedy (chain) + AcceptTree (the token-TREE generalization explicitly mapped to Medusa/EAGLE-2/SpecInfer) + PickDrafter ensemble selection, with single-pass batched + tree-attention verify (model.VerifyForward, #533). But it is the policy/accounting brain + a CPU synthetic verify only — off mainline (not in defconfig, FAK_POLYMODEL default off); no draft-target with a real draft model, no trained EAGLE/-2/-3 head, no native MTP weights, no n-gram/prompt-lookup.

### ○ Retrieval / prompt-lookup (n-gram) drafting for input-grounded tasks — fak: **no-claim**

*Why it matters:* A zero-draft-model, zero-extra-VRAM acceleration that works whenever the output echoes the input (summarization, RAG, code edit/refactor, structured rewrite). It is the cheapest spec-decode to deploy and a key differentiator for grounded agent workloads where outputs reuse prompt substrings.

- **SOTA bar:** Prompt-lookup decoding shows up to 2.8x speedup on summarization (CNN/DailyMail) and 2x-4x on input-grounded tasks with no quality change; requires no separate draft model
- **Leading systems:** vLLM (ngram / prompt-lookup), TensorRT-LLM (Lookahead)
- **Source:** [https://blog.vllm.ai/2024/10/17/spec-decode.html](https://blog.vllm.ai/2024/10/17/spec-decode.html) (2024-10)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure (it is adjacent to fak's thesis). Prompt-lookup is input-grounded reuse — the SAME family as fak's cross-agent prefix reuse — and needs no draft model, so it is a natural fit for a reuse kernel and worth a measured row. fak has zero evidence today. Competitor value is the up-to-2.8x summarization figure.
- **Trace:** No fak evidence: CLAIMS.md / BENCHMARK-AUTHORITY.md contain no n-gram, prompt-lookup, or retrieval-drafting claim. fak's PickDrafter selects among drafters but no n-gram/prompt-lookup drafter exists.

### ≈ Output-distribution losslessness of speculation (greedy and stochastic) — fak: **parity**

*Why it matters:* Speculative decoding must not change what the model would have produced; rejection-sampling verification preserves the target distribution exactly at T=0 and T>0. An engine that silently alters outputs (or only matches at greedy) fails reproducibility, eval parity, and audit requirements - this is the honesty gate operators must verify, not assume.

- **SOTA bar:** vLLM's implementation is algorithmically validated lossless via rejection sampling, preserving the target distribution under both greedy and sampled decoding up to hardware floating-point precision
- **Leading systems:** vLLM, speculative sampling (Leviathan et al.)
- **Source:** [https://docs.vllm.ai/en/latest/features/speculative_decoding/](https://docs.vllm.ai/en/latest/features/speculative_decoding/) (2025-06)
- **fak:** parity — no number (shipped)
- **fak note:** Correctness-oracle => parity (a losslessness property is matched, never beaten). fak's speculative path is proven token-identical to plain greedy decode (the accepted path is bit-exact via KVCache.Evict rollback), the GREEDY-equivalence guarantee. HONEST FENCE vs the SOTA bar: this is proven for GREEDY only on the CPU synthetic PreNorm model; fak has NOT shown distribution-preserving rejection sampling under STOCHASTIC (temperature>0) decoding the way vLLM's implementation does — fak's in-kernel decode is greedy-only, so the sampled-decoding half of the losslessness bar is unaddressed. The greedy half is at parity; the stochastic half is unbuilt.
- **Trace:** CLAIMS.md In-kernel model (#533): spec.SpeculativeGreedy verifies a k-token draft in ONE VerifyForward pass and stays token-identical to plain greedy; spec.VerifyTree/SpeculativeTree commit only the accepted path and are token-identical to plain greedy decode. Witnesses: TestSpeculativeTreeLosslessGreedyPath (21/63 accepted, distractors rejected by the tree mask), TestSpeculativeTreeLosslessArbitrary (lossless regardless of drafter quality), TestVerifyTreeRewindsAndCommitsCleanly (target cache byte-exact to a greedy session after the round), TestVerifyForwardChainMatchesSerial. go test ./internal/model ./internal/spec.

### ○ Speculative-decoding behavior under high concurrency / large batch (break-even) — fak: **no-claim**

*Why it matters:* Spec decode helps latency at low load but can REGRESS throughput once the GPU is compute-bound - the single most common production footgun. Operators must know the break-even batch and whether the engine auto-disables or has a batch-robust drafter, or they pay for slowdowns at peak QPS.

- **SOTA bar:** Naive spec-decode break-even is ~8-16 concurrent requests; beyond ~32 it adds verify overhead with no benefit. MagicDec uses a fixed-window draft so the draft:target cost ratio falls as batch grows, restoring speedup at large batch for moderate-to-long contexts
- **Leading systems:** MagicDec, vLLM continuous batching, SGLang
- **Source:** [https://arxiv.org/html/2408.11049v1](https://arxiv.org/html/2408.11049v1) (2024-08)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel, and a real gap relative to a full serving stack. Spec-decode-under-high-concurrency (MagicDec fixed-window draft restoring speedup at large batch) presupposes continuous batching, which fak does not have on the live path (continuous-batching is [SIMULATED]; the decode lane is serial by construction). fak makes no batch-scaling claim. Reuses the existing 'continuous-batching' / 'served-throughput-vs-sglang' gap framing.
- **Trace:** CLAIMS.md In-kernel model: the polymodel decode lane is explicitly SERIAL (the at-most-one-model-decodes-per-step invariant is asserted); CLAIMS Engine labels continuous-batching [SIMULATED] (read-only telemetry, not on the live serving path). No batch-scaling / break-even measurement exists.

## Structured output (`structured-output`)

### ○ Constrained/guided decoding per-token overhead (JSON-schema, regex, grammar/CFG) — fak: **no-claim**

*Why it matters:* Guaranteed-valid JSON/regex/grammar output is table stakes for tool-calling and structured APIs, but a slow mask-compute step throttles tokens/sec. The per-token mask overhead is the dominant cost in steady-state constrained generation and the number that decides whether structured output is 'free' or a tax.

- **SOTA bar:** XGrammar (default backend in vLLM/SGLang/TensorRT-LLM) achieves under 40 microseconds per-token overhead via precompiled bitmask FSM; SGLang is reported ~3x faster than vLLM on constrained workloads; LLGuidance is competitive/faster on unique (uncached) schemas
- **Leading systems:** XGrammar, LLGuidance, SGLang, vLLM
- **Source:** [https://blog.squeezebits.com/70642](https://blog.squeezebits.com/70642) (2025-03)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP. Constrained/guided decoding (JSON-schema/regex/CFG bitmask per token) is explicitly [STUB] in fak — fak owns its decode loop in the in-kernel model but has not built the logit-mask. XGrammar's <40 microsecond/token is the bar; fak has no per-token mask overhead to report. Competitor value is XGrammar's <40us/token.
- **Trace:** CLAIMS.md Pre-flight ladder: '[STUB] Decode-time logit-mask (grammar-constrained generation) — the strongest form (never-emit a malformed call) — requires owning the decode loop; not in v0.1.' fak's grammar rung is a post-hoc positional->named TRANSFORM at the syscall boundary, not a per-token decode-time bitmask FSM.

### ○ Grammar/schema compilation latency and cache (cold-start for dynamic agentic schemas) — fak: **no-claim**

*Why it matters:* Agentic workloads change the schema/grammar per request (different tool sets, varying response protocols), so the one-time compile cost - not per-token mask - becomes the bottleneck. Compilation latency and cross-grammar caching determine whether constrained decoding is viable for high-churn tool-calling.

- **SOTA bar:** XGrammar-2 compiles in ~10 ms vs >1000 ms for XGrammar (~80x faster), with >6x tool-calling compilation speedup via Earley-based adaptive mask cache + cross-grammar cache + JIT; integrated into vLLM and SGLang
- **Leading systems:** XGrammar-2, XGrammar, LLGuidance/Guidance
- **Source:** [https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation) (2026-05)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP, partial-scope. fak has a content-addressed dedup CACHE for its tool-arg grammar (so a repeated grammar is not re-parsed), conceptually adjacent to XGrammar's cross-grammar cache, but it is a tool-call repair grammar, not a CFG/JSON-schema decode mask, and carries no committed compile-latency number. XGrammar-2's ~10ms (vs >1000ms) is the bar; fak makes no measured claim. Competitor value is XGrammar-2 ~10ms.
- **Trace:** CLAIMS.md Pre-flight ladder grammar rung: 'content-addressed grammar dedup' exists for the TOOL-INVOCATION grammar (positional->named repair), but there is no committed compilation-latency number, and the decode-time grammar FSM is [STUB] (not built). No XGrammar/Earley-style mask-cache compile metric exists.

### ○ Structured-output validity vs reasoning-quality tax (constraint-induced degradation) — fak: **no-claim**

*Why it matters:* Constrained decoding can guarantee 100% schema-valid output yet DEGRADE task accuracy by forcing answer fields before chain-of-thought. Buyers must weigh validity rate against the reasoning hit; this is the dimension a naive 'we always emit valid JSON' scorecard hides.

- **SOTA bar:** Strict format constraints can cause ~10-30% reasoning degradation when the schema forces answer-before-reasoning; JSONSchemaBench (10K real schemas) is the standard cross-framework benchmark for validity/coverage; mitigation is deferring structure until reasoning completes
- **Leading systems:** JSONSchemaBench (Outlines/XGrammar/Guidance/llama.cpp/OpenAI/Gemini), OpenAI Structured Outputs
- **Source:** [https://arxiv.org/pdf/2501.10868](https://arxiv.org/pdf/2501.10868) (2025-01)
- **fak:** no-claim — no number (stub)
- **fak note:** OUT OF SCOPE for a reuse kernel. Structured-output validity-vs-reasoning-tax (JSONSchemaBench, the answer-before-reasoning degradation) is a property of a constrained-generation engine fak does not have. fak claims nothing here; the related answershape consumer witness measures a different thing (degeneration), not schema conformance.
- **Trace:** No fak evidence: fak runs no constrained generation ([STUB] decode-time logit-mask) and has no JSONSchemaBench run. fak's answershape witness grades degeneration/repetition shape, not schema validity, and its grammar rung repairs tool-arg arity, not output schemas.

### ○ Tool/function-call parsing coverage and robustness (per-model formats) — fak: **no-claim**

*Why it matters:* Agents live or die on correctly extracting structured tool calls from model output; each model family uses a different format (Hermes, llama3_json, mistral, pythonic, qwen3_coder XML). Wrong/absent parser yields malformed args or silent empty calls - so parser coverage and graceful failure are a core agent-serving differentiator.

- **SOTA bar:** vLLM ships per-family parsers (hermes, llama3_json, mistral, pythonic, internlm, granite, plus qwen3/qwen3_coder); when constraints aren't enforced it falls back to extracting from raw text, so args can be malformed - making grammar-backed (constrained) tool calling the robustness frontier
- **Leading systems:** vLLM tool-call parsers, SGLang, XGrammar-2 (grammar-backed tool calling)
- **Source:** [https://docs.vllm.ai/en/latest/features/tool_calling/](https://docs.vllm.ai/en/latest/features/tool_calling/) (2025-12)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP on the measured axis; mechanism is adjacent but narrow. fak has a SHIPPED boundary mechanism that hardens tool calls (arity repair + MISROUTE deny + MALFORMED deny on bad writes), which is robustness-shaped, but it is NOT the per-family format-parser coverage the SOTA bar measures (vLLM's hermes/llama3_json/mistral/pythonic/qwen3_coder parsers) and carries no coverage/malformed-rate number. So fak makes no coverage claim. Grammar-backed (constrained) tool calling — the robustness frontier — is the [STUB] decode-time mask fak has not built.
- **Trace:** CLAIMS.md Pre-flight ladder grammar rung [SHIPPED]: positional->named auto-repair (in-syscall TRANSFORM, no model turn) for arity-matched calls; unrepairable => Deny(MISROUTE); fail-open on unknown grammar (units 52-57). CLAIMS adjudicator: write-scoped codelint Deny(MALFORMED) on unparseable Go/JSON (#536). Gateway mints/validates calls. But there is NO per-model-family parser suite and NO measured coverage/robustness number.

### ○ Sampler coverage and correctness (temperature, top-p, top-k, min-p) — fak: **no-claim**

*Why it matters:* Sampling controls the quality/diversity/coherence tradeoff and must be implemented correctly and identically across engines for eval parity. min-p in particular keeps outputs coherent at high temperature and is now an ecosystem default; missing or buggy samplers force operators to over-constrain temperature and lose quality.

- **SOTA bar:** min-p (dynamic confidence-scaled truncation) improves quality and diversity over top-p/top-k at high temperature on GPQA/GSM8K/AlpacaEval and is now natively supported across vLLM, llama.cpp, SGLang, HF Transformers, Ollama, ExLlamaV2; TensorRT-LLM supports top-k/top-p/combined plus beam search
- **Leading systems:** vLLM, llama.cpp, SGLang, TensorRT-LLM
- **Source:** [https://arxiv.org/abs/2407.01082](https://arxiv.org/abs/2407.01082) (2024-07)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP. fak's in-kernel model decodes greedily only (argmax), which is what makes its losslessness proofs tractable, but it ships NO stochastic samplers (temperature/top-p/top-k/min-p). min-p is now native across vLLM/llama.cpp/SGLang/TensorRT-LLM; fak has none. A production serving engine needs samplers; fak makes no claim. (When fronting an external engine via fak guard / fak serve, sampling passes through to that engine untouched — fak's own engine has no sampler.)
- **Trace:** No fak evidence: CLAIMS.md In-kernel model describes 'a real greedy Prefill+Step decode' and 'real greedy speculative decode is token-identical to plain greedy' — the decode path is GREEDY-ONLY. No temperature, top-p, top-k, or min-p sampler is mentioned in CLAIMS.md or BENCHMARK-AUTHORITY.md.

