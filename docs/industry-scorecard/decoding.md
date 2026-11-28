---
title: "fak industry scorecard — decoding"
description: "The decoding dimensions that matter in LLM serving, the current SOTA bar on each, and fak's honest position. Generated from tools/industry_scorecard.data/."
---

# decoding — the dimensions that matter, and where fak stands

[← back to the scorecard index](README.md) · part of the industry-first scorecard. Each dimension is a thing the field competes on; the fak column is honest — mostly `no-claim` gaps for a focused reuse kernel.

## Decoding acceleration (`speculative-decoding`)

### ○ Speculative decoding throughput effect across batch sizes (does it survive at high concurrency?) — fak: **no-claim**

*Why it matters:* Speculative decoding accelerates single-stream decode but consumes extra compute per step, so at the high batch sizes a busy server runs it can REDUCE throughput. The differentiating question for a serving operator is not the single-request speedup but whether the method still nets a throughput gain at production batch sizes; only newer drafters do.

- **SOTA bar:** EAGLE-3 remains the named SOTA: ~1.38x throughput at batch 64 in SGLang on a single H100, peaking at ~1.81x at batch 2 and up to ~6.5x single-stream, ~1.4x over EAGLE-2.
- **Leading systems:** EAGLE-3 (SafeAILab), SGLang speculative backend, vLLM/TensorRT-LLM spec backends
- **Source:** [https://arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840) (2025-03)
- **fak:** no-claim — no number (in-flight)
- **fak note:** SINGLE-STREAM SHIPPED, HIGH-CONCURRENCY UNMEASURED: fak now ships opt-in CUDA DSpark speculative decoding on Inkling-2-Mini (1.43x d2 / 1.51x d4 at C=1, 67bb976926a) alongside NativeScheduler batching, but has NOT measured whether speculative speedup survives at high serving batch concurrency (e.g. batch 16-64 vs EAGLE-3). Honest no-claim.
- **Trace:** 67bb976926a · docs/benchmarks/published/HEADLINES.json · docs/benchmarks/published/HEADLINES.json (67bb976926a C=1 DSpark 171.54/180.92 tok/s); no C=16..64 spec-decode sweep

### ○ Speculative decoding effect on ITL (acceptance-rate-bound) — fak: **no-claim**

*Why it matters:* Speculative decoding (EAGLE-3, Medusa, MTP) verifies multiple draft tokens per forward pass, cutting ITL/TPOT 2-4x WITHOUT touching TTFT - the cheapest way to hit aggressive per-user token-rate SLOs at low batch size. The realized speedup is bound by draft acceptance rate, which is workload-dependent, so acceptance rate is itself a first-class latency dimension.

- **SOTA bar:** EAGLE-3 reports ~0.80-0.88 acceptance on coding/instruction tasks giving ~3-4x decode token-rate (ITL) gains on H100/H200, and outperforms EAGLE-2/Medusa-2 by 15-25% tokens/s at batch size 1. Speedup primarily reduces ITL, not TTFT.
- **Leading systems:** EAGLE-3, Medusa-2, TensorRT-LLM / vLLM speculative decoding
- **Source:** [https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/) (2026-01)
- **fak:** no-claim — no number (in-flight)
- **fak note:** C1 RATE MEASURED ON RTX 5080 (see spec-decode-speedup 1.51x and acceptance-length-rate 0.70), P50/P99 HTTP ITL DISTRIBUTION UNMEASURED: fak measures C=1 in-process DSpark decode speedups (171.54 tok/s d2 / 180.92 tok/s d4 vs 119.72 plain) plus spec oracle depth 4-8 (294-326 tok/s on f32 KV in rtx5080-inkling2-v060), but has no per-token P50/P99 HTTP ITL comparison vs EAGLE-3.
- **Trace:** 67bb976926a · docs/benchmarks/published/HEADLINES.json · docs/benchmarks/published/HEADLINES.json (spec.d2=171.535, spec.d4=180.92); aegis-inkling2-spec-decode/RESULTS.md

### ▼ Speculative-decoding wall-clock speedup (single-stream / low-concurrency) — fak: **trails**

*Why it matters:* The headline reason operators turn on speculative decoding: how much faster each request finishes at interactive (batch~1) load. It directly sets the latency a chat/agent user feels, and is the number buyers compare across engines and draft methods.

- **SOTA bar:** EAGLE-3 reports up to 4.79x speedup on LLaMA-3.3-70B with no quality loss and is the de-facto industrial standard supported in vLLM and SGLang; vLLM's P-EAGLE reaches ~4-5x over standard decoding (20-30% over EAGLE-3 alone) on coding benchmarks
- **Leading systems:** EAGLE-3 (SafeAILab), vLLM, SGLang, P-EAGLE (vLLM)
- **Source:** [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) (2025-09)
- **fak:** trails — 1.51 x wall-clock speedup (shipped)
- **fak note:** HW-WITNESSED on RTX 5080 (Inkling-2-Mini-UD-Q3_K_XL, opt-in FAK_INKLING2_SPEC): DSpark depth 4 reaches 180.92 tok/s (1.51x vs plain 119.715 tok/s, capitals prompt; depth 2 reaches 171.535 tok/s = 1.43x, superseding prior 151.65 tok/s). Opt-in and not bit-exact on every prompt; trails EAGLE-3's up-to-4.79x on LLaMA-3.3-70B.
- **Trace:** 67bb976926a · docs/benchmarks/published/HEADLINES.json · docs/benchmarks/published/HEADLINES.json (67bb976926a: spec.d4=180.92 vs plain 119.715, spec.d2=171.535); docs/benchmarks/published/aegis-inkling2-spec-decode/RESULTS.md

### ▼ Draft acceptance length / acceptance rate (mean accepted tokens per verify step) — fak: **trails**

*Why it matters:* Acceptance length tau is the upstream driver of every spec-decode speedup; it is the engine-independent quality metric of a draft method and the thing that determines whether the verify-step overhead pays off. Operators tune draft depth/tree against measured tau on their own traffic.

- **SOTA bar:** EAGLE-3 reports per-task acceptance rate ~0.80-0.88 (coding/instruction) and mean accepted length ~4.5-5.0 tokens per draft-verify cycle, exceeding the older DeepSeek-V3 MTP ~0.85 second-token figure.
- **Leading systems:** EAGLE-3 (SafeAILab), DeepSeek-V3 MTP, FastMTP / FastEagle (2025 successors)
- **Source:** [https://arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840) (2025-03)
- **fak:** trails — 0.7 draft acceptance rate (fraction; d2=2.40 tokens/verify) (shipped)
- **fak note:** HW-WITNESSED on RTX 5080 (Inkling-2-Mini DSpark depth 2): 74/106 drafts accepted (0.70 acceptance rate, 2.40 tokens/verify in aegis-inkling2-spec-decode prior to the 67bb976926a verify-kernel climb reaching 171.54 tok/s at d2 and 180.92 tok/s at d4). Trails EAGLE-3 (0.80-0.88 rate, ~4.5-5.0 tokens/verify) and DeepSeek-V3 MTP (0.85-0.90 2nd-token rate).
- **Trace:** 67bb976926a · docs/benchmarks/published/HEADLINES.json · docs/benchmarks/published/HEADLINES.json (67bb976926a: spec.d2=171.535, spec.d4=180.92 tok/s superseding 5915ef42be); docs/benchmarks/published/aegis-inkling2-spec-decode/RESULTS.md (74/106 drafts = 0.70, 2.40 tok/verify at depth 2)

### ○ Breadth of speculative methods supported (draft-model, EAGLE/-2/-3, Medusa, MTP, lookahead, n-gram/prompt-lookup) — fak: **no-claim**

*Why it matters:* No single draft method wins on every model/workload, so an engine's menu of methods is a portability and future-proofing axis. Buyers want one engine that covers self-speculative (MTP/Medusa), feature-level (EAGLE), retrieval (n-gram), and external-draft so they can pick per model without re-platforming.

- **SOTA bar:** TensorRT-LLM supports draft-target, Medusa, ReDrafter, Lookahead, and EAGLE(-1/-2, EAGLE-3 via two-model/disaggregated); vLLM supports EAGLE/EAGLE-3, Medusa, MTP, n-gram/prompt-lookup; SGLang supports EAGLE/EAGLE3 and standard draft
- **Leading systems:** TensorRT-LLM, vLLM, SGLang
- **Source:** [https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) (2025-12)
- **fak:** no-claim — no number (in-flight)
- **fak note:** PARTIAL COVERAGE SHIPPED, STILL TRAILS SOTA MENU: fak now ships CUDA DSpark drafting (opt-in on Inkling-2-Mini) and deterministic toolgrammar.ForcedDraft jump-forward decoding (020586f72c) plus CPU tree-verify primitives, yet lacks general external draft-target, EAGLE-3, Medusa, and SuffixDecoding/n-gram lookup across models on fak serve.
- **Trace:** 67bb976926a · docs/benchmarks/published/HEADLINES.json · docs/benchmarks/published/HEADLINES.json (67bb976926a CUDA DSpark MTP); commits 28b7299a29 & 020586f72c (toolgrammar.ForcedDraft jump-forward); internal/polymodel & internal/spec (CPU tree verify)

### ○ Retrieval / prompt-lookup (n-gram) drafting for input-grounded tasks — fak: **no-claim**

*Why it matters:* A zero-draft-model, zero-extra-VRAM acceleration that works whenever the output echoes the input (summarization, RAG, code edit/refactor, structured rewrite). It is the cheapest spec-decode to deploy and a key differentiator for grounded agent workloads where outputs reuse prompt substrings.

- **SOTA bar:** SuffixDecoding (NeurIPS 2025, model-free suffix-tree drafting) is the new prompt-lookup-family SOTA: outperforms classic PLD/n-gram by 1.3-3x and delivers up to 5.3x speculative speedup (and ~4x end-to-end on SWE-Bench agentic tasks) over vanilla decoding.
- **Leading systems:** SuffixDecoding (Snowflake Arctic Inference + vLLM), vLLM PLD/n-gram (prior bar), TensorRT-LLM Lookahead
- **Source:** [https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/) (2025-05)
- **fak:** no-claim — no number (stub)
- **fak note:** REAL GAP fak should measure (it is adjacent to fak's thesis). Prompt-lookup is input-grounded reuse — the SAME family as fak's cross-agent prefix reuse — and needs no draft model, so it is a natural fit for a reuse kernel and worth a measured row. fak has zero evidence today. Competitor value is the up-to-2.8x summarization figure.
- **Trace:** No fak evidence: CLAIMS.md / BENCHMARK-AUTHORITY.md contain no n-gram, prompt-lookup, or retrieval-drafting claim. fak's PickDrafter selects among drafters but no n-gram/prompt-lookup drafter exists.

### ≈ Output-distribution losslessness of speculation (greedy and stochastic) — fak: **parity**

*Why it matters:* Speculative decoding must not change what the model would have produced; rejection-sampling verification preserves the target distribution exactly at T=0 and T>0. An engine that silently alters outputs (or only matches at greedy) fails reproducibility, eval parity, and audit requirements - this is the honesty gate operators must verify, not assume.

- **SOTA bar:** EAGLE 3.1 (vLLM/EAGLE/TorchSpec, May 2026) is the current SOTA lossless speculative-decoding method: up to 2x longer acceptance length and up to ~2.03x per-user output throughput vs EAGLE-3 (Kimi-K2.6-NVFP4, GB200), fixing 'attention drift' robustness; correctness is preserved by rejection-sampling verification that matches the target distribution (greedy spec == greedy non-spec).
- **Leading systems:** EAGLE 3.1 (vLLM + TorchSpec), EAGLE-3, SpecForge training framework
- **Source:** [https://vllm.ai/blog/2026-05-26-eagle-3-1](https://vllm.ai/blog/2026-05-26-eagle-3-1) (2026-05)
- **fak:** parity — no number (shipped)
- **fak note:** Parity on the greedy CPU oracle half, WITH HONEST FENCES: (1) internal/spec proves token-identity vs plain greedy decode on the CPU PreNorm path (via KVCache.Evict rollback); (2) on CUDA RTX 5080 Inkling-2-Mini, opt-in DSpark batched verify is NOT bit-exact to single-step decode on all prompts due to GEMM reduction order (diverges at token 65 on capitals; bit_exact=false in HEADLINES.json); (3) distribution-preserving stochastic (temperature>0) rejection sampling parity is unbuilt.
- **Trace:** internal/spec SpeculativeGreedy/VerifyTree (greedy CPU bit-exact rollback witness); docs/benchmarks/published/aegis-inkling2-spec-decode/RESULTS.md & HEADLINES.json (RTX 5080 DSpark bit_exact=false)

### ○ Speculative-decoding behavior under high concurrency / large batch (break-even) — fak: **no-claim**

*Why it matters:* Spec decode helps latency at low load but can REGRESS throughput once the GPU is compute-bound - the single most common production footgun. Operators must know the break-even batch and whether the engine auto-disables or has a batch-robust drafter, or they pay for slowdowns at peak QPS.

- **SOTA bar:** EAGLE-3's measured high-concurrency behavior: positive throughput uplift sustained to batch ~64 in SGLang (1.38x at batch 64), with the speedup approaching break-even beyond that as baseline throughput rises; EAGLE-2 already regresses (0.93x at batch 24).
- **Leading systems:** EAGLE-3 (SGLang), MagicDec (fixed-window draft, prior reference), vLLM/SGLang continuous batching
- **Source:** [https://arxiv.org/abs/2503.01840](https://arxiv.org/abs/2503.01840) (2025-03)
- **fak:** no-claim — no number (in-flight)
- **fak note:** GAP: NativeScheduler continuous batching and CUDA cohorts are shipped, and C=1 DSpark spec decode is measured (180.92 tok/s d4), but high-concurrency break-even batch sizing for speculative decoding (e.g. batch 16-64 vs MagicDec / EAGLE-3) is unmeasured.
- **Trace:** internal/modelengine/nativesched.go & HEADLINES.json; no high-batch speculative scaling sweep yet

## Structured output (`structured-output`)

### ○ Constrained/guided decoding per-token overhead (JSON-schema, regex, grammar/CFG) — fak: **no-claim**

*Why it matters:* Guaranteed-valid JSON/regex/grammar output is table stakes for tool-calling and structured APIs, but a slow mask-compute step throttles tokens/sec. The per-token mask overhead is the dominant cost in steady-state constrained generation and the number that decides whether structured output is 'free' or a tax.

- **SOTA bar:** XGrammar-2 (arXiv 2601.04426, Jan 2026) drops grammar-mask per-token overhead to ~12-20us (Llama tool-calling ~12-13us, OpenAI Harmony ~15-20us) with <6% end-to-end overhead vs unconstrained and ~10ms (>100x faster) dynamic compilation; it is the new SOTA, far below LLGuidance's ~250us (Harmony) / >1000us (tool-calling) on the agentic workloads.
- **Leading systems:** XGrammar-2 (vLLM/SGLang/TensorRT-LLM default backend), XGrammar, LLGuidance (Rust Earley)
- **Source:** [https://arxiv.org/html/2601.04426](https://arxiv.org/html/2601.04426) (2026-01)
- **fak:** no-claim — no number (in-flight)
- **fak note:** MECHANISM SHIPPED, OVERHEAD BENCH UNMEASURED: fak now ships native decode-time logit bitmask FSMs (BatchBitmaskFSM with prefix-skip vocab trie and speculative rollback, 361b316673/b2a53f9305), structured carriers in internal/agent/inkernel_structured.go, and toolgrammar.ForcedDraft jump-forward decoding (020586f72c). However, no per-token microsecond latency benchmark against XGrammar-2 (<40 us/tok) is committed, so verdict stays no-claim.
- **Trace:** 361b316673 · internal/grammar/batch_bitmask_fsm.go (361b316673) + b2a53f9305 (sorted-vocab trie); internal/agent/inkernel_structured.go; internal/toolgrammar ForcedDraft (28b7299a29, 020586f72c); no XGrammar-2 us/token parity benchmark

### ○ Grammar/schema compilation latency and cache (cold-start for dynamic agentic schemas) — fak: **no-claim**

*Why it matters:* Agentic workloads change the schema/grammar per request (different tool sets, varying response protocols), so the one-time compile cost - not per-token mask - becomes the bottleneck. Compilation latency and cross-grammar caching determine whether constrained decoding is viable for high-churn tool-calling.

- **SOTA bar:** XGrammar-2 compiles in ~10 ms vs >1000 ms for XGrammar (~80x faster), with >6x tool-calling compilation speedup via Earley-based adaptive mask cache + cross-grammar cache + JIT; integrated into vLLM and SGLang
- **Leading systems:** XGrammar-2, XGrammar, LLGuidance/Guidance
- **Source:** [https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation) (2026-05)
- **fak:** no-claim — no number (in-flight)
- **fak note:** CACHE SHIPPED, COMPILE LATENCY UNMEASURED: fak now ships SchemaCompileCache caching compiled *JSONSchemaMask and sorted-vocab tries across decode requests (361b316673, b2a53f9305), alongside content-addressed tool-arg grammar dedup, but has committed no cold compile-time (ms) comparison against XGrammar-2 (~10 ms). Honest no-claim.
- **Trace:** 361b316673 · internal/grammar/batch_bitmask_fsm.go SchemaCompileCache (FIFO schema/vocab compile cache, 361b316673) + internal/grammar/json_schema_mask.go; no cold compile-ms benchmark vs XGrammar-2

### ○ Structured-output validity vs reasoning-quality tax (constraint-induced degradation) — fak: **no-claim**

*Why it matters:* Constrained decoding can guarantee 100% schema-valid output yet DEGRADE task accuracy by forcing answer fields before chain-of-thought. Buyers must weigh validity rate against the reasoning hit; this is the dimension a naive 'we always emit valid JSON' scorecard hides.

- **SOTA bar:** Constrained decoding guarantees ~100% syntactic/schema validity (JSON-Pass/Type-Safety >.95) but does NOT guarantee semantic quality and can REDUCE accuracy: ExtractBench (2026) shows structured mode dropping overall validity 51%->37% and GPT-5 credit-agreement pass rate 86.9%->70.0% vs prompt-based extraction. The honest SOTA bar is 'validity is solved; value accuracy is the open gap (13.7pt spread across models).'
- **Leading systems:** XGrammar (vLLM/SGLang/TRT-LLM default), LLGuidance (guidance-ai), OpenAI Structured Outputs
- **Source:** [https://arxiv.org/pdf/2602.12247](https://arxiv.org/pdf/2602.12247) (2026-02)
- **fak:** no-claim — no number (in-flight)
- **fak note:** NATIVE STRUCTURED CONSTRAINTS SHIPPED, REASONING-TAX BENCH UNMEASURED: fak now enforces bounded JSON-schema, regex, GBNF, and guided_choice bitmasks during native decode (witnessed by go test ./internal/agent and ./internal/grammar), but has NOT run JSONSchemaBench or ExtractBench to quantify validity-vs-reasoning accuracy loss. Honest no-claim.
- **Trace:** 361b316673 · internal/grammar + internal/agent/inkernel_structured.go (TestNativeGuidedJSONConstrainsCompletion, TestNativeGuidedRegexConstrainsCompletion, TestNativeStructuredCompleteAndStream); no JSONSchemaBench run

### ○ Tool/function-call parsing coverage and robustness (per-model formats) — fak: **no-claim**

*Why it matters:* Agents live or die on correctly extracting structured tool calls from model output; each model family uses a different format (Hermes, llama3_json, mistral, pythonic, qwen3_coder XML). Wrong/absent parser yields malformed args or silent empty calls - so parser coverage and graceful failure are a core agent-serving differentiator.

- **SOTA bar:** vLLM ships per-family parsers (hermes, llama3_json, mistral, pythonic, internlm, granite, plus qwen3/qwen3_coder); when constraints aren't enforced it falls back to extracting from raw text, so args can be malformed - making grammar-backed (constrained) tool calling the robustness frontier
- **Leading systems:** vLLM tool-call parsers, SGLang, XGrammar-2 (grammar-backed tool calling)
- **Source:** [https://docs.vllm.ai/en/latest/features/tool_calling/](https://docs.vllm.ai/en/latest/features/tool_calling/) (2025-12)
- **fak:** no-claim — no number (in-flight)
- **fak note:** FORCED-DRAFT + BOUNDARY REPAIR SHIPPED, PER-FAMILY COVERAGE BENCH UNMEASURED: fak now ships deterministic toolgrammar.ForcedDraft jump-forward constraints wired into serving decode (020586f72c), decode-time bitmask FSMs, and syscall-boundary arity repair + MISROUTE/MALFORMED denials, yet has no measured multi-family raw-text parser malformed-call rate benchmark vs vLLM parsers.
- **Trace:** 020586f72c · internal/toolgrammar + commit 020586f72c (ForcedDraft wired into serving decode loop); internal/grammar/batch_bitmask_fsm.go; boundary positional->named repair; no per-family parser benchmark

### ○ Sampler coverage and correctness (temperature, top-p, top-k, min-p) — fak: **no-claim**

*Why it matters:* Sampling controls the quality/diversity/coherence tradeoff and must be implemented correctly and identically across engines for eval parity. min-p in particular keeps outputs coherent at high temperature and is now an ecosystem default; missing or buggy samplers force operators to over-constrain temperature and lose quality.

- **SOTA bar:** Reference samplers cover the full mainstream set: temperature, top_p, top_k, min_p, typical, presence/frequency/repetition penalties, logit_bias, seed/n/best_of, plus pluggable custom logits processors (argmax-invariant classification). vLLM and llama.cpp converge on Temperature+Min-P as the 2026 default power-user combo.
- **Leading systems:** vLLM SamplingParams + logits processors, llama.cpp sampler chain, SGLang sampler
- **Source:** [https://docs.vllm.ai/en/latest/design/logits_processors/](https://docs.vllm.ai/en/latest/design/logits_processors/) (2026-01)
- **fak:** no-claim — no number (shipped)
- **fak note:** Mechanism shipped; cross-engine parity not measured yet. The native `fak serve` route samples with temperature, top-p, top-k, min-p (vLLM order), logit_bias, frequency and presence penalties, and a per-request seed; temperature 0 is greedy. The same lanes mask logits for structured output: a bounded JSON-schema subset (`response_format` json_schema or `guided_json`), `guided_regex`/`regex`, and `guided_choice`. Typical-p, multiplicative repetition penalty, n/best_of and pluggable logits processors are not native yet. With no measured sampler-parity number against vLLM or llama.cpp, the verdict stays no-claim. In ride mode, sampling passes through to the fronted engine untouched.
- **Trace:** internal/agent/inkernel_sampling.go sampleLogitsWithPenaltyCoreWS (temperature, top-p, top-k, min-p, bias, penalties); structured masks in internal/agent/inkernel_structured.go. Witnesses: TestTopKExcludesTokensOutsideTheK, TestTopPNucleusTruncateEquivalence, TestSamplerMinPComposesBeforeTopKAndTopP, TestSamplerRandomizedPITIsUniform, TestSampleLogitsWithPenaltyFrequencySuppressesRepeatedToken, TestNativeStructuredCompleteAndStream, TestNativeGuidedRegexConstrainsCompletion, TestNativeGuidedJSONConstrainsCompletion, TestNativeGuidedChoiceConstrainsCompletion (go test ./internal/agent). No sampler-parity benchmark exists.

