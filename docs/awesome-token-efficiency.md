---
title: "Awesome Token Efficiency — the catalog of token / context / KV-cache efficiency methods"
description: "A maintained awesome-list index of every token-efficiency method we know of — prompt/API-level, KV-cache/serving, and agent/context-engineering — each tagged lossy-vs-lossless, semantic-vs-mechanical, and with fak's status (shipped / partial / plan / not-in-fak / out-of-scope). Methods that can be safely default-on in fak link to a tracking issue."
---

# Awesome Token Efficiency

> A single index of the methods that make an LLM workload cost fewer tokens — billed
> input/output tokens **and** the resource "tokens" of KV-cache memory and recompute.
> This is a reference catalog (an "awesome list"), not a roadmap: it lists methods
> **even if fak does not and will not implement them**, so an operator or contributor
> can see the whole field on one page and know where fak sits on each.

<!-- awesome-token-efficiency: maintained index. Last refreshed 2026-06-28. -->

## How to read this

Every entry carries up to four tags:

- **Loss** — **lossless** (the model sees identical tokens / output is provably
  unchanged) vs **lossy** (the model sees less or different content; outputs may
  change). For trained-in architecture choices we write *native* (lossy vs vanilla
  MHA, but no fidelity loss relative to that model's own weights).
- **Kind** — **mechanical** (no task understanding needed) vs **semantic** (requires
  understanding the content or task). Mechanical + lossless methods are the ones safe
  to turn on by default; semantic/lossy methods need a guardrail and usually a flag.
- **Layer** — where it lives: *prompt construction*, *provider feature*,
  *gateway/proxy*, *inference engine*, *model architecture*, *agent harness*.
- **fak** — fak's status on this method (see legend). Omitted when the method lives at
  a layer fak does not own (e.g. the prompt author's wording, a provider's batch tier).

**fak status legend:** ✅ shipped · 🟡 partial / seam present · 🔭 plan or design doc
only · ❌ not in fak (and maybe never) · ➖ out of fak's layer (client/provider/author
side) · ⭐ a place fak does something the standard stacks do not.

**The load-bearing axis is lossy vs lossless.** The biggest *mechanical, lossless*
wins — prefix caching, paged/radix KV reuse, batch tiers, lazy tool loading, tokenizer
choice — change cost without changing what the model sees, so they are the ones that
can be **default-on**. Lossy methods (compression, eviction, summarization,
quantization) trade fidelity for size and belong behind a flag with a witness.

---

## fak at a glance — what is on by default

fak is an **agent kernel**: a gateway you put in front of the model that keeps the
provider's prompt-cache prefix byte-identical while shedding old turns, and (in the
fused path) runs the model with an addressable, bit-exact KV cache. Its token-saving
*defaults* are audited by `fak token-defaults-scorecard`
([scorecard doc](serving/token-defaults-scorecard.md)). On by default today:

| Saver | Loss | What it does | Lever |
|---|---|---|---|
| Provider prompt-cache passthrough | lossless | ships a byte-identical cache prefix so the provider discount survives | structural |
| Tool-floor pruning | lossless | drops provably-unreachable tool definitions from the request | structural |
| vDSO dedup | lossless | collapses identical repeated calls | `--vdso` |
| Compact-history | bounded | sheds the un-cacheable middle past a 48k-token budget by splicing original bytes | `--compact-history-budget` |
| Oversized-result elision | bounded | shrinks a scrolled-past `tool_result` to head+tail at 16 KB | `--elide-result-bytes` |
| Structured-output passthrough | lossless | forwards `response_format` / `json_schema` / `logit_bias` to the ride engine | structural (#907) |
| ctxplan O(1) view | bounded | re-materializes history under a budget | `--ctx-view-budget` (default-on at 8000) |

The rest of this page is the wider field. Where a method **could be a safe fak
default and isn't yet**, it links to a tracking issue (see
[default-on candidates](#default-on-candidates-tracked-as-issues)).

---

# Part A — Prompt & API-level (request-side)

Methods that shape the request before/around the model. fak owns the gateway here, so
many of these are fak levers; others are the prompt author's or the provider's.

## A1. Prompt caching (lossless, mechanical)

- **Anthropic prompt caching** — explicit `cache_control: ephemeral` breakpoints (≤4)
  over tools→system→messages; min ~1024 tokens; read ≈ 0.1× input, write 1.25×/2.0×
  (5-min/1-h TTL). [docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching). **fak: ✅** byte-faithful passthrough is fak's core thesis.
- **OpenAI automatic prefix caching** — zero-config, caches longest seen prefix ≥1024
  tokens; up to 90% input discount; `prompt_cache_key` pins routing.
  [docs](https://platform.openai.com/docs/guides/prompt-caching). **fak: ✅** passthrough.
- **Gemini context caching** — implicit (auto on 2.5+, no storage fee) and explicit
  (`CachedContent` + TTL, storage fee). [docs](https://ai.google.dev/gemini-api/docs/caching). **fak: ✅** passthrough.
- **Cache-prefix stability ("static-first" layout)** — invariant content (instructions,
  tool schemas, few-shot) first, volatile content (user turn, timestamps) last, to
  maximize the cacheable prefix. *lossless · mechanical · prompt/gateway.* **fak: ✅⭐**
  compact-history splices on the original bytes (a memcpy, never a re-marshal) so the
  prefix stays byte-identical while the middle is shed.
- **Cache-aware routing / sticky sessions** — route same-prefix requests to the same
  warm server. *lossless · mechanical · gateway.* **fak: 🟡** sticky `trace_id` routing
  ([advanced-topics](fak/advanced-topics.md)).
- **Cache economics / break-even** — only cache prefixes read back many times; a
  write-once prefix costs *more* than no cache. *semantic · gateway policy.* **fak: ✅⭐**
  see [O(1) context-window economics](explainers/o1-context-window-economics.md) and
  [compounding benefits of a saved call](explainers/compounding-benefits-of-a-saved-call.md).

## A2. Prompt compression (lossy, semantic)

- **LLMLingua** — small LM scores token perplexity; coarse-to-fine budget controller
  drops low-information tokens (~20×). [arXiv:2310.05736](https://arxiv.org/abs/2310.05736). **fak: ❌** (lossy; tracked as a research item, issue [#110](https://github.com/anthony-chaudhary/fak/issues/110)).
- **LongLLMLingua** — question-aware long-context compression + reordering to fight
  "lost in the middle." [arXiv:2310.06839](https://arxiv.org/abs/2310.06839). **fak: ❌**.
- **LLMLingua-2** — task-agnostic compression as token classification (distilled
  encoder), 3–6× faster. [arXiv:2403.12968](https://arxiv.org/abs/2403.12968). **fak: ❌**.
- **Selective Context** — prune low self-information lexical units (the seminal
  hard-pruning method). [arXiv:2304.12102](https://arxiv.org/abs/2304.12102). **fak: ❌**.
- **Gisting / soft-prompt compression** — train the LM to compress a prompt into a few
  cacheable "gist" tokens (≤26×). [arXiv:2304.08467](https://arxiv.org/abs/2304.08467). **fak: ❌** (needs a gist-trained model).
- **Prompt distillation / instruction rewriting** — offline-rewrite a verbose system
  prompt into a tighter equivalent, reused thereafter. *lossy · semantic · one-time
  authoring.* **fak: ➖** author-side.

## A3. Context pruning / selection (mostly lossy, semantic)

- **Retrieval-Augmented Generation (RAG)** — fetch only top-k relevant chunks instead
  of the full corpus. *lossy vs full corpus · semantic · app.* **fak: ➖** above fak's
  boundary, but [promptmmu (#751)](https://github.com/anthony-chaudhary/fak/issues/751)
  curates *what* enters the context window cache-prefix-safely.
- **Cross-encoder / LLM reranking** — retrieve broadly, jointly score query×passage,
  keep top-K so you pass fewer chunks. [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). **fak: ➖**.
- **Post-rerank evidence pruning** — strip redundant sentences from retrieved passages
  (Provence/XProvence, information-gain pruning). **fak: ➖**.
- **Context deduplication** — collapse repeated/near-identical context blocks (same doc
  pulled twice, repeated tool output). *lossless if exact-dup · mechanical→semantic ·
  gateway.* **fak: ✅** vDSO exact-call dedup + [simhash](observability/trajectory.md)
  near-dup detection; extending exact-dedup to inbound message *content* is a
  [default-on candidate](#default-on-candidates-tracked-as-issues).
- **History truncation / windowing** — drop or summarize old turns past a budget.
  *lossy · mechanical/semantic · gateway.* **fak: ✅** compact-history (shed, not
  summarize, to stay cache-safe).
- **Rolling / hierarchical summarization** — replace long history with a running
  summary + recent verbatim tail. *lossy · semantic · app.* **fak: 🟡** many open issues
  (#65, #459, #1075); fak prefers cache-safe shedding over summarization.
- **Tool-result pruning / elision** — replace a stale/oversized tool output with a
  short reference once consumed. *lossy · semantic · gateway.* **fak: ✅** oversized-result
  elision (head+tail at 16 KB), on by default.

## A4. Output-side control (caps lossless; concision lossy)

- **`max_tokens` cap** — hard output ceiling. *lossy if it truncates real content ·
  mechanical · provider param.* **fak: ➖** passthrough.
- **Stop sequences** — halt at a delimiter. *lossless · mechanical · provider param.* **fak: ➖**.
- **Concise / "be terse" instructions** — tell the model to skip preamble/filler.
  *lossy · semantic · prompt.* **fak: ➖** author-side.
- **Structured Outputs / JSON Schema (strict)** — constrained decoding masks invalid
  next-tokens against a compiled grammar → schema-valid output, no prose wrapper.
  [OpenAI](https://openai.com/index/introducing-structured-outputs-in-the-api/). **fak: ✅**
  structured-output passthrough (#907) forwards the constraint to vLLM `guided_json` /
  SGLang `json_schema`.
- **Grammar-constrained / guided decoding** — GBNF, Outlines, XGrammar, vLLM guided
  decoding force output into a regex/CFG/enum. [XGrammar](https://arxiv.org/abs/2411.15100). **fak: ✅** (passthrough, as above).
- **Reasoning-effort / thinking-budget control** — cap hidden reasoning tokens on
  reasoning models. *lossy · semantic · provider param.* **fak: ➖** passthrough.

## A5. Batching / request shaping (lossless, mechanical)

- **Batch API (50% off)** — async jobs at half price on input *and* output (OpenAI
  Batch, Anthropic Message Batches). [OpenAI Batch](https://platform.openai.com/docs/guides/batch). **fak: ➖** provider feature, client-side (fak is a sync gateway). Stacks with prompt caching.
- **Flex / lower service tier** — same ~50% discount via the live API, latency-tolerant.
  **fak: ➖** passthrough.
- **Multi-item prompts** — pack several independent items into one call sharing a cached
  instruction prefix. *lossless · mechanical · app/gateway.* **fak: ➖** app-side.
- **Embedding-based request dedup / semantic response cache** — serve a cached response
  for exact/near-identical prompts (GPTCache pattern). *lossless (exact) / lossy (near)
  · gateway.* **fak: 🟡** simhash is for trajectory similarity, not yet a response cache.
- **Model cascade / FrugalGPT** — try a cheap model first, escalate on low confidence.
  *lossless if escalation is sound · semantic · gateway.* **fak: 🟡** per-aspect routing +
  ensembles (`vote`/`best_of`) is the [routing spine](model-routing.md); auto-confidence
  escalation is not wired.

## A6. Tool / function-definition efficiency

The tool schemas are a per-turn input tax paid whether or not a tool is called.

- **Tool-schema pruning (load only relevant tools)** — send only the plausibly-needed
  subset. *lossless if pruned tools unused · semantic · gateway.* **fak: ✅⭐** tool-floor
  pruning drops provably-unreachable tool defs by default (structural, not a classifier).
- **Tool Search / lazy schema loading (MCP)** — load a search interface; fetch a tool's
  full schema only when the model picks it (Anthropic reports ~85% tool-token cut).
  [code-execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp).
  *lossless · mechanical · host/protocol.* **fak: ✅** — shipped as the
  `fak_tools_search` MCP tool, supporting name/description/full detail levels with
  query filtering (see `internal/gateway/mcp.go`).
- **Compact tool descriptions / schema minification** — trim verbose docstrings; let
  the tool *name* carry meaning. *lossy · semantic · authoring.* **fak: ❌**.
- **One-router-tool pattern** — fold many tools behind a single dispatcher with an `op`
  param. *lossy (loses per-tool typing/permissions) · mechanical · gateway.* **fak: ➖**
  (fak prefers per-tool adjudication for the capability floor).
- **Code-execution tool exposure (skills)** — present MCP servers as importable code
  files, not upfront JSON (Anthropic/Cloudflare report ≤98.7% cut). **fak: ➖** harness-side.

## A7. Tokenizer-level (lossless, mechanical)

- **Choose a denser tokenizer** — token-per-word varies 2–10× across vocabularies.
  *lossless · mechanical · model selection.* **fak: ➖** model choice.
- **Avoid token-wasteful formats** — prefer terse JSON/CSV/TSV over verbose
  XML/pretty-printed JSON; strip needless whitespace. *lossless · mechanical · prompt.*
  **fak: ➖** author-side.
- **Language/script choice** — express instructions in a tokenizer-efficient language
  where the task allows. *lossless · mechanical · prompt.* **fak: ➖**.

---

# Part B — KV-cache & inference-serving (resource-side)

These reduce the *resource* tokens — KV-cache memory, attention compute, recompute.
fak owns the **governance band** on top of an engine and, in the fused path, runs its
own reference engine with an **addressable, bit-exact KV cache** — so some of these are
shipped in-kernel, many are the engine's job that fak fronts, and a few are uniquely
fak's.

## B1. KV-cache quantization (lossy; int8 ≈ neutral)

- **KIVI** — tuning-free 2-bit: Keys per-channel, Values per-token. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750). **fak: 🟡** KV precision tiers q8/f32 planner half shipped ([#1047](https://github.com/anthony-chaudhary/fak/issues/1047)); engine GPU-gated.
- **KVQuant** — per-channel Key (pre-RoPE) + per-token Value, non-uniform levels; ~10M
  ctx. [arXiv:2401.18079](https://arxiv.org/abs/2401.18079). **fak: 🟡** (as above; q8 keeps pre-RoPE K in f32 for exact eviction).
- **INT8 / INT4 KV (engine-native)** — LMDeploy/TensorRT-LLM/vLLM online KV quant. **fak: ➖** engine-side.
- **FP8 / NVFP4 KV** — hardware-native low-precision KV (Hopper/Blackwell). **fak: ➖** engine-side.

## B2. KV-cache eviction / sparsity (lossy)

- **StreamingLLM (attention sinks)** — keep sink tokens + a sliding window. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453). **fak: ❌** engine-level; fak's eviction is *addressable & exact*, a different axis.
- **H2O (Heavy-Hitter Oracle)** — retain recent + high-attention tokens. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048). **fak: ❌**.
- **Scissorhands** — exploit importance persistence to evict. [arXiv:2305.17118](https://arxiv.org/abs/2305.17118). **fak: ❌**.
- **SnapKV** — vote on important prefix positions from an end-of-prompt window (targets
  prefill). [arXiv:2404.14469](https://arxiv.org/abs/2404.14469). **fak: ❌**.
- **FastGen** — per-head hybrid eviction policy chosen at end-of-prefill. [arXiv:2310.01801](https://arxiv.org/abs/2310.01801). **fak: ❌**.
- **Sliding-window / local attention** — Longformer, Mistral SWA. [arXiv:2310.06825](https://arxiv.org/abs/2310.06825). **fak: ➖** model-native.
- **PyramidKV / PyramidInfer** — larger KV budget to lower layers. [arXiv:2406.02069](https://arxiv.org/abs/2406.02069). **fak: ❌**.
- **Quest** — query-aware sparsity: keep full KV, load only top-K relevant pages per
  step. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774). **fak: ❌**.
- **DuoAttention** — split retrieval heads (full KV) from streaming heads (sinks+recent).
  [arXiv:2410.10819](https://arxiv.org/abs/2410.10819). **fak: ❌**.
- **Addressable, bit-exact mid-run causal eviction** — evict one span (a poisoned tool
  result, an expired secret) and leave the KV cache bit-for-bit identical to a run that
  never saw it (`max|Δ| = 0`). **fak: ✅⭐ UNIQUE** — no shipped serving engine offers
  policy-driven exact mid-run eviction. See [addressable KV cache](explainers/addressable-kv-cache.md).

## B3. Attention architecture for KV reduction (native — lossy vs MHA, no fidelity loss vs own weights)

- **Multi-Query Attention (MQA)** — all query heads share one K/V head. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150). **fak: ✅** in-kernel engine reads `multi_query`.
- **Grouped-Query Attention (GQA)** — `g` K/V groups; MHA↔MQA interpolation. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). **fak: ✅** in-kernel engine reads `num_key_value_heads` (Llama/Qwen/Gemma).
- **Multi-head Latent Attention (MLA)** — low-rank joint KV compression to one latent
  per token (DeepSeek-V2/V3). [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). **fak: 🟡** GLM-MoE proven; MLA-specific kernel not separately witnessed.
- **Cross-Layer Attention (CLA)** — share K/V across adjacent layers. [arXiv:2405.12981](https://arxiv.org/abs/2405.12981). **fak: ❌**.
- **YOCO** — one global KV cache reused by a cross-decoder. [arXiv:2405.05254](https://arxiv.org/abs/2405.05254). **fak: ❌**.
- **MLKV** — share KV heads across layers (below MQA). [arXiv:2406.09297](https://arxiv.org/abs/2406.09297). **fak: ❌**.
- **TPA (Tensor Product Attention)** — factorize Q/K/V into low-rank tensors; cache
  factors. [arXiv:2501.06425](https://arxiv.org/abs/2501.06425). **fak: ❌**.
- **GoldFinch** — RWKV/Transformer hybrid, 756–2550× smaller KV. [arXiv:2407.12077](https://arxiv.org/abs/2407.12077). **fak: ❌**.
- **Sigma / DiffQKV** — asymmetric Q/K/V rescaling. [arXiv:2501.13629](https://arxiv.org/abs/2501.13629). **fak: ❌**.
- **Interleaved local/global + GQA** — Gemma 2/3. [arXiv:2408.00118](https://arxiv.org/abs/2408.00118). **fak: ➖** model-native.

### B3b. Low-rank KV compression (post-hoc SVD, lossy)

- **Palu** — offline SVD of K/V projection weights. [arXiv:2407.21118](https://arxiv.org/abs/2407.21118). **fak: ❌**.
- **Eigen Attention** — attention in a calibrated low-rank space. [arXiv:2408.05646](https://arxiv.org/abs/2408.05646). **fak: ❌**.
- **ShadowKV** — online SVD of pre-RoPE keys + value offload. [arXiv:2410.21465](https://arxiv.org/abs/2410.21465). **fak: ❌**.
- **LoRC / xKV** — progressive / cross-layer SVD KV compression. [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) · [arXiv:2503.18893](https://arxiv.org/abs/2503.18893). **fak: ❌**.

## B4. Prefix / cache sharing (lossless, exact reuse)

- **PagedAttention (vLLM)** — OS-style paging of KV blocks shared via block tables.
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). **fak: ✅⭐** in-kernel `kvmmu`
  context-MMU with span-level management + **policy-aware invalidation** (not just memory
  pressure).
- **RadixAttention (SGLang)** — KV in an LRU radix tree, longest-prefix auto-reuse.
  [arXiv:2312.07104](https://arxiv.org/abs/2312.07104). **fak: ✅⭐** in-kernel `radixkv`
  (86.7% hit on agents) + **policy-driven eviction by quarantine verdict**.
- **Automatic Prefix Caching** — vLLM / TensorRT-LLM hash-keyed block reuse. **fak: ➖**
  engine feature fak fronts; fak's own equivalent is `radixkv`.
- **Prompt Cache (modular attention reuse)** — precompute attention states of reusable
  modules, reuse even non-contiguous. [arXiv:2311.04934](https://arxiv.org/abs/2311.04934). **fak: 🔭** related to the [regenerable-KV plan](serving/regenerable-kv-plan.md).
- **Hydragen** — attention decomposition (shared prefix vs unique suffix) + inter-seq
  batching. [arXiv:2402.05099](https://arxiv.org/abs/2402.05099). **fak: 🟡** `BatchFromPrefix` does shared-prefix batched inference.
- **FlashInfer cascade inference** — hierarchical shared-prefix + per-request suffix
  attention. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005). **fak: ➖** engine kernel.
- **ChunkAttention** — chunked K/V in a prefix tree for runtime prefix sharing. [arXiv:2402.15220](https://arxiv.org/abs/2402.15220). **fak: ➖** engine kernel.
- **Cross-agent / cross-session shared prefill** — do shared prefill once; later agents
  read it for free. **fak: ✅⭐** the [cache-efficient fleets](explainers/kv-cache-agentic-context.md)
  result — ~4× fewer tokens vs a tuned warm-cache stack on a 50-turn × 5-agent run.

## B5. KV offloading / transport / disaggregation (lossless movement)

- **FlexGen** — LP-scheduled GPU+CPU+disk placement of weights/KV/activations. [arXiv:2303.06865](https://arxiv.org/abs/2303.06865). **fak: ➖** engine.
- **vLLM CPU offload** — async GPU→CPU DRAM KV copy to avoid recompute. **fak: ➖** engine.
- **InstInfer** — in-storage attention offload to computational SSDs. [arXiv:2409.04992](https://arxiv.org/abs/2409.04992). **fak: ➖**.
- **LMCache** — external KV layer (GPU→DRAM→disk→Redis), cross-query non-prefix reuse.
  [arXiv:2510.09665](https://arxiv.org/abs/2510.09665). **fak: 🔭** governance only — see
  [KV transport governance](serving/kv-transport-governance-nixl-mooncake-lmcache.md).
- **Mooncake** — KVCache-centric disaggregated architecture (pooled CPU/DRAM/SSD/RDMA).
  [arXiv:2407.00079](https://arxiv.org/abs/2407.00079). **fak: 🔭** governance band; see
  [P/D disaggregation + KV-routing SOTA](serving/pd-disaggregation-kv-routing-sota.md).
- **NIXL + NVIDIA Dynamo** — async P2P KV transport over RDMA/TCP/NVMe-oF. [github](https://github.com/ai-dynamo/nixl). **fak: 🔭** governance.
- **DistServe** — disaggregate prefill (TTFT) and decode (TPOT) onto distinct GPUs.
  [arXiv:2401.09670](https://arxiv.org/abs/2401.09670). **fak: 🔭** see PD doc above.
- **Splitwise** — split prompt vs token-gen phases onto heterogeneous pools. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677). **fak: 🔭**.
- **KV-aware routing (Preble, vLLM Router)** — route to the replica with the best KV
  hit rate. [arXiv:2407.00023](https://arxiv.org/abs/2407.00023). **fak: 🟡** sticky `trace_id` routing; per-aspect [model routing](model-routing.md).
- **CacheGen** — encode KV into compact bitstreams + adaptive network streaming. [arXiv:2310.07240](https://arxiv.org/abs/2310.07240). **fak: ➖**.
- **CacheBlend** — fuse multiple precomputed KV caches with selective recompute (RAG).
  [arXiv:2405.16444](https://arxiv.org/abs/2405.16444). **fak: ➖**.
- **InfiniGen** — full KV in CPU; speculatively prefetch next-layer critical tokens.
  [arXiv:2406.19707](https://arxiv.org/abs/2406.19707). **fak: ➖**.

## B6. Recompute vs storage tradeoffs (lossless)

- **Chunked prefill (Sarathi)** — split big prefills into chunks batched with decodes.
  [arXiv:2308.16369](https://arxiv.org/abs/2308.16369). **fak: ➖** engine scheduler.
- **Recompute-on-preempt** — recompute rather than store when KV space runs out. **fak: ➖** engine.
- **Compute-or-Load (hybrid prefix caching)** — use GPU recompute *and* SSD-load
  bandwidth together. [openreview](https://openreview.net/pdf?id=cK0kUzocJW). **fak: 🔭** the [regenerable-KV plan](serving/regenerable-kv-plan.md) is fak's take on the store-vs-recompute knob.

## B7. Decoding-side compute savers (speculative family — mostly lossless)

- **Speculative decoding** — draft proposes K tokens, target verifies in parallel,
  exact distribution recovered. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). **fak: 🟡** ABI seam (`internal/abi/speculate.go`); no in-kernel engine impl yet.
- **Medusa** — extra heads + tree-attention. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774). **fak: ❌**.
- **EAGLE / EAGLE-2 / EAGLE-3** — feature-level autoregressive drafting. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077). **fak: ❌** (engine feature fak can front).
- **Lookahead decoding** — Jacobi-iteration parallel n-gram generation, no draft model.
  [arXiv:2402.02057](https://arxiv.org/abs/2402.02057). **fak: ❌**.
- **Self-speculative / LayerSkip** — draft with a subset of the model's own layers.
  [arXiv:2309.08168](https://arxiv.org/abs/2309.08168). **fak: ❌**.
- **Prompt Lookup Decoding (PLD)** — draft tokens copied from the input via n-gram match
  (great for RAG/code-edit). [github](https://github.com/apoorvumang/prompt-lookup-decoding). **fak: ❌**.
- **SpecInfer / Sequoia / REST / Ouroboros** — tree- and retrieval-based speculative
  verification. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781). **fak: ❌**.

## B8. Context-length extension (KV-relevant)

- **YaRN / NTK-aware / Position Interpolation** — rescale RoPE frequencies to extend
  context cheaply; zero runtime KV overhead but *amplifies* the KV-footprint problem
  that B1–B5 solve. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071). **fak: ➖** model-native.

---

# Part C — Agent / context-engineering (loop-level)

How agent harnesses keep multi-turn token usage down. "Context engineering" (the term
Cognition and Anthropic popularized) is the discipline of curating the optimal set of
tokens in the model's limited attention budget at each step. fak shares prefill across
agents and sheds old turns cache-safely; most of the rest is the harness's job, though
fak's gateway position lets it do several mechanically. A useful field taxonomy is
[Drew Breunig's "How to Fix Your Context"](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html)
(RAG · Tool Loadout · Context Quarantine · Pruning · Summarization · Offloading).

## C1. Context compaction / summarization (lossy, semantic)

- **Auto-compaction at the window limit** — summarize the transcript into a high-fidelity
  digest and restart the loop from it (one of Anthropic's three long-horizon strategies).
  [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
  *lossy · semantic · harness.* **fak: 🟡** fak prefers cache-safe *shedding*
  (compact-history) over summarization; summarization tracked in #65/#459/#1075. Watch the
  recursive re-nesting failure mode (#880-class).
- **Dedicated compaction model** — a separate fine-tuned LLM whose only job is to compress
  the full action trace, preserving key decisions for a single-threaded agent.
  [Cognition — Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents). **fak: ❌**.
- **Rolling / summary-buffer memory** — last *N* turns verbatim + a continuously-updated
  summary of everything older. LangChain `ConversationSummaryBufferMemory`, LangGraph
  summarization node + `RemoveMessage`. **fak: ➖** harness-side.

## C2. Tool-result management

- **Head+tail elision of large tool outputs** — clip a big output to head+tail with a
  byte marker. *lossy · mechanical · gateway/harness.* **fak: ✅** oversized-result elision,
  on by default. (Claude Code/Cursor cap bash/read/terminal output similarly.)
- **Restorable compression (reference by handle)** — drop the bulky body but keep a stable
  reference (path, URL, artifact id) so it can be re-fetched; *recoverable*, so effectively
  lossless. [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
  (filesystem-as-context). *semantic · harness.* **fak: 🟡** elision keeps head+tail; a full
  handle-store is a candidate.
- **Mask, don't remove (logit masking of tools)** — keep *all* tool definitions in context
  (so the KV-cache and references stay valid) and instead mask the decoder logits to
  constrain which tools are selectable per state. [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus).
  *lossless · semantic · engine/harness.* **fak: ➖** — note the contrast with tool-floor
  pruning (A6): masking trades tokens-saved for cache-stability; fak's tool-floor *removes*
  provably-unreachable defs, which is lossless only because they were unreachable.
- **Observation masking / scrollback dropping** — drop superseded observations while keeping
  the actions/decisions. *lossy · mechanical→semantic · harness.* **fak: 🟡** compact-history
  sheds the middle.

## C3. Sub-agent / context isolation

- **Orchestrator–subagent fan-out** — a lead agent spawns parallel subagents, each with an
  isolated context window, that return only condensed findings; the orchestrator never carries
  the sub-transcripts (map-reduce over agents). [Anthropic multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system).
  *lossy (only conclusions return) · semantic · harness.* **fak: ➖** harness-side (Claude Code,
  this very session) — but fak's **cross-agent shared prefill** makes the fan-out cheap.
- **Context quarantine** — isolate contexts into separate threads so none grows long or
  accrues conflicting content. **fak: ➖** (the *name* collides with fak's security
  *result-quarantine*, which is unrelated — that holds untrusted tool results out of context
  for safety, not for token economy).
- **Counterpoint: single-threaded linear agent** — Cognition argues *against* naive fan-out
  (subagents can't see each other's work, so "actions carry implicit decisions" that clash);
  their fix is one continuous context + compaction. Bounds when isolation is safe (read-heavy,
  parallelizable) vs unsafe (coordinated writes). [Cognition](https://cognition.com/blog/dont-build-multi-agents).

## C4. Retrieval over full-context / external memory

- **Just-in-time / agentic retrieval** — don't frontload data; let the agent fetch it at
  runtime (grep/glob/targeted queries) via "progressive disclosure". *recoverable · semantic ·
  harness.* **fak: 🟡** [promptmmu (#751)](https://github.com/anthony-chaudhary/fak/issues/751)
  curates inbound context; [Context is not memory](CONTEXT-IS-NOT-MEMORY.md) is the durability model.
- **RAG / vector-store retrieval** — retrieve only relevant chunks per query. *lossy · semantic
  · app.* **fak: ➖** above fak's boundary.
- **External / paged memory hierarchy** — context = RAM, external store = disk; the agent pages
  memory blocks in/out (MemGPT/Letta, mem0, LangMem). *recoverable · semantic · harness.* **fak: ➖**;
  fak's [four layers of agent memory](MEMORY-LAYERS-EXPLAINER.md) is the related model.
- **Tool Loadout (RAG over tool definitions)** — when tools number in the dozens, retrieve only
  the relevant subset of tool *descriptions* per turn ("RAG-MCP", [arXiv:2505.03275](https://arxiv.org/abs/2505.03275)).
  *lossy (a needed tool can be unselected) · semantic · gateway.* **fak: ❌** — related to the lazy
  tool-loading [default-on candidate](#default-on-candidates-tracked-as-issues).

## C5. History pruning / dedup

- **Sliding-window buffer** — keep only the last *k* turns, no summary. *lossy · mechanical ·
  harness.* **fak: ✅** compact-history is the cache-safe analogue (sheds the un-cacheable middle).
- **Permanent deletion (RemoveMessage)** vs **non-destructive trim** — LangGraph distinguishes
  dropping turns from the checkpoint vs making a shorter copy for one call. **fak: ➖** harness-side.
- **Tool-call / duplicate-turn dedup** — reuse a prior result for an identical repeated call.
  *lossless (identical) · mechanical · gateway.* **fak: ✅** vDSO; read-once file dedup tracked as
  [#318](https://github.com/anthony-chaudhary/fak/issues/318).
- **Compact the "uncacheable middle"** — target compaction between the cached prefix and the live
  tail to preserve cache hits. *lossy · semantic · gateway.* **fak: ✅⭐** exactly compact-history's design.

## C6. Planning / scratchpad externalization (lossless)

- **Structured note-taking (NOTES.md / memory tool)** — write durable state to a file, re-inject
  only the relevant slice. [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
  *recoverable · semantic · harness.* **fak: ➖**.
- **Recitation (todo.md rewriting)** — rewrite the todo list each step so the global plan stays in
  the most-recent attention span (counters "lost in the middle" over long loops). [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus).
  *lossless (adds, doesn't drop) · semantic · harness.* **fak: ➖**.
- **Context offloading ("think" tool / scratchpad)** — store working notes outside the conversation
  via a tool. *recoverable · semantic · harness.* **fak: ➖**.

## C7. Prompt / system-prompt hygiene

- **Right-altitude system prompts** — concise, high-signal; specific enough to guide, general
  enough not to over-constrain. [Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).
  *lossless · semantic · author.* **fak: ➖**.
- **Keep the wrong stuff in (don't scrub errors)** — leave failed actions + error observations in
  context; seeing its own mistakes reduces repeats. [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus).
  *lossless (intentionally retains) · semantic · harness.* **fak: ➖** (an anti-pruning caution).
- **Stable serialization / minimal boilerplate** — deterministic JSON/tool serialization so the
  cacheable prefix doesn't churn. *lossless · mechanical · gateway/author.* **fak: ✅** byte-faithful
  passthrough preserves whatever stable serialization the client sends.

## C8. Caching at the agent layer

- **KV / prompt-cache reuse via a stable, append-only prefix** — Manus treats KV-cache hit rate as
  the #1 production metric (~10× cost difference on Sonnet); one changed prefix token invalidates it.
  *lossless · mechanical (with prompt discipline) · gateway.* **fak: ✅⭐** the whole thesis — see Part A1.
- **Semantic caching of repeated sub-queries (GPTCache)** — embed the query, vector-search prior
  queries, return a stored response above a cosine threshold. [GPTCache](https://github.com/zilliztech/GPTCache).
  *lossy (near-match) · semantic · gateway.* **fak: 🟡** simhash similarity exists; a response cache is not wired.
- **Response memoization / deterministic tool-call cache** — exact-match response/tool-result
  reuse. *lossless · mechanical · gateway.* **fak: ✅** vDSO (exact-call) is the in-kernel form.

---

## Default-on candidates (tracked as issues)

The honest rule (from the [token-defaults scorecard](serving/token-defaults-scorecard.md)):
**default-on a saver only when it is demonstrably safe** — lossless or in-code-guarded
bounded loss, with a committed witness. The methods below are *mechanical and
lossless-or-bounded*, sit at fak's gateway layer, and aren't on yet:

| Candidate | Why it qualifies | Issue |
|---|---|---|
| Inbound content exact-dedup (same file/tool output sent twice in one request) | lossless, mechanical, extends vDSO from calls to message content | [#1101](https://github.com/anthony-chaudhary/fak/issues/1101) |

*Note: Tool Search / lazy MCP tool-schema loading shipped as `fak_tools_search` in 2026-06 and is no longer a candidate — see [A6. Tool / function-definition efficiency](#a6-tool--function-definition-efficiency).*

Lossy methods (LLMLingua compression, summarization, rerank, KV eviction/quant) are
deliberately **not** default-on candidates — they change what the model sees and stay
behind a flag with a witness.

## Maintenance

This list is meant to be refreshed. To update it:

1. Re-run the field survey (the three research clusters: prompt/API, KV/serving,
   agent/context-engineering) and add new methods with a verified primary reference.
2. Re-derive fak's status from source, not from this page — the
   [`fak token-defaults-scorecard`](serving/token-defaults-scorecard.md) is the
   on/off authority for the defaults; `internal/` packages are the authority for the
   in-kernel methods.
3. For any newly-safe **default-on candidate**, file a tracking issue and link it here.

See also: [SOTA optimizations fak sits on top of](explainers/sota-optimizations.md),
[fak vs vLLM / SGLang / provider KV caching](fak-vs-alternatives-comparison.md),
[the frozen-trajectory cache cliff](explainers/frozen-trajectory-cache-cliff.md), and
the broader [Agent optimization methods survey (370 methods, 16 families)](notes/RESEARCH-agent-optimization-methods-survey-2026-06-23.md)
— this page is the token-efficiency-focused, fak-status-annotated slice of that field map.
