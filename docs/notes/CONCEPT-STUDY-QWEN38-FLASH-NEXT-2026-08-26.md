---
title: "Qwen3.8-Flash-Next: source-pinned architecture and fak opportunity study"
description: "Pinned study of Qwen3.8-Flash-Next architecture, runtime contract, and bounded fak-native adaptation opportunities."
---

# Qwen3.8-Flash-Next: source-pinned study

**Observed:** 2026-08-26  
**Tracker:** [#9122](https://github.com/anthony-chaudhary/fak/issues/9122)  
**Durable receipt:** `study_207f3c56d6e23d2ccfb0d0881fde3a3a8ca1f81d7952897d1a87f61a61a4d383`  
**GitHub source:** [`QwenLM/Qwen3.8-Flash-Next@513aa6e`](https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/513aa6e18a335296fc13e538232a8735b230877d)  
**Checkpoint source:** [`Qwen/Qwen3.8-Flash-Next@f5d0827`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274bafd880402bd16f5e3e6c514136ec06c)

## Verdict

Qwen3.8-Flash-Next is **not a small revision of fak's current Qwen3.8 path**. Its checkpoint declares the new `qwen4_exp_text` architecture: 48 layers alternate three gated-delta linear-attention layers with one sparse-attention layer; every layer has 512 routed experts with 10 active plus one shared expert; sparse attention adds a learned top-2,048 token indexer; and one hybrid MTP layer predicts three future tokens. The 82.8B-total/3.1B-active design is attractive for agent workloads, but fak presently recognizes only the existing Qwen3.8/Qwen3.5-style vocabulary. Loading this checkpoint through that path would be a false compatibility claim.

The smallest honest next move is a **fak-native architecture/config spine** that recognizes `qwen4_exp_text`, rejects unsupported execution explicitly, and locks the tensor/config contract. Native execution should then arrive as separately witnessed packets: gated delta, routed MoE, sparse indexer attention, and hybrid MTP. vLLM/SGLang/Transformers are references and parity oracles only—not an automatic runtime fallback.

## Triggering Unsloth guide and pre-release drift

The operator supplied [Unsloth's Qwen3.8-Flash-Next local-run guide](https://unsloth.ai/docs/models/qwen3.8-next). Its Markdown representation was fetched at `2026-08-26T15:28Z` with SHA-256 `898a0d6d5d073bce68599f25bc20a61c72e3941c0fc82f396c8eeeacf7103db0`. The guide requires [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742), which was still **OPEN** at head `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`; it was not a llama.cpp trunk capability. At that head the PR adds a distinct `QWEN4EXP` converter, metadata/tensor mappings, model graph, hybrid indexed state-memory support, quantization wiring, and architecture tests. These are implementation references only: fak must retain ownership of loading, graph execution, kernels, state/cache, scheduling, and receipts, with no automatic llama.cpp fallback.

The guide recommends `temperature=1.0`, `top_p=0.95`, `top_k=20`, and `presence_penalty=1.5`. Its preserve-thinking recipe retains prior reasoning content and tells the model to continue without restarting reasoning; this is a model-scoped protocol recipe requiring transcript tests, not a global fak default. The guide's “75 GB” 1-bit minimum, “79% smaller,” “80% top-1% accuracy,” RAM/unified-memory performance, and SSD/mmap PLE/Ngram offload statements are **vendor claims**, not fak benchmark evidence.

There is material pre-release drift to preserve rather than explain away. The Unsloth guide describes **125B total / 14B active**, while the pinned official repository/report and checkpoint studied below describe **82.8B total / 3.1B active** and use `qwen4_exp_text`; the linked GGUF PR uses `qwen4exp`. These snapshots may represent a moving artifact, converter vocabulary, or release revision. Until an exact checkpoint-to-GGUF manifest reconciles them, never combine their parameter, memory, quality, or runtime claims into one operating envelope, and never alias either to dense Qwen3.8-27B.
## Pinned source inventory

The linked GitHub repository is unusually small and was read exhaustively: `README.md` and `tech_report.pdf` are the entire tree at `513aa6e`. The checkpoint tree at `f5d0827` was inventoried from the Hugging Face API and the following lightweight artifacts were read directly: model card, `config.json`, `generation_config.json`, `tokenizer_config.json`, `chat_template.jinja`, and the safetensors index. Weight shards were inventoried but not downloaded; no claim below depends on inspecting their bytes.

| Source class | Evidence read | Coverage / honest limit |
|---|---|---|
| Repository tree | `README.md`, `tech_report.pdf` | Exhaustive at `513aa6e`; repository contains no implementation or tests. |
| Architecture | report §§2–4; checkpoint `config.json` | Exact declared architecture and training method; report equations/benchmark tables are author-reported. |
| Runtime recipes | GitHub README and HF model card | Transformers, vLLM, SGLang, Unsloth, quantization, and serving recipes; recipes were not executed locally. |
| Tokenizer/chat/tools | tokenizer config and `chat_template.jinja` | Template structure and special-token contract; tokenizer vocabulary bytes were not exhaustively audited. |
| Weights | safetensors index and HF API sibling list | Shapes/names and shard inventory only; no 160+ GB checkpoint download. |
| Issues/PRs/discussions | GitHub API at observation time | Repository was created the same day and exposed one merged README PR; this is not mature operational evidence. |
| License/provenance | HF model metadata/model card and repository | Apache-2.0 is declared for the checkpoint; GitHub API reports no detected repository license file at the pinned revision. Preserve source attribution and re-check before copying code. |

## What the model actually is

### 1. Four interacting sparsity mechanisms

1. **Gated DeltaNet:** the report's linear-attention recurrence maintains a fixed-size state and adds input-dependent forgetting and update gates. The checkpoint spells out 16 key heads × 128 dimensions, 48 value heads × 128 dimensions, convolution width 4, and FP32 SSM state. Thirty-six of 48 layers use this path.
2. **Sparse full attention:** every fourth layer uses 24 query heads, two KV heads, 256-wide head dimension, and partial rotary position encoding. A four-head indexer scores compressed keys and selects a budget of 2,048 tokens. This is learned token selection, not a drop-in sliding window.
3. **Fine-grained MoE:** all 48 layers declare 512 routed experts, top-10 routing, expert width 640, and a shared expert of width 640. The report says this arrangement yields 82.8B total parameters while activating 3.1B per token.
4. **Hybrid MTP:** one additional full-attention predictor uses parameter-shared three-token heads (`ngram_size: 3`). The report trains the first future token autoregressively and the next two with bidirectional attention, then discards MTP during post-training. At inference it is a speculative-decoding proposal mechanism, not part of ordinary logits correctness.

The sequence is therefore approximately `3 × linear attention + 1 × sparse full attention`, repeated twelve times, with MoE in every block. Peak memory, routing locality, and state residency matter more than the “3.1B active” headline alone.

For maintainers, the key distinction is operational: the recurrent blocks trade a growing KV cache for fixed recurrent state, but the sparse full-attention blocks still need a searchable history and the MoE layers still need access to the full expert bank. A useful implementation plan measures those three memory classes separately instead of projecting one headline “active parameter” number onto the whole runtime.

### 2. Exact checkpoint contract

| Field | Pinned value | Consequence for fak |
|---|---:|---|
| `model_type` / architecture | `qwen4_exp_text` / `Qwen4ExpForConditionalGeneration` | Current Qwen aliases do not prove compatibility. |
| layers / hidden / vocab | 48 / 2,560 / 248,320 | New tensor schema and tokenizer cardinality. |
| context | 262,144 tokens | Requires explicit state/KV accounting and long-context receipts. |
| dense-attention cadence | every fourth layer | 12 sparse-attention blocks, 36 recurrent linear blocks. |
| query / KV / head dim | 24 / 2 / 256 | GQA plus unusually wide heads. |
| indexer | 4 heads, 1 KV head, 128 dim, top 2,048, compression 4 | New gather/index state and kernels. |
| experts | 512 total, 10 routed + 1 shared active | Routing, batching, residency, and expert-weight movement dominate. |
| MTP | one hybrid layer, n-gram 3 | Optional optimization; base decode must not depend on it. |
| dtype | BF16; SSM state FP32 | Mixed-precision state is part of correctness. |
| RoPE | theta 10,000,000; 25% rotary | Existing full-RoPE assumptions are unsafe. |

### 3. Training and post-training facts

The report says pretraining used roughly 16.2T high-quality tokens and introduced four relevant methods: STEPS for early training-stage prediction, a data-mixing law, GSPO for MoE reinforcement learning, and adaptive three-stage RL that shifts from broad sampling to rejection sampling to higher-temperature hard-problem training. Those methods explain claimed model quality but are not inference features to port into fak.

Reported evaluations are source claims, not independently reproduced fak results. The report/model card position the model near or above larger open models on reasoning, coding, agent, tool-use, and long-context suites, and report substantially higher long-context generation throughput than Qwen3.5-Flash under specified SGLang hardware. Treat every such number as author-observed until a matched fak-native campaign reproduces quality, engine identity, prompt length, batch, hardware, and end-to-end accounting.

## Runtime and interaction contract

- The model card requires a current Transformers build and recommends default sampling around temperature 1.0, top-p 0.95, top-k 40; the checked generation config uses temperature 1.0, top-p 0.95, top-k 20 and repeats token IDs for end-of-turn termination. Recipe drift means fak should ingest pinned config rather than copy prose defaults.
- The chat template uses explicit channel tokens (`analysis`, `final`, `commentary`) and recipient/tool-call markup. It distinguishes tool outputs and supports tool definitions. Tool-loop parity must capture rendered prompt bytes and parsed calls, not merely generate plausible text.
- Thinking and non-thinking behavior are template/runtime concerns. Existing fak semantic-stop logic is relevant, but Qwen3.8 stop handling is only partial evidence for this new tokenizer/template.
- vLLM and SGLang examples expose an OpenAI-compatible server and specify tool parsers. They are useful parity/reference paths. The native-performance invariant forbids turning either into a silent convenience fallback.
- The source lists FP8, AWQ, GPTQ, GGUF, and one-click Unsloth routes. These are ecosystem availability claims, not proof that fak can load those formats.

## Source-to-fak seam map

| Upstream behavior | Current fak evidence | Status | Decision and disconfirming check |
|---|---|---|---|
| Recognize Qwen3.8 family names and prefer fak-native engine | `internal/model/qwen38_config.go`; `cmd/fak/serve_qwen38_runtime_test.go` | **PARTIAL** | **DEFAULT:** add exact `qwen4_exp_text` config/tensor contract. Disconfirm by proving current loader parses every pinned field and tensor shape without coercion. |
| Gated-delta recurrent state with FP32 SSM | Existing Qwen3.5/Qwen3.8 hybrid seams and Metal decode kernels | **PARTIAL** | **OPTIONAL-MODULE:** adapt only after numerical traces match a pinned oracle across prefill/decode/state restore. Existing GDN similarity does not prove this model's projections, gates, or head layout. |
| 512-way top-10 routed MoE plus shared expert | Generic model/loading and GEMM infrastructure; no exact Flash-Next routing receipt found | **PARTIAL** | **DEFAULT:** implement a bounded routing/tensor spine before optimized kernels. Reject if tensor inventory shows a different packed layout than config/report imply. |
| Learned top-2,048 sparse-attention indexer | No exact indexer implementation or receipt found | **ABSENT** | **OPTIONAL-MODULE:** first build an index/gather correctness oracle and dense-reference comparison. Reject approximations that alter selected-token sets or quality. |
| Hybrid three-token MTP speculative decode | Existing decode/speculation concepts, no exact Qwen4-Exp MTP contract | **ABSENT** | **OPTIONAL-MODULE:** keep base decode independent; accept only net-true end-to-end speedup with identical accepted-token output and all proposal overhead counted. |
| 262K managed context | Context/KV/state management exists; no Qwen4-Exp long-context receipt | **PARTIAL** | **RECIPE:** stage context-length envelopes and memory plans before allocation. Reject “supports 262K” based only on parsing the config. |
| Channel/tool chat template and stop contract | Qwen semantic-stop and tool surfaces exist | **PARTIAL** | **RECIPE:** add byte-level render/parse fixtures from pinned template. Reject semantic-only tests that never render the prompt. |
| External serving through Transformers/vLLM/SGLang | Reference commands are documented upstream | **PRESENT** externally, **ABSENT** as native proof | **EXCLUDE** from automatic product execution. Permit only explicit reference/parity/benchmark selection with engine named in receipt. |

A repository search found mature Qwen3.8 work and native receipts, but no `qwen4_exp` identifier before this study. That lexical absence is not alone proof of runtime absence; the exact config/tensor mismatch above is the stronger witness.

## Recommended sequence

1. **Architecture spine:** recognize `qwen4_exp_text`; parse and validate all pinned fields; inventory expected tensors; fail closed with a typed unsupported-execution error. This is the smallest end-to-end truth improvement.
2. **Base correctness oracle:** bind a pinned Transformers or sanctioned reference serve explicitly; capture layer/tensor traces for tiny prompts; compare fak-native gated-delta, MoE routing, and sparse selection independently.
3. **Native gated-delta + MoE:** execute one layer, then one 4-layer cadence, then the full model. Keep engine identity in every receipt.
4. **Sparse indexer attention:** add selection/gather correctness before performance specialization.
5. **MTP:** optimize only after ordinary autoregressive decode is correct and measured; count verification and rejected proposals.
6. **Long-context and tool campaigns:** prove 262K envelopes and byte-level chat/tool behavior independently from throughput.

Filed packets: [#9125](https://github.com/anthony-chaudhary/fak/issues/9125) architecture/config spine; [#9124](https://github.com/anthony-chaudhary/fak/issues/9124) chat/tool fixtures; [#9123](https://github.com/anthony-chaudhary/fak/issues/9123) four-layer native oracle; [#9126](https://github.com/anthony-chaudhary/fak/issues/9126) net-true MTP gate. Each carries the durable study receipt.

## What not to borrow

- Do not alias `qwen4_exp_text` to the existing Qwen3.8 architecture merely because the product name matches.
- Do not infer native support from an OpenAI-compatible vLLM/SGLang endpoint.
- Do not market 3.1B active parameters as a 3.1B memory footprint; all 82.8B weights and routing/residency overhead still exist.
- Do not port benchmark tables into fak claims without matched quality, hardware, prompt, batch, and engine receipts.
- Do not make MTP or approximate sparse attention part of correctness-critical base decode.

## Reproducibility

```powershell
# GitHub source pin and complete tree
gh api repos/QwenLM/Qwen3.8-Flash-Next/commits/HEAD --jq .sha
git clone --filter=blob:none --no-tags https://github.com/QwenLM/Qwen3.8-Flash-Next.git source
git -C source checkout 513aa6e18a335296fc13e538232a8735b230877d
git -C source ls-tree -r --name-only HEAD

# Checkpoint pin and inventory
(Invoke-RestMethod https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next).sha
Invoke-WebRequest https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/f5d08274bafd880402bd16f5e3e6c514136ec06c/config.json -OutFile config.json
```

The study intentionally did not download weights or run claimed GPU benchmarks. That boundary keeps a source study from masquerading as a runtime result. Weight-level tensor validation, numerical parity, memory envelopes, and throughput remain implementation/benchmark follow-ons; they must use sanctioned compute with explicit fak-native engine receipts.





