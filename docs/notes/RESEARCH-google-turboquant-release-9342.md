# Google TurboQuant release — what fak should borrow, and what the headline omits

**Issue:** [#9342](https://github.com/anthony-chaudhary/fak/issues/9342)  
**Study date:** 2026-08-26  
**Verdict:** **borrow the rotated scalar-quantization design and fused-query decode shape as a candidate for fak's lossy KV tier; do not adopt an implementation or repeat “5× with no quality loss.”** TurboQuant is credible algorithmic prior art, but Google released a paper and explainer, not a production runtime. The paper's LLM evidence is simulated cache quantization over Llama 3.1 8B and Gemma 2 2B; it does not report an end-to-end serving speedup, a fused production kernel, or a fak-native generation receipt. Its nominal 3-bit key / 2-bit value settings also carry per-vector metadata and padding, and conflict with fak's lossless pre-RoPE `Kraw` exact-eviction path if substituted there.

## Decision frame

- **Centrality:** Enabling. This can lower the native engine's KV bandwidth and capacity cost, but it is not the engine spine.
- **For:** fak native-inference and KV-cache maintainers.
- **Problem:** long-context KV storage and reads grow linearly with sequence length; naive low-bit scalar quantization loses too much dot-product fidelity.
- **Today:** fak keeps lossless `f32 Kraw` for exact eviction and uses `q8_0` materialized K/V, with separate int2/vector-quality evaluation leaves. The earlier [#1266 note](RESEARCH-turboquant-kv-quant-triage-1266.md) examined one unofficial PyTorch repository, before the post-release implementation and critique wave.
- **Better because:** this study separates paper-authored results, Google's relaying, third-party implementation claims, and facts witnessed in source.
- **Witness:** the source ledger, immutable code revisions, equations and matched native-only experiment below.

### P1–P4

| Check | Finding |
|---|---|
| P1 — important problem | **Yes.** KV capacity and bandwidth constrain long-context concurrency and decode throughput. |
| P2 — real next-best alternative | Compare against fak's current `q8_0` K/V and a plain per-vector int2/int3 scalar baseline, not only FP16. |
| P3 — smallest end-to-end spine | One fak-native, fused attention path for a non-authoritative lossy cache tier, with quality, memory and net latency receipts. Keep `Kraw` unchanged. |
| P4 — operating proof | Qwen3.8 served generation on a sanctioned CUDA node; no attention-cosine-only promotion; any explicitly selected llama.cpp reference arm stays outside the native path. |

## Sources and versions

All web sources were accessed 2026-08-26. “Witnessed” means inspected directly for this note; it does not mean benchmark-reproduced.

| Source | Pinned identity | Provenance and use |
|---|---|---|
| [Google Research post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | published 2026-03-24; page retrieved 2026-08-26 | **Google-relayed** overview. It links the paper but no Google code repository. |
| [TurboQuant paper](https://arxiv.org/abs/2504.19874) ([PDF v1](https://arxiv.org/pdf/2504.19874v1)) | arXiv:2504.19874v1, 2025-04-28; Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni | **Author-authored** algorithm, proofs and experiments. Google later says it was accepted at ICLR 2026. |
| [0xSero/turboquant](https://github.com/0xSero/turboquant/tree/7ac9b8d165a3f7d5e6df33b0450bc1f88ec0d4d5) | `7ac9b8d165a3f7d5e6df33b0450bc1f88ec0d4d5` | **Third-party code, witnessed.** Small PyTorch prototype; GPL-3.0 at the pinned tip. |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch/tree/999713889a18c0ffa20c62a65e7cbbe5746794e3) | `999713889a18c0ffa20c62a65e7cbbe5746794e3` | **Third-party code, witnessed.** Reference-oriented PyTorch package; MIT. Baseline of #1266. |
| [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant/tree/c6b2ee90d17eecb43c0afa273ae5ef8ecc8a1a3d) | `c6b2ee90d17eecb43c0afa273ae5ef8ecc8a1a3d` | **Third-party fork, witnessed.** Large vLLM fork with TurboQuant-specific Python/CUDA paths; Apache-2.0. Not upstream vLLM or Google code. |
| [Revisiting RaBitQ and TurboQuant](https://arxiv.org/abs/2604.19528) | arXiv:2604.19528v2, 2026-04-30 | **Independent comparison.** Argues both are instances of randomized Hadamard rotation + scalar quantization; reports symmetric experiments. |
| [Note on TurboQuant and DRIVE/EDEN](https://arxiv.org/abs/2604.18555) | arXiv:2604.18555v1, 2026-04-20 | **Prior-art critique by DRIVE/EDEN authors.** Disputes novelty framing; not an independent neutral replication. |
| fak tree | `internal/model`, `internal/kvint2eval`, `internal/kvquantmeta`, `internal/kvquantquality`, `internal/kvvectoreval` at `61328912f827ec9d2c51ed808d3b16ac4c906f75` | **Directly witnessed** current comparison surface. |

No official Google implementation was located from the post, paper, Google Research GitHub organization, or paper links. Absence of a link is evidence of **no published official code found**, not proof that internal code does not exist.

## Mechanism

For a vector `x ∈ R^d`, TurboQuant applies a randomized fast orthogonal transform before scalar quantization:

1. Compute `z = H D x`, where `D` is a reproducible random sign diagonal and `H` is a normalized Walsh–Hadamard transform. Padding is required when the implementation's Hadamard routine cannot handle `d` directly.
2. Compute a per-vector scale/radius (the paper describes normalization and a finite scalar grid), then round each coordinate of `z` to one of `2^b` levels.
3. Store the packed `b`-bit codes plus the scale/norm and enough seed/sign metadata to reproduce the rotation.
4. For reconstruction, invert the quantizer and orthogonal transform. For a dot product, do not reconstruct every cached key: rotate/scale the query once, then dot it directly with the key codes and correction metadata.

The useful systems asymmetry is step 4. A query is reused against all cached keys, so its `O(d log d)` transform is amortized across sequence length; key codes remain low-bit in memory and can feed a fused low-bit dot-product kernel. Values are different: attention weights are not reused like the query, so the runtime must either decode values during the weighted sum or fuse low-bit value accumulation.

### What is actually novel enough to borrow

The core family is not unique to TurboQuant. The RaBitQ comparison reduces RaBitQ and TurboQuant to the same broad recipe—randomized Hadamard rotation followed by scalar quantization—and finds their distortion broadly similar under symmetric settings. The DRIVE/EDEN note says randomized rotation, scalar quantization and unbiased inner-product estimation were established earlier, while acknowledging differences in grid/normalization choices. Treat the paper as a useful LLM-oriented composition and analysis, not ownership of the entire transform-quantize family.

## Bit and cost accounting

“3-bit keys” is the code width, not automatically the resident-cache width.

For a `d`-component vector, a practical accounting is:

`resident_bits = padded_d × b + scale_bits + norm/correction_bits + seed/sign_bits + alignment_bits + allocator/page slack`.

If signs are derived from a layer/head seed, sign storage can be near zero; if every vector carries them, the design loses its point. A 3-bit vector also needs packing boundaries and usually byte/word alignment. For head dimension 128, raw codes are 384 bits = 48 bytes. Adding two FP16 values gives 52 bytes before alignment, or **3.25 bits/component**, not 3. A 64-byte-aligned allocation becomes **4 bits/component**. Padding a non-power-of-two head dimension increases it further. The same caveat applies more strongly to 2-bit values.

Relative to FP16 payload alone, ideal code ratios are `16/3 = 5.33×` and `16/2 = 8×`; they are not a whole-KV-cache 5.33× result. Keys and values use different widths, metadata differs, and the allocator plus page table remains. Relative to fak's current `q8_0`, the useful baseline is roughly 8-bit payload plus block scales, so ideal payload gain is only `8/3 = 2.67×` for keys and `8/2 = 4×` for values before metadata. A benchmark must report physical bytes from the native allocator, not divide nominal bit widths.

### Compute and kernel consequences

- Encode: fast Hadamard/sign transform plus quantization, `O(d log d)` per new K/V vector. At one token at a time this launch overhead matters.
- Query: one transform per layer/head/query, amortized over all cached keys.
- Attention: packed low-bit load, unpack, scale/correction and accumulation must be fused. Materializing FP16 K/V defeats much of the bandwidth benefit.
- Values: require a second fused decode/weighted-sum path; a key-only kernel cannot support the paper's key+value memory headline.
- Shapes: Hadamard support, padding, grouped-query attention, page/block layout and tail tokens are product constraints, not details.
- Numerical contract: accumulation type, deterministic sign generation, RoPE order, scale precision and unbiased-estimator correction must be receipt fields.

## Claims, with their actual envelope

### Paper-authored

The paper proves near-optimal distortion-rate bounds for its online vector quantizer and reports vector-retrieval and LLM KV experiments. Its LLM section simulates quantize/dequantize around cached tensors for **Llama 3.1 8B** and **Gemma 2 2B**, evaluating LongBench/RULER-style tasks and perplexity. It reports that low-bit TurboQuant is close to full precision and often better than the compared scalar baselines. Those are model/task averages under the paper's software experiment, not a universal “no quality loss” guarantee.

The paper also reports microbenchmarks for transform/quantization components. It does **not** establish a production end-to-end decode speedup including encode, packing, metadata, page management, fused attention and recovery costs. Its theoretical distortion result is not a generation-quality theorem.

### Google-relayed

Google's post says TurboQuant can compress KV caches “by at least 6×” and reports no significant quality degradation in its highlighted experiments. It also discusses a broader vector-database use case. This is an explainer of the authors' work, not a release receipt for a Google serving implementation. The post's ratio and the ecosystem's common “5×” wording use different denominators/settings; neither should be copied without its exact table and byte accounting.

### Third-party-relayed

The repositories advertise combinations such as 3-bit keys, 2-bit values, “5×,” attention fidelity, or vLLM integration. These are maintainer claims. This study inspected source shape but did not reproduce their CUDA measurements or generation scores.

## Implementation reality

### `0xSero/turboquant`

This is a compact, pure-PyTorch prototype: the pinned package has no Triton or CUDA source. It implements transform/quantizer logic and an unfused attention helper, which makes the math readable but repeatedly reconstructs tensors. It is not a complete model server: no scheduler, allocator ownership, production paged-cache lifecycle, fused packed-cache attention, or independently witnessed model-quality suite is supplied by the small package.

### `tonbistudio/turboquant-pytorch`

This is the clearest readable reference and the source studied in #1266. It exposes transform, scalar quantization and KV-cache experiments in ordinary PyTorch. That readability also means it is not evidence of a production fused path. README attention-cosine fidelity is a diagnostic, not a served-generation witness.

### `mitkox/vllm-turboquant`

This is the closest inspected systems integration: a vLLM fork with TurboQuant-specific cache/configuration and CUDA/Python code among the normal vLLM tree. It demonstrates that the feature crosses cache allocation, attention dispatch, model configuration, tests and kernels. It is still an out-of-tree third-party fork, pinned to one vLLM history; it is not evidence that upstream vLLM, Google, or fak supports the format. Importing it would also violate fak's native ownership goal; it is prior art to inspect, not a fallback engine.

### Supply-chain and license

The licenses differ materially: `0xSero/turboquant` changed to **GPL-3.0** at the pinned tip, `tonbistudio/turboquant-pytorch` is MIT, and the vLLM fork is Apache-2.0. License compatibility does not remove provenance work: copy no code without preserving notices and tracing whether generated/copied kernels have additional origins. Low-bit cache bytes are model-derived sensitive state just like FP16 KV; compression is not encryption. New parsers/kernels must reject impossible dimensions, scale NaN/Inf, truncated packed buffers and integer-overflow sizes before device access. Deterministic random signs must be domain-separated by model/layer/head and recorded without introducing secret-dependent behavior.

## Fit with fak-native inference

The existing #1266 conclusion still holds, but the release wave sharpens it:

- **Do not replace `Kraw`.** fak retains lossless pre-RoPE `f32 Kraw` so exact eviction can reconstruct/replay correctly. Lossy rotation/quantization destroys that authority. TurboQuant belongs only in a derived, evictable materialization tier.
- **Compete with `q8_0`, not FP16 marketing.** fak's current K/V materialization already quantizes. The adoption bar is incremental physical-byte and net-throughput gain over that path at equal served quality.
- **Use existing evaluation leaves.** `internal/kvint2eval` and `internal/kvvectoreval` can host deterministic vector diagnostics; `internal/kvquantquality` and `internal/kvquantmeta` already encode quality/metadata discipline. Attention cosine can reject a bad candidate, never promote one by itself.
- **Preserve engine identity.** The final receipt must name the fak-native engine. llama.cpp/vLLM may be reference arms only when explicitly selected; there must be no auto/recovery fallback.
- **RoPE order is load-bearing.** The experiment must state whether rotation quantizes pre- or post-RoPE keys and how the query transform composes with RoPE. Exact eviction continues to consume untouched pre-RoPE `Kraw`.

## Matched native-only experiment

Run on the sanctioned CUDA fleet path in [fleet-compute-nodes.md](../fleet-compute-nodes.md), using **Qwen3.8** and the same model artifact, prompts, scheduler, batch/concurrency, context lengths, sampling, attention backend, allocator and fak-native engine in every arm.

1. **Arms:** current fak `q8_0 K/V`; plain per-vector int3-K/int2-V baseline; TurboQuant-class int3-K/int2-V candidate; optional FP16 diagnostic only.
2. **Contexts:** at least 4K, 16K, 32K and the largest stable model envelope; include GQA heads, partial cache pages and non-power-of-two/tail handling.
3. **Quality gates:** fixed-seed perplexity plus LongBench/RULER/needle and tool-use generation tasks. Set tolerances before running. Record task-level deltas, not only mean score. A candidate that passes cosine but fails generation is rejected.
4. **Net performance:** prefill and decode tokens/s, p50/p95/p99 inter-token latency, time-to-first-token, encode/pack cost, query-transform cost, fused attention cost, allocator/page overhead, peak and steady physical device bytes, concurrency at OOM, and recovery/verification overhead.
5. **Ablations:** key-only, value-only, metadata precision, group size, transform implementation, fused versus materialized decode, and deterministic seed strategy.
6. **Receipts:** engine=`fak-native`, module versions, model hash, GPU/driver/kernel identifiers, exact format (`bK/bV`, scales, correction, padding), physical bytes, quality corpus hashes and all failures. No silent vLLM/llama.cpp arm.
7. **Promotion:** accept only if quality-constrained net throughput or admitted concurrency beats `q8_0` and every exact-eviction round trip remains bit-identical through untouched `Kraw`.

This is ready to dispatch to the lab; lack of a local GPU is not a terminal blocker.

## Borrow / reject / defer

| Decision | Item | Why |
|---|---|---|
| **Borrow** | Reproducible randomized Hadamard preconditioning before low-bit scalar quantization | Strong, simple way to flatten outliers; supported by multiple method families. |
| **Borrow** | Query-side transform amortized over cached keys | The systems idea that can preserve low-bit residency through attention. |
| **Borrow** | Explicit correction/scale metadata and distortion tests | Avoids pretending nominal code bits are the whole format. |
| **Borrow** | Fused packed-code attention shape from the vLLM fork | Architectural reference only; reimplement fak-native after SOTA check. |
| **Reject** | Replacing authoritative `Kraw` | Breaks the exact-eviction/replay invariant. |
| **Reject** | “5×/6×, no quality loss” as a fak claim | Denominator, metadata, model/task and runtime envelope are missing. |
| **Reject** | Automatic runtime fallback to a foreign engine or vendoring a fork | vLLM/llama.cpp are allowed only when explicitly selected for reference, parity, migration or borrowing; the product path remains fak-native. |
| **Reject** | Attention cosine as acceptance evidence | It does not prove generation quality. |
| **Defer pending witness** | Product default below `q8_0` | Requires Qwen3.8 native served quality and net accounting. |
| **Defer pending profile** | 3-bit K / 2-bit V as fixed format | Optimal widths and metadata depend on model, layer, head and hardware. |

## Recommendation

Keep TurboQuant in the SOTA catalog as a **candidate quantizer for a lossy derived KV tier**, not a shipped fak capability. The next implementation ticket should be one end-to-end spine: a fak-native Qwen3.8 experimental arm that preserves `Kraw`, implements one fused low-bit attention path, emits physical-byte/engine receipts, and compares against `q8_0` under predeclared generation-quality gates. Split kernel variants, per-layer adaptation and productization only after that spine wins.

No change is justified to the exact cache or public performance claims today. The study found no security threat requiring defense and no official Google code to adopt. The remaining work is empirical, not another literature note.


