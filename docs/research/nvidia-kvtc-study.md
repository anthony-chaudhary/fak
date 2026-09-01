---
title: "NVIDIA KVTC: transform-coded KV storage, studied for fak"
description: "Observed: 2026-08-26 · Tracker: #9341 · Centrality: Enabling · Disposition: adapt the storage-boundary idea; do not port the available third-party serving patches."
---

# NVIDIA KVTC: transform-coded KV storage, studied for fak

**Observed:** 2026-08-26 · **Tracker:** [#9341](https://github.com/anthony-chaudhary/fak/issues/9341) · **Centrality:** Enabling · **Disposition:** adapt the storage-boundary idea; do not port the available third-party serving patches.

## Decision in one screen

- **For:** operators keeping reusable long-context sessions across scarce HBM, DRAM, disk, or remote tiers.
- **Problem:** a stale but reusable native KV cache is expensive to retain, transfer, or recompute.
- **Today:** fak owns exact float32 `K`, pre-RoPE `Kraw`, and `V` tensors in `model.KVCache`, while `cachemeta` can describe HBM/DRAM/disk/remote residency but deliberately stores no payload (`internal/model/kvcache.go:3-18`; `internal/cachemeta/cachemeta.go:71-83,201-225`). It has eviction, cloning, routing, and metadata, but no transform-coded payload format.
- **Better because:** KVTC supplies a plausible offline-calibrated, storage-only codec: PCA decorrelation, mixed precision, then lossless DEFLATE. The paper reports roughly 20x realized compression at a 16x pre-DEFLATE target while staying within one point of vanilla on its principal non-reasoning table, and avoids recomputation cliffs in one host-memory experiment.
- **Witness:** first ship a fak-native, versioned codec experiment on Qwen3.8 cache captures; require bytes, full encode/decode and transfer time, TTFT, bit-exact structural checks, and downstream quality in one receipt. The execution path remains fak-native; a headline ratio alone cannot pass.

## Problem checks (P1-P4)

- **P1 — real recurring pain: pass.** Reusable long-context KV competes for finite HBM/DRAM and otherwise incurs transfer or recomputation. The paper's capacity-cliff experiment makes that tradeoff observable.
- **P2 — fak owns the seam: pass.** `model.KVCache` owns the native tensors, and `cachemeta` owns tier/derivation metadata. No provider-side or llama.cpp dependency is needed.
- **P3 — measurable better state: pass for a study, unproved for a feature.** Net bytes, restore TTFT, recompute avoidance, decode interference, and task quality are measurable. The experiment gates below prevent nominal compression from standing in for value.
- **P4 — smallest useful spine: pass.** A versioned CPU reference over captured Qwen3.8 cold slabs can prove transfer before fused kernels or production routing. Production enablement remains explicitly out of scope.
**Verdict:** **ADAPT / EXPERIMENT, not production adoption.** KVTC is unusually relevant to fak because it compresses stored reusable state without changing model weights or attention. The paper is credible enough to justify a bounded native prototype. It does **not** establish direct attention over compressed state, decode-loop speedups, general production distributions, or Qwen3.8 quality. The inspected public repository is an independent implementation, not NVIDIA code, and its CUDA serving path still contains placeholder kernels.

## Source and provenance ledger

Event dates are source dates; observation date for every row is 2026-08-26.

| Source | Event/state | What was inspected | Provenance / reuse disposition | Refresh cue |
|---|---|---|---|---|
| [Staniszewski & Łańcucki, *KV Cache Transform Coding for Compact Storage in LLM Inference*, arXiv:2511.01815v2](https://arxiv.org/abs/2511.01815v2), [PDF](https://arxiv.org/pdf/2511.01815v2) | v1 2025-11-03; v2 2026-03-11; accepted ICLR 2026 | Full 46-page PDF, especially §§3-5, Tables 2-5 and 22, appendices B.1-B.17 | NVIDIA/University of Warsaw research publication. Algorithmic inspiration and reported evidence only; the PDF names no accompanying source repository. **INSPIRE-ONLY** unless separate code/licensing appears. | New arXiv version, author code link, ICLR artifact, erratum |
| [OpenReview aNVKROYpLB](https://openreview.net/forum?id=aNVKROYpLB) | ICLR 2026 forum | Publication identity (API was challenge-gated during this pass) | Reviews/rebuttal not available to this pass; this is a ledger gap, not evidence of absence. | OpenReview access becomes available |
| [`OnlyTerp/kvtc@79d290621166`](https://github.com/OnlyTerp/kvtc/tree/79d290621166a9ebcace8b992c2c7ee996d48194) | created 2026-03-25; tip 2026-04-17; observed 22 stars, 5 forks, 3 open issues; no tags/releases | Python pipeline, PCA/quantization/entropy code, CUDA code, tests, benchmark JSON, workflow, issues/PRs, full history | Independent project: no author or NVIDIA affiliation established. MIT license present from initial commit. **ADAPT** for small ideas only, with attribution; do not treat its results as paper reproduction. | New release/tag, resolved integration issue, non-placeholder CUDA path, independent benchmark |
| [OnlyTerp issue #6](https://github.com/OnlyTerp/kvtc/issues/6) | open 2026-04-27 | Reports vLLM V1 prefill bypassing the monkey patch, leaving decode-only interception | Negative maturity evidence. | Issue closes with a tested integration |
| [OnlyTerp PR #3](https://github.com/OnlyTerp/kvtc/pull/3) | open 2026-04-16 | Proposed 93 integration regression tests, not merged at observed tip | Unmerged evidence; not credited to the tip. | PR merges or is superseded |
| Search surfaces | GitHub repository search, paper references, repository metadata/history | No NVIDIA-owned KVTC repository located; several third-party projects found | **“No official code located”**, not “official code does not exist.” | NVIDIA publication/project page appears |

Primary-source facts below cite paper section/table. Repository statements cite immutable paths at `79d2906`.

## What the paper actually proposes

### Lifecycle and tensor geometry

KVTC is a **storage codec**, not a new attention algorithm (paper §5). At a turn boundary, old K/V positions are transformed and stored compactly; before reuse, they are decoded back into the ordinary KV layout. Recent positions and attention sinks remain uncompressed. The paper's simulated multi-turn protocol compresses/decompresses every 16 tokens while preserving a sliding window of 112-128 tokens (§4). It explicitly leaves inference directly in principal-component space to future work (§5).

Calibration forwards a representative corpus and pools token positions. For each sampled position it concatenates the corresponding tensors across `l` layers, `h` heads, and `d_head` into one feature row of width `p=l*h*d_head`. It excludes sinks/recent tokens. For keys, positional rotation is undone before fitting/compression because RoPE obscures low-rank structure (§3.1). Keys and values are calibrated separately; the paper reports keys generally compress better (Appendix Table 7).

This matters for fak: `KVCache.K` is post-RoPE and `Kraw` is already pre-RoPE (`internal/model/kvcache.go:3-18`). A native experiment should encode `Kraw` and regenerate post-RoPE `K` on restore rather than numerically undoing rotation. That is a fak-specific extension, not a paper result.

### 1. PCA decorrelation

From centered calibration matrix `C`, KVTC computes an offline randomized SVD/PCA basis `V` and mean `μ`; inference maps `D=(X-μ)V` and restores `X≈DVᵀ+μ` (§3.1). Unlike prompt-local SVD baselines, one model-specific basis is reused across requests. Trailing components can be truncated. Calibration uses about 200K tokens on one H100 SXM 80GB and completes “within minutes”; larger calibration was not evaluated (§5).

The basis spans layers and heads. That extracts global redundancy but creates deployment coupling: model revision, tensor layout, RoPE convention, layer partition, and calibration identity must all be codec identity. Paper Table 4 shows separate per-GPU chunks for a four-way pipeline-parallel Llama 3.3 70B run; joint compression may be better but changes placement and transfer costs.

### 2. Adaptive mixed-bit quantization

PCA coordinates are ordered by variance. A dynamic program minimizes calibration-set Frobenius reconstruction error under a global bit budget (§3.2, Appendix B.17). Consecutive coordinates share a 16-bit shift and scale; allowed group sizes are `{1,16,64,256,1024}`. The DP chooses bit width and group size, assigns fewer bits to later components, and may assign zero bits, permitting basis truncation. Metadata and PCA parameters must count toward stored bytes; the paper's compression-ratio notation targets payload before DEFLATE and excludes the uncompressed sliding-window tokens (§4).

Frobenius error is only a proxy. The authors explicitly say it does not guarantee task-level quality and may be task-dependent (§5). Therefore a fak gate cannot substitute cosine similarity or reconstruction error for generated-task quality.

### 3. Bit packing and entropy coding

Quantized coordinates are packed into bytes, then losslessly DEFLATE-compressed on GPU through NVIDIA nvCOMP (§3.3). DEFLATE's gain is content-dependent. Appendix Table 14 compares lossless codecs. This last stage saves storage/transfer bytes but adds latency and workspace; it is not responsible for lossy quality change.

### 4. Restore and attention

Restore performs DEFLATE decode, unpack/dequantize, inverse projection, mean addition, and key positional reapplication before ordinary attention. KVTC leaves cache structure and attention computation unchanged (§5). Thus it composes in principle with token eviction, but it does not reduce the resident decode-time attention footprint once fully restored. Its strongest immediate fit is cold/warm tier storage and transfer, not hot-token decode acceleration.

## Quantitative evidence, with envelopes

### Quality and ratio

Paper Table 2 evaluates Llama 3.1 8B, NVIDIA Minitron 8B, and Mistral NeMo 12B on GSM8K, MMLU, Qasper, Lost-in-the-Middle, and RULER variable tracking. A **16x target before DEFLATE** realizes ranges of **17-22x**, **17-21x**, and **17-20x** respectively. The authors report every 16x-target score within <1 accuracy/F1 point of vanilla. Higher targets are not uniformly safe: realized 32x/64x ranges reach 31-46x/51-95x, while long-context quality can fall sharply (for example Minitron LITM 99.8 vanilla, 86.9 at target 32x, 59.5 at 64x). “Up to 40x+” is therefore cohort-specific, not the default claim.

Comparators include GEAR 2-bit and KIVI 2-bit (about 5x), H2O/TOVA token eviction (8x), xKV prompt-local SVD (1-5x), and FP8 (2x). The paper gives xKV an easier prefill-only protocol because recomputing its SVD during decode is prohibitive (§4). Ratios count compressed tokens only and omit the live sliding window, so whole-cache savings are lower for short contexts.

For distilled DeepSeek-R1 Qwen2.5 1.5B/7B, paper Table 3 uses temperature 0.6/top-p 0.95. At target 8x, LiveCodeBench drops 0.3 and 0.2 points; target 16x is less stable (AIME25 40.8→38.3 on 7B). AIME uses eight runs with large standard deviations. This is not evidence for Qwen3.8.

On four pipeline-parallel GPUs, Llama 3.3 70B MATH-500 falls 1.2 points at 10x and 3.0 points at 20x; the paper describes these as within 1.5 standard errors (Tables 4-5 discussion). This demonstrates partition compatibility, not quality equivalence.

### Latency and end-to-end behavior

Paper Table 5 is a simple Transformers implementation on one H100 with Mistral NeMo 12B bf16:

| Batch/context | Compress | Decompress | Vanilla recompute TTFT | KVTC decompress TTFT |
|---|---:|---:|---:|---:|
| BS=8, 8K | 379 ms | 267 ms | 3098 ms | 380 ms |
| BS=2, 16K | 194 ms | 143 ms | 1780 ms | 208 ms |

The module rows attribute projection, quantization, and DEFLATE portions, but totals and overlap should be taken from the table rather than summed naively. The useful comparison is restore versus recomputation; this is not a decode-tokens/s speedup.

Appendix Table 22 is more operational: vLLM+LMCache, Llama 3.3 70B FP8, 2×H100 80GB tensor parallel, 128 GiB host DRAM **per GPU**, 62-66K initial contexts, 16-100-token questions, 100-token answers, no think time. At 1-10 clients, 16x KVTC adds latency (e.g. TTFT at 10: 3.4→4.5 s). At 12 clients vanilla exceeds host capacity and recomputes: TTFT is 136.6 s vanilla versus 5.6 s KVTC; at 16 it is 181.6 versus 7.3 s. This proves a capacity-cliff benefit in that synthetic envelope. It does not prove typical production benefit: the authors note realistic users pause, and compression/decompression share the same GPU.

## Paper limitations that bound a fak decision

- Simulated multi-turn conversations may not reflect production content or interaction patterns; dense decoder-only models span 1.5B-70B (§5).
- Calibration is tested only to ~200K tokens; PCA scaling beyond that is future work (§5).
- Direct compressed-space inference, fused kernels, and hierarchical PCA are future work (§5).
- Reconstruction error is an imperfect quality proxy (§5).
- One basis couples the codec to model and layout. Cross-model, model-update, adapter, architecture, and calibration-drift behavior is not established.
- Ratios exclude recent/sink tokens and typically report compressed payload; codec metadata, basis residency, allocator padding, temporary workspaces, and durability framing must be charged in fak's net accounting.
- The paper predates and does not test Qwen3.8, fak's native float32 plus `Kraw` representation, hybrid/recurrent caches, adversarial contexts, cache poisoning, or exact eviction/reposition semantics.

## Independent implementation audit

`OnlyTerp/kvtc` is useful as executable pseudocode, not authoritative NVIDIA code.

- The MIT-licensed tip implements a CPU/PyTorch path: per-layer/per-head fitting and transform, scalar/NumPy bit packing, and Python `zlib` DEFLATE (`src/pipeline.py`, `src/pca.py`, `src/quantize.py`, `src/entropy.py`). Its grouping and calibration geometry are not a demonstrated exact reproduction of the paper's cross-layer basis and DP.
- Local tip test run on 2026-08-26: `python -m pytest -q` → **38 passed, 1 skipped in 9.48 s**. The skip is the real-model/GPU cohort; this proves basic round trips, not serving quality or claimed GPU throughput.
- The default-branch CI runs Python 3.10/3.11 CPU pytest only (`.github/workflows/test.yml`). No released artifact or tag exists.
- `cuda/kvtc_kernels.cu:464,496,513` labels GPU bit packing, unpacking, and FP32→FP16 conversion as unfinished placeholder work. That disqualifies this CUDA path as a fak performance base.
- Open issue #6 says vLLM V1 prefill bypasses the monkey patch. The repository also contains llama.cpp patch scripts; fak's native-inference invariant forbids adopting that serving route.
- Benchmark JSON is checked-in producer output, mostly self-authored and not cryptographically tied to hardware/runtime/model artifacts. It can suggest tests but cannot substantiate fak claims.
- History is small and concentrated (35 commits by the owner identity, seven by an automation bot, three by another bot identity in local `shortlog`); PR #3's broader integration suite is unmerged. This is maturity context, not a quality judgment.

**License decision:** small MIT ideas could be adapted with attribution, but a from-paper fak-native implementation is preferable: it avoids Python/CUDA/llama.cpp integration debt and keeps fak ownership of kernels, scheduling, memory, cache, and operations.

## Gap witness and concrete seams

Self-query run at trunk `61328912f827`:

```text
fak capabilities "study external NVIDIA KVTC paper repository implementation and derive evidence-backed borrow decisions"
→ cards for resident-context reuse and cache savings, but no transform-coded KV payload codec

go run ./cmd/fak-dev index docs --limit 5 "KVTC PCA entropy"
→ unrelated i18n/study docs; no KVTC design surfaced

go run ./cmd/fak-dev index leaves --limit 5 "KVTC PCA entropy"
→ only unrelated logvault
go run ./cmd/fak-dev index claims --limit 5 "KVTC PCA entropy"
→ no matching claim
```

Verdict: **ABSENT as an indexed capability**, while adjacent primitives are **PARTIAL**:

1. `internal/model/kvcache.go:3-18` owns raw and rotated tensors. Add a codec boundary around immutable old-position slabs; do not mutate hot decode storage in the first spine.
2. `internal/model/kvcache.go:94-185` compacts/repositions on eviction. A restored slab must preserve `pos`, `Kraw`, recurrent-cache fences, and fresh-prefill equivalence before it may re-enter this path.
3. `internal/cachemeta/cachemeta.go:71-95` has identity and derivation fields. A compressed object needs serializer/codec identity including model revision, tokenizer/layout, PCA/calibration digest, target budget, sink/window policy, and format version.
4. `internal/cachemeta/cachemeta.go:201-225` already names HBM, DRAM, disk, remote, provider, and recompute tiers, but explicitly carries no payload. Keep policy/accounting here; place bytes behind a separate native store/engine seam.
5. `internal/gateway/residency_router.go:43-48,83-91,343-354` ingests resident-prefix add/drop events. Compression changes capacity and restore cost, so routing needs witnessed compressed bytes and predicted restore latency, not a boolean residency lie.
6. `docs/native-inference-goal.md` is the acceptance authority: engine receipts must identify fak-native execution. A receipt naming any non-native execution path is rejection, not progress.

## Bounded-superset portfolio

| Priority | Disposition | Candidate | Why / boundary |
|---|---|---|---|
| P0 | **ADAPT** | Versioned cold-slab codec interface plus uncompressed control | Smallest end-to-end spine; format and accounting precede optimization. |
| P0 | **ADAPT** | Offline PCA on fak-captured **Qwen3.8** `Kraw` and `V`, separate by tensor kind; mixed-bit allocator; CPU reference encode/decode | Tests whether the core rate-distortion idea transfers before kernels. Include sink/recent-window bypass and full metadata bytes. |
| P0 | **ADOPT principle** | Compare restore against recompute and raw transfer at DRAM/disk/remote tier boundaries | This is the paper's strongest use case and fak's actual decision boundary. |
| P1 | **ADAPT** | GPU projection/quantization/pack and a lossless codec selected by measured Pareto frontier | nvCOMP/DEFLATE is a candidate, not dogma. Include workspace, synchronization, and contention. |
| P1 | **ADAPT** | Cachemeta serializer identity + fail-closed compatibility checks | Prevent decoding with wrong model/basis/layout or silently serving stale state. |
| P1 | **ADAPT** | Routing/capacity cost model from measured compressed bytes and restore latency | Admit compression only when avoided eviction/recompute outweighs codec cost. |
| P2 | **DEFER** | Hierarchical/per-layer PCA, basis clustering, calibration drift detector | Optimize only after one global basis proves quality and net value. |
| P2 | **DEFER** | Compose with token eviction | Two lossy mechanisms require factorial quality testing and provenance. |
| P3 | **REJECT now** | Attention directly in PCA/compressed space | Paper leaves it future work; it changes hot kernels and numerical semantics. |
| P3 | **REJECT** | OnlyTerp llama.cpp/vLLM monkey patches or placeholder CUDA kernels | Violates native ownership and lacks a complete witnessed path. |
| P3 | **REJECT** | Default-on “20x KV” marketing | Ratio is workload-, target-, window-, metadata-, and quality-dependent. |

This is a bounded superset: it covers reference correctness, storage economics, accelerated implementation, governance, and later composition, while explicitly stopping before speculative compressed attention.

## First experiment and falsifiable gates

**Spine:** capture immutable old-position slabs from a fak-native Qwen3.8 run on the sanctioned GPU path; train one reproducible basis; encode to a versioned blob; evict the source; restore it; continue generation. Preserve an uncompressed arm and recompute arm from the same prompts.

Record in one receipt:

- engine/model/artifact revision, hardware and software stack;
- corpus and calibration digest, sample count, exclusions, basis dimensions/bytes;
- raw `K`, `Kraw`, `V`, metadata, compressed blob, workspace, and peak HBM/DRAM bytes;
- encode, transfer/write, read, decode, recompute, TTFT, inter-token latency, and throughput distributions under concurrency;
- whole-cache ratio including live window, sinks, basis, scales/shifts, framing, allocator padding, and replicas;
- reconstruction metrics **and** task outputs for long-context retrieval, code, math/reasoning, multi-turn editing, and adversarial/out-of-distribution prompts;
- cache compatibility failures and exact engine receipt naming fak-native execution.

**Advance from reference to kernels only if all hold:**

1. Restored shape, positions, `Kraw`→RoPE regeneration, eviction/reposition behavior, and deterministic replay invariants pass; wrong codec/model/calibration identities fail closed.
2. At least **8x net resident-byte reduction** at the chosen safe arm after every overhead, not merely payload bits. (A conservative feasibility gate, not a paper claim.)
3. On a pre-registered evaluation set, no aggregate task metric loses >1 absolute point **and** no critical cohort loses >2 points; confidence intervals and sample counts accompany the decision.
4. At a declared cold-tier reuse interval/concurrency, p95 restore TTFT beats both raw fetch/transfer and recomputation by ≥20%; encode+decode+I/O+verification all count.
5. Hot decode throughput regresses <2% when compression is idle, and the compression cohort stays within its declared SLO under shared-GPU contention.
6. No fallback engine, no excluded failures, and no savings claim based on simulated bytes when physical allocation can be measured.

If quality fails, lower the target or split K/V and layer groups. If latency fails while capacity wins, keep it opt-in for eviction-cliff avoidance. If neither wins end to end, reject the codec despite a high nominal ratio.

## Risks and unanswered questions

- Does a cross-layer basis remain stable across Qwen3.8 checkpoints, adapters, quantization modes, and long-running agent domains?
- Can `Kraw` eliminate the paper's inverse-RoPE cost without inflating the baseline so much that reported ratios become misleading?
- How should hybrid/recurrent layers be fenced? The current cache has typed unsupported eviction paths; do not flatten them into a dense-KV promise.
- Is DEFLATE best after mixed-bit packing on modern NVIDIA hardware, or do LZ4/ANS/no-lossless-stage win net TTFT?
- What is the integrity/authentication format for blobs crossing remote/disk tiers, and how are basis artifacts revoked?
- Does compression amplify rare attention failures, prompt injection persistence, or poisoned-cache effects even when average quality is flat?
- How much GPU scheduling interference does online compression cause during active decode, especially near the very capacity cliff it is meant to avoid?

## Follow-up issue candidates

File only after deduplication:

1. **`gen/now`: Qwen3.8 KVTC reference spine** — capture, calibrate, encode/restore, quality + byte receipt, uncompressed/recompute controls.
2. **Codec identity and cachemeta admission** — versioned format, basis digest, compatibility matrix, fail-closed corruption tests.
3. **GPU rate-distortion kernel shootout** — PCA projection, allocator, packing, DEFLATE/LZ4/ANS comparison with net accounting.
4. **Tier admission simulator and live dogfood** — derive break-even reuse distance from measured tier bandwidth, codec latency, capacity, and recompute.
5. **OOD and adversarial quality suite** — calibration drift, long-context retrieval, code editing, reasoning, poisoned/sink-heavy prompts.

## Reproduction receipt

```powershell
# Primary paper
Invoke-WebRequest https://arxiv.org/pdf/2511.01815v2 -OutFile $env:TEMP\kvtc-2511.01815v2.pdf
pdftotext -layout $env:TEMP\kvtc-2511.01815v2.pdf $env:TEMP\kvtc-2511.01815v2.txt

# Independent implementation, immutable tip observed in this study
git clone https://github.com/OnlyTerp/kvtc $env:TEMP\kvtc-study-src
git -C $env:TEMP\kvtc-study-src checkout 79d290621166a9ebcace8b992c2c7ee996d48194
python -m pytest -q  # 38 passed, 1 skipped in 9.48s on this host
rg -n "T[O]DO|placeholder" $env:TEMP\kvtc-study-src\cuda

gh api repos/OnlyTerp/kvtc
gh api repos/OnlyTerp/kvtc/issues?state=all
gh api repos/OnlyTerp/kvtc/pulls?state=all

# fak gap witness
go run ./cmd/fak-dev index docs --limit 5 "KVTC PCA entropy"
go run ./cmd/fak-dev index leaves --limit 5 "KVTC PCA entropy"
go run ./cmd/fak-dev index claims --limit 5 "KVTC PCA entropy"
```

The study is complete when read as a decision, not as an implementation claim: **KVTC earns a fak-native Qwen3.8 storage-boundary experiment; nothing inspected earns production enablement yet.**
