---
title: "Attention-preserving KV vector quantization evaluation"
description: "Contract: fak.kvvectoreval/v1 Status: research interoperability contract; no kernel integration"
---
# Attention-preserving KV vector quantization evaluation

Issue: [#6259](https://github.com/anthony-chaudhary/fak/issues/6259)  
Contract: `fak.kvvectoreval/v1`  
Status: research interoperability contract; no kernel integration

## Named research target and immutable inputs

This evaluation preserves the full named target rather than reducing it to a generic
"vector quantization" capability:

- Paper: **Spend Bits Where Queries Look: KV Cache Vector Quantization with
  Attention-Preserving Transforms**, `arxiv:2608.04074v1` (NOVA-KV).
- PDF: `https://arxiv.org/pdf/2608.04074v1`, SHA-256
  `7cd51970952e7fd72fd36db00e194dade32cae7c410d45982eb21c148721ab77`.
- arXiv source: `https://export.arxiv.org/e-print/2608.04074v1`, SHA-256
  `72edf6938775c532e63d6703164bbebc8ca7d722aea683cc24c7674601ca17d4`.
- Public recipe: `github.com/Amir-zsh/nova-kv@d81c77b007d7a8e50ed608134fcb0feba0269ef8`.
- Recipe artifact manifest SHA-256:
  `41691496d9628cb2825bbe1fa87470b2159e931fe869ce74d9f786eb733edd98`.
- Delegated runtime: `sglang` at `v0.5.10+nova-kv.d81c77b`. This is the repository's
  research fork and is not represented as stock SGLang support.

The Go contract requires all these IDs and both paper digests. A changed paper version,
artifact, recipe commit, or runtime is **unsupported**, not a compatible alias. If the
exact runtime is recognized but unavailable, the result is **delegate** and names that
runtime. Only the exact tuple with an available runtime is **supported**. The leaf does
not load checkpoints, start SGLang, or create a fak-only artifact format.

## What the method actually evaluates

NOVA-KV derives a non-orthogonal key transform from query and key calibration
statistics so attention-weighted key error becomes MSE in the transform domain. It
then partitions transformed coefficients into equal-volume groups and uses fixed-size
vector-quantizer codebooks. Values use an OSCAR-derived rotation and per-token affine
INT2 scalar quantization. The public recipe describes a shared BF16 prefix, a BF16 recent
window, and low-bit pages elsewhere. These details are part of the named evaluation;
they are not interchangeable with arbitrary codebooks, transforms, or all-scalar KV
quantizers.

## Result ledger

Evidence labels describe provenance, not confidence. **Observed (paper/repository)**
means a value is read from the pinned authors' paper or repository. It does not mean fak
reran it. **Modeled** means the source derives or calculates the value rather than
measuring a hardware run.

| Evidence | Result | Exact envelope and provenance |
|---|---:|---|
| Observed (paper/repository) | NOVA-KV 2-bit NIAH aggregate **0.947**; OSCAR 2-bit scalar baseline **0.524** | Qwen3-8B, 4K/8K/16K/32K NIAH; paper Table 1. This is the named tuned scalar comparison for the Qwen quality claim. |
| Observed (paper/repository) | NOVA-KV 2-bit normalized aggregate **0.474**; TurboQuant INT2 tuned scalar **0.481**; OSCAR INT2 scalar **0.231** | GPT-OSS-20B across AIME25, GPQA, MATH500 and NIAH; paper Table 2. NOVA-KV does **not** beat the strongest tuned scalar aggregate here, so no universal quality gain is claimed. |
| Observed (paper/repository) | **1.6x-3.1x** decode throughput versus BF16 | Qwen3-8B, SGLang, 8x NVIDIA H100 80GB SXM, 16K/32K/64K contexts and batch 1-64; paper Section 4.2/Figure 5. Not portable to another GPU/runtime without a new observation. |
| Modeled | metadata contribution **<0.01 bits/KV element**, versus approximately **2.2 bits/KV element** cache rate | Paper Appendix C: 32-layer Qwen3-8B, 128K sequence, shared transforms/codebooks, amortized at served batch sizes. This is an analytical accounting result, not a memory-profiler observation. |
| Observed (paper/repository) | shipped GPT-OSS-20B `codebook.pt`: **9,936,810 bytes** | Pinned repository `artifacts/MANIFEST.json`; artifact SHA-256 `b65796b3a2628d57b038c5ad70fbd27dc1c7628b6b44fbdf90f4836b6144fdb9`. This records concrete codebook overhead without extrapolating it to other models. |

### Net-true grade

- **Qwen3-8B NIAH quality:** net-true only inside the paper's four NIAH cells against
  the paper's OSCAR 2-bit scalar baseline; both the accuracy result and codebook/transform
  costs above remain in scope.
- **GPT-OSS-20B aggregate quality:** no gain over the tuned TurboQuant INT2 scalar
  baseline (0.474 versus 0.481), so reporting a general quality win would be a strawman.
- **Throughput:** an authors' observed H100/SGLang result, net of their implemented
  transform/codebook path, but not independently reproduced by fak and not transferable
  outside the named hardware/runtime envelope.
- **Codebook cost:** explicitly retained. The fixed 9,936,810-byte GPT-OSS artifact and
  the modeled Qwen rate answer different questions and must not be substituted.

## Reproduction recipe and runtime boundary

The pinned public repository is the recipe authority. Its README installs the local
`python/` SGLang v0.5.10 fork, calibrates per-model transforms/codebooks, runs the method
matrix, and supplies throughput configurations. Model weights and Qwen/Llama bundles are
built or fetched separately and are not redistributed here. The recipe's committed
manifest is the artifact integrity authority; callers must verify its hashes before use.

A fresh hardware reproduction must use the sanctioned private-comms path described in
`docs/private-comms-channel.md`, select a listed lab GPU node from
`docs/fleet-compute-nodes.md`, record the node's GPU/runtime versions, and preserve raw
recipe output under the run archive. It must compare the tuned scalar baseline in the
same model/context/batch cells and account for calibration, transform, codebook, and
runtime costs. Until that run exists, **No fak lab run is claimed** and no local observed
quality or performance numbers are added.

## Typed outcomes

`internal/kvvectoreval.Evaluate` returns exactly one disposition:

- `supported / KV_VECTOR_EVAL_EXACT_MATCH`: exact contract, paper artifacts, recipe and
  delegated runtime are pinned, and that exact runtime is available.
- `delegate / KV_VECTOR_EVAL_EXTERNAL_RUNTIME`: exact inputs are recognized but execution
  belongs to `sglang` at `v0.5.10+nova-kv.d81c77b`, which is unavailable to the caller.
- `unsupported`: malformed input, unknown contract/paper/recipe/runtime, or an artifact
  digest mismatch receives its own stable reason code. There is no silent fallback.

## Independent witness

The repository test reads this document independently, requires the named IDs and evidence
labels, and exercises at least three typed cases (supported, delegate, and multiple
unsupported reasons). The Go ledger separately preserves the tuned scalar comparisons,
codebook cost, runtime envelope, and modeled/observed distinction. The fetched paper,
source, and public repository used during evaluation are retained only in the ignored
operator scratch directory `_scratch/issue-6259`; the durable witness is their pinned
public identity and digest above.

