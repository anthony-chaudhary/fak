# SPECTRA transform-coded KV compression: deep study and FAK disposition

> Studied 2026-08-28. Paper: [arXiv:2608.07915v1](https://arxiv.org/abs/2608.07915v1), published 2026-08-08. Code: [nokia-applied-research/SPECTRA](https://github.com/nokia-applied-research/SPECTRA), pinned at [`272032275106dc5944fbfa7091a1ceb403fa7e28`](https://github.com/nokia-applied-research/SPECTRA/tree/272032275106dc5944fbfa7091a1ceb403fa7e28). Code license: MIT. Durable receipt: `study_13eb7b60b128c49c8575e840bcf36ef0ae625c4a314fb5d36551794c9046c8c4`.

## Verdict

SPECTRA's transferable contribution is a credible **post-training transform codec** for ordinary GQA K/V: measure the activation geometry, find an energy-ordered low-rank latent, and spend bits by reverse water-filling instead of applying the same raw-K/V quantizer everywhere. Its paper provides strong memory/quality evidence around 4x and competitive results around 8x on three 7B-8B models.

It does **not** yet prove a production serving speedup. The published Hugging Face and vLLM paths reconstruct full K/V before attention, the fused latent kernel is a disconnected microbenchmark, and the capacity validation uses analytical accounting plus filler-tensor OOM rather than a complete serving run. The paper explicitly leaves a fused mixed-bit, RoPE-aware kernel to future work and reports no end-to-end throughput numbers.

FAK should therefore borrow the math into an **optional Qwen3.8-native codec oracle**, compare it against dense KV and the current Q4 cold codec, and retain the current default until a live quality-constrained receipt wins on actual resident and transient bytes. [#9789](https://github.com/anthony-chaudhary/fak/issues/9789) is that bounded spine. A production fused kernel remains WATCH, not implied work.

## Value frame

- **For:** Qwen3.8 agent-serving workloads bounded by native KV memory and bandwidth.
- **Problem:** raw low-bit K/V quantization reaches a quality cliff; model-native MLA is unavailable for ordinary GQA checkpoints.
- **Today:** FAK has raw-K/V Q4 cold compression with exact residual recovery, true latent-resident MLA for architectures trained with it, and exact pre-RoPE eviction/reuse.
- **Better because:** SPECTRA supplies a calibration-derived transform and rate allocator that can be tested without changing the current default or surrendering kernel ownership.
- **Witness:** #9789 requires a sanctioned-GPU, fak-native Qwen3.8 comparison with predeclared quality, actual resident and transient bytes, setup cost, latency, throughput, exact eviction, and zero fallback.

Centrality: **Core**. P1 advances owned context capacity; P2 counts the complete path; P3 keeps dense, Q4, and transform arms separately selectable; P4 makes engine, fallback, bytes, quality, and exactness visible.

## What the method actually does

For a layer input `h`, SPECTRA estimates the activation second moment `G = E[h^T h]`. It chooses K/V low-rank factors by minimizing produced-output error in that `G`-weighted geometry rather than unweighted weight reconstruction. Splitting singular values by their square roots makes latent channel second moments energy ordered.

It then applies reverse water-filling: high-energy coordinates receive more bits, and a zero-bit coordinate becomes a rank reduction. Adjacent coordinates are grouped and assigned a shared integer rate. At aggressive compression, K and V may receive different ranks. The value path can optionally refit the output projection (`W_O` healing) from streaming calibration statistics after compression.

There are two conceptual execution modes:

1. **Mode A:** store the quantized latent, reconstruct full K/V, then use conventional attention.
2. **Mode B:** fold reconstruction into attention and never materialize full K/V.

The paper's quality experiments exercise the transform/quantization idea. The repository's actual serving integrations are Mode A. Its Mode-B Triton routine is a standalone benchmark, not the public model path.

Pinned method anchors: [`spectra/core.py`](https://github.com/nokia-applied-research/SPECTRA/blob/272032275106dc5944fbfa7091a1ceb403fa7e28/spectra/core.py), [`spectra/method.py`](https://github.com/nokia-applied-research/SPECTRA/blob/272032275106dc5944fbfa7091a1ceb403fa7e28/spectra/method.py), and [`spectra/quant.py`](https://github.com/nokia-applied-research/SPECTRA/blob/272032275106dc5944fbfa7091a1ceb403fa7e28/spectra/quant.py).

## Paper evidence and claim boundaries

The evaluation uses Llama-3.1-8B, Mistral-7B, and Qwen2.5-7B GQA checkpoints with up to 31,500 tokens. The main calibration uses 32 WikiText-2 windows of 2,048 tokens. LongBench covers eight English tasks, first 150 examples each; NIAH is a deterministic context-length by needle-depth grid.

| Model | Dense | Representative SPECTRA frontier | Boundary |
|---|---:|---|---|
| Llama-3.1-8B | 53.24 | 53.56 @ 3.56x; 54.00 @ 7.66x; 48.81 @ 11.12x; 45.18 @ 12.19x | The headline 12x point loses 8.06 LongBench points; no usability threshold is declared. |
| Mistral-7B | 53.20 | 53.50 @ 3.56x; 52.94 @ 6.31x; 51.17 @ 7.73x; 48.85 @ 11.28x | Quality degrades more steadily beyond about 6x. |
| Qwen2.5-7B | 52.37 | 52.53 @ 3.48x; 52.27 @ 4.92x; 50.72 @ 6.10x | No competing baselines are reported for Qwen; the paper says its energy is less concentrated. |

At roughly 8x on Llama, SPECTRA reports 54.00 @ 7.66x and 52.85 @ 8.13x, versus TurboQuant 47.11 and RotateKV 45.97. Those are valuable quality results, but the selection is best-per-tier, has no uncertainty intervals, and includes non-monotonic points.

The capacity table projects 24 GB dense/SPECTRA sequence limits of 65k / 232k / 427k / 617k / 793k at 1x / 3.56x / 6.56x / 9.48x / 12.19x, and 80 GB limits of 524k / 1.86M / 3.44M / 4.97M / 6.39M. The prose separately reports 68.3k measured versus 68.8k analytical for the 24 GB dense check, which does not match the table's 65k row. Treat the table as capacity modeling, not a serving-throughput receipt.

The introduction mentions RULER, but the experimental results contain LongBench and NIAH, not RULER. Baselines are reported only for Llama and Mistral; Palu is discussed but omitted. The reverse-water-filling optimum is conditional on independent Gaussian/high-rate approximations, which are least trustworthy at the lowest integer rates.

## Code-path maturity

| Surface | What is real at the pin | Claim ceiling |
|---|---|---|
| `spectra.compress()` with ordinary HF generation | The model weights are transformed, then conventional generation reconstructs/caches full K/V. | Algorithm/quality prototype; not compressed resident KV. |
| Packed HF adapter | Stores an int4 latent, concatenates full history, reconstructs full K/V each step, and disables the ordinary HF cache contract. | Single-path memory prototype; batching, masks, beams, prefix reuse, and multi-request safety are unproven. |
| vLLM adapter | Allocates paged `uint8` latent storage and gathers/dequantizes it eagerly in Python before full-K/V attention. | Real packed residency, but not a production latency path; hard-coded Llama shape and global private-class monkeypatches narrow the envelope. |
| Triton `latent_decode` | Computes a latent-attention decomposition in isolation and prints relative error. | Microbenchmark only; no RoPE, paging, variable lengths, or public serving integration. |

The repository actually contains three different codec contracts: fake quantization in the calibration/evaluation path, symmetric one-scale-vector int4 in the HF adapter, and affine per-group int4 in the vLLM adapter. Therefore the paper's quality frontier cannot be assumed to describe either serving format.

## Deep repository study

Observed 2026-08-28. The repository was created 2026-08-05 and last pushed 2026-08-12. It has two unsigned commits; the second changes only the README to add the paper link. The exact tree contains 30 files. GitHub reported 1 star, 1 fork, 0 issues, 0 pull requests, 0 discussions, 0 releases, 0 tags, and 0 Actions workflows. Discussions are disabled. Projects are enabled, but the available token could not enumerate Projects v2, so no absence claim is made for boards.

The artifact has no test suite, CI, lockfile, container recipe, committed result JSON, model/dataset revision pins, release contract, changelog, security policy, or provenance manifest. README/demo commands refer to missing `scripts/run_eval.py` and `scripts/run_ppl_lmeval.py`; the vLLM source refers to a nonexistent README Efficiency section.

The committed figures have no source-data companion or generation manifest. At least one plot script contains data inconsistent with the committed image, and NIAH plot scripts read hard-coded absent `/tmp` inputs. The memory benchmark emulates compressed capacity with shorter fp16 allocations rather than serving the compressed implementation.

Saved deltas retain a mutable base-model name rather than a revision/digest, load with `torch.load(..., weights_only=False)`, and can silently truncate mismatched module lists through `zip`. FAK must use a versioned safe artifact with model, calibration, shape, and codec identity.

License: root MIT, copyright 2026 Nokia. Direct reuse is compatible with notice preservation, subject to ordinary legal review; MIT has no explicit patent grant. Because FAK is Go/CUDA-native and SPECTRA's serving code is Python/Triton/vLLM-specific, adaptation from the documented math is the lower-maintenance route. No upstream code was copied by this study.

## FAK self-query and seam map

Candidate-specific `fak capabilities` queries for spectral latent K/V, water-filling, W_O healing, fused latent attention, and effective-byte accounting returned only generic context/cache cards. Raw source inspection established the finer result:

- `internal/model/kvquant.go`: groupwise affine Q4 for raw K/V, effective six bits including f32 scale/min, explicitly not live-decode-wired.
- `internal/model/coldkv.go`: Q4 cold rows plus sparse exact residual recovery; receipts include original/compressed/read bytes, max error, encode latency, and engine.
- `internal/model/kvlayout.go`: true latent-resident MLA; the absorbed path folds the query and avoids full-K/V materialization.
- `internal/compute/kvprecision.go`: exact pre-RoPE `Kraw` preservation for eviction/reuse semantics.
- `internal/engine/kv_quantization.go`: pressure ladder and codec-control seam, not a calibrated transform implementation.

`fak sota internal/model/kvquant.go` previously resolved to generic `kv-cache-paging`, while `coldkv.go` and the engine controller had no specific prior-art row. The new `kv-cache-transform-compression` row closes that discovery blind spot and binds future work to the SPECTRA pin plus the full Qwen3.8 oracle.

## Candidate matrix

| Axis | FAK verdict | Portfolio | Decision and disconfirming witness |
|---|---|---|---|
| Activation-weighted latent + reverse-water-filling rate/rank allocation | **ABSENT** | **OPTIONAL-MODULE** | Adapt in #9789. Reject promotion if it does not beat current Q4 cold KV on quality-constrained complete bytes/token. |
| `W_O` healing | **ABSENT** | **RECIPE** | Measure as a separate arm inside #9789; exclude if its calibration/setup cost or quality delta is unfavorable. |
| Latent-resident attention without materialized K/V | **PRESENT** | **DEFAULT** | Keep FAK's absorbed MLA implementation; SPECTRA's disconnected microbenchmark does not supersede it. |
| Exact eviction/reuse and residual recovery | **PRESENT** | **DEFAULT** | Preserve as a harder acceptance invariant; any lossy transform that breaks exact pre-RoPE behavior fails. |
| Packed paged mixed-rate GQA latent with fused attention | **PARTIAL** | **WATCH** | FAK owns component seams but lacks the calibrated format. File only after #9789 proves the codec frontier; require live resident/transient bytes and end-to-end gain. |
| Mutable-base pickle delta and global vLLM monkeypatch | **ABSENT** | **EXCLUDE** | Do not borrow. A future artifact must be versioned, hash-bound, safely decoded, shape checked, and native-engine owned. |
| Quality/paging split | **PARTIAL** | **RECIPE** | Retain the deployment lesson: a quality-optimal mixed format and a page-friendly fixed-width format are separate frontiers and must not share claims. |
| Deterministic NIAH matrix and produced-energy diagnostic | **PARTIAL** | **RECIPE** | Borrow as machine-readable witnesses with rotated seeds/needles, retained inputs, and pinned model/dataset identities. |

Build/test/security/docs/maintenance cost is concentrated in the calibration artifact, numerical oracle, sanctioned-GPU evaluation, and eventual format/kernel lifecycle. #9789 owns only the reference codec and matched frontier. #9141 remains the parent performance campaign; #9161 is the existing Q4+recovery baseline; #2236 remains the broader KV-memory epic.

## What not to borrow

- Do not treat a filler allocation or analytical capacity model as a live serving receipt.
- Do not claim throughput from a disconnected Triton routine.
- Do not import the reconstruct-full-K/V path as FAK's performance implementation.
- Do not silently switch native work to llama.cpp.
- Do not conflate the paper's fake-quant quality codec with the distinct HF/vLLM int4 formats.
- Do not use mutable model identifiers, unrestricted pickle, or global private-runtime monkeypatches.
- Do not call a broad, unlocked environment bit-for-bit reproducible.
- Do not publish a figure without its source data, command, revision, and artifact hash.

## Filed and deduplicated trail

- [#9785](https://github.com/anthony-chaudhary/fak/issues/9785): study tracker; owns this note, receipt, monitored-source entry, and SOTA-row correction.
- [#9789](https://github.com/anthony-chaudhary/fak/issues/9789): one dispatchable optional-codec/Qwen3.8 oracle leaf, deduplicated against 1,000 open issues and contract-checked before filing.
- [#9141](https://github.com/anthony-chaudhary/fak/issues/9141): existing parent for the weight-to-KV bandwidth crossover and complete live performance accounting.
- [#9161](https://github.com/anthony-chaudhary/fak/issues/9161): closed current Q4 cold-KV plus exact residual-recovery baseline.
- [#2236](https://github.com/anthony-chaudhary/fak/issues/2236): existing broad KV-memory epic.

## Completion and refresh boundary

This study covers the paper source, all load-bearing repository code, README/demos/evaluation/benchmark/plot surfaces, all three binary assets, both commits, tree inventory, issues, PRs, releases, tags, workflows, discussions, license/provenance, FAK self-query, candidate disposition, deduplication, and a checked implementation packet. Three delegated result sets were independently read back from the pinned source: **3 confirmed, 0 refuted, 0 unwitnessed, 0 no-claim**.

Unresolved source classes: GitHub Projects v2 could not be enumerated with the available token; dependency repositories and the sole public fork were outside the canonical-source scope. Refresh on a new upstream commit, release, issue/PR, fused serving integration, throughput result, Qwen3.8-family result, or a changed FAK Q4/MLA frontier.

This note does not claim that SPECTRA reproduces on FAK hardware, that #9789 is implemented, or that the transform codec should become default. Those are outcome claims for a new receipt after the sanctioned-GPU witness.
