# LightRot bounded low-bit evaluation (arXiv:2607.27704)

Status: **modeled contract shipped; model-quality and hardware-performance claims not yet observed**.
Issue: #6250, child of #6221.

## Pinned research object

The named input is *LightRot: Towards Outlier-Free and Hardware-Efficient
Quantization for Large Language Models*, arXiv:2607.27704. The PDF fetched from
`https://arxiv.org/pdf/2607.27704` for this evaluation has SHA-256
`e9e6093c0b0025e0fa40b575c416d8e40cb287d97d434373d6878ec6f3762696`.
The paper proposes fixed lightweight block-diagonal Hadamard rotations plus
permutations/signs to reduce activation outliers. It reports WikiText-2
perplexity for LLaMA-2 7B/13B/70B under W4A4, W6A6, and W8A8 and reports
throughput/latency on NVIDIA A100. Those paper values are **reported by the
paper, not reproduced by fak**, and therefore are not copied into a shipped
performance claim.

No public implementation or immutable recipe artifact was identified in the
paper or a repository search at evaluation time. Consequently the Go leaf does
not pretend to reproduce paper training, model inference, or kernel timing. It
pins a transparent bounded recipe inspired by the paper's named construction:
block-size-4 normalized Hadamard blocks, seeded column permutation/signs,
per-tensor symmetric low-bit fake quantization, and inverse rotation. This is a
modeled research comparison, not a claim of implementation identity.

## Contract and typed outcomes

`internal/lightroteval` accepts only `lightroteval/v1` and records four distinct
provenance objects:

- artifact: fixture ID/version/SHA-256/source;
- recipe: LightRot ID, recipe version, paper ID/PDF SHA-256, seed and block size;
- runtime: evaluator ID/version and `cpu-reference-f64` backend;
- model: model ID/revision/SHA-256/license, plus OS/architecture/device envelope.

A valid modeled request returns `supported/LIGHTROT_EVALUATED_MODELED`. Unknown
contract or recipe versions, invalid pins, bad shape, and unsupported bit width
return typed `unsupported` reasons. An unknown runtime or any request to label
a result `observed` returns typed `delegate`; the latter explicitly requires a
sanctioned lab runner to attach measured wall time and device read-back. There
is no silent fallback.

## Bounded evaluation and baselines

Every independently readable JSON fixture evaluates the same matrix, bit width,
and quantizer against all three candidates:

1. `lightrot`: seeded fixed lightweight block rotation (candidate);
2. `tuned_rotation`: deterministic dense pairwise rotation with a larger modeled
   preprocessing budget (tuned-rotation baseline);
3. `no_rotation`: identity path (no-rotation baseline).

The result records reconstruction accuracy (`1 - MSE / mean(x^2)`), MSE,
maximum absolute error, modeled preprocessing scalar operations, modeled
runtime scalar operations, and the full provenance object. Candidate ordering
and outputs are deterministic. The fixtures deliberately claim only bounded
synthetic reconstruction: they do not establish language-model perplexity,
downstream task accuracy, GPU latency, throughput, memory, energy, or a
universal quantization winner. The embedded claim-check verdict remains
`not-yet` until those measurements exist against the real tuned alternative.

## Witness and hardware envelope

Three checked-in fixtures under `internal/lightroteval/testdata/` are read back
and compared byte-semantically by `TestWitnessFixtures`; unsupported/delegate
cases and provenance pins have separate assertions. This is the independently
read artifact required by #6250, and it runs on Windows and WSL without model
weights or a GPU.

The sanctioned `fak-realmodel` GCP target was queried during this work. Instance
metadata independently identified one NVIDIA L4 accelerator, but the guest
returned `nvidia-smi: command not found`; therefore **no observed GPU result was
recorded or inferred**. The private lab bridge preflight also returned an
explicit missing-channel configuration error before dispatch, so no private
hardware claim exists. These outcomes are preserved as local issue evidence,
not promoted into result fixtures. A future observed run must pin the exact
model weights and revision, quantization recipe/code revision, runtime/container
revision, calibration/evaluation datasets and digests, driver/device read-back,
warmup/trial protocol, raw samples, accuracy result, preprocessing cost, and
runtime cost before changing the claim-check verdict.

## Reproduction

From the repository root:

```text
go test ./internal/lightroteval -count=1
./test.ps1 ./internal/lightroteval
fak validate --mine internal/lightroteval --mine docs/research/quantization/lightrot.md
```

The checked-in outputs are modeled (`wall_ns` is absent and `wall_evidence` is
`modeled`). A lab adapter must generate a separately pinned observed artifact;
it must never overwrite or reinterpret these fixtures.

