---
title: "Security review — multimodal image-handling & DoS surface (2026-06-26)"
description: "Evidence-backed security review of the governed vision/multimodal seam (internal/model/multimodal.go, #399): each image-handling and DoS threat mapped to its kernel-level mitigation and the test that witnesses it, plus the honest residual/out-of-scope fences and the operator gate for graduated rollout."
---

# Security review — multimodal image-handling & DoS surface

> **Scope of this review.** The governed multimodal forward seam added for
> [issue #399](https://github.com/anthony-chaudhary/fak/issues/399) —
> [`internal/model/multimodal.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/model/multimodal.go) and its tests
> [`internal/model/multimodal_test.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/model/multimodal_test.go).
> This is the **"Security review of image handling pipeline"** that #399 lists itself
> *Blocked By*, and the Phase 4 "Security review completed (image handling, DoS vectors)"
> DoD item. It reviews the **admission/governance boundary**, not the semantic safety of
> a vector (see the out-of-scope fence). Generated 2026-06-26 against `ForwardMultimodal`
> as shipped in `e8335d9`.

This review follows fak's house rule for security claims: **a mitigation is only credited
if a test witnesses it.** Every row below names the code that enforces the control and the
test that proves it fires. All 21 multimodal test cases pass (`go test ./internal/model
-run Multimodal`).

## The architectural decision that shrinks the attack surface

The single most important security property is what is **absent** from the decoder path:

> **Raw image bytes never enter an in-kernel image codec.** The model package consumes
> *precomputed hidden-size vectors* supplied by an external `VisionEncoder` (the CLIP/LLaVA
> image tower stays outside this core). Image `Bytes` are carried only as bounded metadata
> for accounting and the quarantine digest — they are never decoded, decompressed, or
> parsed here.

This is a deliberate trust-boundary choice. The classic image-handling vulnerability class
(a crafted PNG/JPEG triggering a memory-safety bug or a decompression bomb inside `libpng`/
`libjpeg`-style code) **cannot occur in this path** because there is no image-parsing code
in it. The forward path consumes `part.Image.Vectors` only; the bytes are inert. A future
in-process decoder would be a new, separately-reviewable attack surface and is explicitly not
part of this seam.

## Governance posture: fail-closed, opt-in, no auto-detection

| Property | Mechanism | Witness |
|---|---|---|
| **No silent multimodal activation.** The zero value `MultimodalModeDisabled` (`""`) holds all image input. Image-bearing prompts require an explicit `Mode: MultimodalModeQuarantine` opt-in; there is no auto-detection that turns vision on. | `prepareMultimodalRows` → `if verdict.Images > 0 && policy.Mode != MultimodalModeQuarantine { … Quarantine }` | `TestForwardMultimodalDefaultQuarantinesImages` (default holds), `TestForwardMultimodalQuarantineModeAllowsBoundedEmbeddings` (opt-in releases) |
| **Text-only is unchanged.** A prompt with no image parts is bit-identical to `Forward()` under the zero policy, so enabling the seam cannot regress text serving. | `ForwardMultimodal` text path reuses `embedRows`/`forwardHiddenRows` | `TestForwardMultimodalTextOnlyMatchesForward` (bit-equality assert) |
| **A held image is tamper-evident.** A quarantined image's `QuarantineID` is a sha256 over media-type + dimensions + byte length + every embedding bit, so altering the embedding yields a different pointer — no silent swap of held content. | `multimodalQuarantineID` | `TestForwardMultimodalQuarantineIDBindsEmbeddingBits` (a 0.25 delta in one vector changes the id) |

## DoS vectors and their bounds

Every limit has a safe default via `withDefaults()`, so an unconfigured caller is still
bounded; `valid()` rejects non-positive overrides so a limit cannot be disabled by setting it
to zero/negative.

| DoS vector | Bound | Mechanism | Witness |
|---|---|---|---|
| **Too many images** (per-request fan-out) | `MaxImages` (default 4) | `admitVisionEmbedding`: `verdict.Images++; if > policy.MaxImages → deny` | `TestForwardMultimodalRejectsTooManyImages` |
| **Oversized image bytes** (memory) | `MaxImageBytes` (default 10 MiB), per-image **and** cumulative | `nbytes > MaxImageBytes \|\| verdict.ImageBytes > MaxImageBytes → deny` | `TestForwardMultimodalGovernanceLimits/bytes` |
| **Pixel bomb / dimension overflow** | `MaxImagePixels` (default 4096×4096) | **Overflow-safe** check `w > MaxImagePixels/h \|\| w*h > MaxImagePixels` — the division guard prevents `w*h` overflowing `int64` before the comparison | `TestForwardMultimodalGovernanceLimits/pixels` |
| **Embedding-token flooding** (context/compute) | `MaxEmbeddingTokens` (default 1024), cumulative across images | `verdict.EmbeddingTokens += len(Vectors); if > limit → deny` | `TestForwardMultimodalGovernanceLimits/embedding tokens` |
| **Non-image payload smuggled as an image part** (type confusion) | media type must be `image/*` (trimmed, lower-cased) | `if !strings.HasPrefix(media, "image/") → deny` | `TestForwardMultimodalGovernanceLimits/media type` |

## Malformed-input hardening

| Threat | Mitigation | Witness |
|---|---|---|
| **Embedding width mismatch** → out-of-bounds / silent corruption in the forward matmul | every vector length must equal `HiddenSize`, else deny | `TestForwardMultimodalRejectsWrongEmbeddingWidth` |
| **Out-of-vocab token id** → OOB read into the embedding table | `id < 0 \|\| id >= VocabSize → deny` before the table slice | `TestForwardMultimodalRejectsTokenOutOfVocab` |
| **Empty / degenerate prompt** (empty request, empty embedding, a part with both or neither text and image) | explicit deny on each | `TestForwardMultimodalRejectsEmptyPrompt`, `…RejectsEmptyEmbedding`, `…RejectsMixedOrEmptyPart` |
| **Governance-bypass via zero/negative limits** | `valid()` rejects non-positive bounds and unknown modes; `withDefaults()` substitutes safe defaults for zero | `TestForwardMultimodalRejectsInvalidPolicy` (5 cases) |
| **Source-buffer mutation / caller aliasing** | each admitted row is copied (`append([]float32(nil), vec…)`); the caller's embedding is never mutated and never aliased into model state | `TestForwardMultimodalQuarantineModeAllowsBoundedEmbeddings` asserts the source vectors are bit-unchanged after the call |

## Residual risk and out-of-scope fences (honest)

These are **not** mitigated by this seam, by design — they are named so the rollout decision
is made with eyes open:

- **Semantic safety of an embedding is out of scope.** A well-formed, in-bounds vector from a
  poisoned or adversarial upstream encoder is *admitted* — the core governs shape, count, and
  size, not meaning. This is the same trust model fak applies to text tokens: the kernel bounds
  the blast radius, it does not judge content. Trust in the `VisionEncoder` is the caller's to
  establish.
- **No in-core image decoding** means no review of an image codec here — but if a future change
  adds in-process decoding of `Bytes`, that is a new attack surface requiring its own review.
- **The sha256 quarantine digest is an integrity pointer, not a secret-bearing path**; no
  constant-time/side-channel requirement applies.
- **Graduated rollout beyond quarantine is an operator decision, deliberately not automated.**
  There is no code path that promotes `quarantine` to an auto-allow mode; #399 Phase 4's
  "graduated rollout after a validation period" requires explicit human sign-off.
- **End-to-end latency on a real VLM checkpoint is hardware-gated** (Phase 3 benchmark). It is
  independent of this review: the governance/DoS surface above is witnessed deterministically on
  a synthetic CPU model (`NewSynthetic`), needs no GPU or weights, and is what the *Blocked By*
  line asks for.

## Disclosure

A real vulnerability against this boundary should be reported privately per
[`SECURITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/SECURITY.md) (GitHub private advisory), not as a public issue.

## Verdict

The image-handling and DoS surface of the governed multimodal seam is **fail-closed, opt-in,
and bounded on every input dimension reachable in this path, with each control witnessed by a
passing test.** The named residual risks are upstream-encoder trust and the deliberately-manual
rollout gate — neither is a defect in this seam. This satisfies #399's *Blocked By: Security
review of image handling pipeline* and the Phase 4 security-review DoD item; the Phase 3
end-to-end VLM benchmark remains the separate, hardware-gated item.
