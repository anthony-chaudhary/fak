# TICKET-01: V4.1 attention, rotary and indexer numerics are red after waived landing

## Current state

At origin/main `24033b4080f` (GOWORK=off, Windows):

- `go test -count=1 ./internal/model/` fails `TestV41Parity`. Oracle-vs-golden and forward-vs-golden logits[0] are -0.6709014, want -0.6718912 (|delta| 9.9e-4 > tol 1e-4); the prefill delta is 2.6e-3.
- `TestV41SharedAttentionSession` fails in the plain-window-boundary and source-reader-group-boundary cases ("published compressed row lost its BF16 boundary").
- `TestV41CompressorNormSession/0/prefix-1` panics: "V4.1 forward stage embedding ... token id 8 out of range [0,8)" at `prefillV41Suffix`, `v41_forward.go:1241`.
- `go test ./internal/computebuild/` fails `TestVulkanShadersCompleteness`: expected 61 shaders, got 62.

## Covered commits

56346cb850 per-layer rotary tables; cbf95b0303a inverse attention rotation; 72ade8c9ca7 compressed rotary cache publication; ec5b6596d9f F32 indexer rounding; 10237ca5dd6 gated indexer score primitive and shader; 4232253f555, 86e97780f48 index-key normalization; 666dceb652c, f088c0fe85f compressor normalization; fb6724d7efb shared attention device callbacks; c0a18c1c522 shared latent sink attention; 4cd54773544, 31de0e44223 tail RoPE; 0b7df47254a opt-in attention witnesses; docs receipts 24033b4080f, f82fe7ea040.

## Working spine

1. Bisect each red to its commit.
2. For parity, decide which side is correct (the fixture, unchanged since 2026-09-18, or the new rotary tables) from the published reference. Never loosen the tolerance or regenerate the golden without a reference witness.
3. Fix with fail-before/pass-after tests.
4. Reconcile the shader inventory count with the indexer-score shader.

## Gold-plating boundary

No hardware qualification here (that is TICKET-02); no new kernels.

## Witness

`go test -count=1 ./internal/model/ ./internal/computebuild/ ./internal/compute/` exit 0.

## Done condition

- [ ] All four reds green on trunk, with tests that fail on the parent.
- [ ] Each covered commit cited in the fix commits.
