# TICKET-07: FP8 tiled-to-Q8 loader drops the Q8 shape after waived landing

## Current state

At fak `0016a2a6a9f`, `TestFP8TiledQuantLoadMatchesExpanded/canonicalized` (`internal/model/fp8_quantize_test.go:79`) fails: "missing or wrong Q8 shape for model.layers.0.self_attn.q_proj.weight". The suspect commits landed under the operator deferred-verification waiver after `24033b4080f`:

- 1db0c8bbc04 tiled FP8 to Q8 loader conversion
- 8b859be73c4 tiled FP8 conversion for Flash dense weights
- related: 2805e379478 pinned FP8 lookup in the shared decoder; bcbb58aa833 tensor-scaled FP8 output head loading

Parent: #13764 (the V4.1 numerics group, which found this red outside its scope).

## Working spine

1. Bisect the red between `24033b4080f` and `0016a2a6a9f`.
2. Fix the canonicalized-name path so the Q8 shape is recorded for tiled FP8 tensors.
3. Add fail-before/pass-after coverage for each suspect commit.

## Witness

`go test -count=1 -timeout 60m -run 'FP8' ./internal/model/` exit 0.

## Done condition

- [ ] The FP8 tiled-load tests are green on trunk, and the fix commit cites the covered shas.
