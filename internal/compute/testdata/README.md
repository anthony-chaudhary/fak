# internal/compute/testdata — golden fixtures

## nvfp4_swizzle_golden.json

Pinned input-to-swizzled byte pairs for `SwizzleNVFP4Scales` /
`UnswizzleNVFP4Scales` (`internal/compute/nvfp4_scale_layout.go`), the NVIDIA TensorRT-LLM
**128x4 block-scale tile layout**, zero padding included. Consumed by
`internal/compute/nvfp4_golden_fixture_test.go` (`TestNVFP4SwizzleGoldenFixtures`).

### Provenance

Pinned at **NVIDIA/TensorRT-LLM @ `ecae3c97e2b646580415518a45a72bc5a5a67075`**, fetched
2026-11-03:

- `cpp/tensorrt_llm/kernels/quantization.h`,
  sha256 `4a64db27f63ba18c790087bcae314ad9b3eec3f9ca1907cfe314faaf0c3594d7` —
  `QuantizationSFLayout::SWIZZLED` contract: scale blocks organized in 512-byte blocks of
  128x4 FP8 values; rows padded to the nearest multiple of 128, columns to 4.
- `cpp/tensorrt_llm/kernels/quantization.cu`,
  sha256 `141e4a222b0511bd568aebd0b72d3ff3d3aa6340f676eb215502aa512621318f` —
  `block_scale_interleave_kernel`: out-of-range row/col writes **zero**, and the exact
  offset math:

  ```
  tilesK     = PadUp(cols, 4) / 4
  tileOffset = ((r / 128) * tilesK + c / 4) * 512
  dst        = tileOffset + (r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4
  ```

This fixture pins the absolute destination of every scale byte against that formula, so a
tile/stride drift in `nvfp4ScaleLayoutOffset` fails even when swizzle/unswizzle still
invert each other (the round-trip witness in `nvfp4_scale_layout_test.go` cannot see an
offset bug the pair shares).

### Derivation record

- Fixture bytes were derived from the pinned upstream formula by `nvfp4GoldenSwizzle` in
  `nvfp4_golden_fixture_test.go`, which applies the formula through
  `nvfp4ScaleGoldenOffset` (the deliberately duplicated upstream formula already pinned in
  `nvfp4_scale_layout_test.go`) — **not** from `SwizzleNVFP4Scales`, so production drift
  cannot be laundered into the fixture.
- Inputs: `linear[i] = (37*i + 11) mod 256` (gcd(37,256)=1, so the stream covers all 256
  byte values and includes zeros inside the data region — only the PADDING region must be
  all zero).
- Cases: `single-scale-padded-to-tile` (1x1 -> 512 B, maximum padding fraction),
  `exact-single-tile` (128x4, no padding anywhere), `both-axes-padding` (130x6 -> 2048 B,
  row pad into the second tile row plus column pad inside every tile).
- Regenerate (only after a reviewed change to the reference or the shape set):

  ```
  FAK_UPDATE_GOLDEN=1 go test ./internal/compute -run TestNVFP4SwizzleGoldenFixtures -count=1
  ```

- Verify:

  ```
  go test ./internal/compute -run TestNVFP4SwizzleGoldenFixtures -count=1
  ```

The test asserts: committed fixture vs pinned-formula reference (fixture integrity),
committed fixture vs `SwizzleNVFP4Scales` (the regression witness), every padded slot
reads exactly 0 in the production output, and
`UnswizzleNVFP4Scales(SwizzleNVFP4Scales(input)) == input` (round trip strips the padding).
