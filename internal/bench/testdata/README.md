# internal/bench/testdata — golden fixtures

## q4k_interleave_golden.json

Pinned input-to-packed byte pairs for `RepackQ4KInterleaved` /
`UnrepackQ4KInterleaved` (`internal/bench/q4krepack.go`) at column-interleave widths
**1/4/8**, including the short tail group (`out % width != 0`). Consumed by
`internal/bench/golden_fixtures_test.go` (`TestQ4KInterleaveGoldenFixtures`).

### Provenance

- Pattern source: **ggml-org/llama.cpp @ `571d0d540df04f25298d0e159e520d9fc62ed121`**,
  `ggml/src/ggml-cpu/repack.cpp`, fetched 2026-11-03,
  sha256 `b3a1d6953fc61f2a4088337384166fb5102db617afe37a4b92f7cf07992a9f8a`:
  `repack_q4_K_to_q4_K_8_bl` (:3231), `make_block_q4_Kx8` (:2836), picker
  `ggml_repack_get_optimal_repack_type` (~:4520).
- What the fixture pins: **fak's documented column-interleaved byte-permutation contract**
  (the LAYOUT block in `q4krepack.go`): rows grouped `width` at a time; per super-block
  `[d x w][dmin x w][scales x w][qs byte j x w]`; a tail group truncated to the remaining
  rows; pure permutation (same length, nothing derived).
- Divergence from upstream (provenance-honest): this is **NOT** byte-identical to
  llama.cpp's native `block_q4_Kx8`. Upstream 6-bit-repacks the 12-byte scale/min fields
  across lanes, chunk-interleaves qs in 8-byte units, and refuses tensors whose row count
  is not a multiple of 8. fak keeps the GGUF scale fields intact per lane and
  byte-interleaves qs so the resident dequant arithmetic is unchanged — the "no gather"
  property for nibble bytes is preserved either way. See the `q4krepack.go` header and its
  KNOWN NARROWING note.

### Derivation record

- Fixture bytes were derived from the pinned source logic by the independent reference
  `q4kGoldenReference` in `golden_fixtures_test.go` (group-first slice copies, written
  separately from the production row-first scatter) — **not** from
  `RepackQ4KInterleaved`, so production drift cannot be laundered into the fixture.
- Inputs: deterministic 64-bit LCG (Knuth constants `mult=6364136223846793005`,
  `inc=1442695040888963407`), seed mixed per case from `(0x5285, out, in, width)` as in
  `q4kGoldenInput`; per record, `d`/`dmin` are drawn as finite f16 (exponent field in
  [12,16]) and the 12 scale + 128 nibble bytes come straight from the stream.
- Regenerate (only after a reviewed change to the reference or the shape set):

  ```
  FAK_UPDATE_GOLDEN=1 go test ./internal/bench -run TestQ4KInterleaveGoldenFixtures -count=1
  ```

- Verify:

  ```
  go test ./internal/bench -run TestQ4KInterleaveGoldenFixtures -count=1
  ```

The test asserts three-way agreement: committed fixture vs independent reference
(fixture integrity), committed fixture vs `RepackQ4KInterleaved` (the regression witness),
and `UnrepackQ4KInterleaved(reference(input)) == input` (both directions agree on the same
mapping). A plain round-trip or byte-histogram check would miss an offset bug that both
directions share; the pinned absolute bytes do not.
