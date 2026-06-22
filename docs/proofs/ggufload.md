---
title: "fak proof: GGUF loader offset and dequant layout"
description: "Proof that fak's GGUF loader addresses each tensor's own offset window and dequantizes every block format with the correct intra-block layout."
---

# N11 · ggufload

`internal/ggufload` parses GGUF model checkpoints off the inference hot path: it reads the
header magic/version, the metadata key-value table, and the tensor directory; computes each
tensor's absolute file offset from the aligned header end plus its declared in-data offset;
and dequantizes per-tensor payloads (F32/F16/BF16 and the quantized block formats
Q8_0, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K, Q3_K) into float32, then maps GGUF tensor names
to the canonical HF-Llama names `internal/model` consumes. "Correct" here is **regime N
(numerical / layout)**: the bytes addressed for tensor *i* are exactly tensor *i*'s payload
(offset/stride correctness), each quantized block is decoded with the right intra-block
layout (no off-by-block / wrong-stride), and parsing the metadata recovers exactly the
typed values that were encoded. The module is **read-only** — it has no GGUF serializer — so
the round-trip obligation is realizable only as parse-consistency (encode-in-test → `Read` →
equals written), not as a strict `encode∘decode = id` involution.

All witnesses below ran on the macOS node (`go1.26 darwin/arm64`, native `go test`) on
2026-06-20; the full package is `ok` in ~0.22s. Real-weight HF-oracle parity is gated behind
`FAK_GGUF_REAL_SMOKE=1` and is **not** exercised here (it SKIPs), so end-to-end numerical
parity against a reference engine is recorded as OPEN, distinct from the layout theorems
which are fully witnessed by deterministic synthetic fixtures.

---

## Theorem N11.1 — tensor offset/stride layout addresses the correct bytes

**THEOREM.** For every tensor in a parsed GGUF, `FileOffset = align(headerEnd) + Offset`
(with `Offset` required to be an alignment-multiple), and `TensorBytes(name)` returns exactly
the `[FileOffset, FileOffset+payloadBytes)` window for that tensor's declared dtype and dims —
so reading tensor *i* never addresses tensor *j*'s bytes.

**REGIME.** N (numerical / layout).

**PROOF.** `Read` (`fak/internal/ggufload/gguf.go:1095`) sets
`data = alignOffset(headerEnd, align)` (`gguf.go:1368`), rejects any tensor whose
`Offset % align != 0` (`gguf.go:1104`), and assigns
`FileOffset = int64(data + Offset)` (`gguf.go:1110`). `TensorBytes`
(`fak/internal/ggufload/gguf.go:558`) computes `n = tensorPayloadBytes(info)` from the dtype
and dims (`gguf.go:1372`), bounds-checks `FileOffset + n <= size` (`gguf.go:577`), and reads
`r.ReadAt(buf, FileOffset)` (`gguf.go:581`). Each tensor's bytes therefore come from exactly
its own offset window, and the payload length is derived from its own dtype — a transposed or
off-by-one offset, or a stride computed from the wrong dtype, surfaces a neighbour's bytes.

**WITNESS.**
`go test ./internal/ggufload/ -count=1 -timeout 120s -run 'TestReadParsesMetadataTensorDirectoryAndConfig|TestWeightSourceReadsAndDequantizesSimpleTensors' -v`

`TestReadParsesMetadataTensorDirectoryAndConfig` asserts
`Tensors[1].FileOffset == TensorDataOffset + 64` for the Q4_K directory entry declared at
offset 64, plus `Alignment == 64` and `TensorDataOffset % 64 == 0`.
`TestWeightSourceReadsAndDequantizesSimpleTensors` places 11 tensors at distinct declared
offsets (`tinyTensorGGUF`, `gguf_test.go:929`) and reads each by name, asserting bit-exact
`float32` equality — a wrong offset or stride surfaces the wrong tensor's bytes and fails.

**VERDICT.** PROVEN — 2026-06-20, both tests PASS (full package `ok 0.216s`).

**DOS.** bound at ship.

---

## Theorem N11.2 — per-dtype dequant reads the right block layout

**THEOREM.** For each supported dtype (F32, F16, BF16, Q8_0, Q4_K, Q6_K, Q5_0, Q5_1, Q5_K,
Q2_K, Q3_K), `dequantF32` reads the correct intra-block layout (scale/min/qh/ql and
sub-block-scale offsets, per-element strides, nibble order) so the decoded `float32` vector
equals the value each quantized code represents — no off-by-block, wrong-stride, or
swapped-nibble error.

**REGIME.** N (numerical / layout).

**PROOF.** `dequantF32` (`fak/internal/ggufload/gguf.go:1444`) dispatches on `TensorType`;
each branch validates `elems % QK == 0` and `len(raw) == blocks*blockBytes`, then calls the
matching kernel: `dequantQ4K` (`gguf.go:1695`) reads d/min as f16 at the block base, the 12-byte
packed scale array, and `getScaleMinK4` (`gguf.go:1722`) to select each sub-block scale/min,
decoding the low nibble over the first 32 and the high nibble over the next 32; `dequantQ6K`
(`gguf.go:1768`), `dequantQ2K` (`gguf.go:1596`), `dequantQ3K` (`gguf.go:1630`), `dequantQ5K`,
`dequantQ5_0`/`dequantQ5_1` follow their respective GGUF block specs. Bit-exact equality of the
decoded output to a fixture encoded by independent packing math witnesses that every offset and
stride matches the spec.

**WITNESS.**
`go test ./internal/ggufload/ -count=1 -timeout 120s -run 'TestWeightSourceReadsAndDequantizesSimpleTensors' -v`

The test compares `ws.TensorF32` output bit-exactly (`math.Float32bits` equality,
`gguf_test.go:175`) against `want[]` vectors built by independent oracle fixtures —
`q4KFixtureCodes` (`gguf_test.go:988`), `q2KFixtureBlock`, `q3KFixtureBlock`, `q5FixtureBlock`,
`q5KFixtureBlock`, `q6KFixtureBlock`. Each `want[]` is computed by the test's own packing
logic, never by calling the loader, so a layout/stride bug in any `dequant*` kernel fails the
compare.

**VERDICT.** PROVEN — 2026-06-20, PASS for all 11 dtypes.

**DOS.** bound at ship.

> *Scope note.* This witnesses **layout** correctness against synthetic blocks. End-to-end
> numerical parity of the dequantized weights against a reference engine (HF) is a separate,
> stronger claim carried by `TestOptionalSmolLM2F16GGUFGreedyMatchesHFOracle`, which is
> SKIP-gated on `FAK_GGUF_REAL_SMOKE=1` + a local oracle cache and is therefore **OPEN** on
> this node — it did not run.

---

## Theorem N11.3 — metadata parse round-trips (parse then re-read is consistent)

**THEOREM.** Parsing GGUF metadata is consistent with the encoding: for every metadata value
type (string, uint8/16/32/64, int8/16/32/64, float32/64, bool, array-of-string,
array-of-int32) and the tensor directory, `Read` recovers exactly the typed values that were
serialized, and `Config()` derives the documented dimensions from them.

**REGIME.** N (numerical / layout).

**PROOF.** `Read` (`fak/internal/ggufload/gguf.go:1022`) drives `countingReader.value`
(`gguf.go:1868`), which decodes each `ValueType` per the GGUF wire format (LittleEndian,
length-prefixed strings/arrays); the typed accessors `String`/`Uint64`/`Float64`/`StringArray`
(`gguf.go:1235`) and `Config` (`gguf.go:1122`) read them back. Because the witness itself
little-endian-encodes the bytes with the inverse layout, the bit-exact match witnesses parse
consistency.

The stronger `decode∘encode = id` **involution** (witness kind §3.4) is **OPEN** here:
`ggufload` is read-only — a grep for `Write`/`Encode`/`Marshal`/`Serialize` in `gguf.go`
returns nothing — so there is no in-tree encoder to compose with `Read`. The test's encoder is
the de-facto inverse but lives in the test, not the package. Closing the involution form would
require either a `gguf.Write` serializer (then a `Read(Write(f)) == f` property test) or
designating the test encoder as a frozen reference oracle; neither exists today.

**WITNESS.**
`go test ./internal/ggufload/ -count=1 -timeout 120s -run 'TestReadParsesMetadataTensorDirectoryAndConfig|TestReadRejectsBadAlignmentAndBool' -v`

`TestReadParsesMetadataTensorDirectoryAndConfig` hand-encodes 14 KVs + a 3-entry tensor
directory and asserts `Read` recovers Version, `Alignment == 64`,
`StringArray == [<unk> hello world]` (`gguf_test.go:109`), and the full derived `Config`
(`gguf_test.go:118-129`). `TestReadRejectsBadAlignmentAndBool` confirms the parser **fails
closed** on a non-multiple-of-8 alignment and an out-of-range bool byte.

**VERDICT.** PROVEN for parse-consistency (encode-in-test → `Read` → equals written, all KV
types + derived `Config`), 2026-06-20, all subtests PASS. The strict `encode∘decode = id`
involution is **OPEN** — no in-tree GGUF serializer exists to invert.

**DOS.** bound at ship.

---

### Residual / OPEN items

- **End-to-end HF-oracle parity** — `TestOptionalSmolLM2F16GGUFGreedyMatchesHFOracle`,
  `TestOptionalSmolLM2Q4KMGGUFDequantizesAllTensors`, `TestOptionalQwen35GGUFMapsEveryTensorName`
  are gated on `FAK_GGUF_REAL_SMOKE=1` + a local fixture/oracle cache and **SKIP** here. They
  would upgrade Theorem N11.2 from synthetic-layout to real-weight numerical parity. OPEN on
  this node (not REFUTED — they simply did not run).
- **Strict metadata involution** — needs an in-tree `gguf.Write` to make
  `Read(Write(f)) == f` a property test (Theorem N11.3). OPEN; the module is currently
  read-only by design.
