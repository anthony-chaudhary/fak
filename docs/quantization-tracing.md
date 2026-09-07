# Follow Q4_K bytes through a computation

`go run ./cmd/quanttrace --out receipt.json` executes a deterministic two-row
demo and explains each stage. Add `--json` for the machine-readable receipt.
For actual extracted row-major Q4_K bytes:

```sh
go run ./cmd/quanttrace --input weights.raw --rows 8 --columns 256 --width 4 --samples 0,31,32,255,256 --out receipt.json
```

The input is a tensor payload, **not an entire GGUF file**. Maximum input is
8 MiB; columns must be divisible by 256. Samples are zero-based row-major
element indexes (at most 32). The activation is deterministic:
`x[i] = (i % 17 - 8) / 8`. Its float32 little-endian hash is in the receipt.
The output path must differ from the input, including hard links. The receipt
shows the first eight output values (or all rows when fewer); its output hash
covers every row.

| Operation | What actually changes | Witness |
|---|---|---|
| Unpack | 4-bit codes become one byte each; scale/min metadata gives their meaning | Source byte, nibble shift, code, metadata start and subblock |
| Repack | Row-major blocks become width-interleaved fields; no requantization | Restoring row order reproduces every input byte |
| Expand | Decode `w = (d * scale) * code - dmin * min` into float32 | Sample value and full float32 little-endian hash |
| Contract | Matrix-vector product reduces the input dimension | Both layouts produce bit-identical outputs and report maximum absolute error |

A block stores 256 values in 144 bytes: two float16 coefficients (4 bytes),
packed 6-bit scale/min metadata (12 bytes), and 256 nibbles (128 bytes). That is
**4.5 bits per value**, including metadata. Float32 expansion occupies 1,024
bytes per block. This tool deliberately allocates full expanded code and float
arrays; these counts are diagnostic allocations, not measured device traffic,
peak memory, or production residency. The GEMV helpers decode block scratch as
they calculate. Other simultaneous buffers include input, repacked and restored
payloads, activation, two outputs and helper scratch.

Sample `source_byte` and `repacked_byte` are offsets in their respective
payloads. The same `nibble_shift` (0 or 4) extracts the sampled code from either
layout. `scale_metadata_start` names the start of the source block's 12-byte
metadata field; `subblock` selects its decoded 6-bit pair.

Expansion reconstructs values represented by the quantized artifact. It does
not recover original pre-quantization floats. Their error is unknown and
explicitly unmeasured. Matrix contraction is not an inverse quantizer or
storage compression. Byte round-trip equality proves only the layout permutation.

This is an offline software diagnostic using existing
`internal/bench/q4krepack.go` permutation, dequantization and GEMV helpers, not
runtime instrumentation or hardware qualification. Dequantization calls the
model's actual scale/min and float16 conversion primitives. The existing repack
experiment documents its pinned llama.cpp lineage and separates its pure-Go
layout comparison from shipped SIMD kernels. No new codec or kernel policy is
introduced.

Verification: `go test ./internal/bench ./cmd/quanttrace -run QuantTrace -count=1`.
Tracking: [fak#12059](https://github.com/anthony-chaudhary/fak/issues/12059).
