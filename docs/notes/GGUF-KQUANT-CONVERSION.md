# Local GGUF K-quant conversion

CPU-qualified local conversion candidate. Publication requires the native candidate
acceptance and remote-main verification; fixture success is not a release claim.

## Opt-in command

```
fak model quantize --input local.gguf --output new.gguf --type q4_k
```

The command accepts q4_k, q5_k and q6_k. It is a bounded local GGUFv3 producer,
not a model manager: no aliases, downloads, cache installation, split discovery,
GPU requirement or runtime backend change. The input file is never modified.
Only eligible F32 two-dimensional matrices with row width divisible by 256 are
encoded; other supported payloads remain byte-identical. This is not llama.cpp's
model-specific mixed K_M recipe. Importance-matrix weighting and file input are
explicitly deferred; no test-only weighted API is claimed as integration.

Output must be absent. An owned same-directory temporary file is written, synced,
validated and reloaded through existing bounded WeightSource row reads before
an atomic no-overwrite hard link publishes it. Filesystems without this primitive
fail closed; no replacing rename fallback. A conversion failure or cancellation
before publication removes the owned temporary file and publishes no destination.
A concurrently created destination is preserved.

Input names, dimensions, alignment and non-recipe metadata are retained. Raw
metadata spans preserve empty-array element types, float bit payloads and custom
duplicate fields rather than round-tripping through a generic serializer. Tensor
directory offsets are regenerated. General file-type/quantization-version UINT32
scalars must be truthful; stale importance/quantization recipe provenance and
unsupported scalar types are refused. The first-version envelope preserves vectors and already-target rank>=2 tensors
whose row width is divisible by 256, but rejects preserved rank>=2 tensors of
other dtypes or incompatible geometry. It never writes GUESSED1024 or claims
general mixed-model conversion support.

## Bounded envelope

- Single regular local GGUFv3 file; no split metadata.
- At most 65,536 tensors and 65,536 metadata records, checked before allocation.
- At most 1,048,576 aggregate decoded metadata values and 64 MiB header/string
  extent; nested metadata arrays are rejected.
- Alignment at most 1 MiB. Converted rows fit the existing 64 MiB decoded-row
  ceiling; encoding itself streams 256-value blocks.
- Duplicate tensor names, overlapping/out-of-file payloads, unknown byte geometry,
  dimension/offset overflow and no eligible matrices fail before publication.
- Converted values must be finite and inside the documented numeric envelope;
  unrepresentable scales, nonfinite intermediates and invalid reference rounding
  arguments fail closed. Existing preserved auxiliaries are copied as raw bytes.

Parser limits are opt-in to this producer. Ordinary Read/ReadSize behavior and
existing decoder arithmetic are unchanged. Converted values must be zero or have
absolute value between 1e-30 and 65504 inclusive. Intermediate checks may reject
inputs inside that range; the range is not a promise of exhaustive equivalence.

## Provenance and reproduction

Numeric source: ggml-org/llama.cpp at
83078fec0db82d6b5a00d9599062c38c39145755, ggml/src/ggml-quants.c git blob
1ebc50a763f16db909de37090da38cc8c0fdde94. The port preserves float32 evaluation
order, strict search comparisons, reference nearest-integer behavior, local
round-to-nearest-even half serialization and Q4/Q5/Q6 wire layout. Q4_K is 144
bytes/block, Q5_K 176, Q6_K 210; all represent 256 source values, little endian.
This is a modified Go translation and local-artifact integration, not an upstream
endorsement or an alternate inference backend.

The testdata/kquant_refgen/main.go native fixture driver consumes an unchanged
pinned source/header directory, verifies every git blob and compiles the actual
upstream implementation. It records compiler binary hash/version, flags, source
SHA256 and C driver hash. Flags include -ffp-contract=off, -fno-fast-math and
-fexcess-precision=standard. The C driver only replaces the fatal ggml_abort
link dependency; it does not replace any numeric function or macro.

Fixtures store input float32 bit patterns and C-derived encoded bytes/RMSE, not
cross-language RNG or sin output. Cohorts include signed zeros/constants, tiny
values, mixed signs, half boundaries, skewed subblocks and consecutive blocks,
plus 16 direct C half tie/subnormal/overflow cases. The acceptance roots
are TestKQuantReferenceGolden, TestConvertGGUFKQuant,
TestConvertGGUFRejectsMixedRecipe and TestModelQuantizeCLI.
The latter builds/runs the real production CLI independently of cmd/fak's large
test package. On 2026-11-02 these four roots passed on CPU with Go
`-p=1 -trimpath -buildvcs=false`; the test-built CLI is not a stamped release
binary. Exact byte parity covers nine fixed cohorts for all three encoders plus
16 half-boundary cases. These are CPU software correctness witnesses only:
no model-quality, performance, GPU or full-llama.cpp parity claim.

Transitive half-conversion provenance: ggml-impl.h names Maratyszcza/FP16. Its
observed license/header were verified at 782eea126dc5c755827be751a099eb01826175cf;
this is the observed origin-license pin, not a claimed historical import revision.
The reference uses unchanged ggml-impl.h, not the separately fetched FP16 header.
Both complete MIT notices follow.

## llama.cpp / ggml notice

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## FP16 notice

The MIT License (MIT)

Copyright (c) 2017 Facebook Inc.
Copyright (c) 2017 Georgia Institute of Technology
Copyright 2019 Google LLC

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
