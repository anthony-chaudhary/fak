# CUDA selected-row primitives: source and qualification boundary

The optional native embedding fragments directly reuse the MIT-licensed
`ggml-org/llama.cpp` implementation at
`e0f142d5ddc1461ce721644d92af33db23545105`. The upstream runtime, allocator,
stream, scheduler and model execution are not imported. Fak owns execution.

## Source attribution

- [CUDA row gathering](https://github.com/ggml-org/llama.cpp/blob/e0f142d5ddc1461ce721644d92af33db23545105/ggml/src/ggml-cuda/getrows.cu)
  supplies the selected-row and 32-lane IQ4_NL mapping.
- [CUDA dequantization](https://github.com/ggml-org/llama.cpp/blob/e0f142d5ddc1461ce721644d92af33db23545105/ggml/src/ggml-cuda/dequantize.cuh)
  supplies the Q2_K, Q5_K and IQ4_NL lane maps.
- [Block layout and nonlinear codebook](https://github.com/ggml-org/llama.cpp/blob/e0f142d5ddc1461ce721644d92af33db23545105/ggml/src/ggml-common.h)
  define the stored blocks. IQ4_NL reuses Fak's existing
  `fcuda_kvalues_iq4nl` table; there is no second production codebook.
- The full upstream MIT notice is preserved in
  `internal/compute/cuda/v41_embed_gather.LICENSE` and in the IQ4_NL header.
  Fak-owned finite/status/poison glue remains Apache-2.0.

Raw byte loads avoid aligned-struct assumptions. Each IQ4_NL block contains
an f16 scale followed by 16 code bytes: low nibbles produce elements 0–15,
high nibbles 16–31. Eight 18-byte blocks make one 144-byte, 256-value row.
Half widening and separately rounded F32 multiplication preserve the host
reference's ordering. No quantizer or new dialect conversion is introduced.

## Native and serving boundary

`FAK_CUDA_V41_EMBED_GATHER` includes the original native embedding ABI2
(modes 0/1/2 only). IQ4_NL also requires `FAK_CUDA_V41_EMBED_IQ4NL` and has
separate ABI1 exports; it never changes ABI2 mode 3 or Vulkan poison mode 3.
The checked-row entry point returns a fresh F32 panel plus one uint32 status
word per row. Status 0 is clean; a nonfinite value at the lowest element j
publishes `2*j+2`. A selected native error invalidates both outputs and never
authorizes host replay. Numerical status is not a CUDA completion result.

The optional shared-stream guard is integrated additively with current peer
guards, including routed-MMID and failed-fence retention. A terminal uncertain
stream closes admission and prevents allocation recycling. The optional Go witness
caller uses the registered backend under `cudaMu`, exact ABI admission, tracked
device-local allocations, fresh output/status storage and checked completion.

The IQ4_NL dialect is known at 144 bytes, but a legacy broad row-decode
capability cannot admit it. The optional Go adapter additionally requires
`fak_cuda_v41_engram_iq4nl` and `fak_cuda_v41_embed_gather` tags with matching
native feature defines. Both its broad and exact capability methods remain
false, and public decode/poison calls refuse. Only the in-package qualification
witness reaches the adapter. Upload borrows host bytes synchronously and
checks completion of both the call's default stream and the private stream.
No model staging, loader default, host replay or serving activation is added.
Capability must remain false until the actual binding, producer/lifetime
contract and exact module/device are qualified.

## Evidence and remaining checks

The unchanged portable IQ4_NL math and finite/status/poison helpers have
bounded host known-answer witnesses. Those receipts prove neither current
NVCC/archive/cgo linkage nor CUDA atomics, barriers, stream lifetime or a model
request. The two source-bound host tests are retained for changed-source
requalification, not as a reason to rerun unchanged math.

Critical next verification is one exact-module/device selected-row fixture
with mixed clean/fault rows, repeated IDs, exact poison and default/alias
refusal through the actual Go witness; then checked failure/retention. The
portable host fixtures separately cover bounded geometry and refused IDs. Native build
and link checks, device execution, authenticated IQ4_NL model staging,
selected-model first token, quality and throughput remain unverified.
