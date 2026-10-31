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


## Checked Engram row family (source candidate, not serving qualification)

The existing CUDA row adapter also composes packed E4M3/E8M0, Q2_K and F32
rows with IQ4_NL. It is still one registered-backend caller, one checked stream,
one source-retirement policy and the same first-fault/poison contract. The new
family ABI1 has its own encoding namespace: packed=0, Q2_K=1, F32=2, IQ4_NL=3.
It does not extend embedding ABI2's modes or reinterpret Vulkan poison mode 3.
The historical IQ4_NL build tag and both native defines remain required for this
candidate family; a stale archive fails its exact ABI/link check.

The row-parallel mapping is adapted from
[Niko1221/Strata `ds4_kernels.cu`](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/dsv41/ds4_kernels.cu#L1521)
at `c62d5e3a15b40ea51c96aae9e8d36928965f53ab` (SHA256
`c96ba2b906ed54d3e9c7ac22b4a5d970783fe2246e4cfd01af4e8bf748aedf5a`),
including its per-token row mapping at lines 1938–1959. Copyright belongs to
Peasant Smith (mw00), the Project Maya contributors including needmorevram,
and Niko1221 and the Strata contributors. The complete MIT license and Maya
notice are retained in `THIRD_PARTY_NOTICES.md`; no second runtime is imported.

Composition and exclusions are explicit:

- Existing llama-derived Q2_K/IQ4_NL scalar maps and the canonical device
  codebook are reused, not copied a second time. F32 copies every stored word.
- Packed rows use the existing FAK Vulkan/host integer product and BF16 RNE
  contract, not Strata's unrounded F32 output. This preserves subnormals,
  signed zero, overflow, NaN-byte precedence and lowest-element status.
- Strata Q6_K is not FAK Q2_K: its 210-byte layout is excluded. Strata's gate
  reduction/rounding and zero-fill-on-read-error policy are not substituted.
- Already-gathered immutable rows enter in demand order. One bounded grid
  decodes the panel; no global table/ID ownership or whole-table upload is
  added. Each output/status allocation is fresh and device-local.
- Existing checked upload, default/private-stream fences, capture refusal,
  sticky fault latch, quarantine and selected-error/no-host-replay behavior
  remain authoritative. The former IQ4 upload shares that same checked copy.

Both broad and exact public capabilities remain **false**. The actual FAK model
consumer already selects `V41EngramRowDecodeBackend` by exact encoding, but this
CUDA candidate is reachable only through the in-package device witness until
module/device/model qualification is returned. Neither compilation nor this
source import enables default serving, a new model format or a new policy knob.

### Source-bound qualification pickup

On a separately admitted CUDA machine, build the native archive with
`FAK_CUDA_V41_EMBED_GATHER` and `FAK_CUDA_V41_EMBED_IQ4NL`, using the existing
repository archive workflow, then run the actual checked caller:

```sh
CGO_ENABLED=1 go test -tags cuda,fak_cuda_v41_embed_gather,fak_cuda_v41_engram_iq4nl ./internal/compute -run '^TestCUDAV41EngramRows$' -count=1 -v
```

The explicit tags are required; `-tags cuda` alone excludes the test and is not
acceptance. This witness fails rather than skips without registered CUDA. It
checks all 65,536 packed byte/scale pairs, partial-row fault precedence,
Q2_K scalar edges, exact F32 words, the retained IQ4_NL fixture, checked refusal,
immutable input, completed fresh outputs and status/poison composition. It is
not a fault-injection, model, allocator-peak or throughput witness.

Record the source commit, native archive/build receipt, toolkit/driver, device
UUID/architecture and test output. Hardware follow-up must additionally exercise
capture/terminal-error retention and selected failure with no host replay, then
an authenticated model/quantization artifact and ordinary native request. Measure
cold/warm prefill, decode and batch throughput only with matched cache state and
quality; separately report launches, transferred bytes, peak device/host memory,
wall time and status/fence costs. No speedup or end-to-end memory reduction is
claimed from the source-derived grid/copy counts. Compiler, linkage, tests,
device/model parity and performance are **UNRUN** for this source candidate.
