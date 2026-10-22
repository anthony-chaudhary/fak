# Source-only packed main-KV attention candidate

## Scope and evidence boundary

Exactly one candidate shader, `v41_packed_main_attention.comp`, derived from the existing `internal/compute/shaders/v41_shared_attention.comp`. No live code, registry, backend, native ABI, capability flag, generated SPIR-V or default was changed. No compiler, glslc, device, download, index, or runtime validation was performed. This is preparatory consumer source, not enabled GPU support or a speed claim.

The numerical target is existing Fak scalar-order F32 shared attention over reference-dequantized main-KV records, NOT upstream fused-attention equivalence. The local attention contract explicitly excludes BF16 casts within the three attention passes. Decode occurs at the logical KV publication boundary before each scalar use; it does not cast Q, scores, weights, denominator or output accumulators.

Directly read [pinned DeepInfer cache_common.py](https://raw.githubusercontent.com/deepseek-ai/DeepInfer/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cache_common.py): lines108-117 define the main FP4 format; lines334-356 request BF16 from reference cast-back; lines203-219 allocate BF16 for cast-on-load formats. The separately prepared local host layout change uses BF16 RNE followed by F32 widening on decoded FP4 values. This establishes the reference-dequantization contract, not live fused-attention parity or integration into Fak publication.

Independent arithmetic argument: positive finite E4M3 scales0x01..0x7e span2^-9 to448. Nonzero E2M1 magnitudes span0.5 to6. Their nonzero products therefore span2^-10 to2688, all normal in both BF16 and F32. E4M3 has at most4 significant bits (subnormals at most3); E2M1 at most2. Their exact product has at most6 significant bits, below BF16's8. Thus F32 multiplication and BF16 RNE preserve the exact value, including the smallest scale and saturated448 scale. Zero payload preserves its sign under multiplication by strictly positive scale and the explicit integer RNE. No signed/zero/NaN scale is admitted. This proves decoding exactness, not downstream exp or GPU parity.

Baseline observed HEAD: `6ba2d64f4019ae37dd9750f1234c4236a34b7778` (shared HEAD may advance). SHA-256 of source read:

- `internal/compute/v41_shared_attention.go`: `7978dc36712e2f86a817b3750b0a4356e0db506d824fd510dfb6e46683ab55d8`
- `internal/compute/shaders/v41_shared_attention.comp`: `b2259bc3fca8511feb7d080853b857cae66fab70f7b392ebc336b2464bade063`

## Proposed new immutable interface (not implemented)

This shader cannot be passed to the old F32 attention ABI, even though several binding numbers and the first six push fields are retained. A separately admitted native entry point, descriptor/push validator, compiled-module proof and lifecycle implementation are prerequisites. The `V41PackedRowStorage` metadata contract alone provides none of those.

- Binding 0: immutable finite F32 Q, row-major `[heads,512]`.
- Binding 1: immutable backend-owned packed ordered selection panel, accessed as 32-bit unsigned storage words. `selectedRows` initialized contiguous records; no spare capacity reads. Row zero begins at the descriptor base. Each slot supplies both K and V. The caller gathers selected records in the exact existing order, preserving duplicates; no sorting, deduplication or quantization at this seam. A packed gather surface does not yet exist in the proposed storage contract and is a separate implementation prerequisite.
- Binding 2: immutable finite F32 sink `[heads]` when present; otherwise a live F32 `[1]` zero dummy. Sink is unscaled, denominator-only.
- Binding 3: fresh disjoint F32 output `[heads,512]`.
- Binding 4: fresh disjoint unsigned status `[heads,4]`, the existing stage/slot+1/element+1/valueBits protocol. All status is initialized each dispatch. A selected error invalidates all output; do not consume partial writes or retry on CPU/F32.

Inputs, metadata and backing allocation ownership must remain immutable and retained through completion. Upload/copy completion and visibility must be established before dispatch; output/status visibility and completion before readback. Reject foreign/released/incomplete handles and aliased outputs before any side effect.

## Record geometry and alignment

Only width512 is supported:448 NoPE+64 RoPE. One record is288 bytes, exactly72 uint words, without inter-record padding: NoPE payload `[0,224)`, RoPE payload `[224,256)`, NoPE scale bytes `[256,284)`, RoPE scale bytes `[284,288)`. Element d uses payload byte `d/2`, low nibble for even d, high nibble for odd d, and scale byte `256+d/16`. E2M1 magnitudes are0,0.5,1,1.5,2,3,4,6 with signed-zero-preserving sign bit. No global scale exists. Indexer per32 UE8M0 is unsupported.

Canonical byte order is little-endian within each storage uint. The host must establish that byte upload to uint storage has this meaning on the selected backend, or serialize words explicitly; unsupported endianness is not admitted. Descriptor base/offset must meet both4-byte access alignment and device storage-buffer-offset requirements. Arbitrary unaligned byte origins are unsupported. Rows need only4-byte alignment:288-byte stride does not preserve64/128-byte alignment. Allocation alignment overhead is separate from logical packed bytes and must be accounted for honestly.

Canonical generated scales are positive finite E4M3 bytes0x01..0x7e. Zero, signed or NaN scales are rejected at admission, not repaired. The shader additionally produces NaN for these malformed scale bytes so existing score fault reporting catches corruption; this is not a replacement for a new validated host boundary. Payload nibbles all have finite E2M1 meanings. There is no claim that the existing general host decoder rejects malformed scales.

## Push constants and mandatory pre-dispatch checks

New fixed32-byte block, explicit4-byte offsets:

- offset0: signed heads, positive.
- offset4: signed headDim, exactly512.
- offset8: signed selectedRows, positive. Empty selection is host-owned zeros with no dispatch.
- offset12: signed mode, exactly0 plain or1 compressed.
- offset16: signed hasSink, exactly0 or1.
- offset20: F32 scale, finite and nonzero.
- offset24: unsigned rowStrideBytes, exactly288.
- offset28: unsigned formatVersion, exactly1, matching main per16 E4M3/no-global format.

These are host admission prerequisites, not shader self-validation. The shader is unsafe to dispatch with arbitrary push constants; checking or returning after untrusted addressing would not establish a safe ABI. `formatVersion` is host-validated metadata, not a shader capability assertion.

Before narrowing, overflow-check products in a sufficiently wide unsigned host type. Require heads*512 <= INT32_MAX; selectedRows*288 <= UINT32_MAX; selectedRows <= INT32_MAX-2047 to avoid signed chunk-loop overflow; heads*4 <= UINT32_MAX. Check all byte sizes against host limits and actual live descriptor extents: Q/Out >=heads*512*4, PackedKV >=selectedRows*288, Sink >=(hasSink?heads:1)*4, Status >=heads*16. Also check device maximum storage-buffer ranges, workgroup count, push limit,128-lane workgroup limit and shared-memory requirement. Reject invalid dispatch geometry before allocation/upload/dispatch, not by reading opaque device memory through a host pointer.

## Preserved arithmetic and remaining qualification

The original shared-attention body is unchanged except the two KV reads (score dot and weighted output) now call the same decoder. It keeps ascending element-order scalar dot products, separate product/sum rounding, ordered selected-slot maximum and denominator, sink term first, plain -Inf versus compressed -MaxFloat32 seeds, score validation before sentinel skip, valid exp(-Inf), three scalar-order passes, resident versus recomputed chunks, and existing first-fault ordering. Different heads/dimensions may run in parallel; no per-scalar reduction tree, FMA or online-softmax substitution is introduced.

`precise` source is necessary but is NOT a compiled NoContraction or binary32 proof. Before any promotion, compile under an owned compiler lane; inspect exact SPIR-V for descriptor types, push offsets, scalar NoContraction dependency chains, and binary32 RTE/denorm/signed-zero-Inf-NaN controls matching existing admission. Then implement bounded new native admission and exercise actual upload/dispatch/status/readback against the F32 reference for both modes, sinks, duplicate slots, finite extremes/fault order, selectedRows around2048, signed zero and all supported packed decode codes. No hardware/runtime outcome is currently established. Existing exp implementation precision qualifications remain required.

Next checkable step: independently review this source and immutable ABI, then authorize a separately scoped compiler/native integration witness; do not enable a capability from this artifact alone.
