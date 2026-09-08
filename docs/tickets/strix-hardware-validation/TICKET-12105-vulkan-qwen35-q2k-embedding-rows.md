<!-- fak-model-key: vulkan-qwen35-q2k-embedding-rows -->
# feat(model): admit retained Q2_K embedding rows to Vulkan sequence prefill

```routing
lane: model
paths: ["internal/model/qwen35_hal.go", "internal/model/embedding_q2k.go", "internal/compute/qwen35_sequence_contract.go", "internal/compute/vulkan_qwen35_sequence.go"]
expected_steps: 6
```

GitHub: [#12105](https://github.com/anthony-chaudhary/fak/issues/12105)

## Current state

The Qwen sequence path previously needed a single `[vocab, hidden]` f32 input
embedding allocation. For Qwen3.8 geometry that allocation is 5,085,593,600
bytes, larger than the Strix Vulkan device's 4,294,967,292-byte single-resource
cap. The loader already retains Q2_K embeddings, but sequence prefill could not
consume them without recreating the forbidden f32 table.

The admitted route now dequantizes only the requested rows in exact token order,
uploads one `[tokens, hidden]` activation panel, and identifies that request mode
with a separate backend capability. CUDA and Vulkan copy the panel into their
sequence-owned transient before execution; the caller frees the uploaded panel
exactly once when the synchronous operation returns.

## Core through-line

Retained Q2_K bytes -> ordered bounded host row gather -> one device row panel ->
exact capability/shape/vocabulary preflight -> device-local sequence transient ->
unchanged resident Qwen hybrid stack -> logits and durable route receipt.

## Gold-plating boundary

- Do not upload or shard a full f32 embedding table.
- Do not add another quantization format or alter the output head.
- Do not treat scalar per-token fallback as native sequence performance.
- Do not stop or reload the operator-owned Strix service for an exact-checkpoint test.

## Done condition

- First, middle, last, and repeated token rows preserve order and CPU row parity.
- Q2_K models enter whole-sequence prefill with memory proportional to prompt length.
- Missing or wrong row-panel capability fails explicitly before backend submission.
- Request lifetime has one owner and one free for the uploaded panel.
- The physical Vulkan path matches an independent CPU oracle without host activation traffic after admission.

## Verifiable Witness

```powershell
go test -v ./internal/model -run "TestQ2KEmbedding|TestQwen35SequencePrefill.*Q2K" -count=1
go test ./internal/compute -count=1
go vet ./internal/model ./internal/compute
```

Physical source-bound witness:

```sh
FAK_VULKAN_SPIRV="$build/spirv" \
FAK_VULKAN_REQUIRE_DEVICE=1 \
FAK_VULKAN_EXPECT_DEVICE=8060S \
FAK_VULKAN_DISPATCH_PROFILE=1 \
FAK_VULKAN_SOURCE_REV=65e42a81fc7c80e6f55ffbc4ca258db2fb56dc26 \
timeout 180s "$build/model.test" \
  -test.run '^TestVulkanQwen35SequencePrefillQ2KEmbeddingRowsMatchCPU$' \
  -test.v -test.timeout 170s
```

## Likely files

- `internal/model/embedding_q2k.go`
- `internal/model/qwen35_hal.go`
- `internal/compute/qwen35_sequence_contract.go`
- `internal/compute/vulkan_qwen35_sequence.go`
- `internal/compute/cuda_qwen35_sequence.go`

## Lane

`model` with a bounded public `compute` contract seam

## Scoped acceptance criteria

- [x] [SW-VERIFIED] Batched row gathering preserves first/middle/last/repeated order and rejects invalid IDs.
- [x] [SW-VERIFIED] Whole-table Q2_K expansion is refused at the HAL helper.
- [x] [SW-VERIFIED] Table and row-panel request modes are mutually exclusive and share one CUDA/Vulkan contract.
- [x] [SW-VERIFIED] Missing row support is an explicit non-qualifying fallback; a wrong capability identity is a typed error.
- [x] [HW-WITNESSED] RADV STRIX_HALO executed the native sequence path from source commit `65e42a81fc7c80e6f55ffbc4ca258db2fb56dc26`.
- [x] [HW-WITNESSED] A 4,096-byte row panel matched CPU logits at cosine 1.0 and exact argmax with zero post-admission H2D/D2H bytes.
- [ ] [HW-WITNESSED] Exact Qwen3.8-27B-Q4_K_M generation remains gated on a non-disruptive full-checkpoint admission window in #11959.

Physical receipt:
[`strix-vulkan-qwen35-q2k-embedding-rows-20260908.json`](../../benchmarks/receipts/strix-vulkan-qwen35-q2k-embedding-rows-20260908.json).
