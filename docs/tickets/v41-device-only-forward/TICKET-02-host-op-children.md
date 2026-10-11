# V4.1 Flash forward: host-op child issues

Ref #13668. Parent contract: [TICKET-01](TICKET-01.md). An op leaves
`v41DeviceOnlyHostOps` (`internal/model/v41_device_only_inventory.go`) only in the
change that moves it to a device seam. There is no host escape hatch.

Sites are at public `origin/main` 03ef6ec788.

## Listed in the inventory

| Op | Host site | Issue |
| --- | --- | --- |
| engram-stream-injection | `Model.v41EngramInjectPrepared` (`internal/model/v41_forward_engram.go:357`) | #13811 |
| compressor-gated-pool | `V41CompressorPool.PushNormalized` (`internal/model/v41_compress_index.go:77`) | #13812 |
| attention-input-norm | `Model.v41AttentionInputNormWithLeaf` | #13810 |

## Missing from the inventory (process cause: scoping-failure)

These ops run on the host, but neither `v41DeviceOnlyHostOps` nor
`v41DeviceOnlyBindings` lists them, so the gate would admit V4.1 with a hidden host
arm. Each child adds the inventory entry first, so the gate refuses honestly. It then
moves the op to the device and removes the entry in the same change.

| Op | Host site | Issue |
| --- | --- | --- |
| bf16-boundary | `v41RoundBF16` (`internal/model/v41_compress_index.go:178`) through `v41LatentNormBF16Copy` (`v41_forward.go:1928`), `v41RoutedBF16`, `v41MHCBF16`, `v41RoPEBF16Copy`, `v41IndexBF16Copyback`, `Model.v41FullFinalNorm` | #13813 |
| logit-scale | `Model.v41HeadWithFinalNorm` calls `logitScaleInPlace` (`internal/model/v41_forward.go:1448`); also `Model.v41OutputHead` (`v41_decoder.go:226`) | #13814 |
| routed-swiglu-clamp | `Model.v41FullRoutedExpert` declined gate/up arm (`internal/model/v41_routed_expert.go:110`); `Model.v41ContractRoutedGrouped` (`v41_expert_groups.go:278`) | #13815 |
| compressed-publication-rope | `Model.v41CompressedPublicationRoPE` (`internal/model/v41_indexer_projection.go:116`) | #13816 |
| indexer-query-rope | `Model.v41IndexRoPE` (`internal/model/v41_indexer_projection.go:141`) | #13817 |
| dense-lora-apply | `Session.v41DenseProjectionFunc` calls `Model.loraApply` (`internal/model/v41_forward_weights.go:1466`); also `Model.v41PackedHeadRows` (`v41_head_projection.go:136`) | #13818 |

## Witness per child

`go test ./internal/model -run '<pattern>' -count=1` checks parity against the CPU
reference. `go test -tags vulkan ./internal/model -run '<pattern>' -count=1` checks
the device dispatch. A selected-op failure must return a typed error that
`errors.Is` can match, and it must never replay on the host.
