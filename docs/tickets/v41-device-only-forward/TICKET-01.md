# V4.1 Flash forward: device-only on Halo

Ref #13294, #13668.

## Problem

`fak serve` on Halo runs `ExecutionPolicyDeviceOnly`. `validateDeviceOnlyExecution`
(`internal/model/qwen35_hal.go`) refuses every DeepSeek V4.1 session, so warmup
fails closed and `/healthz` stays `warmup_pending`. The refusal is correct:
`forwardV41` is a host-orchestrated loop. Its activations live in host slices,
and each device callback (projections, norms, RoPE, indexer score, sparse sink,
expert gate/up/down) uploads a row and reads the result back.

## Contract (landed in slice 1)

- The refusal cause is a typed `v41DeviceOnlyHostOpError` (`errors.Is` ->
  `errV41DeviceOnlyHostOp`) that lists every op still on the host, in forward order.
- The refusal reports forward `deepseek41`, not `attnSeq-gqa`.
- The gate admits V4.1 only when `v41DeviceOnlyHostOps`
  (`internal/model/v41_device_only_inventory.go`) is empty. Remove an entry
  only in the change that moves that op onto the device seam. There is no
  host-compute escape hatch.

## Host-op inventory (2026-10-10, base 4e6ebc6218e)

| Op | Host site | Notes |
| --- | --- | --- |
| embedding-panel | `Model.v41EmbeddingPanel` | host gather into the activation panel |
| mhc-initial-streams | `v41FullInitialStreams` | four-stream BF16 init |
| mhc-sinkhorn-split | `v41MHCSplit` | Sinkhorn over the projected mix (projection itself has a device seam) |
| mhc-pre-collapse | `v41MHCPre` | weighted four-stream collapse |
| attention-index-list | `Model.v41AttentionIndexList` | top-k row selection after the device indexer score |
| attention-output-rotation | `v41AttentionOutputForProjection` | inverse output RoPE before the grouped output projection |
| moe-router-topk | `v41RouteForGeometry` | router softmax/bias/top-k (router logits projection has a device seam) |
| routed-expert-weighted-sum | `v41FullRoutedSum` | weighted sum of the expert outputs |
| shared-expert-add | `v41SharedExpertAdd` | routed + shared sum |
| mhc-post-mix | `v41MHCPostBF16` | post/comb mix back into four streams |
| residual-stream-carry | `Model.v41Layer` | activations and streams stay in host slices between ops |

Ops that already have a device seam with a typed fail-closed error and no host retry:
dense/grouped/head projections, the mHC and FFN-mHC projections, the
query/KV/FFN/final/compressor/index-key norms, tail RoPE, the indexer score, the
sparse-sink and shared attention, the shared-expert activation, and the routed
expert gate/up/down.

## Next slices

1. Keep the residual streams and the mHC split/pre/post on device. This is the
   prerequisite for every row below it, because otherwise each op still round-trips.
2. Router top-k and the routed/shared expert sums on device, feeding the
   activated-expert device ring.
3. Embedding gather and the attention index list/output rotation on device.
4. Once the inventory is empty, run the Halo witness: `fak serve` warmup passes
   on strix1 or strix3 under a lease, and `/healthz` reports ready.
