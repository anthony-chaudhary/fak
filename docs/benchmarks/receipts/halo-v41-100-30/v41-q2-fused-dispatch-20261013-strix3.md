# V4.1 fused routed-FFN dispatch (P4) on strix3: two device dispatches per routed layer instead of five, 161 -> 152 compute dispatches per decode step with bitwise-equal logits (2026-10-13)

- **Dispatch count.** `[HW-WITNESSED]` On the real strix3 device (AMD Radeon 8060S
  Graphics, RADV STRIX_HALO), the fused two-dispatch routed layer replaced the id
  path's five in the model-level `TestV41RoutedLayerBatchVulkan` `batch_id` arm:
  **161.0 -> 152.0 compute dispatches per decode step** over 6 steps
  (compressed=false). The same run reports `bitwise_equal_windows=7`, i.e. every
  prefill and decode logit window is bitwise equal between the fused and id arms.
- **Parity.** `[HW-WITNESSED]` The P3 id path's five dispatches and the P4 fused two
  are compared directly by `TestVulkanV41RoutedFusedMatchesID`: the fused
  down+sum epilogue is **bitwise** the id down+publish+sum over the same `mid`, every
  status verdict is exactly the id verdict (including both fault-injection cases), and
  the gate/up `mid` rows are within a measured BF16-ULP/relative bound. The
  end-to-end model logits are bitwise equal.
- **Gate/up GEMV parity is ULP-bounded, not bitwise.** `[HW-WITNESSED]` The fused
  gate/up kernel compiles one program producing both projections, so its FP32 FMA
  contraction differs from the id path's two separate GEMV dispatches. The witness
  geometry (hidden 4096 / inter 2048 / picks 6) measured a worst `maxRelMid=2.7778e-02`
  and `maxMidULP=5`; `permuted_slots` 9.17e-03 / 2 ULP; `eight_picks` 5.38e-03 / 1 ULP.
  Small geometries (`hidden 512`) and the `single_pick`/`small_tail_three_picks` cases
  are **bitwise** equal and are asserted as such. The ticket's P4 witness permits
  "Bit or ULP parity"; this receipt records the measured bound rather than a bit claim.
- **Down+sum fusion is bitwise.** `[HW-WITNESSED]` Relative deviation of the fused
  `sum` against the id `sum` is a reported diagnostic only; the pass/fail gate is the
  bitwise replay of the id down+sum epilogue over the fused `mid` (same inputs), which
  holds for every non-fault case.

## Binding

| Field | Value |
| --- | --- |
| Base | public `02beb0159` (the trunk tip the lane rebased onto; the change was first witnessed on `58daea354cd4`). |
| Change | this lane's delta: `internal/compute/v41_routed_fused.go`, `internal/compute/vulkan_v41_routed_fused.go`, `internal/compute/shaders/v41_routed_gate_up_swiglu_id.comp`, `internal/compute/shaders/v41_routed_down_sum.comp`, `internal/compute/vulkan_shim.cpp`, `internal/compute/vulkan_backend.h`, `internal/compute/build_vulkan.ps1`, `internal/computebuild/{types.go,vulkan_registry_successor.go,vulkan_receipt_v16_test.go}`, `internal/model/v41_routed_layer_device.go`, `internal/devcmd/build_vulkan.go`, `THIRD_PARTY_NOTICES.md`. |
| Shader registry | V17/complete81 (`fak.vulkan-shader-registry.v17.current81`, sha256 `c06385812cd06425e906c1ab1e469cf29a42e54481483d00fd9c44ab6ff7198b`), V16/79 and V15/78 retained as historical identities. A concurrent peer lane landed V16/79 (`v41_shared_attention_fast`), so this change carries V17 on top of it. |
| Build | strix3 clone at `/tmp/fak-commit`, `glslc -O --target-env=vulkan-1.2 -fshader-stage=comp` over the 81 registry stems, then `go run ./cmd/fak-dev build-vulkan lib` -> `libfakvulkan.a` 431,816 B. |
| Device | AMD Radeon 8060S Graphics (RADV STRIX_HALO), Vulkan Runtime 1.4.354, device memory 86.3 GB total / 73.0 GB free at test time. |
| Model path | Synthetic pinned Q2_K gate/up and Q3_K down expert slate (the model-level witness's `v41PinnedSlateFixture`), not the full checkpoint. |
| Invocation | `FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/compute -run '^TestVulkanV41RoutedFusedMatchesID$' -count=1 -v` and the same env with `./internal/model -run '^TestV41RoutedLayerBatchVulkan/compressed_false$'`. |
| Fused on/off | `FAK_V41_ROUTED_FUSED` (unset = fused preferred; `0` = five-dispatch id path), alongside the existing `FAK_V41_ROUTED_ID` / `FAK_V41_LAYER_EXPERT_BATCH`. |

## Parity witness `[HW-WITNESSED]`

`TestVulkanV41RoutedFusedMatchesID` (all 11 cases PASS):

| Case | Result | maxRelMid | maxMidULP |
| --- | --- | ---: | ---: |
| v41_flash_geometry (4096/2048/6) | PASS (bounded) | 2.7778e-02 | 5 |
| permuted_slots (4096/2048/6) | PASS (bounded) | 9.1743e-03 | 2 |
| small_tail_three_picks (512/256/3) | PASS (bitwise) | 0 | 0 |
| eight_picks (1024/512/8) | PASS (bounded) | 5.3763e-03 | 1 |
| activation_fault_pick_2 | PASS (exact fault) | n/a | n/a |
| publication_fault_pick_4 | PASS (exact fault) | n/a | n/a |
| reversed_sum_order | PASS | 0 | 0 |
| per_pick_activation | PASS | 0 | 0 |
| single_pick | PASS (bitwise) | 0 | 0 |
| per_pick_reversed_permuted | PASS | 0 | 0 |
| fused_down_fault_reversed_per_pick | PASS (exact fault) | n/a | n/a |

- The fused `sum` is bitwise the id down+publish+sum epilogue over the same `mid`
  (`v41RoutedIDBitsEqual`) in every non-fault case.
- Publication and activation verdicts are exactly the id verdicts, and a poisoned
  pick faults only in its own record.

## Dispatch economy `[HW-WITNESSED]`

`TestV41RoutedLayerBatchVulkan/compressed_false`, 6 decode steps:

| Arm | Compute dispatches/step | submits/step | bitwise logit windows |
| --- | ---: | ---: | ---: |
| per_pick_seam | 254.0 | 192.5 | 7 |
| batch_per_pick (id off) | 254.0 | 138.5 | 7 |
| batch_id, `FAK_V41_ROUTED_FUSED=0` (P3) | **161.0** | 138.5 | 7 |
| batch_id, fused on (P4) | **152.0** | 138.5 | 7 |

- The fused path removes 9 compute dispatches per step in this 3-layer fixture:
  one fewer per routed layer (two instead of three id dispatches once the
  publication+sum is folded), i.e. the P4 target of 2 dispatches per routed layer.
- Every arm's logits are bitwise equal, so the change is exact at the model boundary.
- The model-level `submits/step` is unchanged in this fixture: the routed dispatch
  term is already recorded into the step command buffer, so P4 removes dispatches,
  not submits, at this geometry.

## Provenance

- ds4 anchors re-pinned: antirez/ds4 `fc80bd695da76ee14bcb86820bf900a5cdfda806`
  (MIT, `Copyright (c) 2025-2026 antirez`), `ds4_cuda.cu:22143`
  `moe_gate_up_mid_decode_lut_qwarp32_kernel`, `:23869` `moe_down_sum_qwarp32_kernel`,
  `:24099` `moe_down_q4K_sum6_qwarp32_kernel`. Notice added to
  `THIRD_PARTY_NOTICES.md`.
- llama.cpp `mul_mat_vec_id` @`ccdbe290533f68154ede46e0618622eebcb6c070` and
  `llama-moe-cache.h` PR #29887 remain the id/slot-map binding the fused kernels
  reuse; that notice was already present.
- SGLang batching/draft-shape anchors (design reference only, no code copied):
  `sgl-project/sglang@45418d7dc648d0d71501b8b905a2499a40e9b971` (Apache-2.0),
  `python/sglang/srt/models/deepseek_v4.py`, `deepseek_v4_dspark.py`,
  `deepseek_v2.py` resolve at that revision.

## Not shown yet

- The **full V4.1 Flash Q2_K checkpoint** serve-mode A/B (`fak serve` on strix3,
  `v41-expert-residency-s1` protocol) has NOT been re-run for this change, so the
  end-to-end warm tok/s delta at 12G ring is unmeasured. This receipt is a
  synthetic-slate device witness plus the dispatch count, per the P4 "experiment"
  completion standard.
- The gate/up ULP bound was measured on the witness geometries only (4096/2048/6,
  1024/512/8, 512/256/3, 2048/512/5) and 8 seed trials. A different weight slate
  or activation distribution could widen it.
- `stagings/token` and `ms/staging` are **unchanged** by P4 by construction: the
  fused kernels read the same ring slots through the same ids buffer. Ring-fill
  bytes remain the P2 lever.

## Next

1. Run the strix3 serve A/B on the real Q2_K checkpoint with `FAK_V41_ROUTED_FUSED`
   unset vs `0` to put warm tok/s and dispatches/token on the S1 protocol.
2. The fused gate/up kernel still spends one workgroup per (row-pair, pick); the
   next dispatch cut is a single grid over all picks with the slot map as a grid
   dimension, targeting the remaining routed dispatch term.

## Climb points

<!-- fak-climb: metric=v41_routed_layer_compute_dispatches_per_step value=161 host=strix3 build=02beb0159 arm=batch_id_p3 workload=pinned-slate-6steps class=HW-WITNESSED at=2026-10-13T05:40:00Z -->
<!-- fak-climb: metric=v41_routed_layer_compute_dispatches_per_step value=152 host=strix3 build=02beb0159 arm=batch_id_p4_fused workload=pinned-slate-6steps class=HW-WITNESSED at=2026-10-13T05:40:00Z -->
<!-- fak-climb: metric=v41_fused_max_rel_mid_deviation value=2.7778e-02 host=strix3 build=02beb0159 arm=fused-vs-id workload=4096x2048x6 class=HW-WITNESSED at=2026-10-13T05:40:00Z -->
<!-- fak-climb: metric=v41_fused_max_bf16_ulp value=5 host=strix3 build=02beb0159 arm=fused-vs-id workload=4096x2048x6 class=HW-WITNESSED at=2026-10-13T05:40:00Z -->
