# TPU / Neural-Engine backend — C-004 / issue #261 — scaffold + named blocker

**Status:** host-tractable scaffold SHIPPED (the accelerator→compiler-lane taxonomy, exact +
unit-witnessed on any host). **Blocked:** the XLA/CoreML compilation and every acceptance bullet
that runs on silicon are deferred to a node with the accelerator — this dev host has no Google
TPU and no Apple Neural Engine, and no XLA/PJRT or CoreML toolchain.

## Why one issue is two backends

The issue title ("TPU/Neural Engine") lumps two accelerators that do **not** share a code path,
so the load-bearing first decision is *which compiler lane each one lowers through*:

| Accelerator | Lane | How it reaches the HAL | Native tier | HAL cap |
|---|---|---|---|---|
| Google Cloud TPU (v2–v6e Trillium) | **XLA** | record the in-process op-list → lower to StableHLO → XLA/PJRT compiles & places it (the whole-graph **Dataflow** lens) | bf16 (MXU) | `Caps.GraphCompile` |
| Apple Neural Engine (A17/M3 … M4) | **CoreML** | map coarse blocks (a whole MLP) to a CoreML MLProgram; CoreML places ops on the ANE; stage weights device-native (the **Edge-NPU** lens) | fp16 | `Caps.FusedFFN` + `WeightSource` |

The ANE is **not** the Apple Metal GPU backend already shipped (`metal.go`) — it is a separate
fixed-op-menu accelerator reached through CoreML, not Metal compute shaders.

## What shipped here (compute lane, `internal/compute/tpu_arch.go` + test)

The issue's scope is a 10–14-day experimental backend; three of its four acceptance bullets can
only be witnessed on the silicon. What *can* be built and proven on any host is the part the
compiler build needs **before** a kernel ever runs: the accelerator→lane taxonomy + native tier +
canonical target token. That is shipped here, exact and tested.

| Scope bullet (#261) | Shipped here | Witnessed by |
|---|---|---|
| XLA compilation path | The `LaneXLA` classification + `AccelLane.GraphCompiled()`/`PrimaryCap()=="GraphCompile"`: every TPU generation routes to the whole-graph op-list→StableHLO→XLA path, the existing `Caps.GraphCompile` seam, never an edit to the forward loop. | `TestAccelLaneInvariant` |
| ONNX import | Reframed honestly: fak does **not** import an external ONNX/StableHLO graph — it captures its OWN op sequence as a portable in-process op-list and lowers that (the documented Dataflow-lens stance in `docs/explainers/hardware-portability.md`: "no ONNX/StableHLO importer"). The taxonomy records the lane that op-list is lowered to; an external-graph importer is the inverse direction and out of fak's architecture. | doc + `LaneXLA` mapping |
| Device bridging | `LookupAccelArch`/`AccelTarget` normalize the noisy strings a runtime reports (XLA device-kind like `"TPU v5 lite"`, Apple chip names like `"Apple M3 Pro (ANE)"`) to one canonical target, failing closed on an unsupported part — the exact input the deferred build path feeds the compiler. | `TestAccelTargetNormalization` |
| Proof-of-concept benchmark | No new gate needed — the within-N acceptance reuses the existing `WithinTarget(measured, baseline, factor)` (prefill.go). The device node feeds `WithinTarget(tpuSeconds, cpuRefSeconds, factor)`; the rule lives in one tested place. | `prefill.go:WithinTarget` (existing) |
| (HAL integration) | The taxonomy is the always-compiled half; the cgo `//go:build xla` / `//go:build coreml` halves register an `Approx` backend that plugs into the **existing** `compute.Register`/`Pick` seam unchanged — adding a backend is a registration, never a forward-loop edit, exactly as Vulkan and Metal did. | seam proven by `cuda.go`/`vulkan.go`/`metal.go` |
| Native tier ties to the HAL dtype enum | `AccelArch.NativeDtype` is a real `compute.Dtype` (`BF16`/`F16`), so the deferred `Upload(t, NativeDtype)` narrows weights to a dtype the contract already dispatches on, not a parallel vocabulary. | `TestAccelDtypeIsHALDtype` |

`go build ./...` (default, no accelerator tag) is clean; `go test ./internal/compute/ -run Accel`
is green on the CPU path.

## The compiler-reuse strategy (for the accelerator node)

1. **TPU (XLA lane):** `internal/compute/tpu_xla.go` — `//go:build xla`, an `Approx` backend named
   `"tpu"` that advertises `Caps{GraphCompile,Async,DeviceMemory,UploadDtype,CapacityProbe}`. It
   runs the Backend methods in record-only mode to capture the in-process op-list (the exact
   mechanism the Dataflow lens already describes), lowers it to StableHLO, and executes it via
   PJRT. `cpu-ref` stays the `Reference` Default so nothing runs on the TPU unless explicitly
   selected (`FAK_BACKEND=tpu` / `--backend tpu`). The CPU reference eagerly interprets that *same*
   op-list, so the recorded-graph replay stays bit-identical (the GraphCompile contract).
2. **Apple Neural Engine (CoreML lane):** `internal/compute/ane_coreml.go` — `//go:build coreml`,
   an `Approx` backend named `"ane"` advertising `Caps{FusedFFN,DeviceMemory,UploadDtype}`, using
   `WeightSource` to stage an fp16 device-native layout and `Caps.FusedFFN` to map an MLP block to
   one CoreML op. Distinct from `metal.go` (Metal GPU).
3. The op-level + full-forward witnesses (`-tags xla` / `-tags coreml`) mirroring
   `cuda_test.go` / `vulkan_test.go`: argmax EXACT, the rest cosine + small max|Δ| (`Approx`).

## The named blocker (do NOT fake this on a host with no TPU / Neural Engine)

The four acceptance bullets

- [ ] Run on TPU/neural engine
- [ ] Correct forward pass
- [ ] Baseline performance
- [ ] Documentation  ← satisfied by this notes file + the TPU/Neural-Engine lens in
      `docs/explainers/hardware-portability.md`

The first three are **measurements on the accelerator** (a TPU under XLA/PJRT, or an ANE under
CoreML). They cannot be witnessed on this win32 CPU host (no TPU, no Neural Engine, no XLA/CoreML
toolchain). **No TPU/ANE throughput or correctness number was run, estimated, or fabricated
here** — the shipped deliverable is the exact, hardware-independent lane taxonomy + target
selection + the documentation. Dependencies: none (per the issue).

## Next agent, on a node with the accelerator

1. **TPU:** provision a Cloud TPU (v5e/v5p/v6e) with `libtpu`/PJRT; confirm the runtime reports a
   device-kind string and feed it to `AccelTarget` (it accepts `"TPU v5 lite"`, `"tpu-v5e"`, etc.).
   Write `tpu_xla.go` (`//go:build xla`) per the strategy above; register the `"tpu"` Approx
   backend. Witness numerical parity op-by-op then the full forward (`-run HALTPU`); hold `argmax`
   EXACT, the rest to cosine + small max|Δ| (`Approx`).
2. **Apple Neural Engine:** on an Apple-silicon Mac (M3/M4) with CoreML, write `ane_coreml.go`
   (`//go:build coreml`); register the `"ane"` Approx backend; witness parity the same way. (The
   fleet's Mac verify node is the natural host — see the always-on dogfood server doc.)
3. Benchmark vs the `cpu-ref` baseline on the same node and grade with
   `WithinTarget(accelSeconds, cpuRefSeconds, factor)`. Record lineage (toolchain version + UTC +
   commit + sanitized machine + harness), run on a node NOT also running the agent (placement law).
