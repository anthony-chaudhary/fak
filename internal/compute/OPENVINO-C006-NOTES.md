# OpenVINO backend (Intel CPU/GPU/NPU) — C-006 / issue #257 — scaffold + named blocker

**Status:** host-tractable scaffold SHIPPED (the OpenVINO device-plugin taxonomy, exact +
unit-witnessed on any host). **Blocked:** the OpenVINO IR export + runtime integration and every
acceptance bullet that runs on Intel silicon are deferred to an Intel CPU/GPU/NPU node — this dev
host has no OpenVINO toolchain (the OpenVINO runtime / `openvino` package + the cgo C API) and no
Intel NPU.

## What shipped here (compute lane, `internal/compute/openvino_arch.go` + test)

The issue's scope is a 10–14-day backend; three of its four acceptance bullets can only be witnessed
on Intel silicon through the OpenVINO runtime. What *can* be built and proven on any host is the part
the OpenVINO integration needs **before** any model runs: the device-plugin taxonomy — which Intel
device classes fak targets, the canonical plugin token each maps to, the device-native inference
precision, and which device-selection strings are virtual meta-plugins. That is shipped here, exact
and tested.

| Scope bullet (#257) | Shipped here | Witnessed by |
|---|---|---|
| Device selection | `OVDeviceClass` + the `ovArches` table: every Intel device fak targets classified CPU / GPU / NPU, each with its native inference precision and a `PrimaryCap` naming the HAL `Caps` field it keys on. `LookupOVDevice`/`OVDeviceToken` normalize the noisy strings OpenVINO's `available_devices` reports (case, whitespace, the `GPU.0`/`GPU.1` instance ordinal) to one canonical plugin token, failing closed on an unsupported device. `IsVirtualOVDevice` recognizes the AUTO/HETERO/MULTI/BATCH meta-plugins (including the `AUTO:GPU,CPU` candidate-list form) as device-selection directives that delegate to physical devices — never a compile target. The CPU/GPU/NPU split is the load-bearing one. | `TestOVDeviceLookupKnown`, `TestOVDeviceClassInvariant`, `TestOVDeviceNormalization`, `TestOVVirtualDevices` |
| OpenVINO model export | The taxonomy records the target device + its `NativePrecision` (a real `compute.Dtype`: F32 on the CPU plugin, F16 on GPU/NPU) the IR export targets, so the deferred `Upload(t, NativePrecision)` narrows weights to a dtype the HAL already dispatches on, not a parallel vocabulary. fak exports its OWN in-process op-list to OpenVINO IR rather than importing an external ONNX graph — the same Dataflow-lens stance the TPU/Neural-Engine notes take. | `TestOVPrecisionIsHALDtype` |
| Runtime integration | The taxonomy is the always-compiled half; the cgo `//go:build openvino` half (registration as an `Approx` backend named `"openvino"`) plugs into the **existing** `compute.Register`/`Pick` seam unchanged — adding a backend is a registration, never a forward-loop edit, exactly as Vulkan and Metal did. | seam already proven by `cuda.go`/`vulkan.go`/`metal.go` |
| NPU support | `OVClassNPU` + `IsNPU()` + `FixedGraph()`: the Intel NPU is the unique reach of this backend, classified as the whole-model ahead-of-time-compiled fixed graph (`PrimaryCap()=="GraphCompile"`) that the deferred backend stages as a precompiled device blob. The host-independent classification is shipped; the on-NPU run is the deferred half. | `TestOVDeviceClassInvariant` |
| Benchmark (within 1.5× native CPU) | No new gate needed — the within-N acceptance reuses the existing `WithinTarget(measured, baseline, factor)` (prefill.go). The device node feeds `WithinTarget(openvinoSeconds, cpuRefSeconds, 1.5)`; the 1.5× rule lives in one tested place. | `prefill.go:WithinTarget` (existing) |

`go build ./...` (default, non-OpenVINO) is clean; `go test ./internal/compute/ -run OV` is green on
the CPU path (run under WSL / an isolated archive on the native-Windows dev box, per AGENTS.md).

## Why this is NOT the Intel XPU / oneDNN-SYCL lens (#264)

Both reach Intel hardware, but through different stacks and at different levels:

- **Intel XPU / oneDNN-SYCL (#264)** reaches an Intel **Arc discrete GPU** through the oneAPI/SYCL
  runtime with hand-lowered oneDNN primitives (matmul / SDPA / softmax onto a SYCL queue) — a
  structural sibling of the CUDA/Vulkan backends, op-by-op.
- **OpenVINO (#257, here)** is Intel's higher-level **inference runtime**: it ingests an IR and
  dispatches the whole model across the CPU / GPU / NPU plugins, with AUTO/HETERO/MULTI doing the
  device selection. Its load-bearing decision is the plugin/device selection, and its unique reach
  is the **NPU** — which oneDNN-SYCL does not target.

So this is a distinct backend, not a duplicate of the XPU lens.

## The integration strategy (for the Intel node)

1. `internal/compute/openvino.go` — `//go:build openvino`, an `Approx` backend named `"openvino"`
   over the OpenVINO C API. It exports the recorded in-process op-list to an OpenVINO IR (`ov::Model`),
   compiles it for the selected device via `core.compile_model(model, OVDeviceToken(dev))`, and runs
   inference. `cpu-ref` stays the `Reference` Default so nothing runs through OpenVINO unless
   explicitly selected (`FAK_BACKEND=openvino` + a device token / `--backend openvino --device NPU`).
   The host already reports an unbuilt backend honestly (`serve.go`: needs both a matching build tag
   and a reachable device).
2. The device caps follow the taxonomy: a discrete GPU advertises `Caps.DeviceMemory` (+ `Async`);
   the NPU advertises `Caps.GraphCompile` (it compiles the whole IR to a static blob); the CPU plugin
   is the programmable parity floor (the within-1.5× baseline).
3. The op-level + full-forward witnesses (`-tags openvino`) mirroring `cuda_test.go` /
   `vulkan_test.go`: argmax EXACT, the rest cosine + small max|Δ| (`Approx`).

## The named blocker (do NOT fake this on a host with no OpenVINO toolchain / no Intel NPU)

The four acceptance bullets

- [ ] Run via OpenVINO
- [ ] Within 1.5× native CPU
- [ ] NPU support
- [ ] Documentation  ← satisfied by this notes file + the OpenVINO lens in
      `docs/explainers/hardware-portability.md`

The first three are **measurements on Intel silicon through the OpenVINO runtime** (an IR compiled
and run on the CPU/GPU plugin, the within-1.5×-CPU throughput grade, and a run on a real Intel NPU).
They cannot be witnessed on this win32 host (no OpenVINO runtime, no Intel NPU). **No OpenVINO
throughput or correctness number was run, estimated, or fabricated here** — the shipped deliverable
is the exact, hardware-independent device-plugin taxonomy + device selection + the documentation.
Dependencies: none (per the issue).

## Next agent, on an Intel node (OpenVINO runtime installed)

1. Confirm the toolchain + devices: install the OpenVINO runtime, then enumerate
   `core.available_devices` → it reports `"CPU"`, `"GPU"`/`"GPU.0"`/`"GPU.1"`, `"NPU"`. Feed each to
   `OVDeviceToken` (it accepts the case noise and the `.N` instance ordinal) to get the canonical
   plugin token; check `IsVirtualOVDevice` for an AUTO/HETERO/MULTI directive.
2. Write `openvino.go` (`//go:build openvino`) per the strategy above; register the `"openvino"`
   Approx backend. The Register/Pick seam and the IR-export-then-compile flow are the only new parts;
   the device selection is already decided by the taxonomy.
3. Witness numerical parity op-by-op (mirror `vulkan_test.go`) then the full forward
   (`-run HALOpenVINO`); hold `argmax` EXACT, the rest to cosine + small max|Δ| (`Approx`).
4. Benchmark vs the `cpu-ref` baseline on the same node and grade with
   `WithinTarget(openvinoSeconds, cpuRefSeconds, 1.5)`. For the NPU bullet, run the same parity +
   throughput pass on a Meteor/Lunar/Arrow Lake NPU. Record lineage (OpenVINO version + device +
   UTC + commit + sanitized machine + harness), run on a node NOT also running the agent (placement
   law).
