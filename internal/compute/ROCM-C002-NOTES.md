# ROCm backend (AMD Linux) — C-002 / issue #266 — scaffold + named blocker

**Status:** host-tractable scaffold SHIPPED (the device-arch taxonomy, exact + unit-witnessed
on any host). **Blocked:** the HIP kernel compilation and every acceptance bullet that runs on
silicon are deferred to an AMD-GPU-on-Linux node — this dev host has no AMD GPU and no ROCm
toolchain (`hipcc`).

## What shipped here (compute lane, `internal/compute/rocm_arch.go` + test)

The issue's scope is a 14–21-day GPU backend; none of its acceptance bullets can be witnessed
without AMD silicon. What *can* be built and proven on any host is the part the HIP build needs
**before** a kernel ever runs: the device-arch taxonomy. That is shipped here, exact and tested.

| Scope bullet (#266) | Shipped here | Witnessed by |
|---|---|---|
| CDNA/RDNA support | `ROCmFamily` + the `KnownROCmArches` table: every supported AMD target (`gfxNNNN`) classified GCN5 / CDNA1-3 / RDNA1-3, each with the family's native wavefront (64 on GCN/CDNA, 32 on RDNA) and a `Datacenter()` predicate. The CDNA↔RDNA split is the load-bearing one for kernel tuning. | `TestROCmArchLookupKnown`, `TestROCmCDNARDNAInvariant` |
| HIP kernel compilation | `ROCmOffloadArch(gfx)` returns the canonical `hipcc --offload-arch=<gfx>` token, normalizing the noisy device strings ROCm reports (case, whitespace, `:sramecc+:xnack-` feature suffix) and failing closed on an unsupported part — the exact input the deferred `build_rocm.sh` feeds hipcc. | `TestROCmOffloadArchNormalization` |
| ROCm integration | The arch table is the always-compiled half of the backend; the cgo `//go:build rocm` half (registration as an `Approx` backend named `"rocm"`) plugs into the **existing** `compute.Register`/`Pick` seam unchanged — adding a backend is a registration, never a forward-loop edit, exactly as Vulkan and Metal did. | seam already proven by `cuda.go`/`vulkan.go`/`metal.go` |
| Benchmark vs Vulkan | No new gate needed — the within-N acceptance reuses the existing `WithinTarget(measured, baseline, factor)` (prefill.go). The device node feeds `WithinTarget(rocmSeconds, vulkanSeconds, 1.5)`; the 1.5× rule lives in one tested place. | `prefill.go:WithinTarget` (existing) |

`go build ./...` (default, non-ROCm) is clean; `go test ./internal/compute/ -run ROCm` is green
on the CPU path (run under WSL / an isolated archive on the native-Windows dev box, per AGENTS.md).

## The HIP-reuse strategy (for the AMD node)

HIP is source-compatible with CUDA by design (`hipMalloc`↔`cudaMalloc`, …), and AMD ships
`hipify` to translate CUDA→HIP. The mature path — the one llama.cpp's ROCm backend takes — is to
compile the **existing** `cuda_kernels.cu` through `hipcc`/`hipify` rather than hand-write a
second kernel set. So the deferred backend is expected to be:

1. `internal/compute/rocm.go` — `//go:build rocm`, a near-clone of `cuda.go`'s cgo wrapper that
   registers an `Approx` backend named `"rocm"` over a flat C ABI (`rocm_backend.h`), keeping
   `cpu-ref` the `Reference` Default so nothing runs on the GPU unless explicitly selected
   (`FAK_BACKEND=rocm` / `--backend rocm`). The host already reports an unbuilt backend honestly
   (`serve.go`: "needs both a matching build tag (e.g. `-tags rocm`) and a reachable device").
2. `build_rocm.sh` — the `build_cuda.sh` twin: `hipify` (or hipcc directly) over `cuda_kernels.cu`
   → `librocfak.a`, with `--offload-arch` taken from `ROCmOffloadArch(<device gfx>)`.
3. The op-level + full-forward witnesses (`-tags rocm`) mirroring `cuda_test.go` / `vulkan_test.go`.

## The named blocker (do NOT fake this on a host with no AMD GPU)

The four acceptance bullets

- [ ] Run on AMD GPU on Linux
- [ ] Within 1.5× Vulkan throughput
- [ ] Bit-exact results
- [ ] Multi-GPU support

are all **measurements on AMD silicon under ROCm**. They cannot be witnessed on this host (no AMD
GPU, no `hipcc`). **No ROCm throughput or correctness number was run, estimated, or fabricated
here** — the shipped deliverable is the exact, hardware-independent arch taxonomy + offload-target
selection. The dependency #282 (B-004, continuous batching) is CLOSED, so it is not a blocker.

Useful prior art on AMD: the **Vulkan** backend already executes the full forward pass on a real
AMD Radeon RX 7600 (gfx1102 / RDNA3) with **numerical parity** (argmax-exact, prefill cosine 1.0)
— see `docs/benchmarks/VULKAN-AMD-RESULTS.md`. That is the cross-vendor AMD path and the exact
throughput baseline this ROCm backend benchmarks against; ROCm is the AMD-**native** path whose
point is to beat Vulkan's per-dispatch tax (Vulkan-native ≈ 2 ms/op vs CUDA-on-WSL ≈ 0.07 ms/op).

## Next agent, on an AMD-GPU-on-Linux node (ROCm installed)

1. Confirm the toolchain: `hipcc --version`, `rocminfo | grep gfx` → feed the reported `gfxNNNN`
   to `ROCmOffloadArch` (it accepts the `:sramecc+:xnack-` suffix `rocminfo` may print).
2. Write `rocm.go` (`//go:build rocm`) + `rocm_backend.h` + `build_rocm.sh` as above; register the
   `"rocm"` Approx backend. Mirror `cuda.go` — the C ABI seam and the registration are identical.
3. Witness numerical parity op-by-op (mirror `vulkan_test.go`'s 7-op suite) then the full forward
   (`-run HALRocm`); hold `argmax` to EXACT, the rest to cosine + small max|Δ| (`Approx`).
4. Benchmark vs the Vulkan baseline on the same card and grade with
   `WithinTarget(rocmSeconds, vulkanSeconds, 1.5)`. Record lineage (ROCm version + UTC + commit +
   sanitized machine + harness), run on a node NOT also running the agent (placement law).
5. Multi-GPU: implement `CollectiveBackend` (the cross-rank seam in compute.go) over RCCL; the
   single-rank identity is witnessable on one card before any second GPU is needed.
