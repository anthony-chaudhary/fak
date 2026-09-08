# Pure-Go host bindings: remove CGo control shims, retain native device kernels

**Observed:** 2026-09-08  
**Decision:** ADAPT incrementally behind fak-owned interfaces  
**Tracker:** [#12210](https://github.com/anthony-chaudhary/fak/issues/12210)  
**First spine:** [#12212](https://github.com/anthony-chaudhary/fak/issues/12212), shipped as `52ac2d962d24dd7d15678d001b2aea8c02321c14`  
**Existing CUDA loader leaf:** [#11657](https://github.com/anthony-chaudhary/fak/issues/11657)  
**Receipt:** `study_3cd46be3b861ab9872e7c8cb01f81def505a9cecbbc6cf80e728087c689558dc`

## Verdict

Make fak's optional accelerator **host/control plane** build with the Go toolchain and
`CGO_ENABLED=0`. Do not translate GPU device programs into Go, obscure vendor driver
identity, or silently change an explicitly selected accelerator into a CPU or llama.cpp
route. CUDA/PTX/cubin, Metal Shading Language, GLSL/SPIR-V, and Go assembly remain
legitimate implementation artifacts; the migration target is the C/C++/Objective-C
host binding layer and its CGo build dependency.

The current default/release binary is already CGo-free. This program removes the
remaining CGo tax from opt-in acceleration tags while preserving fak-native execution,
typed failures, resource lifetime discipline, and receipts that name the real engine.

## For / Problem / Today / Better because / Witness

- **For:** users and operators who want the one-binary fak path plus optional native
  accelerators without a platform C/C++ toolchain.
- **Problem:** several optional CUDA, Vulkan, Metal, Accelerate, and IOKit paths still
  cross CGo shims, increasing build variance and hiding lifetime and thread-affinity
  contracts behind compiled glue.
- **Today:** the ordinary binary is pure Go, but accelerator tags can pull in 37
  production `import "C"` files and multiple host shims.
- **Better because:** the Go binary owns loading, symbol resolution, errors, lifetime
  ledgers, and receipts directly while continuing to run the same native device code.
- **Witness:** each migrated leaf must build with `CGO_ENABLED=0`, prove zero CGo files,
  execute a real device/driver path, and fail with a typed error when the requested
  accelerator is unavailable. Unit-only or silent-fallback evidence is insufficient.

## Repository inventory

Tracked native/device-language files total **99**: 2 `.c`, 3 `.cpp`, 5 `.cu`, 6 `.h`,
12 `.m`, 3 `.metal`, 44 `.comp`, and 24 Go assembly `.s`. There are no tracked `.a`,
`.o`, `.so`, `.dll`, `.dylib`, `.lib`, `.spv`, `.syso`, or `.wasm` artifacts.

Production `internal/` packages contain **37** direct `import "C"` files:

| Package | Files | Primary role | Migration disposition |
|---|---:|---|---|
| `internal/compute` | 19 | CUDA, Vulkan, and Metal host bindings | Migrate backend-by-backend; Vulkan first, CUDA under #11657 |
| `internal/metalgemm` | 14 | Objective-C/MPS Metal GEMM control | Split basic Metal ABI/lifetime proof from MPSMatrixMultiplication |
| `internal/model` | 2 | CUDA AWQ and Apple Accelerate helpers | Reuse backend loaders; measure Accelerate replacement independently |
| `internal/power` | 2 | Darwin IOKit listener and wakelock | Early pure-Go FFI leaf after callback/thread proof |

The mobile C archive is an intentional consumer ABI, and the optional Python L3
RDMA/CXL client is a separate integration boundary. Neither is permitted to dictate
the core runtime's build mode. Synthetic fixture strings are not production CGo.

## Source-pinned prior art

All source observations below were read at the named revision; API shape and project
activity may move after the observation date.

| Source | Pin and license | What was actually read | Transfer boundary |
|---|---|---|---|
| `ebitengine/purego` | [`72b0dcd291d424a09c2a56411ac60f488bacbe7d`](https://github.com/ebitengine/purego/tree/72b0dcd291d424a09c2a56411ac60f488bacbe7d), Apache-2.0 | Tier-1 platform support and dynamic-call contracts | Useful portable FFI substrate; fak still owns signatures, structure padding, callbacks, lifetimes, and typed errors |
| `townsendmerino/aikit` | [`c214070f53c0675202885c231747bd8b2e59af44`](https://github.com/townsendmerino/aikit/tree/c214070f53c0675202885c231747bd8b2e59af44), MIT | `gpu/metal.go`, real vadd smoke, and Metal leak tests | Strong Metal bootstrap precedent; does not prove fak's MPS GEMM seam |
| `eitamring/gocudrv` | [`498ca99de542a2e9bd1ce15b57dcc1263dd65a46`](https://github.com/eitamring/gocudrv/tree/498ca99de542a2e9bd1ce15b57dcc1263dd65a46), MIT | dynamic loader, CUDA Driver API binding, executor, PTX vadd, and no-CGo test script | Adapt mechanisms behind a fak-owned adapter; no cuBLAS/NCCL or kernel compiler is provided and the API is not treated as frozen |
| `townsendmerino/goinfer` | [`78671469f4bc8562088ede3342022a2f2c12640a`](https://github.com/townsendmerino/goinfer/tree/78671469f4bc8562088ede3342022a2f2c12640a), MIT | application-scale accelerator cleanup and close/leak tests | Evidence that lifecycle proof scales; Go 1.27 makes it unsuitable as a direct dependency today |

### Metal details worth adapting

At `aikit@c214070`, `gpu/metal.go` loads Metal and calls
`MTLCreateSystemDefaultDevice` without CGo (lines 41-49), keeps explicit object and
buffer release ledgers (96-141), declares the MSL language version (165-182), passes
the `MTLSize` ABI-sensitive value by reference (308-312), builds queues, pipelines,
and buffers (341-420), and performs real pipeline dispatch (575-605). Autorelease
pools are bounded while the goroutine is pinned to its OS thread (668-681, 817-827).
`gpu/smoke_test.go` proves real vector addition; `gpu/metal_leak_test.go` checks drain,
idempotence, and isolation. Fak should copy the ownership discipline, not the package
API, and should keep MPS as its own risk-gated leaf.

### CUDA details worth adapting

At `gocudrv@498ca99d`, `internal/dynload` loads the real CUDA driver library;
`cudasys/syscall_bind.go` binds module load and launch; the executor pins CUDA context
work to an OS thread and quarantines foreign TLS; the integration test runs a real PTX
vector addition; and the no-CGo script checks the build boundary. This is a credible
host-control precedent, not a kernel implementation. Fak must separately bind and
witness cuBLAS, NCCL, module/argument lifetimes, and its engine receipts.

## Migration map

| Order | Leaf | Current status | Required witness |
|---:|---|---|---|
| 1 | Windows Vulkan loader and symbol probe | **SHIPPED** in #12212 / `52ac2d9` | Native Windows `CGO_ENABLED=0` call to System32 `vulkan-1.dll`, `VK_SUCCESS`, Vulkan 1.4.341 decode, typed failure cases, `CgoFiles=[]` |
| 2 | Vulkan instance and physical-device selection | **OPEN** #12283 | Real device identity, compute queue-family discovery, deterministic choice, repeated teardown |
| 3 | Vulkan logical-device and queue lifecycle | **OPEN** #12284 | Real create/get-queue/wait-idle/close loop plus typed partial-init unwind |
| 4 | Vulkan mapped-buffer round trip | **OPEN** #12285 | Exact sentinel bytes, allocation failure unwind, and double-close under no CGo |
| 5 | Vulkan SPIR-V add dispatch | **OPEN** #12286 | Real no-CGo add dispatch, bounded result parity, object drain, and explicit engine identity |
| 6 | CUDA Driver API loader | **PARTIAL**, tracked by #11657 | Real PTX launch, explicit driver identity, no silent fallback, no CGo files |
| 7 | Metal device/queue/pipeline/vadd bootstrap | **ABSENT** | Apple Silicon real dispatch, autorelease/thread/lifetime tests, no CGo files |
| 8 | Darwin IOKit power bindings | **ABSENT** | Real assertion/listener lifecycle and callback-thread proof |
| 9 | Accelerate and MPS high-level math | **ABSENT / high risk** | Matched correctness and performance evidence before removing the measured native exception |
| 10 | cuBLAS/NCCL and remaining CUDA host APIs | **ABSENT / very high risk** | Separate ABI, lifecycle, collective-failure, and performance receipts |

The order deliberately proves loader, ABI, lifetime, and typed-failure primitives before
touching high-risk GEMM or collective paths. Each backend remains explicit: missing
native support returns a typed refusal instead of selecting another engine.

## Candidate decisions

| Candidate | FAK seam | Disposition | Why |
|---|---|---|---|
| Standard-library Windows DLL loading | Vulkan host bootstrap | **IMPLEMENT first** | Smallest real spine; platform precedent already exists in fak |
| `purego` portable calls | Metal, Darwin frameworks, Linux driver bindings | **ADAPT behind internal interface** | Removes CGo but does not remove ABI/lifetime responsibility |
| `gocudrv` design | CUDA Driver API | **ADAPT under #11657** | Strong real-device and context-affinity evidence; keep fak API/provenance ownership |
| Rewrite device shaders/kernels in Go | CUDA/MSL/SPIR-V payloads | **REJECT** | Confuses host build purity with device execution and would weaken native performance ownership |
| Silent CPU/llama.cpp recovery | Accelerator selection | **REJECT** | Violates the native-inference and explicit-route contracts |
| Directly depend on `goinfer` | Whole runtime | **WATCH** | Useful lifecycle evidence, incompatible Go version and wrong ownership boundary |

## Completeness critique

The inventory covers tracked production CGo and native/device-language extensions, and
the prior-art study covers portable FFI, Metal lifecycle/dispatch, CUDA driver loading,
and application-scale cleanup. It does **not** yet prove Vulkan device dispatch, Darwin
framework callbacks, MPS structure ABIs, cuBLAS/NCCL callbacks and collectives, or
performance parity under fak workloads. Those gaps are intentionally tickets with
real-device witnesses, not claims inferred from loader success.

This study also does not redefine "pure Go" to mean "no non-Go bytes anywhere." GPU
instruction payloads and Go assembly stay visible, attributed, and receipt-bound. The
promise is a Go-toolchain-owned host/control plane with no hidden CGo dependency.
