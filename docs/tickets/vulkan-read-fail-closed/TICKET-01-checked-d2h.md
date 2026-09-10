# fix(compute): fail closed when Vulkan device readback fails

<!-- fak-compute-key: vulkan-d2h-read-fail-closed -->

GitHub issue: [#12709](https://github.com/anthony-chaudhary/fak/issues/12709).

```routing
lane: compute
paths: ["internal/compute/vulkan.go", "internal/compute/vulkan_backend.h", "internal/compute/vulkan_shim.cpp", "internal/compute/vulkan_device_debug.go", "internal/compute/vulkan_read_checked_test.go", "docs/tickets/vulkan-read-fail-closed/TICKET-01-checked-d2h.md"]
expected_steps: 5
```

## Parent context

Related feature issue: #12687. That target-verification work exposed the silent
readback failure mode, while this ticket remains a separate general Vulkan
backend correctness repair.

## Current state

At public revision `d0ca6b55e81834809d1329a1e7f4fdb89b8022f1`,
`internal/compute/vulkan_backend.h:51` declares `fvk_d2h` with a `void`
return type. `internal/compute/vulkan_shim.cpp:886-899`
(`copyDeviceToHost`) returns without copying when its staging allocation fails,
and `:1559-1562` (`fvk_d2h`) cannot report that failure. The Go caller at
`internal/compute/vulkan.go:1038-1047` (`vulkanBackend.Read`) therefore returns
the newly allocated zero-filled slice as though it contained device data.

This is the deterministic pre-fix source witness. The post-fix staging and
pre-existing sticky-status paths were subsequently exercised on physical Vulkan
hardware as recorded below; same-call native submit/fence aborts remain outside
the repair.

## Why now

Vulkan readback is the host fence for logits and other device-resident tensors.
A silent zero-filled return can turn a backend resource or submission failure
into plausible numerical output and conceal the real device error from callers.

## Problem frame

- Centrality: Core.
- P1 Context: advanced - preserve the native D2H failure at the public compute boundary.
- P2 Net value: advanced - prevent invalid zero data from reaching model decisions.
- P3 Adaptation: preserved - keep the existing Go `Read(Tensor) []float32` API.
- P4 Operations: advanced - classify allocation, device-loss and submission failures.

For Vulkan inference callers / Problem: failed device readback can look like valid
zeros / Today: the native ABI has no status and the Go caller cannot fail closed /
Better because: the original failure becomes a typed backend error before data is
returned / Witness: source audit plus an independently authored tagged Vulkan test.

## Working spine

1. Make the native D2H helper return a status without changing the Go backend API.
2. Return success only after the requested bytes reach host memory.
3. Surface the pre-existing sticky native status and use one explicit status when staging allocation has no native cause.
4. Map the status to the existing typed `BackendError` classes at `Site: Read`.
5. Exercise both sticky submission failure and one-shot staging failure on a Vulkan device.

## Core through-line

Native staging or submission failure -> `fvk_d2h` status ->
`vulkanBackend.Read` -> direct typed `BackendError` panic -> no tensor slice is
returned as valid data.

The status contract is narrow: zero means the requested bytes reached host
memory. A pre-existing sticky `VK_ERROR_DEVICE_LOST` maps to `ErrVulkanDeviceLost` and
`VulkanClassDeviceLost`; host/device allocation failures and the explicit
unattributed staging-allocation status map to `ErrVulkanAllocationFailed` and
`VulkanClassAllocationFailed`; other nonzero statuses map to
`ErrVulkanSubmissionFailed` and `VulkanClassSubmissionFailed`.

## Gold-plating boundary

No asynchronous read API, retry loop, buffer-pool redesign, transfer scheduler,
model-path changes, performance claim, or speculative-decoding policy change.
Other direct native D2H call sites are outside this bounded leaf unless the ABI
signature requires a mechanical compile adjustment. Same-call Vulkan
submit/fence failures still follow the existing native `VKCHECK` abort behavior;
this leaf does not make those failures recoverable through Go.

## Done condition

- [x] [SW-VERIFIED] `fvk_d2h` returns zero only after the requested bytes are copied to host memory.
- [x] [SW-VERIFIED] A staging allocation failure cannot return a zero-filled slice as successful tensor data.
- [x] [SW-VERIFIED] `vulkanBackend.Read` fails with a direct `*BackendError` at `Site: Read`.
- [x] [SW-VERIFIED] Returned sticky device-loss, allocation and other nonzero statuses retain distinct existing classes and sentinels.
- [x] [SW-VERIFIED] The one-shot staging fault is consumed: the first read fails and the next read returns the original nonzero values.
- [x] [HW-WITNESSED] The tagged independent test passes on a physical Vulkan device.

## Definition of done

The five compute files land with independent tests and a signed issue reference.
Software completion requires native compilation and the focused test. Physical
completion requires the tagged test on a real Vulkan device; source inspection
or an in-memory fake cannot satisfy that checkbox.

## Witness

The witness is the pre-fix ABI/control-flow audit followed by the independently
authored `TestVulkanReadFailsClosed` tagged test on a physical Vulkan device.

### Verifiable Witness

Pre-fix deterministic source witness at
`d0ca6b55e81834809d1329a1e7f4fdb89b8022f1`:

```text
git grep -n 'fvk_d2h' d0ca6b55e81834809d1329a1e7f4fdb89b8022f1 -- internal/compute/vulkan.go internal/compute/vulkan_backend.h internal/compute/vulkan_shim.cpp
git blame -L 886,899 d0ca6b55e81834809d1329a1e7f4fdb89b8022f1 -- internal/compute/vulkan_shim.cpp
```

The first command shows a `void` native ABI and a Go call with no status check.
The second shows the staging-allocation branch returning before `memcpy`.

Post-fix behavioral witness:

```text
GOWORK=off go test -tags vulkan ./internal/compute -run '^TestVulkanReadFailsClosed$' -count=1
```

`TestVulkanReadFailsClosed` uses isolated `sticky-submission` and `d2h-staging`
child modes. It first round-trips known nonzero values, then asserts that a sticky
device-loss status produces the device-loss class and sentinel. The one-shot
staging injection must make the first read fail with the allocation class and
sentinel, while the second read returns the original values exactly.

Physical execution status: passed on an independent RX7600 Vulkan runner. Both
`sticky-submission` and `d2h-staging` child modes passed against the frozen five
source blobs. The device receipt digest is
`sha256:b31b650a26019b78cd664497d02a0116bcc57210192dbbfdc72acec6d68ac591`.
This witness covers the checked staging and pre-existing sticky-status paths; it
does not change or qualify same-call native submit/fence abort behavior.

### File:Line seams

- `internal/compute/vulkan.go:1038-1047` (`vulkanBackend.Read`)
- `internal/compute/vulkan_backend.h:51` (`fvk_d2h` ABI)
- `internal/compute/vulkan_shim.cpp:886-899` (`copyDeviceToHost`)
- `internal/compute/vulkan_shim.cpp:1559-1562` (`fvk_d2h`)
- `internal/compute/vulkan_device_debug.go:248-265` (native failure injection seam)

### Blast radius and fallback

One public package, `internal/compute`, and its Vulkan C ABI shim. CPU and Metal
backends are unaffected. Existing successful reads retain the same Go API and
data path. Same-call native submit/fence abort behavior is unchanged. Before the
fix lands, callers may avoid treating fault-injected Vulkan readback as valid
evidence; no fallback value is fabricated by the repair.

## Likely files

- `internal/compute/vulkan.go`
- `internal/compute/vulkan_backend.h`
- `internal/compute/vulkan_shim.cpp`
- `internal/compute/vulkan_device_debug.go`
- `internal/compute/vulkan_read_checked_test.go`

## Lane

compute; one package; five implementation steps plus this tracking ticket.

## Acceptance gate

The source witness remains reproducible, C++ and tagged Go builds pass, the
independent fault-injection test passes on physical Vulkan hardware, formatting
and diff checks pass, and the public issue text passes leak scrubbing.

## Closure binding

The independent physical test raises this repair to an integrated,
hardware-witnessed standard. Close after one signed public resolving commit
carries the separate issue reference.

## Work estimate

Estimate: 1 point. Uncertainty: the physical fault-injection run may expose a
driver-specific status that needs one additional mapping case.

## Overall completion contribution

Contribution: 1/1 point for this standalone backend-correctness leaf.

## Completion standard

integrated
