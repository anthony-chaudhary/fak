//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#include "vulkan_backend.h"
*/
import "C"
import "fmt"

// BackendCounters reads the same cumulative counters as
// BackendExecutionSnapshot without its identity strings, memory-budget query
// or upload attribution, so a per-stage bracket can difference them cheaply.
func (v *vulkanBackend) BackendCounters() (BackendCounterSnapshot, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	dispatch := vulkanDebugDispatchProfileSnapshotLocked()
	if dispatch.Q4KMatmulDispatches > dispatch.ComputeDispatches {
		return BackendCounterSnapshot{}, fmt.Errorf("compute: Vulkan Q4_K dispatch count exceeds compute total")
	}
	var h2dCount, h2dBytes, d2hCount, d2hBytes, d2dCount, d2dBytes C.uint64_t
	if C.fvk_transfer_counters(&h2dCount, &h2dBytes, &d2hCount, &d2hBytes, &d2dCount, &d2dBytes) == 0 {
		return BackendCounterSnapshot{}, fmt.Errorf("compute: Vulkan transfer counters are unavailable")
	}
	goSide := [...]int64{v.q4kStagedCalls, v.q4kStagedBytes, v.q4kStageFallbacks, v.homeHits, v.homeMisses, v.homeBypasses, v.homeCopied}
	for _, value := range goSide {
		if value < 0 {
			return BackendCounterSnapshot{}, fmt.Errorf("compute: Vulkan backend counter is negative")
		}
	}
	return BackendCounterSnapshot{
		ComputeDispatches: dispatch.ComputeDispatches, Q4KMatmulDispatches: dispatch.Q4KMatmulDispatches,
		OtherDispatches: dispatch.ComputeDispatches - dispatch.Q4KMatmulDispatches,
		DispatchSubmits: dispatch.BatchSubmits + dispatch.OneShotSubmits,
		H2DBytes:        uint64(h2dBytes), H2DCount: uint64(h2dCount),
		D2HBytes: uint64(d2hBytes), D2HCount: uint64(d2hCount),
		D2DCopies: uint64(d2dCount), D2DBytes: uint64(d2dBytes),
		Q4KStageCalls: uint64(goSide[0]), Q4KStageBytes: uint64(goSide[1]), Fallbacks: uint64(goSide[2]),
		TensorHomeHits: uint64(goSide[3]), TensorHomeAdmissions: uint64(goSide[4]),
		TensorHomeBypasses: uint64(goSide[5]), TensorHomeCopiedBytes: uint64(goSide[6]),
	}, nil
}

// DispatchCountsLive reports the native profiler's startup decision: without
// FAK_VULKAN_DISPATCH_PROFILE=1 the dispatch and submit counts stay zero.
func (v *vulkanBackend) DispatchCountsLive() bool { return nativeVulkanDispatchProfileEnabled }

// DeviceSubmits reads the shim's always-on submit count without vulkanMu.
func (v *vulkanBackend) DeviceSubmits() uint64 { return uint64(C.fvk_device_submits()) }

// DeviceWorkCounters reads readVulkanDeviceWork's scalars without vulkanMu and
// without building the host-sync cells a per-stage bracket never uses.
func (v *vulkanBackend) DeviceWorkCounters() DeviceWork {
	var dispatches, readback, submitWait C.uint64_t
	C.fvk_device_work_counters(&dispatches, &readback, &submitWait)
	return DeviceWork{Dispatches: uint64(dispatches), ReadbackBytes: uint64(readback), SubmitWaitNS: uint64(submitWait)}
}
