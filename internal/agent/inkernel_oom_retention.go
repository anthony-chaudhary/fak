package agent

import (
	"context"
	"errors"
	"log"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inKernelOOMReleaseSite names this boundary in device-fault latch transitions.
const inKernelOOMReleaseSite = "inkernel-oom-release"

// classifiedDeviceOOM reports whether err is a device allocation miss: the planner's
// InKernelOOMError, or a DeviceAllocError carried inside a typed forward failure (a
// model.BackendForwardOperationError around a compute.Qwen35SequenceError, for example).
// recoverDevicePanic returns those forward failures as they are, so the retry seam never
// sees them as InKernelOOMError and never trims after them.
func classifiedDeviceOOM(err error) bool {
	if err == nil {
		return false
	}
	var oom *InKernelOOMError
	if errors.As(err, &oom) {
		return true
	}
	var alloc *compute.DeviceAllocError
	return errors.As(err, &alloc)
}

// releaseFailedOOMRequest runs after a request's final device-allocation failure. The failed
// attempt returns its transients to the device pools, which stay cached at their failure-time
// sizes (about 1 GiB on the Raider 5090 Qwen3.8-27B cold prefill). Every later request then
// starts with less free VRAM than idle. This drops those idle pools, then reopens a fault latch
// that only allocation misses poisoned, once the backend proves its context still fences.
// Execution, launch and context faults stay latched.
//
// It declines when a shared coalesced decode pass or a speculative engram selection may still
// own device state. trimBackendIdlePools also declines a busy execution owner.
func (p *InKernelPlanner) releaseFailedOOMRequest(ctx context.Context, err error) {
	if p == nil || p.backend == nil || !classifiedDeviceOOM(err) {
		return
	}
	if attempt, _ := ctx.Value(inKernelEngramAttemptContextKey{}).(*inKernelEngramAttempt); attempt != nil && attempt.selected {
		return
	}
	if req, ok := ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest); ok && req.decodePass.Load() > 0 {
		return
	}
	trimmed := p.trimBackendIdlePools()
	cleared := false
	if latch := compute.BackendFaultLatch(p.backend); latch != nil && latch.Health().Refusing() {
		if prober, ok := p.backend.(compute.DeviceContextProber); ok {
			cleared = latch.ClearAllocationSuspect(inKernelOOMReleaseSite, prober.DeviceContextProbe) == nil
		}
	}
	if trimmed || cleared {
		log.Printf("inkernel_chat oom-release model=%s backend=%s trimmed=%t cleared_allocation_suspect=%t",
			p.modelID, p.backend.Name(), trimmed, cleared)
	}
}
