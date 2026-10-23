//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"unsafe"
)

// C ABI format selectors of fcuda_routed_expert_gemv_e8m0.
const (
	cudaRoutedExpertFP8Block32 = 0
	cudaRoutedExpertMXFP4      = 1
)

// cudaRoutedExpertTiming is the per-call phase split the microbench reads: host-wall H2D of
// the packed weight/scale/x bytes, device-event kernel time, host-wall D2H of y.
type cudaRoutedExpertTiming struct {
	H2DMillis, KernelMillis, D2HMillis float64
}

// GemvExpertFP8Block32E8M0 runs a routed-expert float8_e4m3fn GEMV with one raw E8M0 scale
// per 32x32 tile on the device. The PACKED weight bytes are uploaded as-is and decoded inline,
// and y is bit-identical to GemvFP8Block32E8M0 (a NaN output is NaN on both sides). Inputs the
// oracle refuses are refused here by the same predicate before any device work.
func (c *cudaBackend) GemvExpertFP8Block32E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	return c.routedExpertGemv(cudaRoutedExpertFP8Block32, O, I, weight, scales, x, nil)
}

// GemvExpertMXFP4E8M0 runs a routed-expert MXFP4 GEMV (I8-packed E2M1, low nibble = even K,
// row-major [O,ceil(I/32)] E8M0 scales) on the device, bit-identical to GemvMXFP4E8M0 and
// refusing exactly the inputs it refuses.
func (c *cudaBackend) GemvExpertMXFP4E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	return c.routedExpertGemv(cudaRoutedExpertMXFP4, O, I, weight, scales, x, nil)
}

// routedExpertGemv validates with the oracle's own acceptance predicate, then makes one
// synchronous device call. It never falls back to the scalar oracle: a refused or faulted
// call returns a nil y and a non-nil error. A non-nil timing receives the phase split.
func (c *cudaBackend) routedExpertGemv(format, O, I int, weight, scales []byte, x []float32, timing *cudaRoutedExpertTiming) ([]float32, error) {
	op, site := "GemvExpertFP8Block32E8M0", "routed-expert-fp8-block32"
	if format == cudaRoutedExpertMXFP4 {
		op, site = "GemvExpertMXFP4E8M0", "routed-expert-mxfp4"
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return nil, err
	}
	var err error
	if format == cudaRoutedExpertMXFP4 {
		_, err = checkGemvMXFP4E8M0("cuda", O, I, weight, scales, x)
	} else {
		_, err = checkGemvFP8Block32E8M0("cuda", O, I, weight, scales, x)
	}
	if err != nil {
		return nil, err
	}

	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c.capturing {
		return nil, &CUDAOpError{Op: op, Site: site, Msg: "refused while a CUDA graph capture is open: the per-call upload and read-back are not capturable"}
	}
	y := make([]float32, O)
	var phases [3]C.double
	var phasesPtr *C.double
	if timing != nil {
		phasesPtr = &phases[0]
	}
	status := int(C.fcuda_routed_expert_gemv_e8m0(
		C.int(format),
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(weight))), C.size_t(len(weight)),
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(scales))), C.size_t(len(scales)),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(x))),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(y))),
		C.int64_t(O), C.int64_t(I), phasesPtr,
	))
	switch {
	case status == 0:
	case status == -3:
		allocErr := &DeviceAllocError{Bytes: len(weight) + len(scales) + 4*(len(x)+O), Site: site, Class: MemoryScratchpad}
		c.faultLatch.ObserveError(allocErr, site)
		return nil, allocErr
	case status < 0:
		return nil, &CUDAOpError{Op: op, Site: site, Code: status, Msg: fmt.Sprintf("device ABI refused the call (status %d)", status)}
	default:
		c.faultLatch.Observe(DeviceFaultExecution, site, status)
		return nil, &CUDALaunchError{CUDAOpError{Op: op, Site: site, Code: status, Msg: "routed-expert GEMV copy, launch or execution failed"}}
	}
	// The call fenced g_stream before its read-back, so every earlier enqueue is complete.
	atomic.AddUint64(&c.fenceGen, 1)
	if timing != nil {
		*timing = cudaRoutedExpertTiming{H2DMillis: float64(phases[0]), KernelMillis: float64(phases[1]), D2HMillis: float64(phases[2])}
	}
	return y, nil
}

// cudaRoutedExpertLaunches reports completed routed-expert GEMV kernels: the witness that a
// refused call did no device work and an accepted one ran exactly one launch.
func cudaRoutedExpertLaunches() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_routed_expert_gemv_launches())
}

// C ABI kinds of fcuda_routed_expert_gemv_test_fault.
const (
	cudaRoutedExpertFaultNone   = 0
	cudaRoutedExpertFaultAlloc  = 1
	cudaRoutedExpertFaultLaunch = 2
)

// routedExpertInjectFaultForTest arms a one-shot C-side failure of the next routed-expert
// GEMV call (allocation -> status -3, launch -> a positive cudaError with no kernel run);
// cudaRoutedExpertFaultNone disarms. Unexported like qwen35GDNInjectFaultForTest: only
// package-local -tags cuda tests reach the fail-closed paths.
func routedExpertInjectFaultForTest(kind int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_routed_expert_gemv_test_fault(C.int(kind))
}

var _ RoutedExpertDeviceKernel = (*cudaBackend)(nil)
