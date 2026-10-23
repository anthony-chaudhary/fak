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

// cudaRoutedExpertResident is the CUDA RoutedExpertResidentWeight: the C-side id of pooled
// device copies of one projection's packed weight and E8M0 scale bytes. id, and the C table it
// names, are guarded by cudaMu; id is 0 once released. latch and faults pin the fault epoch the
// bytes were uploaded in, so a handle that predates an observed fault is refused as stale.
type cudaRoutedExpertResident struct {
	be     *cudaBackend
	format int
	o, i   int
	id     uint64
	latch  *DeviceFaultLatch
	faults int
}

func cudaRoutedExpertResidentNames(format int, verb string) (op, site string) {
	if format == cudaRoutedExpertMXFP4 {
		return verb + "ExpertMXFP4E8M0", "routed-expert-resident-mxfp4"
	}
	return verb + "ExpertFP8Block32E8M0", "routed-expert-resident-fp8-block32"
}

// UploadExpertFP8Block32E8M0 makes one float8_e4m3fn 32x32-tile E8M0 projection resident. It
// refuses exactly the weight/scale inputs GemvFP8Block32E8M0 refuses, before any device work.
func (c *cudaBackend) UploadExpertFP8Block32E8M0(O, I int, weight, scales []byte) (RoutedExpertResidentWeight, error) {
	h, err := c.uploadRoutedExpertResident(cudaRoutedExpertFP8Block32, O, I, weight, scales)
	if err != nil {
		return nil, err
	}
	return h, nil
}

// UploadExpertMXFP4E8M0 makes one MXFP4 E8M0 projection resident, refusing exactly the
// weight/scale inputs GemvMXFP4E8M0 refuses.
func (c *cudaBackend) UploadExpertMXFP4E8M0(O, I int, weight, scales []byte) (RoutedExpertResidentWeight, error) {
	h, err := c.uploadRoutedExpertResident(cudaRoutedExpertMXFP4, O, I, weight, scales)
	if err != nil {
		return nil, err
	}
	return h, nil
}

func (c *cudaBackend) uploadRoutedExpertResident(format, O, I int, weight, scales []byte) (*cudaRoutedExpertResident, error) {
	op, site := cudaRoutedExpertResidentNames(format, "Upload")
	// Pin the epoch before Admit: a fault observed at any point after this upload began
	// makes the handle stale.
	latch := c.faultLatch
	faults := latch.faultCount()
	if err := c.faultLatch.Admit(site); err != nil {
		return nil, err
	}
	var err error
	if format == cudaRoutedExpertMXFP4 {
		_, err = checkMXFP4E8M0Weight("cuda", O, I, weight, scales)
	} else {
		_, err = checkFP8Block32E8M0Weight("cuda", O, I, weight, scales)
	}
	if err != nil {
		return nil, err
	}

	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c.capturing {
		return nil, &CUDAOpError{Op: op, Site: site, Class: MemoryWeights, Msg: "refused while a CUDA graph capture is open: the resident upload is a synchronous copy"}
	}
	var id C.uint64_t
	status := int(C.fcuda_routed_expert_resident_upload(
		C.int(format),
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(weight))), C.size_t(len(weight)),
		(*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(scales))), C.size_t(len(scales)),
		C.int64_t(O), C.int64_t(I), &id,
	))
	switch {
	case status == 0:
	case status == -3:
		allocErr := &DeviceAllocError{Bytes: len(weight) + len(scales), Site: site, Class: MemoryWeights}
		c.faultLatch.ObserveError(allocErr, site)
		return nil, allocErr
	case status < 0:
		return nil, &CUDAOpError{Op: op, Site: site, Code: status, Class: MemoryWeights, Msg: fmt.Sprintf("device ABI refused the upload (status %d)", status)}
	default:
		c.faultLatch.Observe(DeviceFaultExecution, site, status)
		return nil, &CUDALaunchError{CUDAOpError{Op: op, Site: site, Code: status, Class: MemoryWeights, Msg: "routed-expert resident upload copy failed"}}
	}
	return &cudaRoutedExpertResident{
		be: c, format: format, o: O, i: I, id: uint64(id),
		latch: latch, faults: faults,
	}, nil
}

// Gemv runs the routed-expert GEMV over the resident bytes, bit-identical to the scalar oracle
// for the uploaded weight. Only x and y cross the bus.
func (h *cudaRoutedExpertResident) Gemv(x []float32) ([]float32, error) {
	return h.gemv(x, nil)
}

func (h *cudaRoutedExpertResident) gemv(x []float32, timing *cudaRoutedExpertTiming) ([]float32, error) {
	c := h.be
	op, site := cudaRoutedExpertResidentNames(h.format, "Gemv")
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if h.id == 0 {
		return nil, &CUDAOpError{Op: op, Site: site, Err: ErrRoutedExpertResidentReleased}
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return nil, err
	}
	if c.faultLatch != h.latch || c.faultLatch.faultCount() != h.faults {
		return nil, &CUDAOpError{Op: op, Site: site, Err: ErrRoutedExpertResidentStale}
	}
	prefix := "fp8 block32"
	if h.format == cudaRoutedExpertMXFP4 {
		prefix = "mxfp4"
	}
	if err := routedExpertXError(prefix, "cuda", h.o, h.i, x); err != nil {
		return nil, err
	}
	if c.capturing {
		return nil, &CUDAOpError{Op: op, Site: site, Msg: "refused while a CUDA graph capture is open: the x upload and y read-back are not capturable"}
	}
	y := make([]float32, h.o)
	var phases [3]C.double
	var phasesPtr *C.double
	if timing != nil {
		phasesPtr = &phases[0]
	}
	status := int(C.fcuda_routed_expert_resident_gemv_e8m0(
		C.uint64_t(h.id), C.int(h.format),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(x))),
		(*C.float)(unsafe.Pointer(unsafe.SliceData(y))),
		C.int64_t(h.o), C.int64_t(h.i), phasesPtr,
	))
	switch {
	case status == 0:
	case status == -3:
		allocErr := &DeviceAllocError{Bytes: 4 * (len(x) + h.o), Site: site, Class: MemoryScratchpad}
		c.faultLatch.ObserveError(allocErr, site)
		return nil, allocErr
	case status < 0:
		return nil, &CUDAOpError{Op: op, Site: site, Code: status, Msg: fmt.Sprintf("device ABI refused the call (status %d)", status)}
	default:
		c.faultLatch.Observe(DeviceFaultExecution, site, status)
		return nil, &CUDALaunchError{CUDAOpError{Op: op, Site: site, Code: status, Msg: "routed-expert resident GEMV copy, launch or execution failed"}}
	}
	// The call fenced g_stream before its read-back, so every earlier enqueue is complete.
	atomic.AddUint64(&c.fenceGen, 1)
	if timing != nil {
		*timing = cudaRoutedExpertTiming{H2DMillis: float64(phases[0]), KernelMillis: float64(phases[1]), D2HMillis: float64(phases[2])}
	}
	return y, nil
}

// Release returns the resident bytes to the device pool. It is not gated by the fault latch,
// so a poisoned or stale handle can still be freed; it is refused while a graph capture is
// open, and the handle then stays usable.
func (h *cudaRoutedExpertResident) Release() error {
	op, site := cudaRoutedExpertResidentNames(h.format, "Release")
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if h.id == 0 {
		return &CUDAOpError{Op: op, Site: site, Err: ErrRoutedExpertResidentReleased}
	}
	if h.be.capturing {
		return &CUDAOpError{Op: op, Site: site, Class: MemoryWeights, Msg: "refused while a CUDA graph capture is open; the resident weight stays live"}
	}
	if status := int(C.fcuda_routed_expert_resident_release(C.uint64_t(h.id))); status != 0 {
		return &CUDAOpError{Op: op, Site: site, Code: status, Class: MemoryWeights, Msg: fmt.Sprintf("device ABI refused the release (status %d)", status)}
	}
	h.id = 0
	return nil
}

// cudaRoutedExpertResidentCounters reports cumulative successful resident uploads and ids
// currently resident: the witnesses that GEMVs never re-upload and Release frees.
func cudaRoutedExpertResidentCounters() (uploads, live uint64) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_routed_expert_resident_uploads()), uint64(C.fcuda_routed_expert_resident_live())
}

var _ RoutedExpertResidentKernel = (*cudaBackend)(nil)
