//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

var _ V41MHCPrenormBackend = (*vulkanBackend)(nil)

// Self-check verdicts, indexed by v41MHCPrenormQ2K. fvk_init runs once per
// process, so these are per device. Guarded by vulkanMu.
const (
	vulkanV41MHCPrenormUnknown  = 0
	vulkanV41MHCPrenormEqual    = 1
	vulkanV41MHCPrenormDiffered = 2
)

var vulkanV41MHCPrenormVerdict [2]int

func v41MHCPrenormQ2K(weight Tensor) C.int {
	if weight.Dtype == Q2_K {
		return 1
	}
	return 0
}

// V41MHCPrenormAvailable reports the prenorm modes of the loaded mHC module.
// A weight is admitted only by SupportsV41MHCPrenormWeight.
func (v *vulkanBackend) V41MHCPrenormAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_mhc_prenorm(0) != 0
}

func (v *vulkanBackend) SupportsV41MHCPrenormWeight(weight Tensor, hidden int) bool {
	if v == nil || !v41MHCPrenormWeightServed(weight, hidden) {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.v41MHCPrenormAdmittedLocked(v41MHCPrenormQ2K(weight))
}

// v41MHCPrenormAdmittedLocked admits a weight dtype only after its device
// self-check found the prenorm bit-identical to MatMul + rms-split. The check
// runs at most once per dtype and device, outside any foreign batch; a sticky
// fault or a device error during it records no verdict and admits nothing.
// Caller holds vulkanMu.
func (v *vulkanBackend) v41MHCPrenormAdmittedLocked(q2k C.int) bool {
	if C.fvk_have_v41_mhc_prenorm(q2k) == 0 {
		return false
	}
	// The opt-in gfx1151 Q2_K decode GEMV is not the kernel this order copies.
	if q2k != 0 && v.strixWave32DecodeGEMVRequested() {
		return false
	}
	switch vulkanV41MHCPrenormVerdict[q2k] {
	case vulkanV41MHCPrenormEqual:
		return true
	case vulkanV41MHCPrenormDiffered:
		return false
	}
	if C.fvk_submission_status() != 0 || bool(C.fvk_batch_foreign_active()) {
		return false
	}
	equal, err := v.v41MHCPrenormSelfCheckLocked(q2k != 0)
	if err != nil {
		return false
	}
	vulkanV41MHCPrenormVerdict[q2k] = vulkanV41MHCPrenormDiffered
	if equal {
		vulkanV41MHCPrenormVerdict[q2k] = vulkanV41MHCPrenormEqual
	}
	return equal
}

// V41MHCPrenormSplit records the projection dispatch and the split dispatch
// under one lock. The scratch row panel is released after recording; a
// pending batch parks it until the flush. The output is caller owned. A
// recorded call emits one computetrace kernel record.
func (v *vulkanBackend) V41MHCPrenormSplit(weight, streams, params Tensor, rows, hidden, iters int, eps, rmsEps float32) (out Tensor, err error) {
	const site = "V41MHC/prenorm-split"
	// As in MatMul: probe before the lock, record after its release.
	emitKernel := v != nil && computetrace.Emitting()
	var kernelStarted time.Time
	var kernelTiming vulkanKernelTiming
	if emitKernel {
		kernelTiming = vulkanKernelTimingFrom(ObservePhasePerformanceQuery(v), vulkanKernelPhase, nil)
		defer func() {
			if err == nil {
				recordV41MHCPrenormKernel(v.Name(), weight, rows, hidden, kernelStarted, kernelTiming)
			}
		}()
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if emitKernel {
		kernelStarted = time.Now()
	}
	var scratch, outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, site)
		}
		if scratch != nil && scratch.ptr != nil {
			C.fvk_free(scratch.ptr)
			scratch.ptr = nil
		}
		if err != nil {
			if outBuf != nil && outBuf.ptr != nil {
				C.fvk_free(outBuf.ptr)
				outBuf.ptr = nil
				outBuf.n = 0
			}
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_mhc() == 0 {
		return Tensor{}, vulkanV41MHCStatusError(site, 3)
	}
	if verr := validateV41MHCPrenorm(weight, streams, params, rows, hidden, iters, eps, rmsEps); verr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: verr.Error(), Recovered: verr}
	}
	q2k := v41MHCPrenormQ2K(weight)
	if !v.v41MHCPrenormAdmittedLocked(q2k) {
		return Tensor{}, vulkanV41MHCStatusError(site, 3)
	}
	width := V41MHCStreams * hidden
	weightBytes := 4 * V41MHCCoeffWidth * width
	if q2k != 0 {
		weightBytes = V41MHCCoeffWidth * (width / q2kSuper) * q2kSuperBlock
	}
	operands := []struct {
		name  string
		t     Tensor
		bytes int
	}{
		{"weight", weight, weightBytes},
		{"streams", streams, 4 * rows * width},
		{"params", params, 4 * V41MHCParamWidth},
	}
	var bufs [3]*vulkanBuf
	for i, operand := range operands {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: operand.name + " must be a live, full-sized buffer owned by this backend"}
		}
		bufs[i] = b
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41MHCStatusError(site, status)
	}
	scratch = v.dallocForClass(4*rows*v41MHCPrenormStride, MemoryActivation, "V4.1 mHC prenorm scratch")
	outBuf = v.dallocForClass(4*rows*V41MHCCoeffWidth, MemoryActivation, "V4.1 mHC prenorm-split output")
	if status := v41MHCPrenormStatusLocked(q2k != 0, bufs[0], bufs[1], bufs[2], scratch, outBuf, rows, hidden, iters, eps, rmsEps); status != 0 {
		return Tensor{}, vulkanV41MHCStatusError(site, status)
	}
	return makeTensor(v, F32, RowMajor, []int{rows, V41MHCCoeffWidth}, nil, outBuf), nil
}

// Caller holds vulkanMu and owns the live handles; the native extent and alias
// refusals are witnessed through this adapter.
func v41MHCPrenormStatusLocked(q2k bool, weight, streams, params, scratch, out *vulkanBuf, rows, hidden, iters int, eps, rmsEps float32) int {
	flag := C.int(0)
	if q2k {
		flag = 1
	}
	return int(C.fvk_v41_mhc_prenorm_split_f32(flag, weight.ptr, streams.ptr, params.ptr, scratch.ptr, out.ptr,
		C.int(rows), C.int(hidden), C.int(iters), C.float(eps), C.float(rmsEps)))
}

// v41MHCPrenormSelfCheckLocked compares the probe's raw mixes and coefficients
// from the prenorm with those from the MatMul kernel the model would dispatch
// (matmul.comp for F32 rows and panels, q2k_matvec for a Q2_K row) followed by
// the rms-split. Caller holds vulkanMu.
func (v *vulkanBackend) v41MHCPrenormSelfCheckLocked(q2k bool) (bool, error) {
	probe, image, err := v.v41MHCPrenormSelfCheckImageLocked(q2k)
	if err != nil {
		return false, err
	}
	return v41MHCPrenormSelfCheckEqual(probe, image), nil
}

// v41MHCPrenormSelfCheckImageLocked records the reference and prenorm arms of
// every probe row count and reads them back in one transfer.
func (v *vulkanBackend) v41MHCPrenormSelfCheckImageLocked(q2k bool) (probe v41MHCPrenormProbe, image [][]byte, err error) {
	dtype := F32
	if q2k {
		dtype = Q2_K
	}
	probe = newV41MHCPrenormProbe(dtype)
	var owned []*vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			image, err = nil, ClassifyVulkanPanic(recovered, "V41MHC/prenorm self-check")
		}
		for _, b := range owned {
			if b != nil && b.ptr != nil {
				C.fvk_free(b.ptr)
				b.ptr = nil
				b.n = 0
			}
		}
	}()
	alloc := func(n int, what string) *vulkanBuf {
		b := v.dallocForClass(n, MemoryActivation, what)
		owned = append(owned, b)
		return b
	}
	upload := func(src unsafe.Pointer, n int, what string) *vulkanBuf {
		b := alloc(n, what)
		C.fvk_h2d(b.ptr, src, C.size_t(n))
		return b
	}
	hidden, width := probe.hidden, V41MHCStreams*probe.hidden
	wb := upload(unsafe.Pointer(&probe.weight[0]), len(probe.weight), "V4.1 mHC prenorm self-check weight")
	pb := upload(unsafe.Pointer(&probe.params[0]), 4*len(probe.params), "V4.1 mHC prenorm self-check params")
	var srcs []*vulkanBuf
	var sizes []int
	for _, rows := range probe.rows {
		sb := upload(unsafe.Pointer(&probe.streams[0]), 4*rows*width, "V4.1 mHC prenorm self-check streams")
		refRaw := alloc(4*rows*V41MHCCoeffWidth, "V4.1 mHC prenorm self-check reference raw")
		refOut := alloc(4*rows*V41MHCCoeffWidth, "V4.1 mHC prenorm self-check reference coeffs")
		scratch := alloc(4*rows*v41MHCPrenormStride, "V4.1 mHC prenorm self-check scratch")
		out := alloc(4*rows*V41MHCCoeffWidth, "V4.1 mHC prenorm self-check coeffs")
		if q2k {
			C.fvk_q2k_matmul_f32(wb.ptr, sb.ptr, refRaw.ptr, C.int(V41MHCCoeffWidth), C.int(width), C.int(rows))
		} else {
			C.fvk_matmul_f32(wb.ptr, sb.ptr, refRaw.ptr, C.int(V41MHCCoeffWidth), C.int(width), C.int(rows))
		}
		if code := int(C.fvk_submission_status()); code != 0 {
			return probe, nil, vulkanV41MHCStatusError("V41MHC/prenorm self-check", code)
		}
		if code := v41MHCStatusLocked(V41MHCOpRMSSplit, refRaw, sb, pb, refOut, rows, hidden, v41MHCPrenormProbeIters, v41MHCPrenormProbeEps, v41MHCPrenormProbeRMSEps); code != 0 {
			return probe, nil, vulkanV41MHCStatusError("V41MHC/prenorm self-check", code)
		}
		if code := v41MHCPrenormStatusLocked(q2k, wb, sb, pb, scratch, out, rows, hidden, v41MHCPrenormProbeIters, v41MHCPrenormProbeEps, v41MHCPrenormProbeRMSEps); code != 0 {
			return probe, nil, vulkanV41MHCStatusError("V41MHC/prenorm self-check", code)
		}
		srcs = append(srcs, refRaw, refOut, scratch, out)
		sizes = append(sizes, 4*rows*V41MHCCoeffWidth, 4*rows*V41MHCCoeffWidth, 4*rows*v41MHCPrenormStride, 4*rows*V41MHCCoeffWidth)
	}
	image, err = vulkanV41ReadMany("V41MHC/prenorm self-check", srcs, sizes)
	return probe, image, err
}
