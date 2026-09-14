// Package strix: real, fail-closed device-dispatch path for BF16 Wave32 WMMA.
//
// This file is deliberately hostile to the most common measurement fraud in a
// simulator-first codebase: a software emulation run quietly earning a hardware
// throughput label. RunWMMAMeasured NEVER falls back to MatMulBF16, NEVER accepts
// a caller-supplied evidence class, and NEVER returns [HW-WITNESSED] unless a
// device executor independently attests to both physical execution and a physical
// device probe AND reports a non-zero kernel duration.
package strix

import (
	"errors"
	"fmt"
	"time"
)

// EvidenceClass labels HOW a WMMA measurement was obtained. It is set
// internally by RunWMMAMeasured and is never accepted from a caller.
type EvidenceClass string

const (
	// EvidenceClassSWVerified marks a result produced by software emulation /
	// simulation. It carries no physical-device authority.
	EvidenceClassSWVerified EvidenceClass = "SW-VERIFIED"

	// EvidenceClassHWWitnessed marks a result measured on a physical device by
	// an executor that attests both kernel execution and a physical probe.
	EvidenceClassHWWitnessed EvidenceClass = "HW-WITNESSED"
)

// ErrWMMADeviceKernelAbsent is the single fail-closed sentinel returned whenever
// a caller requires a real device-resident WMMA kernel and none is present.
//
// The dispatch contract is deliberately asymmetric: a missing device kernel is
// an error, never a silent downgrade to the simulator. RunWMMAMeasured wraps it
// with the request identity (prim/M/N/K) so callers can still attribute the
// refusal to a specific dispatch.
var ErrWMMADeviceKernelAbsent = errors.New("strix/wmma: device-resident v_wmma_f32_16x16x16_bf16 kernel is absent")

// ErrWMMAUnsupportedPrimitive is returned when a caller requests a WMMA
// primitive the device executor does not advertise support for. The dispatch
// refuses rather than silently downshifting to a different tile shape.
var ErrWMMAUnsupportedPrimitive = errors.New("strix/wmma: requested WMMA primitive not supported by device executor")

// WMMAPrimitive identifies an RDNA 3.5 hardware WMMA tile primitive.
//
// NOTE: this is a package-local mirror of the identically-named type in the
// parent internal/compute package. internal/compute imports internal/compute/strix
// (vulkan_strix*.go), so strix cannot import it back without creating an import
// cycle. The two values below are byte-for-byte identical to the parent's
// WMMAPrimitive16x16x16 / WMMAPrimitive16x16x32 constants.
type WMMAPrimitive string

const (
	// WMMAPrimitive16x16x16 is the standard 16x16x16 tile primitive (FP16, BF16).
	WMMAPrimitive16x16x16 WMMAPrimitive = "16x16x16"

	// WMMAPrimitive16x16x32 is the dual-issue 16x16x32 tile primitive (INT8, FP8, INT4).
	WMMAPrimitive16x16x32 WMMAPrimitive = "16x16x32"
)

// WMMACapabilities is the device executor's self-description. The two boolean
// attestations are the ONLY inputs (besides duration) that can earn
// [HW-WITNESSED]; a simulator sets both false.
type WMMACapabilities struct {
	DeviceName          string          `json:"device_name"`
	GFX                 string          `json:"gfx"`
	ExecutesKernel      bool            `json:"executes_kernel"`
	PhysicalDeviceProbe bool            `json:"physical_device_probe"`
	SupportedPrimitives []WMMAPrimitive `json:"supported_primitives"`
}

// WMMADeviceExecutor is the seam a physical gfx1151 executor implements. It is
// the ONLY way to obtain a hardware-witnessed WMMA measurement.
type WMMADeviceExecutor interface {
	Capabilities() WMMACapabilities
	// ExecuteWMMATile executes C = A*B for the given primitive and dims. A_bf16
	// is M*K, B_bf16 is K*N, both BF16 (uint16). C is M*N float32.
	ExecuteWMMATile(prim WMMAPrimitive, M, N, K int, A_bf16, B_bf16 []uint16) (C []float32, kernelDuration time.Duration, err error)
}

// WMMARequest is a single WMMA tile-measurement request.
type WMMARequest struct {
	M, N, K int
	A_bf16  []uint16
	B_bf16  []uint16
	Prim    WMMAPrimitive
}

// WMMAResult is the measured outcome. Class is always assigned internally by
// RunWMMAMeasured; a caller cannot inject it.
type WMMAResult struct {
	C              []float32     `json:"-"`
	Class          EvidenceClass `json:"evidence_class"`
	Prim           WMMAPrimitive `json:"primitive"`
	M              int           `json:"m"`
	N              int           `json:"n"`
	K              int           `json:"k"`
	KernelDuration time.Duration `json:"kernel_duration"`
	MeasuredKnown  bool          `json:"measured_known"`
	MeasuredTFLOPs float64       `json:"measured_tflops"`
	DeviceName     string        `json:"device_name,omitempty"`
}

// classifyWMMA is a PURE function of the device attestations and the observed
// kernel duration. No field from WMMARequest participates: a caller cannot make
// a simulated run look hardware-witnessed by anything it puts in the request.
func classifyWMMA(caps WMMACapabilities, dur time.Duration) EvidenceClass {
	if caps.ExecutesKernel && caps.PhysicalDeviceProbe && dur > 0 {
		return EvidenceClassHWWitnessed
	}
	return EvidenceClassSWVerified
}

// RunWMMAMeasured dispatches a WMMA tile through a device executor and returns a
// fail-closed measurement. A nil/absent device yields ErrWMMADeviceKernelAbsent
// and NEVER a simulator result.
func RunWMMAMeasured(exec WMMADeviceExecutor, req WMMARequest) (WMMAResult, error) {
	if req.M <= 0 || req.N <= 0 || req.K <= 0 {
		return WMMAResult{}, ErrInvalidDimensions
	}
	if len(req.A_bf16) != req.M*req.K || len(req.B_bf16) != req.K*req.N {
		return WMMAResult{}, fmt.Errorf("%w: A_bf16 len %d (want %d), B_bf16 len %d (want %d)",
			ErrDimensionMismatch, len(req.A_bf16), req.M*req.K, len(req.B_bf16), req.K*req.N)
	}

	if exec == nil {
		return WMMAResult{}, fmt.Errorf("%w: prim=%s M=%d N=%d K=%d",
			ErrWMMADeviceKernelAbsent, req.Prim, req.M, req.N, req.K)
	}

	caps := exec.Capabilities()

	if req.Prim != "" && len(caps.SupportedPrimitives) > 0 {
		supported := false
		for _, p := range caps.SupportedPrimitives {
			if p == req.Prim {
				supported = true
				break
			}
		}
		if !supported {
			return WMMAResult{}, fmt.Errorf("%w: prim=%s device=%s supported=%v",
				ErrWMMAUnsupportedPrimitive, req.Prim, caps.DeviceName, caps.SupportedPrimitives)
		}
	}

	C, dur, err := exec.ExecuteWMMATile(req.Prim, req.M, req.N, req.K, req.A_bf16, req.B_bf16)
	if err != nil {
		return WMMAResult{}, fmt.Errorf("strix/wmma: device executor failed for prim=%s: %w", req.Prim, err)
	}

	class := classifyWMMA(caps, dur)

	res := WMMAResult{
		C:              C,
		Class:          class,
		Prim:           req.Prim,
		M:              req.M,
		N:              req.N,
		K:              req.K,
		KernelDuration: dur,
		DeviceName:     caps.DeviceName,
	}

	if class == EvidenceClassHWWitnessed {
		res.MeasuredKnown = true
		res.MeasuredTFLOPs = (2.0 * float64(req.M) * float64(req.N) * float64(req.K)) / dur.Seconds() / 1e12
	}

	return res, nil
}

// DefaultWMMARequest builds a deterministic square BF16 request for dims.
// A and B are filled with small, exactly-representable BF16 patterns so the
// request is reproducible across hosts.
func DefaultWMMARequest(M, N, K int) WMMARequest {
	A := make([]uint16, M*K)
	for i := range A {
		A[i] = FP32ToBF16(float32((i%9)-4) * 0.5)
	}
	B := make([]uint16, K*N)
	for i := range B {
		B[i] = FP32ToBF16(float32((i%7)-3) * 0.5)
	}
	return WMMARequest{M: M, N: N, K: K, A_bf16: A, B_bf16: B}
}
