package compute

import (
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// cudaMatMulTraceEvent describes the operands consumed by one successful CUDA
// dispatch. BytesRead/BytesWritten (and the legacy Bytes alias for BytesRead)
// are MODELED LOGICAL PAYLOAD, not measured HBM/PCIe/cache traffic: packed
// weights are counted once, activations use the dispatched row count, and
// internal scratch, repeated loads and allocator capacity are excluded.
// Timing is supplied by the existing opt-in CUDA event pair, not modeled here.
func (w Tensor) cudaMatMulTraceEvent(backend string, x, y Tensor, rows int, started time.Time, elapsedMS float64, batched bool) (computetrace.Event, bool) {
	if len(w.Shape) != 2 || rows <= 0 || (!batched && rows != 1) || x.Dtype != F32 || y.Dtype != F32 ||
		math.IsNaN(elapsedMS) || math.IsInf(elapsedMS, 0) || elapsedMS < 0 || elapsedMS >= float64(math.MaxInt64)/1e6 {
		return computetrace.Event{}, false
	}
	weightBytes, okW := w.cudaMatMulTraceWeightBytes()
	inputElements, okX := cudaTraceProduct(rows, w.Shape[1])
	outputElements, okY := cudaTraceProduct(rows, w.Shape[0])
	inputCapacity, okXC := cudaTraceProduct(x.Shape...)
	outputCapacity, okYC := cudaTraceProduct(y.Shape...)
	if !okW || !okX || !okY || !okXC || !okYC || inputCapacity < inputElements || outputCapacity < outputElements ||
		inputElements > math.MaxInt64/4 || outputElements > math.MaxInt64/4 || weightBytes > math.MaxInt64-inputElements*4 {
		return computetrace.Event{}, false
	}
	flops, ok := cudaTraceProduct(2, rows, w.Shape[0], w.Shape[1])
	if !ok {
		return computetrace.Event{}, false
	}
	operation := "matmul"
	inputShape := []int{w.Shape[1]}
	if batched {
		operation = "batched_matmul"
		inputShape = []int{rows, w.Shape[1]}
	}
	readBytes := weightBytes + inputElements*4
	duration := int64(elapsedMS * 1e6)
	return computetrace.Event{
		Operation: "matmul", Phase: "kernel", Backend: backend, Device: "cuda:0",
		Kernel: w.Dtype.String() + "_" + operation, Route: "device",
		StartedAt: started.UTC(), DurationNS: duration, DeviceDurationNS: duration,
		TimerDomain: "cuda_event", Status: "ok",
		InputDType: x.Dtype.String(), WeightDType: w.Dtype.String(), OutputDType: y.Dtype.String(),
		Bytes: readBytes, BytesRead: readBytes, BytesWritten: outputElements * 4,
		EstimatedFLOPs: flops, Shapes: [][]int{{w.Shape[0], w.Shape[1]}, inputShape},
		ProvenanceDigest: computetrace.Digest(backend, w.Dtype.String(), operation),
	}, true
}

func cudaTraceProduct(shape ...int) (int64, bool) {
	if len(shape) == 0 {
		return 0, false
	}
	n := int64(1)
	for _, d := range shape {
		if d <= 0 || int64(d) > math.MaxInt64/n {
			return 0, false
		}
		n *= int64(d)
	}
	return n, true
}

func (w Tensor) cudaMatMulTraceWeightBytes() (int64, bool) {
	if len(w.Shape) != 2 {
		return 0, false
	}
	n, ok := cudaTraceProduct(w.Shape...)
	if !ok {
		return 0, false
	}
	switch w.Dtype {
	case F32, F16:
		width := int64(w.Dtype.Bytes())
		if n > math.MaxInt64/width {
			return 0, false
		}
		return n * width, true
	case Q8_0, Q2_0:
		if w.Quant == nil || w.Quant.Block <= 0 || w.Shape[1]%w.Quant.Block != 0 {
			return 0, false
		}
		codes := n
		if w.Dtype == Q2_0 {
			if w.Shape[1]%4 != 0 {
				return 0, false
			}
			codes /= 4
		}
		scales := n / int64(w.Quant.Block)
		if scales > (math.MaxInt64-codes)/4 {
			return 0, false
		}
		return codes + 4*scales, true
	case NVFP4:
		packed, scales, err := nvfp4PayloadShape(w.Shape)
		if err != nil || int64(packed) > math.MaxInt64-int64(scales) {
			return 0, false
		}
		return int64(packed) + int64(scales), true
	}
	var blockBytes int64
	switch w.Dtype {
	case IQ2_XXS, IQ2_XS, IQ1_M, IQ4_NL:
		bytes, found := RawIQBlockBytes(w.Dtype)
		if !found {
			return 0, false
		}
		blockBytes = int64(bytes)
	default:
		// Reuse the existing pure 256-element K-quant geometry lookup; this
		// does not enable a Vulkan pipeline or change CUDA dtype admission.
		var found bool
		blockBytes, found = qwen35SequenceKQuantBlockBytes(w.Dtype)
		if !found {
			return 0, false
		}
	}
	if w.Shape[1]%rawIQSuper != 0 || blockBytes <= 0 || n/rawIQSuper > math.MaxInt64/blockBytes {
		return 0, false
	}
	return n / rawIQSuper * blockBytes, true
}
