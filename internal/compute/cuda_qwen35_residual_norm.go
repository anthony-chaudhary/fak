//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
int fak_qwen35_residual_norm_capability(int rows, int n);
int fak_qwen35_residual_norm_f32(float *X, const float *Delta, const float *W, float *Y, int rows, int n, float eps);
unsigned long long fak_qwen35_residual_norm_launch_count(void);
*/
import "C"

import (
	"fmt"
	"math"
	"os"
)

var cudaQwen35ResidualNormEnabled = os.Getenv("FAK_CUDA_QWEN35_RESIDUAL_NORM") == "1"

// cudaMu is held. A known unsupported/off path declines before allocation or
// mutation. Once selected, error is fatal: X might already contain its residual
// update, so the caller must not retry Add then RMSNorm. No public API is added.
func (c *cudaBackend) qwen35SequenceResidualNormLocked(x, delta, weight Tensor, rows int, eps float32) (Tensor, *cudaBuf, bool, error) {
	if !cudaQwen35ResidualNormEnabled || c.capturing || rows <= 1 {
		return Tensor{}, nil, false, nil
	}
	fail := func(reason string) (Tensor, *cudaBuf, bool, error) {
		return Tensor{}, nil, true, &Qwen35SequenceError{Stage: "residual-post-norm", Layer: -1, Reason: reason}
	}
	if len(x.Shape) != 2 || x.Shape[0] != rows || x.Shape[1] <= 0 || !(eps > 0) || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return fail("invalid panel or RMS epsilon")
	}
	width := x.Shape[1]
	if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(rows), int64(width)); !ok {
		return fail("panel exceeds CUDA signed index range")
	}
	if _, ok := qwen35GDNShapeBytes([]int{rows, width}, F32.Bytes()); !ok {
		return fail("panel byte extent overflow")
	}
	for _, operand := range []struct {
		name   string
		tensor Tensor
		shape  []int
	}{
		{"residual", x, []int{rows, width}}, {"branch", delta, []int{rows, width}}, {"post norm", weight, []int{width}},
	} {
		if err := c.validateQwen35SequenceTensor(operand.name, operand.tensor, false, operand.shape...); err != nil {
			return Tensor{}, nil, true, err
		}
	}
	capability := int(C.fak_qwen35_residual_norm_capability(C.int(rows), C.int(width)))
	if capability == 0 {
		return Tensor{}, nil, false, nil
	}
	if capability != 1 {
		return fail(fmt.Sprintf("CUDA capability status %d", capability))
	}
	y, buf, err := c.qwen35SequenceAllocLocked([]int{rows, width}, "residual-post-norm")
	if err != nil {
		return Tensor{}, nil, true, err
	}
	status := int(C.fak_qwen35_residual_norm_f32(c.cf(x), c.cf(delta), c.cf(weight), c.cf(y), C.int(rows), C.int(width), C.float(eps)))
	if status != 0 {
		if drainErr := c.qwen35SequenceSyncLocked(); drainErr != nil {
			c.faultLatch.ObserveError(drainErr, "residual-post-norm-drain")
			return Tensor{}, nil, true, drainErr
		}
		c.releaseTransientBuffers([]*cudaBuf{buf})
		return fail(fmt.Sprintf("CUDA status %d", status))
	}
	return y, buf, true, nil
}

func cudaQwen35ResidualNormLaunchesLocked() uint64 {
	return uint64(C.fak_qwen35_residual_norm_launch_count())
}
