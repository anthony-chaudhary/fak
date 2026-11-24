//go:build darwin && arm64 && cgo

package metalgemm

/*
#include "packed_gguf.h"
*/
import "C"

import (
	"errors"
	"math"
	"unsafe"
)

// RoutedFFN returns false for expert stacks split across multiple GPU views.
// The caller can then run selected projections through those views separately.
func (s *PackedGGUFStore) RoutedFFN(job PackedRoutedFFN, x, dst []float32) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		return false, ErrPackedGGUFClosed
	}
	router, gu, down := job.Router, job.GateUp, job.Down
	if router == nil || gu == nil || down == nil || router.store != s || gu.store != s || down.store != s {
		return false, ErrPackedGGUFShape
	}
	n, k, h, w := gu.experts, job.Slots, gu.cols, down.cols
	if k < 1 || k > n || router.experts != 1 || router.rows != n || router.cols != h ||
		gu.rows != 2*w || down.rows != h || down.experts != n || len(x) != h || len(dst) != h {
		return false, ErrPackedGGUFShape
	}
	for _, v := range [][]float32{job.Bias, job.GateUpScale, job.DownScale} {
		if len(v) != 0 && len(v) != n {
			return false, ErrPackedGGUFShape
		}
	}
	if len(gu.views) != 1 || len(down.views) != 1 {
		return false, nil
	}
	limit := min(s.maxBuffer/4, math.MaxInt32)
	stride := (w + 3) &^ 3
	idsAt, tableAt := h+3*k, h+4*k
	actAt := (tableAt + 3*n + 3) &^ 3
	gateAt := (n + 3) &^ 3
	if actAt > limit || k > (limit-actAt)/stride || gateAt+h > limit || k > (limit-gateAt-h)/(2*w+h) {
		return false, ErrPackedGGUFShape
	}
	downAt := gateAt + k*2*w
	sumAt := downAt + k*h
	jobs := [3]C.mg_packed_job{router.job(0, 0, 0), gu.job(0, 0, gateAt), down.job(0, actAt, downAt)}
	var xp, yp *C.float
	if C.mg_packed_prepare(s.ctx, C.size_t(actAt+k*stride), C.size_t(sumAt+h), &xp, &yp) == 0 {
		return false, errors.New("metalgemm: routed FFN activation allocation failed")
	}
	input := unsafe.Slice((*float32)(unsafe.Pointer(xp)), actAt)
	copy(input, x)
	for e := 0; e < n; e++ {
		params := input[tableAt+3*e:]
		params[0], params[1], params[2] = 0, 1, 1
		if len(job.Bias) > 0 {
			params[0] = job.Bias[e]
		}
		if len(job.GateUpScale) > 0 {
			params[1] = job.GateUpScale[e]
		}
		if len(job.DownScale) > 0 {
			params[2] = job.DownScale[e]
		}
	}
	scale := job.WeightScale
	if scale == 0 {
		scale = 1
	}
	shape := C.mg_packed_moe_shape{slots: C.uint32_t(k), hidden: C.uint32_t(h), width: C.uint32_t(w),
		activation_stride: C.uint32_t(stride), beta: C.float(job.Beta), linear_beta: C.float(job.LinearBeta)}
	route := C.mg_packed_route_shape{experts: C.uint32_t(n), slots: C.uint32_t(k), weight_scale: C.float(scale)}
	if C.mg_packed_routed_ffn(s.ctx, &jobs[0], shape, route, C.size_t(idsAt), C.size_t(tableAt)) == 0 {
		return false, errors.New("metalgemm: routed FFN command failed")
	}
	output := unsafe.Slice((*float32)(unsafe.Pointer(yp)), sumAt+h)
	copy(dst, output[sumAt:])
	return true, nil
}
