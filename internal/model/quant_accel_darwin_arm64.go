//go:build fakaccel && darwin && arm64 && cgo

package model

/*
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

static void fak_q8_sgemm(const float *x, const float *w, int p, int out, int in, float *y) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		p, out, in,
		1.0f, x, in,
		w, in,
		0.0f, y, out);
}
*/
import "C"

import "unsafe"

func qgemmAccelDefault() bool { return true }

func q8PrepareAccelWeight(qt *q8Tensor) {
	if qt == nil || qt.out <= 0 || qt.in <= 0 || qt.nblk <= 0 {
		return
	}
	if len(qt.q) < qt.out*qt.in || len(qt.d) < qt.out*qt.nblk {
		return
	}
	qt.accelOnce.Do(func() {
		qt.accelF32 = dequantQ8ForAccel(qt)
	})
}

func q8RememberAccelPanel(qp *q8Panel, X []float32) {
	qp.f32 = X
}

func dequantQ8ForAccel(qt *q8Tensor) []float32 {
	out, in, nblk := qt.out, qt.in, qt.nblk
	w := make([]float32, out*in)
	parFor(out, numWorkers, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			qrow := qt.q[o*in : (o+1)*in]
			drow := qt.d[o*nblk : (o+1)*nblk]
			wrow := w[o*in : (o+1)*in]
			for b := 0; b < nblk; b++ {
				d := drow[b]
				base := b * qBlk
				for i := 0; i < qBlk; i++ {
					wrow[base+i] = float32(qrow[base+i]) * d
				}
			}
		}
	})
	return w
}

func qGemm8AccelInto(qt *q8Tensor, qp *q8Panel, Y []float32) bool {
	if qt == nil || qp == nil || qt.out <= 0 || qt.in <= 0 || qp.P <= 0 || qp.in != qt.in {
		return false
	}
	if len(qp.f32) < qp.P*qp.in || len(Y) < qp.P*qt.out {
		return false
	}
	q8PrepareAccelWeight(qt)
	if len(qt.accelF32) < qt.out*qt.in {
		return false
	}
	C.fak_q8_sgemm(
		(*C.float)(unsafe.Pointer(&qp.f32[0])),
		(*C.float)(unsafe.Pointer(&qt.accelF32[0])),
		C.int(qp.P), C.int(qt.out), C.int(qt.in),
		(*C.float)(unsafe.Pointer(&Y[0])),
	)
	return true
}
