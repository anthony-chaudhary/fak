//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
int fak_rmsnorm_width_class(int n);
int fak_rmsnorm_width_dispatched_f32(int cls, const float *dX, const float *dW, float *dY, int rows, int n, float eps);
*/
import "C"

import "os"

// Width classes of fak_rmsnorm_width_dispatched_f32; -1 selects by row width.
const (
	cudaRMSNormWidthAuto    = -1
	cudaRMSNormWidthSubwarp = 0
	cudaRMSNormWidthWarp    = 1
	cudaRMSNormWidthBlock   = 2
)

// Off by default: the narrow classes reduce in a different order than the
// block kernel, so outputs are not bit-identical to the existing RMSNorm.
var cudaRMSNormWidthDispatchEnabled = os.Getenv("FAK_CUDA_RMSNORM_WIDTH_DISPATCH") == "1"

func cudaRMSNormWidthClass(width int) int {
	return int(C.fak_rmsnorm_width_class(C.int(width)))
}

// rmsNormWidthDispatchLocked enqueues RMSNorm with an explicit or width-selected
// class. cudaMu is held. Returns the launched class or a negative/CUDA status.
func (c *cudaBackend) rmsNormWidthDispatchLocked(class int, x, weight, y Tensor, rows, width int, eps float32) int {
	return int(C.fak_rmsnorm_width_dispatched_f32(C.int(class), c.cf(x), c.cf(weight), c.cf(y), C.int(rows), C.int(width), C.float(eps)))
}
