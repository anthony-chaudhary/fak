//go:build cuda && cgo && fak_cuda_v41_compressor_pool

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
int fak_v41_pool_readback(float *, const float *, uint64_t);
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// Qualification only. A successful pool call has already drained its inputs;
// this copies only the logical output, excluding the trailing finite flag.
func (c *cudaBackend) cudaV41PoolWitnessRead(t Tensor, width int) ([]float32, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c == nil || c.capturing || width <= 0 || width > int(^uint(0)>>1)/4-1 || t.be != c || t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || len(t.Shape) != 1 || t.Shape[0] != width {
		return nil, c.cudaV41PoolError(-2, fmt.Errorf("invalid witness output"))
	}
	if err := c.faultLatch.Admit("V41CompressorPoolReadback"); err != nil {
		return nil, err
	}
	b, ok := t.buf.(*cudaBuf)
	if !ok || b == nil || b.ptr == nil || b.n < (width+1)*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
		return nil, c.cudaV41PoolError(-2, fmt.Errorf("invalid witness storage"))
	}
	if err := b.invalidStateError("V41CompressorPoolReadback"); err != nil {
		return nil, err
	}
	values := make([]float32, width)
	status := int(C.fak_v41_pool_readback((*C.float)(unsafe.Pointer(&values[0])), (*C.float)(b.ptr), C.uint64_t(width)))
	runtime.KeepAlive(t)
	runtime.KeepAlive(values)
	if status != 0 {
		return nil, c.cudaV41PoolError(status, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return values, nil
}
