//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

// SupportsQ3KMatMul exposes packed GGUF Q3_K GEMV and token-panel multiplication.
// This is software capability, not hardware parity or throughput evidence.
func (c *cudaBackend) SupportsQ3KMatMul() bool { return true }

// Check the packed storage and activation extent before narrowing Go dimensions
// to the C ABI or issuing an asynchronous read. No failed Q3_K path falls back.
func (c *cudaBackend) checkQ3KOperands(w, x Tensor, out, in, p int, panel bool) {
	fail := func() {
		panic(&CUDAOpError{Op: "Q3_K MatMul", Site: "q3k-operands", Msg: "requires owned packed row-major Q3_K, F32 activations and valid bounded dimensions", Class: MemoryActivation})
	}
	if out <= 0 || in <= 0 || p <= 0 || out > cudaNVFP4IntMax || in > cudaNVFP4IntMax || p > 4*65535 || in%256 != 0 || w.Layout != RowMajor || x.Dtype != F32 || x.Layout != RowMajor {
		fail()
	}
	if panel {
		if len(x.Shape) != 2 || x.Shape[0] != p || x.Shape[1] != in {
			fail()
		}
	} else {
		if len(x.Shape) != 1 || x.Shape[0] != in {
			fail()
		}
	}
	wb := c.cudaBufForSubmit(w)
	xb := c.cudaBufForSubmit(x)
	if uint64(wb.n) < uint64(out)*uint64(in/256)*110 || uint64(xb.n) < uint64(p)*uint64(in)*4 {
		fail()
	}
}

// Check rank before generic MatMul reads Shape[0:2].
func (c *cudaBackend) checkQ3KWeight(w Tensor) {
	if len(w.Shape) != 2 || w.Quant == nil || w.Quant.Block != 256 || w.Quant.Bits != 3 || w.Quant.Axis != 2 || w.Layout != RowMajor {
		panic(&CUDAOpError{Op: "Q3_K MatMul", Site: "q3k-weight-descriptor", Msg: "requires a rank-two row-major 256-element/3-bit Q3_K descriptor", Class: MemoryActivation})
	}
}

// q3kMMVQLaunchesForTest reports Q3_K MMVQ kernel launches, so the parity witness can tell
// which arm produced an output.
func (c *cudaBackend) q3kMMVQLaunchesForTest() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_q3k_mmvq_launch_count())
}
