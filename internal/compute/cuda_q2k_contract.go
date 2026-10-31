package compute

import "fmt"

// cudaQ2KOperandBytes validates the actual Q2-only submission contract without
// a CUDA dependency. Capacity is checked separately against resident buffers.
func cudaQ2KOperandBytes(w, x Tensor, p int, panel bool) (uint64, uint64, error) {
	fail := func() (uint64, uint64, error) {
		return 0, 0, fmt.Errorf("requires rank-two row-major Q2_K block-256/bits-2 weights and bounded F32 activations")
	}
	const intMax = 1<<31 - 1
	if w.Dtype != Q2_K || w.Layout != RowMajor || len(w.Shape) != 2 || w.Quant == nil || w.Quant.Block != 256 || w.Quant.Bits != 2 || w.Quant.Axis != 2 || x.Dtype != F32 || x.Layout != RowMajor {
		return fail()
	}
	// Legacy GEMV computes (out+7)/8 in signed int.
	out, in := w.Shape[0], w.Shape[1]
	if out <= 0 || in <= 0 || p <= 0 || out > intMax-7 || in > intMax || p > intMax || in%256 != 0 {
		return fail()
	}
	if panel {
		if len(x.Shape) != 2 || x.Shape[0] != p || x.Shape[1] != in {
			return fail()
		}
	} else if p != 1 || len(x.Shape) != 1 || x.Shape[0] != in {
		return fail()
	}
	// The unchanged P>=128 fallback uses signed-int weight indexing and a
	// one-dimensional activation split grid. Admission must protect that route
	// even when the optional MMQ arm is disabled or refuses before enqueue.
	if p >= 128 && (uint64(out)*uint64(in) > intMax || (uint64(p)*uint64(in)+255)/256 > intMax) {
		return fail()
	}
	wbytes := uint64(out) * uint64(in/256) * 84
	xbytes := uint64(p) * uint64(in) * 4
	ybytes := uint64(p) * uint64(out) * 4
	maxHost := uint64(^uint(0) >> 1)
	if wbytes > maxHost || xbytes > maxHost || ybytes > maxHost {
		return fail()
	}
	return wbytes, xbytes, nil
}

func cudaQ2KCheckCapacity(w, x Tensor, p int, panel bool, weightBytes, activationBytes int) error {
	wn, xn, err := cudaQ2KOperandBytes(w, x, p, panel)
	if err != nil {
		return err
	}
	if weightBytes < 0 || activationBytes < 0 || uint64(weightBytes) < wn || uint64(activationBytes) < xn {
		return fmt.Errorf("Q2_K resident allocation is shorter than its operand extent")
	}
	return nil
}
