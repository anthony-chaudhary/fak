//go:build cuda && cgo

package compute

import "fmt"

// Called under cudaMu, before tracing, output allocation, or any C submission.
// Use the existing shared-runtime buffer admission (including poison) rather
// than rejecting valid weight-cache tensors owned by another session backend.
func (c *cudaBackend) validateQ2KOperands(w, x Tensor, p int, panel bool) error {
	if _, _, err := cudaQ2KOperandBytes(w, x, p, panel); err != nil {
		return err
	}
	for _, t := range []Tensor{w, x} {
		owner, owned := t.Backend().(*cudaBackend)
		b, ok := t.buf.(*cudaBuf)
		if !owned || owner == nil || !ok || b == nil || b.ptr == nil || b.device != 0 {
			return fmt.Errorf("requires live CUDA operands on the owned default-device runtime")
		}
	}
	wb, xb := c.cudaBufForSubmit(w), c.cudaBufForSubmit(x)
	if err := cudaQ2KCheckCapacity(w, x, p, panel, wb.n, xb.n); err != nil {
		return err
	}
	return nil
}

func (c *cudaBackend) checkQ2KOperands(w, x Tensor, p int, panel bool) {
	if err := c.validateQ2KOperands(w, x, p, panel); err != nil {
		panic(&CUDAOpError{Op: "Q2_K MatMul", Site: "q2k-operands", Msg: err.Error(), Class: MemoryActivation})
	}
}
