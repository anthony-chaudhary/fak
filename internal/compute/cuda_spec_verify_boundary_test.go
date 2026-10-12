//go:build cuda && cgo

package compute

import (
	"errors"
	"strings"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=1ms lane=default
// Unmeasured estimate. Host operands must be rejected before the sentinel CUDA
// pointers can reach a native call; no device allocation or submission is needed.
func TestCUDASpecVerifyAttentionRejectsHostOperands(t *testing.T) {
	for index, name := range []string{"q", "k", "v", "out"} {
		t.Run(name, func(t *testing.T) {
			c := &cudaBackend{}
			var storage [4]float32
			var tensors [4]Tensor
			for i := range tensors {
				tensors[i] = makeTensor(c, F32, RowMajor, []int{1, 1, 1}, nil, &cudaBuf{
					ptr: unsafe.Pointer(&storage[i]), n: F32.Bytes(),
				})
			}
			tensors[index] = NewF32(nil, []int{1, 1, 1}, []float32{1})
			output := tensors[3].buf
			err := c.SpecVerifyAttention(&tensors[0], &tensors[1], &tensors[2], &tensors[3], 1, 1, 1, 1, 1)
			var opErr *CUDAOpError
			if !errors.As(err, &opErr) || opErr.Op != "SpecVerifyAttention" || opErr.Site != "spec-verify-preflight" || !strings.Contains(opErr.Msg, name+" ") {
				t.Fatalf("host %s refusal = %T %v", name, err, err)
			}
			if tensors[3].buf != output || len(c.transient) != 0 {
				t.Fatal("rejected operand changed output or transient allocations")
			}
		})
	}
}
