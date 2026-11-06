//go:build cuda && cgo

package compute

import (
	"errors"
	"testing"
	"unsafe"
)

// Both standalone sequence and decode share this pre-allocation validator.
// Opaque host storage is never submitted to CUDA: validation only inspects
// metadata. The linked CUDA package build is a separate qualification gate.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAQwen35GDNStateRequiresF32(t *testing.T) {
	for _, operand := range []string{"conv_state", "recurrent_state"} {
		for _, tc := range []struct {
			name      string
			dtype     Dtype
			capacity  int
			wantError bool
		}{
			{"f32", F32, 32, false},
			{"f32-short", F32, 31, true},
			{"f16", F16, 16, true},
			{"f16-overallocated", F16, 32, true},
			{"bf16", BF16, 16, true},
		} {
			t.Run(operand+"/"+tc.name, func(t *testing.T) {
				be := &cudaBackend{}
				storage := make([]byte, 32)
				buffer := &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: tc.capacity, class: MemoryKVCache}
				state := Tensor{Dtype: tc.dtype, Layout: RowMajor, Shape: []int{8}, buf: buffer, be: be}
				err := be.validateQwen35GDNTensor(operand, state)
				if !tc.wantError {
					if err != nil {
						t.Fatal(err)
					}
					return
				}
				var refused *Qwen35GDNResidencyError
				if !errors.As(err, &refused) || refused.Operand != operand {
					t.Fatalf("state dtype %s capacity %d: error = %v, want typed %s residency refusal", tc.dtype, tc.capacity, err, operand)
				}
			})
		}
	}
}
