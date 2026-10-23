//go:build cuda && cgo

package compute

import (
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"unsafe"
)

// These pointers are non-device sentinels. Only refusing wrapper calls and the
// allocation-free admission helper may consume this fixture, never a kernel.
func cudaTreeVerifyAdmissionFixture() (*cudaBackend, [4]Tensor) {
	c := &cudaBackend{fenceGen: 7}
	var tensors [4]Tensor
	for i := range tensors {
		tensors[i] = makeTensor(c, F32, RowMajor, []int{1, 1, 1}, nil, &cudaBuf{
			ptr: unsafe.Pointer(new(float32)), n: F32.Bytes(), be: c, bornGen: 7,
		})
	}
	c.transient = []*cudaBuf{tensors[0].buf.(*cudaBuf)}
	return c, tensors
}

func cudaTreeVerifyAdmissionUnchanged(t *testing.T, c *cudaBackend, out *Tensor) func() {
	t.Helper()
	original := *out
	original.Shape = append([]int(nil), out.Shape...)
	transient := append([]*cudaBuf(nil), c.transient...)
	fence := atomic.LoadUint64(&c.fenceGen)
	return func() {
		t.Helper()
		if !reflect.DeepEqual(*out, original) || !reflect.DeepEqual(c.transient, transient) || atomic.LoadUint64(&c.fenceGen) != fence {
			t.Fatal("operand admission changed output, transient allocations, or fence generation")
		}
	}
}

func requireCUDATreeVerifyAdmissionError(t *testing.T, err error, operand, reason string) {
	t.Helper()
	var opErr *CUDAOpError
	if !errors.As(err, &opErr) || opErr.Op != "TreeVerifyAttention" || opErr.Site != "tree-verify-preflight" || opErr.Msg != operand+" "+reason {
		t.Fatalf("%s refusal = %T %v", operand, err, err)
	}
}

// fak-test:runtime fast est=1ms lane=default
// Unmeasured estimate. Each host operand must reject before any native call;
// same-sized output sentinels also keep the pre-fix reproducer allocation-free.
func TestCUDATreeVerifyAttentionRejectsHostOperands(t *testing.T) {
	for index, name := range []string{"q", "k", "v", "out"} {
		t.Run(name, func(t *testing.T) {
			c, tensors := cudaTreeVerifyAdmissionFixture()
			tensors[index] = NewF32(nil, []int{1, 1, 1}, []float32{1})
			unchanged := cudaTreeVerifyAdmissionUnchanged(t, c, &tensors[3])
			err := c.TreeVerifyAttention(&tensors[0], &tensors[1], &tensors[2], &tensors[3], []uint32{1}, 1, 1, 1, 1, 1)
			requireCUDATreeVerifyAdmissionError(t, err, name, "must be owned by the executing CUDA backend; upload host tensors before verification")
			unchanged()
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
// Unmeasured estimate. Malformed metadata is exercised only through the helper,
// so a missing check cannot send a sentinel pointer to CUDA.
func TestCUDATreeVerifyOperandAdmissionRejectsMalformedMetadata(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Tensor)
		reason string
	}{
		{"foreign-backend", func(v *Tensor) { v.be = &cudaBackend{} }, "must be owned by the executing CUDA backend; upload host tensors before verification"},
		{"dtype", func(v *Tensor) { v.Dtype = F16 }, "must be row-major f32"},
		{"layout", func(v *Tensor) { v.Layout = ColMajor }, "must be row-major f32"},
		{"host-buffer", func(v *Tensor) { v.buf = &hostBuf{f32: []float32{1}} }, "must have a live default-device CUDA allocation"},
		{"nil-buffer", func(v *Tensor) { v.buf = (*cudaBuf)(nil) }, "must have a live default-device CUDA allocation"},
		{"freed-buffer", func(v *Tensor) { v.buf.(*cudaBuf).ptr = nil }, "must have a live default-device CUDA allocation"},
		{"nondefault-device", func(v *Tensor) { v.buf.(*cudaBuf).device = 1 }, "must have a live default-device CUDA allocation"},
		{"poisoned", func(v *Tensor) { v.buf.(*cudaBuf).invalid = 1 }, "has invalid CUDA state"},
		{"short-allocation", func(v *Tensor) { v.buf.(*cudaBuf).n = F32.Bytes() - 1 }, "CUDA allocation is too small for its f32 elements"},
	}
	for _, tc := range cases {
		for index, operand := range []string{"q", "k", "v", "out"} {
			t.Run(tc.name+"/"+operand, func(t *testing.T) {
				c, tensors := cudaTreeVerifyAdmissionFixture()
				tc.mutate(&tensors[index])
				unchanged := cudaTreeVerifyAdmissionUnchanged(t, c, &tensors[3])
				cudaMu.Lock()
				err := c.validateVerifyAttentionOperandsLocked("TreeVerifyAttention", "tree-verify-preflight", &tensors[0], &tensors[1], &tensors[2], &tensors[3], 1)
				cudaMu.Unlock()
				requireCUDATreeVerifyAdmissionError(t, err, operand, tc.reason)
				if tc.name == "poisoned" {
					var invalid *Qwen35GDNInvalidStateError
					if !errors.As(err, &invalid) || invalid.Operand != operand {
						t.Fatalf("poison refusal lost its operand or cause: %v", err)
					}
				}
				unchanged()
			})
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
// Unmeasured estimate. Successful sentinel cases stop at the admission helper.
func TestCUDATreeVerifyOperandAdmissionPreservesPendingAndReplacement(t *testing.T) {
	for _, name := range []string{"pending", "synchronous", "empty-output", "replacement-output"} {
		t.Run(name, func(t *testing.T) {
			c, tensors := cudaTreeVerifyAdmissionFixture()
			for _, tensor := range tensors {
				if tensor.Ready() {
					t.Fatal("fixture must start with pending stream-ordered buffers")
				}
			}
			switch name {
			case "synchronous":
				for _, tensor := range tensors {
					tensor.buf.(*cudaBuf).be = nil
					if !tensor.Ready() {
						t.Fatal("synchronous buffer must be ready without an async backend")
					}
				}
			case "empty-output":
				tensors[3] = Tensor{}
			case "replacement-output":
				// An incompatible old output is not consumed and must not be
				// validated. The wrapper replaces it after input admission.
				tensors[3] = NewF32(nil, []int{2}, []float32{1, 2})
			}
			unchanged := cudaTreeVerifyAdmissionUnchanged(t, c, &tensors[3])
			cudaMu.Lock()
			err := c.validateVerifyAttentionOperandsLocked("TreeVerifyAttention", "tree-verify-preflight", &tensors[0], &tensors[1], &tensors[2], &tensors[3], 1)
			cudaMu.Unlock()
			if err != nil {
				t.Fatalf("valid operand admission: %v", err)
			}
			unchanged()
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
// Unmeasured estimate. Extraction must preserve the linear verifier's error contract.
func TestCUDAVerifyOperandAdmissionPreservesSpecErrorContract(t *testing.T) {
	c, tensors := cudaTreeVerifyAdmissionFixture()
	tensors[1].buf.(*cudaBuf).invalid = 1
	cudaMu.Lock()
	err := c.validateSpecVerifyOperandsLocked(&tensors[0], &tensors[1], &tensors[2], &tensors[3], 1)
	cudaMu.Unlock()
	var opErr *CUDAOpError
	var invalid *Qwen35GDNInvalidStateError
	if !errors.As(err, &opErr) || opErr.Op != "SpecVerifyAttention" || opErr.Site != "spec-verify-preflight" || opErr.Msg != "k has invalid CUDA state" || !errors.As(err, &invalid) || invalid.Operand != "k" || !strings.Contains(err.Error(), "spec-verify-preflight") {
		t.Fatalf("linear verifier refusal changed: %T %v", err, err)
	}
}
