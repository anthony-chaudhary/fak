package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestQwen35SequenceIQ2PackedContract(t *testing.T) {
	for _, tc := range []struct {
		dtype Dtype
		block int
	}{{IQ2_XXS, 66}, {IQ2_XS, 74}} {
		w := Tensor{Dtype: tc.dtype, Shape: []int{17408, 5120}}
		if got, ok := qwen35SequenceIQ2MatrixBytes(w); !ok || got != 17408*20*tc.block {
			t.Fatalf("%s stock FFN bytes=%d valid=%v", tc.dtype, got, ok)
		}
		for _, p := range []int{1, 2, 4, 128, 65535} {
			if err := validateQwen35SequenceIQ2(w, p, true); err != nil {
				t.Fatal(err)
			}
		}
		for _, p := range []int{-1, 0, 65536} {
			if err := validateQwen35SequenceIQ2(w, p, true); err == nil {
				t.Fatalf("accepted P=%d", p)
			}
		}
		for _, shape := range [][]int{nil, {256}, {1, 1, 256}, {0, 256}, {-1, 256}, {2, 128}, {1, 257}, {int(qwen35GDNMaxCInt), 256}, {1, int(qwen35GDNMaxCInt)}} {
			bad := Tensor{Dtype: tc.dtype, Shape: shape}
			if _, ok := qwen35SequenceIQ2MatrixBytes(bad); ok {
				t.Fatalf("accepted malformed %v", shape)
			}
			if err := validateQwen35SequenceIQ2(bad, 2, true); err == nil {
				t.Fatalf("launch accepted %v", shape)
			}
		}
		var typed *Qwen35SequenceError
		if err := validateQwen35SequenceIQ2(w, 2, false); !errors.As(err, &typed) || typed.Stage != "iq2-preflight" {
			t.Fatalf("default-off error=%v", err)
		}
		if got := validateQwen35SequenceIQ2(w, 2, CUDAQwen35IQ2SequenceQualificationEnabled()); (got == nil) != qwen35SequenceIQ2Qualification {
			t.Fatalf("compiled selection and admission disagree: %v", got)
		}
	}
	// Other dtype policies are owned by the existing preflight. In particular,
	// generic IQ1_M/IQ4_NL kernels do not silently become sequence capability.
	for _, dt := range []Dtype{F32, Q8_0, IQ4_XS, IQ1_M, IQ4_NL, IQ2_S} {
		w := Tensor{Dtype: dt, Shape: []int{2, 256}}
		if _, ok := qwen35SequenceIQ2MatrixBytes(w); ok {
			t.Fatalf("expanded scope to %s", dt)
		}
		if err := validateQwen35SequenceIQ2(w, 2, false); err != nil {
			t.Fatalf("changed %s policy: %v", dt, err)
		}
	}
}
