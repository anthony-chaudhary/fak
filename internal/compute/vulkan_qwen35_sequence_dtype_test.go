package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=1ms lane=unit
func TestVulkanQwen35SequenceRejectsUnsupportedDtype(t *testing.T) {
	all := qwen35SequenceKQuantPipelines{q6k: true, q5k: true, q3k: true}
	for _, dt := range []Dtype{F32, Q8_0, Q4_K, Q2_K, Q6_K, Q5_K, Q3_K} {
		if err := qwen35VulkanSequenceMatrixDtypeErr(dt, all); err != nil {
			t.Errorf("%s with every pipeline: got %v, want admitted", dt, err)
		}
	}
	for _, tc := range []struct {
		name string
		dt   Dtype
		have qwen35SequenceKQuantPipelines
	}{
		{"Q6_K_without_pipeline", Q6_K, qwen35SequenceKQuantPipelines{q5k: true, q3k: true}},
		{"Q5_K_without_pipeline", Q5_K, qwen35SequenceKQuantPipelines{q6k: true, q3k: true}},
		{"Q3_K_without_pipeline", Q3_K, qwen35SequenceKQuantPipelines{q6k: true, q5k: true}},
		{"F16", F16, all},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := qwen35VulkanSequenceMatrixDtypeErr(tc.dt, tc.have); !errors.Is(err, ErrQwen35SequenceUnsupportedMatrixDtype) {
				t.Fatalf("got %v, want errors.Is ErrQwen35SequenceUnsupportedMatrixDtype", err)
			}
		})
	}
	wrapped := &Qwen35SequenceError{Stage: "tensor-preflight", Layer: 0, Cause: qwen35VulkanSequenceMatrixDtypeErr(F16, all)}
	if !errors.Is(wrapped, ErrQwen35SequenceUnsupportedMatrixDtype) {
		t.Fatal("Qwen35SequenceError must unwrap to the dtype sentinel")
	}
}
